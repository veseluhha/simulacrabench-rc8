import os
import sys
import time

import numpy as np

from .select import distributions, fit, gpu
from .post import meta_of, refine
from .stack import bayes_act, blend, item_weights, logpool, logpool_weights, mean_logp, temper, temperature, truth_probs

START = time.time()
SAFETY = 0.8
HARD = 0.92
PAIRFOLDS = True
SHARES = {19846: 0.12, 10349: 0.68, 6927: 0.20}
STRENGTH = 100.0
SHIFT_CAP = 20.0
POSITION = ()
POOL = "auto"
BAYES = True
REFINE = True
BOOST = {2: ("tabpfn_p16", "tabicl_p32")}
BOOST_MIN = 5
SUBPOP = False
GATEFOCUS = False
FOCUSFIT = False
FOCUS_SHARE = 0.95
SUBPOP_PAIRS = "tabpfn_num:tabpfn_sub,backoff2:backoff2_sub,tabm:tabm_sub"
PANEL = ("panel_copy", "panel_soft", "tabpfn_panel_in")
PANEL_FILL = ("tabpfn_num", "tabm", "backoff2", "marginal")
ORDER = ("backoff2", "tabpfn_panel_in", "panel_copy", "panel_soft", "tabpfn_num", "tabicl", "tabicl4", "tabicl_struct", "tabm_direct", "tabm")
PLAN = {1: (("backoff2", "tabm", "tabm_direct", "tabicl4", "tabpfn_num"), 1, 0.2),
        2: (("backoff2", "tabm", "tabm_direct", "tabicl", "tabicl_struct", "tabpfn_num"), 3, None)}
_done = set()
_cache = {}


def log(message):
    if os.environ.get("SB_DEBUG"):
        print("[pipe] " + message, file=sys.stderr, flush=True)


def phase_of(inst, schema, n_hidden):
    split = schema.get("split") or {}
    total = len(inst.X)
    if total == (schema.get("dataset") or {}).get("n_rows") or n_hidden == split.get("n_test"):
        return 2
    return 2 if n_hidden > 0.13 * total else 1


def allowance(schema, total):
    n = (schema.get("dataset") or {}).get("n_rows")
    if n in SHARES:
        mine = SHARES[n]
        rest = sum(v for k, v in SHARES.items() if k not in _done and k != n)
    else:
        mine, rest = 1.0, max(0, 2 - len(_done))
    left = START + SAFETY * total - time.time()
    return max(0.0, left * mine / (mine + rest))


def folds(seen, k, frac, seed=0, partner=None):
    rng = np.random.default_rng(seed)
    order = rng.permutation(seen)
    if partner is None:
        if k == 1:
            return [order[:int(round(frac * len(order)))]]
        return np.array_split(order, k)
    root = np.arange(len(partner))

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    visible = np.zeros(len(partner), bool)
    visible[seen] = True
    for r in seen:
        p = partner[r]
        if p >= 0 and visible[p]:
            root[find(r)] = find(p)
    comp = np.array([find(r) for r in range(len(partner))])
    if k == 1:
        size, taken, hold = int(round(frac * len(order))), set(), []
        for r in order:
            if len(hold) == size:
                break
            if comp[r] not in taken:
                taken.add(comp[r])
                hold.append(r)
        return [np.array(hold)]
    groups = {}
    for r in order:
        groups.setdefault(comp[r], []).append(r)
    parts = [[] for _ in range(k)]
    for members in groups.values():
        slots = sorted(range(k), key=lambda f: len(parts[f]))
        for i, r in enumerate(members):
            parts[slots[i % k]].append(r)
    return [np.array(p) for p in parts]


def estimate(name, inst, passes, spent):
    if name in spent:
        return spent[name]
    scale = len(inst.scored) * passes * max(1.0, len(inst.seen) / 8000)
    if name.startswith("tabicl"):
        return spent.get("tabicl", spent.get("tabicl4", (0.15 if name == "tabicl4" else 0.3) * scale))
    if name.startswith("tabpfn_panel") and "tabpfn_num" in spent:
        return 1.25 * spent["tabpfn_num"]
    if name.startswith("tabpfn"):
        return 2.0 * spent.get("tabicl", 0.3 * scale)
    if name.startswith("tabm"):
        return spent.get("tabm", 20.0 * passes * max(1.0, len(inst.seen) * len(inst.scored) / 1e6))
    return 3.0 * passes


def shift_weights(inst, rows, target):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore").fit(inst.X)
    y = np.r_[np.zeros(len(inst.seen)), np.ones(len(rows))]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(enc.transform(inst.X[np.r_[inst.seen, rows]]), y)
    p = clf.predict_proba(enc.transform(inst.X[target]))[:, 1]
    rw = np.clip(p / np.maximum(1.0 - p, 1e-9), 0.0, SHIFT_CAP)
    rw = rw / rw.mean()
    log("shift weights: ess %.2f, max %.1f" % (rw.sum() ** 2 / (rw ** 2).sum() / len(rw), rw.max()))
    return rw


def subpopulation(inst, rows, share=None, gap=0.2, least=500):
    share = float(os.environ.get("SB_FOCUS_SHARE", FOCUS_SHARE)) if share is None else share
    best = None
    for g in range(inst.X.shape[1]):
        levels, counts = np.unique(inst.X[rows, g], return_counts=True)
        top = int(levels[counts.argmax()])
        hidden = counts.max() / len(rows)
        visible = float(np.mean(inst.X[inst.seen, g] == top))
        n = int(np.sum(inst.X[inst.seen, g] == top))
        if hidden >= share and hidden - visible >= gap and n >= least and (best is None or hidden - visible > best[0]):
            best = (hidden - visible, g, top)
    return None if best is None else (best[1], best[2])


def parent_items(inst, least):
    index = {s: j for j, s in enumerate(inst.scored)}
    count = np.zeros(len(inst.scored), int)
    for s in inst.scored:
        seen, gate = set(), inst.gate.get(s)
        while gate and gate["parent"] in index and gate["parent"] not in seen:
            seen.add(gate["parent"])
            count[index[gate["parent"]]] += 1
            gate = inst.gate.get(gate["parent"])
    return [j for j in range(len(inst.scored)) if count[j] >= least]


def pool_cost(inst, n, c):
    return 2e-7 * n * c * float(np.sum(inst.widths)) + 1.0


def choose_pool(oof, Y):
    half = np.random.default_rng(1).random(len(Y)) < 0.5
    score = {"linear": 0.0, "log": 0.0}
    cf = {k: [np.zeros_like(p) for p in oof[0]] for k in score}
    for fit_rows, ev_rows in ((half, ~half), (~half, half)):
        part = [[p[fit_rows] for p in cand] for cand in oof]
        held = [[p[ev_rows] for p in cand] for cand in oof]
        W, _ = item_weights(truth_probs(part, Y[fit_rows]), STRENGTH)
        T = temperature(blend(part, W), Y[fit_rows])
        preds = {"linear": temper(blend(held, W), T), "log": logpool(held, logpool_weights(part, Y[fit_rows]))}
        for k, pr in preds.items():
            score[k] += mean_logp(bayes_act(pr), Y[ev_rows]) * ev_rows.sum()
            for j, q in enumerate(pr):
                cf[k][j][ev_rows] = q
    log("pool scores %s" % {k: round(v / len(Y), 5) for k, v in score.items()})
    pick = max(score, key=score.get)
    return pick, cf[pick]


def cached(name, inst, rows, phase, k, compute, parts=()):
    folder = os.environ.get("SB_CACHE")
    if not folder:
        return compute()
    import hashlib
    import pickle
    from pathlib import Path
    h = hashlib.blake2b(digest_size=12)
    for a in (inst.X, inst.Y, rows):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(("%d %d %s" % (phase, k, name)).encode())
    if pairfolds() and getattr(inst, "panel", None) is not None and len(parts):
        h.update(np.ascontiguousarray(np.concatenate(parts)).tobytes())
    panel = getattr(inst, "panel", None)
    if panel is not None and "panel" in name:
        h.update(("v2 %s" % panel["levels"]).encode())
        h.update(np.ascontiguousarray(panel["partner"]).tobytes())
    path = Path(folder) / (h.hexdigest() + "_" + name + ".pkl")
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    from . import tfm
    hits = tfm.HITS[0]
    out = compute()
    if tfm.HITS[0] != hits:
        return out
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".tmp", "wb") as fh:
        pickle.dump(out, fh)
    os.replace(str(path) + ".tmp", path)
    return out


def pairfolds():
    return os.environ.get("SB_PAIRFOLDS", "1" if PAIRFOLDS else "0") == "1"


def pfn_ok():
    if "pfn" not in _cache:
        try:
            from .tfm import pfn_available
            _cache["pfn"] = pfn_available()
        except Exception:
            _cache["pfn"] = False
    return _cache["pfn"]


def release():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run(inst, schema, rows):
    phase = phase_of(inst, schema, len(rows))
    total = float(os.environ.get("SB_TOTAL") or (3600 if phase == 2 else 900))
    deadline = time.time() + allowance(schema, total)
    from . import tfm
    tfm.STOP = START + float(os.environ.get("SB_HARD", HARD)) * total
    position = POSITION if os.environ.get("SB_POSITION") is None else tuple(int(x) for x in os.environ["SB_POSITION"].split(",") if x)
    if phase in position:
        inst.add_position()
        log("position feature on")
    if os.environ.get("SB_ROSTER") == "1" and inst.add_roster():
        log("roster features on")
    names, k, frac = PLAN[phase]
    if os.environ.get("SB_PLAN"):
        names = tuple(os.environ["SB_PLAN"].split(","))
    if os.environ.get("SB_FOLDS"):
        k = int(os.environ["SB_FOLDS"])
    if not gpu() and not os.environ.get("SB_CACHE"):
        names = ("marginal", "backoff2")
    if not pfn_ok():
        names = tuple(n for n in names if not n.startswith("tabpfn"))
    panel = tuple(n for n in os.environ.get("SB_PANEL", ",".join(PANEL)).split(",") if n)
    if panel:
        from .panel import link
        try:
            inst.panel = link(inst, rows, force=bool(os.environ.get("SB_PANEL_FORCE")), log=log, soft="panel_soft" in panel)
        except Exception as exc:
            log("panel failed: %r" % (exc,))
            inst.panel = None
            release()
        if inst.panel is not None:
            found = inst.panel["partner"] >= 0
            names = tuple(names) + tuple(n for n in panel if gpu() or os.environ.get("SB_CACHE") or not n.startswith("tabpfn"))
            if phase == 1 and "tabpfn_panel_in" in names:
                names = tuple(n for n in names if n != "tabpfn_num")
            log("panel on column %d, levels %s: %d rows matched, %d of them hidden" % (
                inst.panel["column"], inst.panel["levels"], found.sum(), found[rows].sum()))
    names = tuple(sorted(names, key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER)))
    from . import select as chooser
    chooser.GATE_FOCUS = subpopulation(inst, rows) if os.environ.get("SB_GATEFOCUS", "1" if GATEFOCUS else "0") == "1" else None
    if chooser.GATE_FOCUS is not None:
        log("gate rates from column %d level %d" % chooser.GATE_FOCUS)
    pairs = inst.panel["partner"] if getattr(inst, "panel", None) is not None and pairfolds() else None
    parts = folds(inst.seen, k, frac, partner=pairs)
    oof_rows = np.concatenate(parts)
    oof, test, spent = {}, {}, {}
    for name in names:
        need = estimate(name, inst, len(parts) + 1, spent)
        if test and time.time() + need > deadline:
            log("skip %s: needs ~%.0fs, %.0fs left" % (name, need, deadline - time.time()))
            continue
        t = time.time()
        try:
            def compute():
                pieces = [distributions(inst, fit(inst, name, np.setdiff1d(inst.seen, part)), part) for part in parts]
                return [np.concatenate([p[j] for p in pieces]) for j in range(len(inst.scored))], distributions(inst, fit(inst, name, inst.seen), rows)
            oof[name], test[name] = cached(name, inst, rows, phase, k, compute, parts)
            if "panel" not in name and any(np.isnan(p).any() for p in oof[name] + test[name]):
                base = next(b for b in PANEL_FILL if b in test and b != name)
                oof[name] = [np.where(np.isnan(p), q, p) for p, q in zip(oof[name], oof[base])]
                test[name] = [np.where(np.isnan(p), q, p) for p, q in zip(test[name], test[base])]
        except Exception as exc:
            log("%s failed: %r" % (name, exc))
            oof.pop(name, None)
            release()
            continue
        spent[name] = time.time() - t
        release()
        log("%s done in %.1fs" % (name, spent[name]))
    base = next((b for b in PANEL_FILL if b in test), None)
    for name in [n for n in test if "panel" in n]:
        if base is None:
            oof.pop(name)
            test.pop(name)
            continue
        if name == "panel_soft":
            fill = lambda P, B: [p + np.clip(1.0 - p.sum(1, keepdims=True), 0.0, 1.0) * b for p, b in zip(P, B)]
            oof[name], test[name] = fill(oof[name], oof[base]), fill(test[name], test[base])
        if any(np.isnan(p).any() for p in oof[name] + test[name]):
            oof[name] = [np.where(np.isnan(p), q, p) for p, q in zip(oof[name], oof[base])]
            test[name] = [np.where(np.isnan(p), q, p) for p, q in zip(test[name], test[base])]
    boost = BOOST.get(phase, ()) if os.environ.get("SB_BOOST", "0") == "1" else ()
    parents = parent_items(inst, BOOST_MIN) if boost and "tabpfn_num" in test else []
    for name in boost if parents else ():
        need = estimate(name, inst, len(parts) + 1, spent) * 4.0 * len(parents) / len(inst.scored)
        if time.time() + need > deadline:
            log("skip %s: needs ~%.0fs, %.0fs left" % (name, need, deadline - time.time()))
            continue
        t = time.time()
        try:
            pieces = [distributions(inst, fit(inst, name, np.setdiff1d(inst.seen, part), only=set(parents)), part) for part in parts]
            full = distributions(inst, fit(inst, name, inst.seen, only=set(parents)), rows)
            oof[name] = [np.concatenate([p[j] for p in pieces]) if pieces[0][j] is not None else oof["tabpfn_num"][j] for j in range(len(inst.scored))]
            test[name] = [full[j] if full[j] is not None else test["tabpfn_num"][j] for j in range(len(inst.scored))]
            log("%s on %d parents done in %.1fs" % (name, len(parents), time.time() - t))
        except Exception as exc:
            log("%s failed: %r" % (name, exc))
            oof.pop(name, None)
            test.pop(name, None)
        release()
    pairs = [pair.split(":") for pair in os.environ.get("SB_SUBPOP_PAIRS", SUBPOP_PAIRS).split(",") if pair]
    focus = subpopulation(inst, rows) if os.environ.get("SB_SUBPOP", "1" if SUBPOP else "0") == "1" else None
    for base, expert in pairs if focus is not None else ():
        if base not in test:
            continue
        need = 2.0 * spent.get(base, 0.0)
        if time.time() + need > deadline:
            log("skip %s: needs ~%.0fs, %.0fs left" % (expert, need, deadline - time.time()))
            continue
        t = time.time()
        try:
            inner = inst.X[:, focus[0]] == focus[1]
            pieces = []
            for part in parts:
                train = np.setdiff1d(inst.seen, part)
                pieces.append(distributions(inst, fit(inst, expert, train[inner[train]], focus=focus), part))
            full = distributions(inst, fit(inst, expert, inst.seen[inner[inst.seen]], focus=focus), rows)
            stacked = [np.concatenate([p[j] for p in pieces]) if pieces[0][j] is not None else None for j in range(len(inst.scored))]
            oof[expert] = [np.where(np.isnan(stacked[j]), oof[base][j], stacked[j]) if stacked[j] is not None else oof[base][j] for j in range(len(inst.scored))]
            test[expert] = [np.where(np.isnan(full[j]), test[base][j], full[j]) if full[j] is not None else test[base][j] for j in range(len(inst.scored))]
            log("%s on column %d level %d (%d visible rows) done in %.1fs" % (expert, focus[0], focus[1], int(inner[inst.seen].sum()), time.time() - t))
        except Exception as exc:
            log("%s failed: %r" % (expert, exc))
            oof.pop(expert, None)
            test.pop(expert, None)
        release()
    if not test:
        return bayes_act(distributions(inst, fit(inst, "marginal", inst.seen), rows))
    kept = list(test)
    if len(kept) == 1:
        return bayes_act(test[kept[0]])
    target = subpopulation(inst, rows) if os.environ.get("SB_FOCUSFIT", "1" if FOCUSFIT else "0") == "1" else None
    if target is not None:
        sel = inst.X[oof_rows, target[0]] == target[1]
        if sel.sum() >= 300:
            oof = {c: [p[sel] for p in v] for c, v in oof.items()}
            oof_rows = oof_rows[sel]
            log("stacking on %d out-of-fold rows like the hidden ones" % len(oof_rows))
    panel = getattr(inst, "panel", None)
    if panel is not None and os.environ.get("SB_PANELSTACK") == "1":
        has_o, has_h = panel["partner"][oof_rows] >= 0, panel["partner"][rows] >= 0
        if min(has_o.sum(), (~has_o).sum()) >= 300 and has_h.any() and (~has_h).any():
            out = [np.zeros((len(rows), w)) for w in inst.widths]
            for go, gh in ((has_o, has_h), (~has_o, ~has_h)):
                part = aggregate(inst, phase, {c: [q[go] for q in v] for c, v in oof.items()}, {c: [q[gh] for q in v] for c, v in test.items()},
                                 kept, oof_rows[go], rows[gh], deadline)
                for j, q in enumerate(part):
                    out[j][gh] = q
            log("stacked separately for %d / %d hidden rows with / without a partner" % (has_h.sum(), (~has_h).sum()))
            return out
    return aggregate(inst, phase, oof, test, kept, oof_rows, rows, deadline)


def aggregate(inst, phase, oof, test, kept, oof_rows, rows, deadline):
    Yo = inst.Y[oof_rows]
    rw = None
    if os.environ.get("SB_SHIFT"):
        try:
            rw = shift_weights(inst, rows, oof_rows)
        except Exception as exc:
            log("shift weights failed: %r" % (exc,))
    P = truth_probs([oof[c] for c in kept], Yo)
    W, shared = item_weights(P, STRENGTH, rw)
    T = temperature(blend([oof[c] for c in kept], W), Yo, rw=rw)
    log("phase %d, weights %s, T %.2f" % (phase, dict(zip(kept, np.round(shared, 3).tolist())), T))
    out = temper(blend([test[c] for c in kept], W), T)
    pool, cf = os.environ.get("SB_POOL", POOL), None
    if pool != "linear" and time.time() + 4 * pool_cost(inst, len(oof_rows), len(kept)) < deadline:
        try:
            t = time.time()
            if pool == "auto":
                pool, cf = choose_pool([oof[c] for c in kept], Yo)
            if pool == "log":
                out = logpool([test[c] for c in kept], logpool_weights([oof[c] for c in kept], Yo))
            log("pool %s in %.1fs" % (pool, time.time() - t))
        except Exception as exc:
            log("log pool failed: %r" % (exc,))
    if cf is not None and os.environ.get("SB_REFINE", "1" if REFINE else "0") == "1" and time.time() + 6 * pool_cost(inst, len(oof_rows), len(kept)) < deadline:
        try:
            t = time.time()
            out = refine([oof[c] for c in kept], Yo, [test[c] for c in kept], cf, out, meta_of(inst, oof_rows, rows))
            log("refine in %.1fs" % (time.time() - t))
        except Exception as exc:
            log("refine failed: %r" % (exc,))
    return bayes_act(out) if os.environ.get("SB_BAYES", "1" if BAYES else "0") == "1" else out


def warm():
    if not gpu():
        return
    try:
        from .tfm import warm as warm_models
        warm_models(pfn=any(n.startswith("tabpfn") for n in PLAN[2][0] + PLAN[1][0]))
    except Exception as exc:
        log("warm failed: %r" % (exc,))


def finish(schema):
    _done.add((schema.get("dataset") or {}).get("n_rows"))
