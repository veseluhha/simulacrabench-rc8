import numpy as np

from .tfm import TabPFN


def round_columns(inst, rows, share=0.95, least=500):
    import re
    word = re.compile(r"\b(round|wave)s?\b", re.I)
    out = []
    for g, name in enumerate(inst.given):
        width = len(inst.options[name])
        levels = np.unique(inst.X[:, g][inst.X[:, g] < width])
        if len(levels) != 2 or not len(rows) or min(np.sum(inst.X[inst.seen, g] == l) for l in levels) < least:
            continue
        top = int(levels[np.argmax([np.sum(inst.X[rows, g] == l) for l in levels])])
        one_way = np.mean(inst.X[rows, g] == top) >= share
        if one_way or word.search(name) or word.search(inst.questions.get(name, "")):
            out.append((g, top, int(levels[levels != top][0]), one_way))
    return out


def link(inst, rows, factor=1.8, margin=0.03, force=False, log=None, soft=False, k=5):
    best = None
    for g, top, other, one_way in round_columns(inst, rows):
        small, large = sorted((top, other), key=lambda l: np.sum(inst.X[:, g] == l))
        cross, null, hid = evidence(inst, g, small, large)
        if log:
            log("panel evidence on column %d: exact matches across levels %.3f (hidden rows %s), within %.3f" % (
                g, cross, "-" if hid is None else "%.3f" % hid, null))
        need = max(factor * null, null + margin)
        if (force or (cross >= need and (hid is None or hid >= need))) and (best is None or cross - null > best[0]):
            best = (cross - null, g, top, other, small, large, one_way)
    if best is None:
        return None
    _, g, top, other, small, large, one_way = best
    found = match(inst, g, small, large, everyone=True, dense=soft)
    n = len(inst.X)
    seen = np.zeros(n, bool)
    seen[inst.seen] = True
    partner, dist, gap = np.full(n, -1), np.full(n, 99.0), np.zeros(n)
    for t in np.flatnonzero(found["partner"] >= 0):
        s = found["partner"][t]
        if seen[s]:
            partner[t], dist[t], gap[t] = s, found["dist"][t], found["gap"][t]
        if seen[t]:
            partner[s], dist[s], gap[s] = t, found["dist"][t], found["gap"][t]
    panel = {"partner": partner, "dist": dist, "gap": gap, "column": g, "level": top, "levels": [top] if one_way else [top, other]}
    if soft:
        W = posteriors(found["S"])
        del found["S"]
        cand, wts = np.full((n, k), -1), np.zeros((n, k), np.float32)
        for side, Wside, other_rows in ((found["targets"], W, found["sources"]), (found["sources"], W.T, found["targets"])):
            Wv = np.where(seen[other_rows][None, :], Wside, 0.0)
            kk = min(k, Wv.shape[1])
            idx = np.argpartition(-Wv, kk - 1, axis=1)[:, :kk]
            vals = np.take_along_axis(Wv, idx, 1)
            cand[side, :kk] = np.where(vals > 1e-4, other_rows[idx], -1)
            wts[side, :kk] = np.where(vals > 1e-4, vals, 0.0)
        panel.update(cand=cand, w=wts)
        if log:
            log("panel posteriors: expected pairs %.0f, mean top weight %.3f" % (W.sum(), float(wts[:, 0][cand[:, 0] >= 0].mean())))
    return panel


def evidence(inst, g, target, source, sample=2000, chunk=256):
    cols = [k for k in range(inst.X.shape[1]) if k != g]
    X, weight = inst.X[:, cols], np.log(inst.cards[cols].astype(np.float32))
    sources = np.intersect1d(np.flatnonzero(inst.X[:, g] == source), inst.seen)
    rng = np.random.default_rng(0)
    level = np.flatnonzero(inst.X[:, g] == target)
    targets = rng.choice(level, min(len(level), sample), replace=False)
    null = rng.choice(sources, min(len(sources), sample), replace=False)
    hidden = np.setdiff1d(level, inst.seen)
    hidden = rng.choice(hidden, min(len(hidden), sample), replace=False)

    def exact(rows, own):
        hits = []
        for i in range(0, len(rows), chunk):
            t = rows[i:i + chunk]
            d = ((X[t][:, None, :] != X[sources][None, :, :]) * weight).sum(-1)
            if own:
                d[t[:, None] == sources[None, :]] = np.inf
            hits.append(d.min(1) < 1e-6)
        return float(np.mean(np.concatenate(hits)))

    return exact(targets, False), exact(null, True), exact(hidden, False) if len(hidden) >= 50 else None


def match(inst, g, target, source, iters=3, top=5, chunk=128, everyone=False, dense=False):
    import re
    band = re.compile(r"^\s*\d+(\s*[-–]\s*\d+|\+)?\s*$")
    X = inst.X
    cols = [k for k in range(X.shape[1]) if k != g]
    targets = np.flatnonzero(X[:, g] == target)
    sources = np.flatnonzero(X[:, g] == source)
    if not everyone:
        sources = np.intersect1d(sources, inst.seen)
    Xt, Xs = X[targets][:, cols], X[sources][:, cols]
    K, n, width = len(cols), len(sources), int(X.max()) + 2
    fe, fn = np.zeros(Xt.shape), np.zeros(Xt.shape)
    ons = np.zeros(Xs.shape, bool)
    near = np.zeros(K, bool)
    for k, c in enumerate(cols):
        flags = [bool(band.match(str(x))) for x in inst.values[inst.given[c]]]
        scale = np.zeros(width, bool)
        if sum(flags) >= 3:
            scale[:len(flags)], near[k] = flags, True
        fr = np.bincount(Xs[:, k], minlength=width)[:width] / n
        x = Xt[:, k]
        fe[:, k] = np.maximum(fr[x], 1.0 / n)
        if near[k]:
            lo, hi = np.maximum(x - 1, 0), np.minimum(x + 1, width - 1)
            side = np.where(scale[lo] & (x > 0), fr[lo], 0.0) + np.where(scale[hi] & (x + 1 < width), fr[hi], 0.0)
            fn[:, k] = np.where(scale[x], np.maximum(side, 1.0 / n), 0.0)
            ons[:, k] = scale[Xs[:, k]]
    me, mn = np.full(K, 0.85), np.where(near, 0.08, 0.0)
    for _ in range(iters):
        A = np.log(me / fe)
        N = np.where(fn > 0, np.log(np.maximum(mn, 1e-6) / np.maximum(fn, 1e-9)), 0.0)
        D = np.log(np.maximum(1 - me - mn, 1e-3) / np.maximum(1 - fe - fn, 1e-6))
        last = (A, N, D)
        cand, gap = [], np.zeros(len(targets))
        for i in range(0, len(targets), chunk):
            t = slice(i, i + chunk)
            a, b = Xt[t][:, None, :], Xs[None, :, :]
            eq = a == b
            nr = (np.abs(a - b) == 1) & (fn[t][:, None, :] > 0) & ons[None, :, :]
            s = D[t].sum(1)[:, None] + np.einsum("tsk,tk->ts", eq, A[t] - D[t]) + np.einsum("tsk,tk->ts", nr, N[t] - D[t])
            k = min(top, s.shape[1])
            best = np.argpartition(-s, k - 1, axis=1)[:, :k]
            vals = -np.sort(-np.take_along_axis(s, best, 1), 1)
            gap[t] = vals[:, 0] - vals[:, 1] if k > 1 else 0.0
            for r in range(len(best)):
                for c in best[r]:
                    cand.append((-float(s[r, c]), i + r, int(c)))
        cand.sort()
        pt, score, used = np.full(len(targets), -1), np.full(len(targets), -np.inf), set()
        for v, r, c in cand:
            if pt[r] < 0 and c not in used:
                pt[r], score[r] = c, -v
                used.add(c)
        ok = (pt >= 0) & (score > 0)
        if ok.sum() < 30:
            break
        x, y = Xt[ok], Xs[pt[ok]]
        me = np.clip((x == y).mean(0), 0.3, 0.995)
        mn = np.where(near, np.clip((np.abs(x - y) == 1).mean(0), 0.005, 0.5), 0.0)
    partner, dist, gaps = np.full(len(X), -1), np.full(len(X), 99.0), np.zeros(len(X))
    hit = pt >= 0
    partner[targets[hit]], dist[targets[hit]], gaps[targets[hit]] = sources[pt[hit]], -score[hit], gap[hit]
    out = {"partner": partner, "dist": dist, "gap": gaps, "column": g, "level": target}
    if dense:
        A, N, D = last
        S = np.empty((len(targets), len(sources)), np.float32)
        for i in range(0, len(targets), chunk):
            t = slice(i, i + chunk)
            a, b = Xt[t][:, None, :], Xs[None, :, :]
            eq = a == b
            nr = (np.abs(a - b) == 1) & (fn[t][:, None, :] > 0) & ons[None, :, :]
            S[t] = D[t].sum(1)[:, None] + np.einsum("tsk,tk->ts", eq, A[t] - D[t]) + np.einsum("tsk,tk->ts", nr, N[t] - D[t])
        out.update(S=S, targets=targets, sources=sources)
    return out


def sinkhorn(S, tau, iters=40):
    try:
        import torch
        if torch.cuda.is_available():
            K = torch.exp(torch.clamp(torch.as_tensor(S, device="cuda") - float(tau), max=50.0))
            b = torch.ones(K.shape[1], device="cuda")
            for _ in range(iters):
                a = 1.0 / (1.0 + K @ b)
                b = 1.0 / (1.0 + a @ K)
            W = (a[:, None] * K * b[None, :]).cpu().numpy()
            del K
            torch.cuda.empty_cache()
            return W
    except Exception:
        pass
    K = np.exp(np.minimum(S - np.float32(tau), 50.0)).astype(np.float32)
    b = np.ones(K.shape[1], np.float32)
    for _ in range(iters):
        a = 1.0 / (1.0 + K @ b)
        b = 1.0 / (1.0 + a @ K)
    return a[:, None] * K * b[None, :]


def posteriors(S, iters=5):
    n_a, n_b = S.shape
    L = 0.5 * min(n_a, n_b)
    for _ in range(iters):
        tau = np.log(n_a) + np.log(n_b) - np.log(L)
        W = sinkhorn(S, tau, 30)
        L = float(np.clip(W.sum(), 10.0, min(n_a, n_b) - 1.0))
    return sinkhorn(S, np.log(n_a) + np.log(n_b) - np.log(L), 60)


def parent_index(inst):
    index = {s: j for j, s in enumerate(inst.scored)}
    return [index.get((inst.gate.get(s) or {}).get("parent"), -1) for s in inst.scored]


class PanelCopy:
    def __init__(self, alpha=2.0, bins=3):
        self.alpha, self.bins = alpha, bins

    def attach(self, inst, mode):
        self.inst = inst

    def fit(self, X, T, widths, cards):
        self.T, self.widths = T, list(widths)
        return self

    def predict(self, Xq):
        inst, panel = self.inst, self.inst.panel
        rows, query = np.asarray(self.rows), np.asarray(self.query)
        own, theirs = panel["partner"][rows], panel["partner"][query]
        groups = []
        for level in panel["levels"]:
            lt = (inst.X[rows, panel["column"]] == level) & (own >= 0)
            lq = (inst.X[query, panel["column"]] == level) & (theirs >= 0)
            d = panel["dist"][rows][lt]
            edges = np.quantile(d, np.linspace(0, 1, self.bins + 1)[1:-1]) if len(d) >= 30 * self.bins else np.array([])
            bt, bq = np.searchsorted(edges, panel["dist"][rows]), np.searchsorted(edges, panel["dist"][query])
            groups += [(lt & (bt == b), lq & (bq == b)) for b in range(len(edges) + 1)]
        out = []
        for j, w in enumerate(self.widths):
            y = self.T[:, j]
            ok = y >= 0
            m = np.bincount(y[ok], minlength=w)[:w] + 0.5
            m = m / m.sum()
            q = np.full((len(query), w), np.nan)
            for st, sq in groups:
                sel = st & ok
                C = np.zeros((w, w))
                np.add.at(C, (inst.Y[own[sel], j], y[sel]), 1.0)
                P = (C + self.alpha * m) / (C.sum(1, keepdims=True) + self.alpha)
                q[sq] = P[inst.Y[theirs[sq], j]]
            out.append(q)
        return out


class SoftPanelCopy(PanelCopy):
    def predict(self, Xq):
        inst, panel = self.inst, self.inst.panel
        rows, query = np.asarray(self.rows), np.asarray(self.query)
        inq = np.zeros(len(inst.X), bool)
        inq[query] = True

        def masked(r):
            c, w = panel["cand"][r].copy(), panel["w"][r].copy()
            bad = (c >= 0) & inq[np.maximum(c, 0)]
            c[bad], w[bad] = -1, 0.0
            return c, w

        ct, wt = masked(rows)
        cq, wq = masked(query)
        lt, lq = inst.X[rows, panel["column"]], inst.X[query, panel["column"]]
        out = []
        for j, width in enumerate(self.widths):
            y = self.T[:, j]
            ok = y >= 0
            m = np.bincount(y[ok], minlength=width)[:width] + 0.5
            m = m / m.sum()
            q = np.zeros((len(query), width))
            for level in panel["levels"]:
                st, sq = (lt == level) & ok, lq == level
                c, w = ct[st], wt[st]
                use = c >= 0
                C = np.zeros((width, width))
                np.add.at(C, (inst.Y[np.maximum(c, 0), j][use], np.repeat(y[st][:, None], c.shape[1], 1)[use]), w[use])
                P = (C + self.alpha * m) / (C.sum(1, keepdims=True) + self.alpha)
                c, w = cq[sq], wq[sq]
                q[sq] = (P[inst.Y[np.maximum(c, 0), j]] * (w * (c >= 0))[:, :, None]).sum(1)
            out.append(np.minimum(q, 1.0))
        return out


class PanelTabPFN(TabPFN):
    def __init__(self, inside=False, extra=None, **kwargs):
        super().__init__(**kwargs)
        self.inside, self.extra = inside, extra

    def attach(self, inst, mode):
        super().attach(inst, mode)
        self.inst = inst
        self.parents = parent_index(inst)

    def fit(self, X, T, widths, cards):
        self.cards, self.T, self.widths = np.asarray(cards), T, list(widths)
        return self

    def profile(self, dims=8):
        inst = self.inst
        seen = inst.seen
        offsets = np.concatenate([[0], np.cumsum(inst.widths)])
        onehot = np.zeros((len(seen), offsets[-1]), np.float32)
        for j in range(len(inst.widths)):
            onehot[np.arange(len(seen)), offsets[j] + inst.Y[seen, j]] = 1.0
        center = onehot.mean(0)
        basis = np.linalg.svd(onehot - center, full_matrices=False)[2][:dims].T
        out = np.zeros((len(inst.X), dims))
        out[seen] = (onehot - center) @ basis
        return out

    def related(self, train, y, j, count=4):
        inst = self.inst
        p = self.inst.panel["partner"][train]
        has = (p >= 0) & (y >= 0)
        if has.sum() < 50:
            return []
        P, t = inst.Y[p[has]], y[has]
        w = self.widths[j]
        scores = np.full(P.shape[1], -1.0)
        pt = np.bincount(t, minlength=w) / len(t)
        for i in range(P.shape[1]):
            if i == j or i == self.parents[j]:
                continue
            wi = int(P[:, i].max()) + 1
            joint = np.bincount(P[:, i] * w + t, minlength=wi * w).reshape(wi, w) / len(t)
            pi = joint.sum(1, keepdims=True)
            nz = joint > 0
            scores[i] = (joint[nz] * np.log(joint[nz] / (pi @ pt[None, :])[nz])).sum()
        return list(np.argsort(-scores)[:count])

    def features(self, rows, j, related=()):
        inst, panel = self.inst, self.inst.panel
        p = panel["partner"][rows]
        has = p >= 0
        cols = [inst.X[rows].astype(np.float64)]
        for k in [j, self.parents[j]] + list(related):
            cols.append(np.where(has, inst.Y[np.maximum(p, 0), k], self.widths[k]) if k >= 0 else np.zeros(len(rows)))
        cols += [panel["dist"][rows], panel["gap"][rows]]
        if self.extra == "svd":
            cols.append(np.where(has[:, None], self.prof[np.maximum(p, 0)], 0.0))
        return np.column_stack(cols)

    def predict(self, Xq):
        import os
        import torch
        from tabpfn import TabPFNClassifier
        from .tfm import _patch_pfn, pfn_path
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        _patch_pfn()
        path, device = pfn_path(self.variant), "cuda" if torch.cuda.is_available() else "cpu"
        inst, panel = self.inst, self.inst.panel
        if self.extra == "svd":
            self.prof = self.profile()
        G = inst.X.shape[1]

        def fit_predict(train, T, query, j, w):
            y = T[:, j]
            ok = y >= 0
            rel = self.related(train[ok], y[ok], j) if self.extra == "mi" else []
            cats = [k for k in range(G) if not (self.numeric and self.ordinal[k])] + list(range(G, G + 2 + len(rel)))
            q = np.full((len(query), w), 1e-4)
            if len(np.unique(y[ok])) < 2:
                q += np.bincount(y[ok], minlength=w)[None, :w] + 0.5
            else:
                clf = TabPFNClassifier(model_path=path, device=device, n_estimators=self.n_estimators,
                                       categorical_features_indices=cats, ignore_pretraining_limits=True,
                                       random_state=self.seed)
                clf.fit(self.features(train[ok], j, rel), y[ok])
                q[:, np.asarray(clf.classes_, dtype=int)] += clf.predict_proba(self.features(query, j, rel))
            return q / q.sum(1, keepdims=True)

        return self.by_level(fit_predict)

    def by_level(self, fit_predict):
        inst, panel = self.inst, self.inst.panel
        train, query = np.asarray(self.rows), np.asarray(self.query)
        if self.inside:
            parts = [(inst.X[train, panel["column"]] == level, inst.X[query, panel["column"]] == level) for level in panel["levels"]]
        else:
            parts = [(np.ones(len(train), bool), np.ones(len(query), bool))]
        from .tfm import stopped
        out = []
        for j, w in enumerate(self.widths):
            p = np.full((len(query), w), np.nan)
            for keep, within in parts:
                if within.any() and keep.any() and not stopped():
                    p[within] = fit_predict(train[keep], self.T[keep], query[within], j, w)
            out.append(p)
        return out


class PanelTabICL(PanelTabPFN):
    def __init__(self, inside=True, extra=None, n_estimators=8, temperature=0.9, seed=0):
        self.inside, self.extra, self.n_estimators, self.temperature, self.seed = inside, extra, n_estimators, temperature, seed
        self.numeric = False

    def attach(self, inst, mode):
        self.inst = inst
        self.parents = parent_index(inst)

    def predict(self, Xq):
        import torch
        from .tfm import classifier, reset
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.extra == "svd":
            self.prof = self.profile()

        def fit_predict(train, T, query, j, w):
            y = T[:, j]
            ok = y >= 0
            rel = self.related(train[ok], y[ok], j) if self.extra == "mi" else []
            q = np.full((len(query), w), 1e-4)
            if len(np.unique(y[ok])) < 2:
                q += np.bincount(y[ok], minlength=w)[None, :w] + 0.5
            else:
                clf = classifier(device=device, n_estimators=self.n_estimators, random_state=self.seed, softmax_temperature=self.temperature)
                clf.fit(self.features(train[ok], j, rel), y[ok])
                q[:, np.asarray(clf.classes_, dtype=int)] += clf.predict_proba(self.features(query, j, rel))
            return q / q.sum(1, keepdims=True)

        out = self.by_level(fit_predict)
        reset()
        return out
