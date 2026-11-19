import os
import sys
import time

import numpy as np

from .gates import Gates
from .models import GBDT, Backoff, DistilTabM, LatentClass, Marginal, Net, SemNet, SubBackoff, SubTabM, TabMNet
from .panel import PanelCopy, PanelTabICL, PanelTabPFN, SoftPanelCopy
from .tfm import Exaone, RepTabPFN, SubTabPFN, TabDPT, TabICL, TabPFN

FLOOR = 1e-3

CANDIDATES = {
    "marginal": (Marginal, "structural"),
    "backoff1": (lambda: Backoff(depth=1, alpha=8.0), "structural"),
    "backoff2": (lambda: Backoff(depth=2, alpha=16.0), "structural"),
    "logreg": (lambda: Net(hidden=(), wd=1e-3, seeds=3), "structural"),
    "encoder": (lambda: Net(hidden=(256, 64), dropout=0.1, wd=1e-4, seeds=3), "structural"),
    "encoder_direct": (lambda: Net(hidden=(256, 64), dropout=0.1, wd=1e-4, seeds=3), "direct"),
}
EXTRA = {
    "encoder_sem": (lambda: SemNet(hidden=(256,), dropout=0.1, wd=1e-4, seeds=3), "structural"),
    "encoder_sem_direct": (lambda: SemNet(hidden=(256,), dropout=0.1, wd=1e-4, seeds=3), "direct"),
    "tabicl": (lambda: TabICL(), "direct"),
    "tabicl_struct": (lambda: TabICL(), "structural"),
    "tabicl4": (lambda: TabICL(n_estimators=4), "direct"),
    "tabm": (lambda: TabMNet(hidden=(256, 64), dropout=0.1, seeds=2), "structural"),
    "tabm_direct": (lambda: TabMNet(hidden=(256, 64), dropout=0.1, seeds=2), "direct"),
    "tabpfn_fast": (lambda: TabPFN("fast"), "direct"),
    "tabpfn_base": (lambda: TabPFN("base"), "direct"),
    "tabicl_n16": (lambda: TabICL(n_estimators=16), "direct"),
    "tabicl_n32": (lambda: TabICL(n_estimators=32), "direct"),
    "tabicl_t11": (lambda: TabICL(temperature=1.1), "direct"),
    "tabicl_oh": (lambda: TabICL(onehot=True), "direct"),
    "gbdt": (lambda: GBDT(), "structural"),
    "gbdt_direct": (lambda: GBDT(), "direct"),
    "lca": (lambda: LatentClass(), "structural"),
    "lca_direct": (lambda: LatentClass(), "direct"),
    "lca_k8": (lambda: LatentClass(classes=8), "direct"),
    "lca_k96": (lambda: LatentClass(classes=96), "direct"),
    "tabpfn_svd": (lambda: TabPFN("fast", view="svd"), "direct"),
    "tabpfn_cross": (lambda: TabPFN("fast", view="cross"), "direct"),
    "tabpfn_struct": (lambda: TabPFN("fast"), "structural"),
    "tabdpt": (lambda: TabDPT(), "direct"),
    "tabdpt_struct": (lambda: TabDPT(), "structural"),
    "exaone": (lambda: Exaone(), "direct"),
    "tabpfn_rep": (lambda: RepTabPFN(), "direct"),
    "tabpfn_num": (lambda: TabPFN("fast", numeric=True), "direct"),
    "tabpfn_n8": (lambda: TabPFN("fast", n_estimators=8), "direct"),
    "tabpfn_nofp": (lambda: TabPFN("fast", config={"FINGERPRINT_FEATURE": False}), "direct"),
    "tabpfn_p16": (lambda: TabPFN("fast", numeric=True, n_estimators=16), "direct"),
    "tabicl_p32": (lambda: TabICL(n_estimators=32), "direct"),
    "tabpfn_sub": (lambda: SubTabPFN("fast", numeric=True), "direct"),
    "backoff2_sub": (lambda: SubBackoff(depth=2, alpha=16.0), "structural"),
    "tabm_sub": (lambda: SubTabM(hidden=(256, 64), dropout=0.1, seeds=2), "structural"),
    "panel_copy": (lambda: PanelCopy(), "direct"),
    "panel_soft": (lambda: SoftPanelCopy(), "direct"),
    "tabpfn_panel": (lambda: PanelTabPFN(variant="fast", numeric=True), "direct"),
    "tabpfn_panel_in": (lambda: PanelTabPFN(inside=True, variant="fast", numeric=True), "direct"),
    "tabpfn_panel_in_mi": (lambda: PanelTabPFN(inside=True, extra="mi", variant="fast", numeric=True), "direct"),
    "tabpfn_panel_in_svd": (lambda: PanelTabPFN(inside=True, extra="svd", variant="fast", numeric=True), "direct"),
    "tabpfn_panel_mi": (lambda: PanelTabPFN(extra="mi", variant="fast", numeric=True), "direct"),
    "tabicl_panel_in": (lambda: PanelTabICL(), "direct"),
    "tabm_distil": (lambda: DistilTabM(lam=0.5, hidden=(256, 64), dropout=0.1, seeds=2), "structural"),
    "tabm_distil_direct": (lambda: DistilTabM(lam=0.5, hidden=(256, 64), dropout=0.1, seeds=2), "direct"),
    "tabm_distil0_direct": (lambda: DistilTabM(lam=0.0, hidden=(256, 64), dropout=0.1, seeds=2), "direct"),
    "encoder_s8": (lambda: Net(hidden=(256, 64), dropout=0.1, wd=1e-4, seeds=8), "structural"),
    "encoder_direct_s8": (lambda: Net(hidden=(256, 64), dropout=0.1, wd=1e-4, seeds=8), "direct"),
}
HYBRID = ("encoder", "encoder_direct")
GATE_FOCUS = None


def fit(inst, name, rows, only=None, focus=None):
    make, mode = CANDIDATES[name] if name in CANDIDATES else EXTRA[name]
    T, widths = inst.targets(mode)
    model = make()
    if hasattr(model, "attach"):
        model.attach(inst, mode)
    model.rows = rows
    model.only = only
    model.focus = focus
    model = model.fit(inst.X[rows], T[rows], widths, inst.cards)
    gate_rows = rows
    if GATE_FOCUS is not None:
        inner = rows[inst.X[rows, GATE_FOCUS[0]] == GATE_FOCUS[1]]
        gate_rows = inner if len(inner) >= 300 else rows
    return model, (Gates(inst, gate_rows) if mode == "structural" else None)


def distributions(inst, fitted, rows):
    model, gates = fitted
    model.query = rows
    cond = model.predict(inst.X[rows])
    return gates.compose(cond, inst.X[rows]) if gates else cond


def item_logp(inst, full, rows):
    out = np.empty(len(full))
    for j, p in enumerate(full):
        p = p / p.sum(1, keepdims=True)
        p = FLOOR + (1.0 - p.shape[1] * FLOOR) * p
        out[j] = np.log(p[np.arange(len(rows)), inst.Y[rows, j]]).sum()
    return out


def folds(seen, k, frac, seed=0):
    order = np.random.default_rng(seed).permutation(seen)
    size = int(round(frac * len(seen)))
    return [order[f * size:(f + 1) * size] for f in range(k)]


def gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def usable(names):
    return [n for n in names if gpu() or not isinstance({**CANDIDATES, **EXTRA}[n][0](), (Net, TabICL, TabPFN))]


def choose(inst, names, k=3, frac=0.15):
    per, seconds = {}, {}
    for name in usable(names):
        t = time.time()
        total, count = np.zeros(len(inst.scored)), 0
        try:
            for test in folds(inst.seen, k, frac):
                train = np.setdiff1d(inst.seen, test)
                total += item_logp(inst, distributions(inst, fit(inst, name, train), test), test)
                count += len(test)
        except Exception as exc:
            if os.environ.get("SB_DEBUG"):
                print("[cv] %s failed: %r" % (name, exc), file=sys.stderr, flush=True)
            continue
        per[name], seconds[name] = total / count, time.time() - t
    report = {name: 1.0 + v.mean() / inst.uniform for name, v in per.items()}
    pick = None
    if all(h in per for h in HYBRID):
        gated = np.array([bool(inst.gate[s]) for s in inst.scored])
        pick = gated & (per[HYBRID[1]] > per[HYBRID[0]])
        report["hybrid"] = 1.0 + np.where(pick, per[HYBRID[1]], per[HYBRID[0]]).mean() / inst.uniform
    if os.environ.get("SB_DEBUG"):
        for name, s in sorted(report.items(), key=lambda kv: -kv[1]):
            print("[cv] %-16s %.4f  %.1fs" % (name, s, seconds.get(name, 0.0)), file=sys.stderr, flush=True)
        if pick is not None:
            print("[cv] hybrid takes direct on %d of %d gated items" % (pick.sum(), gated.sum()), file=sys.stderr, flush=True)
    return (max(report, key=report.get) if report else "marginal"), pick, report


def predict_rows(inst, name, pick, rows):
    for attempt in (name, "backoff1", "marginal"):
        try:
            if attempt != "hybrid":
                return distributions(inst, fit(inst, attempt, inst.seen), rows)
            structural, direct = (distributions(inst, fit(inst, h, inst.seen), rows) for h in HYBRID)
            return [direct[j] if pick[j] else structural[j] for j in range(len(inst.scored))]
        except Exception as exc:
            if attempt == "marginal":
                raise
            if os.environ.get("SB_DEBUG"):
                print("[fit] %s failed: %r" % (attempt, exc), file=sys.stderr, flush=True)
