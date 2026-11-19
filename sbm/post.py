import numpy as np

from .stack import FLOOR, bayes_act

PULL = 0.03
MIX_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
ROUTE_GRID = np.linspace(0, 1, 11)


def meta_of(inst, oof_rows, rows):
    return {"scored": inst.scored, "gate": {s: inst.gate[s] for s in inst.scored},
            "options": {s: inst.options[s] for s in inst.scored}, "nvalues": inst.nvalues, "widths": inst.widths,
            "Y_seen": inst.Y[inst.seen], "X_seen": inst.X[inst.seen], "X_oof": inst.X[oof_rows], "X_hidden": inst.X[rows]}


def norm(p):
    return p / np.maximum(p.sum(1, keepdims=True), 1e-300)


def item_logp(P, y):
    q = bayes_act([P])[0]
    q = FLOOR + (1.0 - q.shape[1] * FLOOR) * norm(q)
    return np.log(q[np.arange(len(y)), y])


def _nll(w, L, y, m):
    rows = np.arange(len(y))
    z = np.tensordot(w, L, axes=1)
    z -= z.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    p = np.exp(z - lse[:, None])
    v = -((z[rows, y] - lse) * m).sum()
    g = -((L[:, rows, y] - np.einsum("cnk,nk->cn", L, p)) * m).sum(1)
    return v, g


def pool_fit(Ls, Ys, Ms):
    from scipy.optimize import minimize
    C = Ls[0].shape[0]
    tot = sum(max(m.sum(), 1.0) for m in Ms)

    def shared(w):
        v, g = 0.0, np.zeros(C)
        for L, y, m in zip(Ls, Ys, Ms):
            a, b = _nll(w, L, y, m)
            v, g = v + a, g + b
        return v / tot, g / tot

    bounds = [(0.0, 3.0)] * C
    w0 = minimize(shared, np.full(C, 1.0 / C), jac=True, bounds=bounds, method="L-BFGS-B").x
    W = []
    for L, y, m in zip(Ls, Ys, Ms):
        n = max(m.sum(), 1.0)

        def obj(w):
            a, b = _nll(w, L, y, m)
            d = w - w0
            return a / n + PULL * (d @ d), b / n + 2 * PULL * d
        W.append(minimize(obj, w0.copy(), jac=True, bounds=bounds, method="L-BFGS-B").x)
    return W


def pool_apply(Ls, W):
    out = []
    for L, w in zip(Ls, W):
        z = np.tensordot(w, L, axes=1)
        z -= z.max(1, keepdims=True)
        out.append(norm(np.exp(z)))
    return out


def _logs(parts):
    return np.stack([np.log(np.clip(p, 1e-9, 1.0)) for p in parts]).astype(np.float64)


def split_pool(fit_preds, Y, app_preds, gated, nv):
    C = len(fit_preds)
    Lb, Yb, Mb, Lc, Yc, Mc, cond = [], [], [], [], [], [], []
    for j in gated:
        s = [np.clip(fit_preds[c][j][:, -1], 1e-9, 1 - 1e-9) for c in range(C)]
        Lb.append(_logs([np.stack([1 - x, x], 1) for x in s]))
        y = Y[:, j]
        Yb.append((y == nv[j]).astype(int))
        Mb.append(np.ones(len(y)))
        if nv[j] >= 2:
            Lc.append(_logs([norm(np.clip(fit_preds[c][j][:, :-1], 1e-12, 1.0)) for c in range(C)]))
            asked = y < nv[j]
            Yc.append(np.where(asked, y, 0))
            Mc.append(asked.astype(float))
            cond.append(j)
    Wb, Wc = pool_fit(Lb, Yb, Mb), (pool_fit(Lc, Yc, Mc) if cond else [])
    sb = pool_apply([_logs([np.stack([1 - x, x], 1) for x in [np.clip(app_preds[c][j][:, -1], 1e-9, 1 - 1e-9) for c in range(C)]]) for j in gated], Wb)
    qc = dict(zip(cond, pool_apply([_logs([norm(np.clip(app_preds[c][j][:, :-1], 1e-12, 1.0)) for c in range(C)]) for j in cond], Wc)))
    out = {}
    for r, j in enumerate(gated):
        s = sb[r][:, 1:2]
        q = qc[j] if j in qc else np.ones((len(s), 1))
        out[j] = np.hstack([q * (1 - s), s])
    return out


def routing_groups(meta, agree=0.995):
    Y, nv, widths = meta["Y_seen"], meta["nvalues"], meta["widths"]
    gated = [j for j in range(len(nv)) if widths[j] > nv[j]]
    S = {j: (Y[:, j] == nv[j]) for j in gated}
    parent = {j: j for j in gated}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(gated):
        if S[a].mean() in (0.0, 1.0):
            continue
        for b in gated[i + 1:]:
            if (S[a] == S[b]).mean() >= agree:
                parent[find(a)] = find(b)
    groups = {}
    for j in gated:
        groups.setdefault(find(j), []).append(j)
    return [g for g in groups.values() if len(g) > 1]


def group_pool(full, groups):
    out = [p.copy() for p in full]
    for g in groups:
        s = np.stack([full[j][:, -1] for j in g], 1)
        z = np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1))
        q = 1 / (1 + np.exp(-z.mean(1)))
        for j in g:
            rest = norm(full[j][:, :-1])
            out[j] = np.hstack([rest * (1 - q)[:, None], q[:, None]])
    return out


def fit_mix(Pcf, Y, aux, items, ws=(0.0, 0.25, 0.5, 0.75, 1.0), seed=777):
    J, n = len(Pcf), len(Y)
    half = np.random.default_rng(seed).random(n) < 0.5
    H = [half, ~half]
    grid = np.asarray(MIX_GRID)
    G = np.zeros((J, len(grid), 2))
    base = np.zeros((J, 2))
    for j in items:
        for g, u in enumerate(grid):
            lp = item_logp((1 - u) * Pcf[j] + u * aux[j], Y[:, j])
            G[j, g] = [lp[H[0]].sum(), lp[H[1]].sum()]
        lp = item_logp(Pcf[j], Y[:, j])
        base[j] = [lp[H[0]].sum(), lp[H[1]].sum()]
    cv = {w: 0.0 for w in ws}
    for a, b in ((0, 1), (1, 0)):
        Ga = G[items][:, :, a]
        ug, ui = grid[Ga.sum(0).argmax()], grid[Ga.argmax(1)]
        for w in ws:
            for jj, j in enumerate(items):
                u = w * ui[jj] + (1 - w) * ug
                rows = H[b]
                lp = item_logp((1 - u) * Pcf[j][rows] + u * aux[j][rows], Y[rows, j])
                cv[w] += lp.sum() - base[j, b]
    wbest = max(ws, key=lambda w: cv[w])
    Gf = G[items].sum(2)
    ug, ui = grid[Gf.sum(0).argmax()], grid[Gf.argmax(1)]
    U = np.zeros(J)
    for jj, j in enumerate(items):
        U[j] = wbest * ui[jj] + (1 - wbest) * ug
    return U


def artifact(meta, j, k, gate):
    Y, X, nv = meta["Y_seen"], meta["X_seen"], meta["nvalues"]
    opts = meta["options"][gate["parent"]]
    closed = np.isin(Y[:, k], [i for i, o in enumerate(opts) if o not in set(gate["observed_if"])])
    sent = Y[:, j] == nv[j]
    best = None
    for g in range(X.shape[1]):
        bad = [l for l in np.unique(X[closed, g]) if (closed & (X[:, g] == l)).sum() >= 30 and sent[closed & (X[:, g] == l)].mean() < 0.01]
        if not bad:
            continue
        rest = closed & ~np.isin(X[:, g], bad)
        if rest.sum() >= 30 and sent[rest].mean() >= 0.5:
            cover = (closed & np.isin(X[:, g], bad)).sum()
            if best is None or cover > best[0]:
                best = (cover, g, np.array(bad))
    return None if best is None else best[1:]


def clean_rows(X, art):
    return np.ones(len(X), bool) if art is None else ~np.isin(X[:, art[0]], art[1])


def route_plan(meta):
    scored = {n: j for j, n in enumerate(meta["scored"])}
    Y, nv = meta["Y_seen"], meta["nvalues"]
    depth = {}

    def level(n):
        if n not in depth:
            g = meta["gate"].get(n)
            depth[n] = 0 if not g or g["parent"] not in scored else 1 + level(g["parent"])
        return depth[n]

    plan = []
    for n in sorted(meta["scored"], key=level):
        g = meta["gate"].get(n)
        if not g or g["parent"] not in scored:
            continue
        j, k = scored[n], scored[g["parent"]]
        art = artifact(meta, j, k, g)
        keep = clean_rows(meta["X_seen"], art)
        opts = meta["options"][g["parent"]]
        width = len(opts)
        asked = (Y[keep, j] < nv[j]).astype(float)
        code = Y[keep, k]
        prior = np.array([o in set(g["observed_if"]) for o in opts], float)
        cnt = np.bincount(code, minlength=width)[:width]
        a = np.bincount(code, weights=asked, minlength=width)[:width]
        plan.append((j, k, (a + 2.0 * prior + 0.5) / (cnt + 3.0), art))
    return plan


def route(full, plan, lam, X):
    out = [p.copy() for p in full]
    for j, k, rates, art in plan:
        m = clean_rows(X, art)
        s_par = 1.0 - np.clip(out[k][m] @ rates, 0, 1)
        s = lam[j] * s_par + (1 - lam[j]) * out[j][m, -1]
        rest = norm(out[j][m, :-1])
        out[j][m] = np.hstack([rest * (1 - s)[:, None], s[:, None]])
    return out


def route_fit(full_oof, Y, plan, J, X):
    lam = np.zeros(J)
    out = [p.copy() for p in full_oof]
    for j, k, rates, art in plan:
        m = clean_rows(X, art)
        if m.sum() < 30:
            continue
        rows = np.arange(m.sum())
        s_par = 1.0 - np.clip(out[k][m] @ rates, 0, 1)
        rest = norm(out[j][m, :-1])
        best = None
        for l in ROUTE_GRID:
            s = l * s_par + (1 - l) * out[j][m, -1]
            q = np.hstack([rest * (1 - s)[:, None], s[:, None]])
            q = FLOOR + (1 - q.shape[1] * FLOOR) * q
            v = np.log(q[rows, Y[m, j]]).sum()
            if best is None or v > best[0]:
                best = (v, l)
        lam[j] = best[1]
        s = lam[j] * s_par + (1 - lam[j]) * out[j][m, -1]
        out[j][m] = np.hstack([rest * (1 - s)[:, None], s[:, None]])
    return lam


def refine(oof, Y, test, base_cf, base_test, meta):
    J, n = len(base_test), len(Y)
    nv, widths = np.asarray(meta["nvalues"]), np.asarray(meta["widths"])
    gated = [j for j in range(J) if widths[j] > nv[j]]
    if not gated:
        return base_test
    Pt = list(base_test)
    Pcf = list(base_cf)
    h1t = split_pool(oof, Y, test, gated, nv)
    h1cf = {j: np.zeros((n, widths[j])) for j in gated}
    half = np.random.default_rng(1).random(n) < 0.5
    for A, B in ((half, ~half), (~half, half)):
        part = split_pool([[p[A] for p in cand] for cand in oof], Y[A], [[p[B] for p in cand] for cand in oof], gated, nv)
        for j in gated:
            h1cf[j][B] = part[j]
    for j in gated:
        Pt[j], Pcf[j] = h1t[j], h1cf[j]
    groups = routing_groups(meta)
    if groups:
        items = sorted(j for g in groups for j in g)
        gcf, gt = group_pool(Pcf, groups), group_pool(Pt, groups)
        U = fit_mix(Pcf, Y, gcf, items)
        for j in items:
            Pcf[j] = (1 - U[j]) * Pcf[j] + U[j] * gcf[j]
            Pt[j] = (1 - U[j]) * Pt[j] + U[j] * gt[j]
    plan = route_plan(meta)
    if plan:
        lam = route_fit(Pcf, Y, plan, J, meta["X_oof"])
        Pt = route(Pt, plan, lam, meta["X_hidden"])
    return Pt
