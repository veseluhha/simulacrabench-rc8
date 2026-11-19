import numpy as np

FLOOR = 1e-3


def truth_probs(preds, Y):
    rows = np.arange(len(Y))
    return np.stack([np.stack([p[rows, Y[:, j]] for j, p in enumerate(cand)]) for cand in preds])


def mixture_weights(P, prior=None, strength=0.0, iters=200, rw=None):
    C = P.shape[0]
    rw = np.ones(P[0].size) if rw is None else np.broadcast_to(rw, P[0].shape).reshape(-1)
    w = np.full(C, 1.0 / C) if prior is None else prior.copy()
    for _ in range(iters):
        mix = np.tensordot(w, P, axes=1)
        resp = (w[:, None] * P.reshape(C, -1) * (rw / np.maximum(mix.reshape(-1), 1e-300))).sum(1)
        new = (resp + (strength * prior if prior is not None else 0.0)) / (rw.sum() + (strength if prior is not None else 0.0))
        if np.abs(new - w).max() < 1e-7:
            return new
        w = new
    return w


def item_weights(P, strength, rw=None):
    shared = mixture_weights(P, rw=rw)
    return np.stack([mixture_weights(P[:, j:j + 1], prior=shared, strength=strength, rw=rw) for j in range(P.shape[1])]), shared


def blend(preds, W):
    C, J = len(preds), len(preds[0])
    return [sum(W[j, c] * preds[c][j] for c in range(C)) for j in range(J)]


def temper(full, T):
    out = []
    for p in full:
        q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / T)
        out.append(q / q.sum(1, keepdims=True))
    return out


def mean_logp(full, Y, rw=None):
    rows, total = np.arange(len(Y)), 0.0
    rw = np.ones(len(Y)) if rw is None else rw
    for j, p in enumerate(full):
        p = p / p.sum(1, keepdims=True)
        p = FLOOR + (1.0 - p.shape[1] * FLOOR) * p
        total += (rw * np.log(p[rows, Y[:, j]])).sum()
    return total / (rw.sum() * Y.shape[1])


def temperature(full, Y, grid=np.linspace(0.6, 1.6, 21), rw=None):
    return float(max(grid, key=lambda T: mean_logp(temper(full, T), Y, rw)))


def bayes_act(full, eps=FLOOR):
    out = []
    for P in full:
        P = P / P.sum(1, keepdims=True)
        K = P.shape[1]
        clamp = np.zeros(P.shape, dtype=bool)
        for _ in range(K):
            s = (1 - clamp.sum(1, keepdims=True) * eps) / np.maximum((P * ~clamp).sum(1, keepdims=True), 1e-300)
            new = clamp | (s * P < eps)
            if (new == clamp).all():
                break
            clamp = new
        s = (1 - clamp.sum(1, keepdims=True) * eps) / np.maximum((P * ~clamp).sum(1, keepdims=True), 1e-300)
        out.append(np.maximum((np.where(clamp, eps, s * P) - eps) / (1 - K * eps), 0.0))
    return out


def _pool_nll(w, L, y):
    rows = np.arange(len(y))
    z = np.tensordot(w, L, axes=1)
    z -= z.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    p = np.exp(z - lse[:, None])
    return -(z[rows, y] - lse).sum(), -(L[:, rows, y] - np.einsum("cnk,nk->cn", L, p)).sum(1)


def logpool_weights(preds, Y, shrink=0.03):
    from scipy.optimize import minimize
    C, J, n = len(preds), len(preds[0]), len(Y)
    L = [np.stack([np.log(np.clip(p[j], 1e-9, 1.0)) for p in preds]).astype(np.float32) for j in range(J)]
    bounds = [(0.0, 3.0)] * C

    def shared_obj(w):
        v, g = 0.0, np.zeros(C)
        for j in range(J):
            a, b = _pool_nll(w, L[j], Y[:, j])
            v, g = v + a, g + b
        return v / (n * J), g / (n * J)

    w0 = minimize(shared_obj, np.full(C, 1.0 / C), jac=True, bounds=bounds, method="L-BFGS-B").x
    W = []
    for j in range(J):
        def obj(w):
            a, b = _pool_nll(w, L[j], Y[:, j])
            d = w - w0
            return a / n + shrink * (d @ d), b / n + 2 * shrink * d
        W.append(minimize(obj, w0.copy(), jac=True, bounds=bounds, method="L-BFGS-B").x)
    return np.array(W)


def logpool(preds, W):
    out = []
    for j in range(len(preds[0])):
        z = sum(W[j, c] * np.log(np.clip(p[j], 1e-9, 1.0)) for c, p in enumerate(preds))
        z -= z.max(1, keepdims=True)
        q = np.exp(z)
        out.append(q / q.sum(1, keepdims=True))
    return out
