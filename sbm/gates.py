import os

import numpy as np

STRENGTH = 2.0
COND_STRENGTH = 5.0


def loglik(a, n, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float((a * np.log(p) + (n - a) * np.log(1 - p)).sum())


def covariate(inst, rows, code, width, asked, rates, skip):
    base = loglik(np.bincount(code, weights=asked, minlength=width)[:width],
                  np.bincount(code, minlength=width)[:width], rates)
    best = None
    for k in range(len(inst.given)):
        if k == skip:
            continue
        c, card = inst.X[rows, k], int(inst.cards[k])
        key = code * card + c
        n = np.bincount(key, minlength=width * card)[:width * card].reshape(width, card)
        a = np.bincount(key, weights=asked, minlength=width * card)[:width * card].reshape(width, card)
        table = (a + COND_STRENGTH * rates[:, None]) / (n + COND_STRENGTH)
        gain = loglik(a, n, table) - base - 0.5 * (n > 0).sum() * np.log(max(len(rows), 2))
        if gain > 0 and (best is None or gain > best[0]):
            best = (gain, k, table)
    return None if best is None else (best[1], best[2])


class Gates:
    def __init__(self, inst, rows):
        given = {g: k for k, g in enumerate(inst.given)}
        scored = {s: j for j, s in enumerate(inst.scored)}
        depth = {}

        def level(name):
            if name not in depth:
                gate = inst.gate.get(name)
                depth[name] = 0 if not gate or gate["parent"] not in scored else 1 + level(gate["parent"])
            return depth[name]

        cond = os.environ.get("SB_GATECOND")
        self.plan = []
        for j in sorted(range(len(inst.scored)), key=lambda j: level(inst.scored[j])):
            gate = inst.gate[inst.scored[j]]
            if not gate:
                self.plan.append((j, None, None, None))
                continue
            asked = inst.Y[rows, j] < inst.nvalues[j]
            parent = gate["parent"]
            if parent in given or parent in scored:
                opts = inst.options[parent]
                width = len(opts) + 1
                code = inst.X[rows, given[parent]] if parent in given else inst.Y[rows, scored[parent]]
                prior = np.array([o in set(gate["observed_if"]) for o in opts] + [0.5], float)
                n = np.bincount(code, minlength=width)[:width]
                a = np.bincount(code, weights=asked, minlength=width)[:width]
                rates = (a + STRENGTH * prior + 0.5) / (n + STRENGTH + 1.0)
                kind, index = ("given", given[parent]) if parent in given else ("scored", scored[parent])
                if cond:
                    cov = covariate(inst, rows, code, width, asked, rates, index if kind == "given" else -1)
                    if cov is not None:
                        rates = cov
                        kind = kind + "_cov"
            else:
                rates = np.array([(asked.sum() + 0.5) / (asked.size + 1.0)])
                kind, index = "constant", None
            self.plan.append((j, kind, index, rates))

    def compose(self, cond, X):
        full = [None] * len(cond)
        for j, kind, index, rates in self.plan:
            if kind is None:
                full[j] = cond[j]
                continue
            if kind == "given":
                asked = rates[X[:, index]]
            elif kind == "scored":
                parent = full[index]
                asked = parent @ rates[:parent.shape[1]]
            elif kind == "given_cov":
                k, table = rates
                asked = table[X[:, index], X[:, k]]
            elif kind == "scored_cov":
                k, table = rates
                parent = full[index]
                asked = (parent * table[:parent.shape[1], X[:, k]].T).sum(1)
            else:
                asked = np.full(len(X), rates[0])
            asked = np.clip(asked, 0.0, 1.0)[:, None]
            full[j] = np.hstack([asked * cond[j], 1.0 - asked])
        return full
