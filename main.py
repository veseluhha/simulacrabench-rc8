import importlib.util
import os

import numpy as np

if importlib.util.find_spec("sbm.preflight"):
    import sbm.preflight

from sbm import pipeline
from sbm.data import Instrument
from sbm.select import CANDIDATES, HYBRID, choose, predict_rows

pipeline.warm()


def predict(frame, schema):
    inst = Instrument(frame, schema)
    rows = np.flatnonzero(inst.blank.any(1))
    name, pick = os.environ.get("SB_MODEL", "stack"), None
    if len(inst.seen) < 50 or not inst.given:
        full = predict_rows(inst, "marginal", None, rows)
    elif name == "stack":
        try:
            full = pipeline.run(inst, schema, rows)
        except Exception as exc:
            pipeline.log("stack failed: %r" % (exc,))
            pipeline.release()
            full = predict_rows(inst, "backoff2", None, rows)
    else:
        if name == "auto":
            name, pick, report = choose(inst, list(CANDIDATES))
            name = name if report else "marginal"
        elif name == "hybrid":
            _, pick, _ = choose(inst, list(HYBRID))
        full = predict_rows(inst, name, pick, rows)
    pipeline.finish(schema)
    full = [clean(p) for p in full]
    return [full[j][r] for r in range(len(rows)) for j in range(len(inst.scored)) if inst.blank[rows[r], j]]


def clean(p):
    p = np.clip(np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    total = p.sum(1, keepdims=True)
    return np.where(total > 0, p / np.where(total > 0, total, 1.0), 1.0 / p.shape[1])
