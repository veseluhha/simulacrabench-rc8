import unicodedata

import numpy as np
import pandas as pd

POS_BINS = 10
GATED = "NA_GATED"


def key(v):
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return " ".join(unicodedata.normalize("NFKC", str(v)).replace("\u2019", "'").split()).lower()


def codes(column, options, missing=-1):
    s = pd.Series(column, dtype=object)
    out = s.map({o: k for k, o in enumerate(options)})
    miss = out.isna() & s.notna()
    if miss.any():
        loose = {key(o): k for k, o in enumerate(options)}
        out[miss] = s[miss].map(lambda v: loose.get(key(v), np.nan))
    return out.fillna(missing).astype(np.int64).to_numpy()


class Instrument:
    def __init__(self, frame, schema):
        records = schema["items"]
        self.items = [n for n, r in records.items() if r["class"] in ("GIVEN", "PREDICT")]
        self.given = [n for n in self.items if records[n]["class"] == "GIVEN"]
        self.scored = [n for n in self.items if records[n]["class"] == "PREDICT"]
        self.gate = {n: records[n].get("gate") or None for n in self.items}
        self.values = {n: list(records[n]["values"]) for n in self.items}
        self.options = {n: self.values[n] + ([schema.get("gated_value", GATED)] if self.gate[n] else []) for n in self.items}
        self.questions = {n: (records[n].get("question") or n).strip() for n in self.items}
        self.gated_value = schema.get("gated_value", GATED)
        self.description = schema.get("dataset", {}).get("description", "")
        n = len(frame)
        self.cards = np.array([len(self.options[g]) + 1 for g in self.given], dtype=np.int64)
        self.X = np.zeros((n, len(self.given)), dtype=np.int64)
        for k, g in enumerate(self.given):
            self.X[:, k] = codes(frame[g].to_numpy(), self.options[g], missing=len(self.options[g]))
        self.Y = np.stack([codes(frame[s].to_numpy(), self.options[s]) for s in self.scored], 1)
        self.widths = np.array([len(self.options[s]) for s in self.scored])
        self.nvalues = np.array([len(self.values[s]) for s in self.scored])
        self.blank = pd.DataFrame(frame[self.scored]).isna().to_numpy()
        visible = ~self.blank.any(1)
        unknown = visible[:, None] & (self.Y < 0)
        if unknown.any():
            self.Y = np.where(unknown, self.widths[None, :] - 1, self.Y)
        self.seen = np.flatnonzero(visible)

        self.uniform = float(np.mean(np.log(self.widths)))

    def add_position(self, bins=POS_BINS):
        pos = np.full(len(self.X), bins - 1)
        pos[self.seen] = np.arange(len(self.seen)) * bins // len(self.seen)
        self.X = np.hstack([self.X, pos[:, None]])
        self.given = self.given + ["__position"]
        self.cards = np.append(self.cards, bins + 1)
        self.values["__position"] = self.options["__position"] = [str(b) for b in range(bins)]
        self.questions["__position"] = "position in file"
        self.gate["__position"] = None

    def add_roster(self):
        import re
        counts = []
        for k, g in enumerate(self.given):
            vals = self.values.get(g) or []
            mids = []
            for v in self.options[g]:
                m = re.fullmatch(r"(\d+)(?:-(\d+)|(\+))?", str(v).strip())
                mids.append(np.nan if m is None else (int(m.group(1)) + int(m.group(2))) / 2 if m.group(2) else float(m.group(1)))
            if "0" in vals and np.isfinite(mids).sum() >= 3:
                counts.append((k, np.array(mids + [np.nan])))
        if len(counts) < 3:
            return False
        size = np.zeros(len(self.X))
        groups = np.zeros(len(self.X), dtype=np.int64)
        for k, mids in counts:
            m = np.nan_to_num(mids[self.X[:, k]])
            size += m
            groups += m > 0
        total = np.minimum(np.round(size).astype(np.int64), 15)
        groups = np.minimum(groups, len(counts))
        for name, code, width in (("__roster_total", total, 16), ("__roster_groups", groups, len(counts) + 1)):
            self.X = np.hstack([self.X, code[:, None]])
            self.given = self.given + [name]
            self.cards = np.append(self.cards, width + 1)
            self.values[name] = self.options[name] = [str(b) for b in range(width)]
            self.questions[name] = name
            self.gate[name] = None
        return True

    def targets(self, mode):
        if mode == "direct":
            return self.Y, self.widths
        return np.where(self.Y < self.nvalues, self.Y, -1), self.nvalues
