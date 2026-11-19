import hashlib
from pathlib import Path

import numpy as np

_shared = {}
_memo = {}
STOP = None
HITS = [0]


def stopped():
    import time
    if STOP is not None and time.time() > STOP:
        HITS[0] += 1
        return True
    return False


def _key(*parts):
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str((p.shape, p.dtype)).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()


def _patch():
    if _memo.get("patched"):
        return
    from tabicl._model.inference import InferenceManager
    from tabicl._sklearn.preprocessing import PreprocessingPipeline

    fit, transform, free = PreprocessingPipeline.fit, PreprocessingPipeline.transform, InferenceManager.get_available_gpu_memory

    def cached_fit(self, X, y=None):
        key = _key("fit", sorted(self.get_params().items()), np.asarray(X))
        if key not in _memo:
            fit(self, X, y)
            _memo[key] = dict(self.__dict__)
        else:
            self.__dict__.update(_memo[key])
        self._memo_key = key
        return self

    def cached_transform(self, X):
        key = _key("transform", getattr(self, "_memo_key", None), np.asarray(X))
        if key not in _memo:
            _memo[key] = transform(self, X)
        return _memo[key].copy()

    def cached_free(self):
        state = _memo.setdefault("free", {"calls": 0, "value": None})
        if state["value"] is None or state["calls"] % 64 == 0:
            state["value"] = free(self)
        state["calls"] += 1
        return state["value"]

    PreprocessingPipeline.fit, PreprocessingPipeline.transform = cached_fit, cached_transform
    InferenceManager.get_available_gpu_memory = cached_free
    _memo["patched"] = True


ICL_FILE = "tabicl-classifier-v2-20260212.ckpt"


def icl_path():
    local = Path(__file__).resolve().parent.parent / "weights" / ICL_FILE
    return str(local) if local.exists() else None


def classifier(**kwargs):
    from tabicl import TabICLClassifier
    _patch()
    if icl_path():
        kwargs = dict(kwargs, model_path=icl_path(), checkpoint_version=ICL_FILE, allow_auto_download=False)

    class Shared(TabICLClassifier):
        def _load_model(self):
            key = (str(self.model_path), self.checkpoint_version)
            if key not in _shared:
                super()._load_model()
                _shared[key] = self.model_
            self.model_ = _shared[key]

    return Shared(**kwargs)


def reset():
    patched = _memo.get("patched")
    _memo.clear()
    if patched:
        _memo["patched"] = True


class TabICL:
    def __init__(self, n_estimators=8, max_context=None, seed=0, temperature=0.9, onehot=False):
        self.n_estimators, self.max_context, self.seed, self.temperature = n_estimators, max_context, seed, temperature
        self.onehot = onehot

    def expand(self, X):
        if not self.onehot:
            return X
        wide = [X] + [(X[:, k:k + 1] == np.arange(c)[None, :]).astype(X.dtype) for k, c in enumerate(self.cards) if 4 <= c <= 51]
        return np.hstack(wide)

    def fit(self, X, T, widths, cards):
        self.cards = cards
        self.X, self.T, self.widths = self.expand(X), T, list(widths)
        return self

    def predict(self, Xq):
        import torch
        Xq = self.expand(Xq)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rng = np.random.default_rng(self.seed)
        out = []
        for j, w in enumerate(self.widths):
            if getattr(self, "only", None) is not None and j not in self.only:
                out.append(None)
                continue
            if stopped():
                out.append(np.full((len(Xq), w), np.nan))
                continue
            y = self.T[:, j]
            rows = np.flatnonzero(y >= 0)
            if self.max_context and len(rows) > self.max_context:
                rows = rng.choice(rows, self.max_context, replace=False)
            p = np.full((len(Xq), w), 1e-4)
            if len(np.unique(y[rows])) < 2:
                p += np.bincount(y[rows], minlength=w)[None, :w] + 0.5
            else:
                clf = classifier(device=device, n_estimators=self.n_estimators, random_state=self.seed,
                                 softmax_temperature=self.temperature)
                clf.fit(self.X[rows], y[rows])
                p[:, np.asarray(clf.classes_, dtype=int)] += clf.predict_proba(Xq)
            out.append(p / p.sum(1, keepdims=True))
        reset()
        return out


PFN_REPO, PFN_REVISION = "Prior-Labs/tabpfn_3_5", "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"
PFN_FILES = {"fast": "tabpfn-v3.5-fast-20260909.safetensors", "base": "tabpfn-v3.5-20260909.safetensors"}


def pfn_path(variant="fast"):
    import os
    local = Path(__file__).resolve().parent.parent / "weights" / PFN_FILES[variant]
    if local.exists() or not os.environ.get("SB_PFN_CACHE"):
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(PFN_REPO, PFN_FILES[variant], revision=PFN_REVISION, local_files_only=True)


def pfn_available(variant="fast"):
    try:
        import tabpfn
        return Path(pfn_path(variant)).exists()
    except Exception:
        return False


def _patch_pfn():
    from tabpfn.checkpoint import Checkpoint
    if getattr(Checkpoint, "_sniffed", False):
        return

    def is_safetensors(self):
        if self.path.suffix == ".safetensors":
            return True
        with open(self.path, "rb") as fh:
            head = fh.read(9)
        return len(head) == 9 and head[8:9] == b"{"

    Checkpoint.is_safetensors = property(is_safetensors)
    Checkpoint._sniffed = True


class TabPFN:
    def __init__(self, variant="fast", n_estimators="auto", seed=0, view=None, numeric=False, config=None):
        self.variant, self.n_estimators, self.seed, self.view, self.numeric = variant, n_estimators, seed, view, numeric
        self.config = config

    def attach(self, inst, mode):
        import re
        band = re.compile(r"^\s*\d+(\s*[-–]\s*\d+|\+)?\s*$")
        self.ordinal = [sum(bool(band.match(str(v))) for v in inst.values[g]) >= max(3, len(inst.values[g]) - 1) for g in inst.given]
        self.offscale = [[k for k, v in enumerate(inst.values[g]) if not band.match(str(v))] + [len(inst.options[g])] for g in inst.given]

    def numeric_view(self, X):
        return X

    def transform(self, X):
        if self.view is None:
            return X
        if self.view == "svd":
            onehot = np.zeros((len(X), self.offsets[-1]))
            onehot[np.arange(len(X))[:, None], X + self.offsets[:-1]] = 1.0
            return np.hstack([X, (onehot - self.center) @ self.basis])
        return np.hstack([X] + [(X[:, a] * self.cards[b] + X[:, b])[:, None] for a, b in self.pairs])

    def fit(self, X, T, widths, cards):
        self.cards = np.asarray(cards)
        self.ncat = X.shape[1] + (len(self.pairs_for(X)) if self.view == "cross" else 0)
        if self.view == "svd":
            self.offsets = np.concatenate([[0], np.cumsum(self.cards)]).astype(np.int64)
            onehot = np.zeros((len(X), self.offsets[-1]))
            onehot[np.arange(len(X))[:, None], X + self.offsets[:-1]] = 1.0
            self.center = onehot.mean(0)
            k = max(2, int(np.sqrt(self.offsets[-1])))
            self.basis = np.linalg.svd(onehot - self.center, full_matrices=False)[2][:k].T
        self.X, self.T, self.widths = self.transform(X), T, list(widths)
        return self

    def pairs_for(self, X):
        rng = np.random.default_rng(self.seed + 1)
        G = X.shape[1]
        allp = [(a, b) for a in range(G) for b in range(a + 1, G)]
        self.pairs = [allp[i] for i in rng.choice(len(allp), min(4, len(allp)), replace=False)]
        return self.pairs

    def predict(self, Xq):
        import os
        import torch
        from tabpfn import TabPFNClassifier
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        _patch_pfn()
        path = pfn_path(self.variant)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Xq = self.transform(Xq)
        Xc, Xn = self.numeric_view(self.X), self.numeric_view(Xq)
        out = []
        for j, w in enumerate(self.widths):
            if getattr(self, "only", None) is not None and j not in self.only:
                out.append(None)
                continue
            if stopped():
                out.append(np.full((len(Xq), w), np.nan))
                continue
            y = self.T[:, j]
            rows = np.flatnonzero(y >= 0)
            p = np.full((len(Xq), w), 1e-4)
            if len(np.unique(y[rows])) < 2:
                p += np.bincount(y[rows], minlength=w)[None, :w] + 0.5
            else:
                cats = [k for k in range(self.ncat) if not (self.numeric and k < len(getattr(self, "ordinal", [])) and self.ordinal[k])]
                extra = {"inference_config": dict(self.config)} if self.config else {}
                clf = TabPFNClassifier(model_path=path, device=device, n_estimators=self.n_estimators, **extra,
                                       categorical_features_indices=cats,
                                       ignore_pretraining_limits=True, random_state=self.seed)
                clf.fit(Xc[rows], y[rows])
                p[:, np.asarray(clf.classes_, dtype=int)] += clf.predict_proba(Xn)
            out.append(p / p.sum(1, keepdims=True))
        return out


def warm(pfn=False):
    import torch
    X = np.random.default_rng(0).integers(0, 3, (64, 3))
    y = (X[:, 0] + X[:, 1]) % 2
    classifier(device="cuda", random_state=0).fit(X, y).predict_proba(X[:2])
    reset()
    if pfn and pfn_available():
        TabPFN("fast").fit(X, y[:, None], [2], None).predict(X[:2])


class TabDPT:
    def __init__(self, n_ensembles=8, seed=0):
        self.n_ensembles, self.seed = n_ensembles, seed

    def fit(self, X, T, widths, cards):
        self.X, self.T, self.widths = X.astype(np.float32), T, list(widths)
        return self

    def predict(self, Xq):
        import os
        import torch
        from tabdpt import TabDPTClassifier
        path = os.environ.get("SB_DPT_PATH") or str(Path(__file__).resolve().parent.parent / "weights" / "tabdpt1_3.safetensors")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Xq = Xq.astype(np.float32)
        out = []
        for j, w in enumerate(self.widths):
            y = self.T[:, j]
            rows = np.flatnonzero(y >= 0)
            p = np.full((len(Xq), w), 1e-4)
            present = np.unique(y[rows])
            if len(present) < 2:
                p += np.bincount(y[rows], minlength=w)[None, :w] + 0.5
            else:
                remap = np.full(w, -1)
                remap[present] = np.arange(len(present))
                clf = TabDPTClassifier(device=device, compile=False, model_weight_path=path, verbose=False)
                clf.fit(self.X[rows], remap[y[rows]])
                p[:, present] += clf.ensemble_predict_proba(Xq, n_ensembles=self.n_ensembles, context_size=None, seed=self.seed)
            out.append(p / p.sum(1, keepdims=True))
        return out


class Exaone:
    def __init__(self, ensemble=4, seed=0):
        self.ensemble, self.seed = ensemble, seed

    def fit(self, X, T, widths, cards):
        self.X, self.T, self.widths = X.astype(np.float32), T, list(widths)
        return self

    def predict(self, Xq):
        import torch
        from exaonetabular import EXAONETabularClassifier
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clf = EXAONETabularClassifier.from_pretrained(device=device, ensemble_count=self.ensemble, seed=self.seed)
        Xq = Xq.astype(np.float32)
        out = []
        for j, w in enumerate(self.widths):
            y = self.T[:, j]
            rows = np.flatnonzero(y >= 0)
            p = np.full((len(Xq), w), 1e-4)
            present = np.unique(y[rows])
            if len(present) < 2:
                p += np.bincount(y[rows], minlength=w)[None, :w] + 0.5
            else:
                remap = np.full(w, -1)
                remap[present] = np.arange(len(present))
                clf.fit(self.X[rows], remap[y[rows]])
                p[:, present] += clf.predict_proba(Xq)
            out.append(p / p.sum(1, keepdims=True))
        return out


class RepTabPFN(TabPFN):
    def __init__(self, dims=16, folds=3, **kwargs):
        super().__init__(**kwargs)
        self.dims, self.folds = dims, folds

    def profile(self, models, X):
        return np.mean([np.hstack(m.predict(X)) for m in models], 0)

    def fit(self, X, T, widths, cards):
        from .models import TabMNet
        n = len(X)
        parts = np.array_split(np.random.default_rng(self.seed).permutation(n), self.folds)
        self.reps, P = [], None
        for part in parts:
            rest = np.setdiff1d(np.arange(n), part)
            m = TabMNet(hidden=(256, 64), dropout=0.1, seeds=1).fit(X[rest], T[rest], widths, cards)
            out = np.hstack(m.predict(X[part]))
            P = np.zeros((n, out.shape[1])) if P is None else P
            P[part] = out
            self.reps.append(m)
        self.mu = P.mean(0)
        self.basis = np.linalg.svd(P - self.mu, full_matrices=False)[2][:self.dims].T
        super().fit(X, T, widths, cards)
        self.X = np.hstack([X, (P - self.mu) @ self.basis])
        return self

    def predict(self, Xq):
        self.ncat = len(self.cards)
        return super().predict(np.hstack([Xq, (self.profile(self.reps, Xq) - self.mu) @ self.basis]))


class SubTabPFN(TabPFN):
    def fit(self, X, T, widths, cards):
        g, level = self.focus
        keep = X[:, g] == level
        super().fit(X[keep], T[keep], widths, cards)
        return self

    def predict(self, Xq):
        g, level = self.focus
        inside = Xq[:, g] == level
        out = [None] * len(self.widths)
        if inside.any():
            part = super().predict(Xq[inside])
            for j, p in enumerate(part):
                if p is not None:
                    full = np.full((len(Xq), p.shape[1]), np.nan)
                    full[inside] = p
                    out[j] = full
        return out
