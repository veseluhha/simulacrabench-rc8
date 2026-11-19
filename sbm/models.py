import numpy as np


def smoothed(y, width, pseudo=0.5):
    counts = np.bincount(y[y >= 0], minlength=width).astype(float) + pseudo
    return counts / counts.sum()


class Marginal:
    def fit(self, X, T, widths, cards):
        self.p = [smoothed(T[:, j], w) for j, w in enumerate(widths)]
        return self

    def predict(self, X):
        return [np.tile(p, (len(X), 1)) for p in self.p]


def mutual_information(x, y, nx, ny):
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    total = joint.sum()
    if total == 0:
        return 0.0
    joint /= total
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())
    return mi - ((px > 0).sum() - 1) * ((py > 0).sum() - 1) / (2 * total)


class Backoff:
    def __init__(self, depth=2, alpha=8.0, max_cells=200000):
        self.depth, self.alpha, self.max_cells = depth, alpha, max_cells

    def fit(self, X, T, widths, cards):
        self.cards, self.levels = cards, []
        for j, w in enumerate(widths):
            y = T[:, j]
            ok = y >= 0
            base = smoothed(y, w)
            gain = [mutual_information(X[ok, g], y[ok], cards[g], w) for g in range(X.shape[1])]
            ladder, tables = [g for g in np.argsort(gain)[::-1][:self.depth] if gain[g] > 0], []
            key, size = np.zeros(int(ok.sum()), dtype=np.int64), 1
            for g in ladder:
                if size * cards[g] * w > self.max_cells:
                    break
                key, size = key * cards[g] + X[ok, g], size * cards[g]
                tables.append((g, np.bincount(key * w + y[ok], minlength=size * w).reshape(size, w).astype(float)))
            self.levels.append((base, tables))
        return self

    def predict(self, X):
        out = []
        for base, tables in self.levels:
            run = np.tile(base, (len(X), 1))
            key = np.zeros(len(X), dtype=np.int64)
            for g, table in tables:
                key = key * self.cards[g] + X[:, g]
                counts = table[key]
                run = (counts + self.alpha * run) / (counts.sum(1, keepdims=True) + self.alpha)
            out.append(run)
        return out


class Net:
    def __init__(self, hidden=(), dropout=0.0, wd=1e-4, lr=3e-3, seeds=3, max_steps=4000, every=20, patience=15):
        self.hidden, self.dropout, self.wd, self.lr = hidden, dropout, wd, lr
        self.seeds, self.max_steps, self.every, self.patience = seeds, max_steps, every, patience

    def _inputs(self, X):
        torch = self.torch
        rows = torch.arange(len(X), device=self.device).repeat_interleave(X.shape[1])
        cols = torch.as_tensor((X + self.offsets).reshape(-1), device=self.device)
        x = torch.zeros(len(X), self.dim, device=self.device)
        x[rows, cols] = 1.0
        return x

    def _build(self):
        nn = self.torch.nn
        layers, width = [], self.dim
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.GELU(), nn.Dropout(self.dropout)]
            width = h
        layers.append(nn.Linear(width, self.J * self.K))
        return nn.Sequential(*layers).to(self.device)

    def _optimizer(self, net):
        return self.torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)

    def _score(self, net, x, t):
        return self._loss(net, x, t)

    def _loss(self, net, x, t):
        torch = self.torch
        logp = torch.log_softmax(net(x).view(-1, self.J, self.K) + self.mask, -1)
        valid = t >= 0
        picked = logp.gather(-1, t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return -(picked * valid).sum() / valid.sum().clamp(min=1)

    def fit(self, X, T, widths, cards):
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cards = np.asarray(cards)
        self.offsets = np.concatenate([[0], np.cumsum(cards)[:-1]]).astype(np.int64)
        self.dim, self.J, self.K, self.widths = int(cards.sum()), len(widths), int(max(widths)), list(widths)
        self.mask = torch.full((self.J, self.K), -1e9, device=self.device)
        for j, w in enumerate(widths):
            self.mask[j, :w] = 0.0
        x = self._inputs(X)
        t = torch.as_tensor(T, device=self.device)
        n = len(X)
        self.nets = []
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            order = np.random.default_rng(seed).permutation(n)
            hold = max(1, n // 10)
            val, train = torch.as_tensor(order[:hold], device=self.device), torch.as_tensor(order[hold:], device=self.device)
            net = self._build()
            opt = self._optimizer(net)
            best, state, bad = np.inf, None, 0
            for step in range(self.max_steps):
                net.train()
                loss = self._loss(net, x[train], t[train])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % self.every == 0:
                    net.eval()
                    with torch.no_grad():
                        score = float(self._score(net, x[val], t[val]))
                    if score < best - 1e-5:
                        best, bad = score, 0
                        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    else:
                        bad += 1
                        if bad > self.patience:
                            break
            net.load_state_dict(state)
            net.eval()
            self.nets.append(net)
        return self

    def predict(self, X):
        torch = self.torch
        x = self._inputs(X)
        with torch.no_grad():
            p = sum(torch.softmax(net(x).view(-1, self.J, self.K) + self.mask, -1) for net in self.nets) / len(self.nets)
        p = p.double().cpu().numpy()
        return [p[:, j, :w] for j, w in enumerate(self.widths)]


class SemNet(Net):
    def __init__(self, rank=64, wd_free=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.rank, self.wd_free, self.E = rank, wd_free, None

    def attach(self, inst, mode):
        from .text import option_matrix
        self.E = option_matrix(inst, mode)

    def _build(self):
        torch, nn = self.torch, self.torch.nn
        E = torch.as_tensor(self.E, device=self.device)
        rank, dropout, hidden, dim = self.rank, self.dropout, self.hidden, self.dim

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                layers, width = [], dim
                for h in tuple(hidden) + (rank,):
                    layers += [nn.Linear(width, h), nn.GELU(), nn.Dropout(dropout)]
                    width = h
                self.body = nn.Sequential(*layers)
                self.A = nn.Linear(E.shape[1], rank, bias=False)
                self.R = nn.Parameter(torch.zeros(E.shape[0], rank))
                self.b = nn.Parameter(torch.zeros(E.shape[0]))
                self.register_buffer("E", E)

            def forward(self, x):
                return self.body(x) @ (self.A(self.E) + self.R).T + self.b

        return Head().to(self.device)

    def _optimizer(self, net):
        free = [net.R]
        shared = [p for n, p in net.named_parameters() if n != "R"]
        return self.torch.optim.AdamW([{"params": shared, "weight_decay": self.wd},
                                       {"params": free, "weight_decay": self.wd_free}], lr=self.lr)


class TabMNet(Net):
    def __init__(self, k=16, clip=1.0, **kwargs):
        kwargs.setdefault("wd", 3e-4)
        super().__init__(**kwargs)
        self.k, self.clip = k, clip

    def _build(self):
        torch, nn = self.torch, self.torch.nn
        k, J, K, dropout = self.k, self.J, self.K, self.dropout
        widths = [self.dim] + list(self.hidden)
        signs = torch.cat([torch.where(torch.rand(k, 1) < 0.5, -1.0, 1.0).expand(k, int(c)) for c in self.cards], 1)

        class Layer(nn.Module):
            def __init__(self, din, dout, r0=None):
                super().__init__()
                bound = din ** -0.5
                self.W = nn.Parameter(torch.empty(dout, din).uniform_(-bound, bound))
                self.r = nn.Parameter(r0.clone() if r0 is not None else torch.ones(k, din))
                self.s = nn.Parameter(torch.ones(k, dout))
                self.b = nn.Parameter(torch.empty(dout).uniform_(-bound, bound).repeat(k, 1))

            def forward(self, x):
                return ((x * self.r) @ self.W.T) * self.s + self.b

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(Layer(widths[i], widths[i + 1], signs if i == 0 else None)
                                            for i in range(len(widths) - 1))
                self.drop = nn.Dropout(dropout)
                bound = widths[-1] ** -0.5
                self.head_W = nn.Parameter(torch.empty(k, widths[-1], J * K).uniform_(-bound, bound))
                self.head_b = nn.Parameter(torch.empty(k, J * K).uniform_(-bound, bound))

            def forward(self, x):
                h = x.unsqueeze(1).expand(-1, k, -1)
                for layer in self.layers:
                    h = self.drop(torch.nn.functional.gelu(layer(h)))
                return torch.einsum("nkd,kdo->nko", h, self.head_W) + self.head_b

        return Model().to(self.device)

    def _logp(self, net, x):
        return self.torch.log_softmax(net(x).view(len(x), self.k, self.J, self.K) + self.mask, -1)

    def _loss(self, net, x, t):
        torch = self.torch
        logp = self._logp(net, x)
        valid = (t >= 0).unsqueeze(1).expand(-1, self.k, -1)
        picked = logp.gather(-1, t.clamp(min=0).unsqueeze(1).expand(-1, self.k, -1).unsqueeze(-1)).squeeze(-1)
        return -(picked * valid).sum() / valid.sum().clamp(min=1)

    def _score(self, net, x, t):
        torch = self.torch
        logp = torch.logsumexp(self._logp(net, x), 1) - np.log(self.k)
        valid = t >= 0
        picked = logp.gather(-1, t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return -(picked * valid).sum() / valid.sum().clamp(min=1)

    def _optimizer(self, net):
        opt = self.torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        params, step = list(net.parameters()), opt.step

        def clipped(*args, **kwargs):
            self.torch.nn.utils.clip_grad_norm_(params, self.clip)
            return step(*args, **kwargs)

        opt.step = clipped
        return opt

    def predict(self, X):
        torch = self.torch
        x = self._inputs(X)
        with torch.no_grad():
            p = sum(torch.softmax(net(x).view(len(x), self.k, self.J, self.K) + self.mask, -1).mean(1)
                    for net in self.nets) / len(self.nets)
        p = p.double().cpu().numpy()
        return [p[:, j, :w] for j, w in enumerate(self.widths)]


class GBDT:
    def __init__(self, leaves=15, rate=0.05, rounds=600, holdout=0.15, seed=0, threads=8):
        self.leaves, self.rate, self.rounds, self.holdout, self.seed, self.threads = leaves, rate, rounds, holdout, seed, threads

    def fit(self, X, T, widths, cards):
        import lightgbm as lgb
        rng = np.random.default_rng(self.seed)
        cats = list(range(X.shape[1]))
        self.models = []
        for j, w in enumerate(widths):
            y = T[:, j]
            rows = np.flatnonzero(y >= 0)
            present = np.unique(y[rows])
            if len(present) < 2 or len(rows) < 30:
                self.models.append(("const", smoothed(y, w)))
                continue
            remap = np.full(w, -1)
            remap[present] = np.arange(len(present))
            params = {"objective": "multiclass", "num_class": len(present), "learning_rate": self.rate,
                      "num_leaves": self.leaves, "min_data_in_leaf": 20, "lambda_l2": 1.0, "feature_fraction": 0.8,
                      "bagging_fraction": 0.8, "bagging_freq": 1, "cat_smooth": 10, "min_data_per_group": 20,
                      "max_cat_to_onehot": 8, "num_threads": self.threads, "seed": self.seed, "verbose": -1}
            if len(present) == 2:
                params.update(objective="binary", num_class=1)
            valid = rng.random(len(rows)) < self.holdout
            fit_rows, val_rows = rows[~valid], rows[valid]
            train = lgb.Dataset(X[fit_rows], remap[y[fit_rows]], categorical_feature=cats, free_raw_data=False)
            val = lgb.Dataset(X[val_rows], remap[y[val_rows]], categorical_feature=cats, reference=train)
            booster = lgb.train(params, train, self.rounds, valid_sets=[val], callbacks=[lgb.early_stopping(30, verbose=False)])
            best = max(1, booster.best_iteration or self.rounds)
            full = lgb.Dataset(X[rows], remap[y[rows]], categorical_feature=cats)
            booster = lgb.train(params, full, int(best * 1.1) + 1)
            self.models.append(("gbdt", (booster, present, w, smoothed(y, w))))
        return self

    def predict(self, X):
        out = []
        for kind, m in self.models:
            if kind == "const":
                out.append(np.tile(m, (len(X), 1)))
                continue
            booster, present, w, prior = m
            raw = booster.predict(X)
            raw = np.column_stack([1 - raw, raw]) if raw.ndim == 1 else raw
            p = np.full((len(X), w), 1e-4)
            p[:, present] += raw
            out.append(0.98 * p / p.sum(1, keepdims=True) + 0.02 * prior[None, :])
        return out


class LatentClass(Net):
    def __init__(self, classes=32, hidden=256, dropout=0.1, lr=3e-3, wd=1e-4, seeds=2, joint_steps=300,
                 max_steps=3000, every=20, patience=15, batch=1024):
        super().__init__(hidden=(hidden,), dropout=dropout, wd=wd, lr=lr, seeds=seeds, max_steps=max_steps, every=every, patience=patience)
        self.classes, self.joint_steps, self.batch = classes, joint_steps, batch

    def _build(self):
        torch, nn = self.torch, self.torch.nn

        class Model(nn.Module):
            def __init__(inner):
                super().__init__()
                inner.enc = nn.Sequential(nn.Linear(self.dim, self.hidden[0]), nn.GELU(), nn.Dropout(self.dropout),
                                          nn.Linear(self.hidden[0], self.classes))
                inner.theta = nn.Parameter(0.01 * torch.randn(self.J, self.classes, self.K))

        return Model().to(self.device)

    def _parts(self, net, x, t):
        torch = self.torch
        logpi = torch.log_softmax(net.enc(x), -1)
        logth = torch.log_softmax(net.theta + self.mask[:, None, :], -1).permute(0, 2, 1)
        valid = t >= 0
        picked = logth[torch.arange(self.J, device=self.device)[None, :], t.clamp(min=0)]
        return logpi, picked, valid

    def _joint(self, net, x, t):
        torch = self.torch
        logpi, picked, valid = self._parts(net, x, t)
        return -torch.logsumexp(logpi + (picked * valid[..., None]).sum(1), -1).mean() / self.J

    def _loss(self, net, x, t):
        torch = self.torch
        logpi, picked, valid = self._parts(net, x, t)
        cell = torch.logsumexp(logpi[:, None, :] + picked, -1)
        return -(cell * valid).sum() / valid.sum().clamp(min=1)

    def fit(self, X, T, widths, cards):
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.offsets = np.concatenate([[0], np.cumsum(cards)[:-1]]).astype(np.int64)
        self.dim, self.J, self.K, self.widths = int(np.sum(cards)), len(widths), int(max(widths)), list(widths)
        self.mask = torch.full((self.J, self.K), -1e9, device=self.device)
        for j, w in enumerate(widths):
            self.mask[j, :w] = 0.0
        x = self._inputs(X)
        t = torch.as_tensor(T, device=self.device)
        n = len(X)
        self.nets = []
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            order = rng.permutation(n)
            hold = max(1, n // 10)
            val, train = torch.as_tensor(order[:hold], device=self.device), order[hold:]
            net = self._build()
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            for step in range(self.joint_steps):
                b = torch.as_tensor(rng.choice(train, min(self.batch, len(train)), replace=False), device=self.device)
                net.train()
                loss = self._joint(net, x[b], t[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            best, state, bad = np.inf, None, 0
            for step in range(self.max_steps):
                b = torch.as_tensor(rng.choice(train, min(self.batch, len(train)), replace=False), device=self.device)
                net.train()
                loss = self._loss(net, x[b], t[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % self.every == 0:
                    net.eval()
                    with torch.no_grad():
                        score = float(self._loss(net, x[val], t[val]))
                    if score < best - 1e-5:
                        best, bad = score, 0
                        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    else:
                        bad += 1
                        if bad > self.patience:
                            break
            net.load_state_dict(state)
            net.eval()
            self.nets.append(net)
        return self

    def predict(self, X):
        torch = self.torch
        x = self._inputs(X)
        out = 0
        with torch.no_grad():
            for net in self.nets:
                pi = torch.softmax(net.enc(x), -1)
                th = torch.softmax(net.theta + self.mask[:, None, :], -1)
                out = out + torch.einsum("bk,jkw->bjw", pi, th)
        p = (out / len(self.nets)).double().cpu().numpy()
        return [p[:, j, :w] for j, w in enumerate(self.widths)]


def teacher_soft(X, Y, widths, cards, folds=5, seed=0, steps=2000, batch=512, hidden=(512, 256), lr=2e-3, device="cuda"):
    import torch
    nn = torch.nn
    n, J = Y.shape
    K = int(max(widths))
    G = int(np.sum(cards))
    goff = torch.as_tensor(np.concatenate([[0], np.cumsum(cards)[:-1]]), device=device)
    ioff = torch.as_tensor(np.concatenate([[0], np.cumsum(widths)[:-1]]), device=device)
    D = G + int(np.sum(widths)) + J
    mask = torch.full((J, K), -1e9, device=device)
    for j, w in enumerate(widths):
        mask[j, :w] = 0.0
    Xt, Yt = torch.as_tensor(X, device=device), torch.as_tensor(Y, device=device)

    def inputs(rows, vis):
        B = len(rows)
        x = torch.zeros(B, D, device=device)
        r = torch.arange(B, device=device)[:, None]
        x[r.expand(-1, X.shape[1]), Xt[rows] + goff] = 1.0
        y = Yt[rows]
        show = vis & (y >= 0)
        cols = G + ioff + y.clamp(min=0)
        x[r.expand(-1, J)[show], cols[show]] = 1.0
        x[:, G + int(np.sum(widths)):] = vis.float()
        return x

    def build():
        layers, width = [], D
        for h in hidden:
            layers += [nn.Linear(width, h), nn.GELU(), nn.Dropout(0.1)]
            width = h
        layers.append(nn.Linear(width, J * K))
        return nn.Sequential(*layers).to(device)

    def loss_on(net, rows, gen):
        r = torch.rand(len(rows), 1, device=device, generator=gen)
        vis = torch.rand(len(rows), J, device=device, generator=gen) > r
        logp = torch.log_softmax(net(inputs(rows, vis)).view(-1, J, K) + mask, -1)
        y = Yt[rows]
        target = (~vis) & (y >= 0)
        picked = logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return -(picked * target).sum() / target.sum().clamp(min=1)

    Q = np.zeros((n, J, K))
    order = np.random.default_rng(seed).permutation(n)
    for f, part in enumerate(np.array_split(order, folds)):
        torch.manual_seed(seed * 100 + f)
        gen = torch.Generator(device=device).manual_seed(seed * 100 + f)
        rest = np.setdiff1d(order, part)
        hold = rest[:max(1, len(rest) // 10)]
        train = torch.as_tensor(rest[len(hold):], device=device)
        hold = torch.as_tensor(hold, device=device)
        net = build()
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        best, state, bad = np.inf, None, 0
        for step in range(steps):
            net.train()
            b = train[torch.randint(len(train), (batch,), device=device, generator=gen)]
            loss = loss_on(net, b, gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 49:
                net.eval()
                with torch.no_grad():
                    v = float(loss_on(net, hold, torch.Generator(device=device).manual_seed(7)))
                if v < best - 1e-5:
                    best, bad, state = v, 0, {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad > 6:
                        break
        net.load_state_dict(state)
        net.eval()
        eye = torch.eye(J, dtype=torch.bool, device=device)
        with torch.no_grad():
            for chunk in np.array_split(part, max(1, len(part) // 32)):
                rows = torch.as_tensor(chunk, device=device).repeat_interleave(J)
                vis = (~eye).repeat(len(chunk), 1)
                logp = torch.softmax(net(inputs(rows, vis)).view(len(chunk), J, J, K) + mask, -1)
                Q[chunk] = logp[:, torch.arange(J), torch.arange(J)].double().cpu().numpy()
    return Q


class DistilTabM(TabMNet):
    def __init__(self, lam=0.5, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam

    def attach(self, inst, mode):
        self.inst, self.mode = inst, mode

    def fit(self, X, T, widths, cards):
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        inst = self.inst
        Q = teacher_soft(X, inst.Y[self.rows], inst.widths, cards, device=self.device)
        self.cards = np.asarray(cards)
        self.offsets = np.concatenate([[0], np.cumsum(cards)[:-1]]).astype(np.int64)
        self.dim, self.J, self.K, self.widths = int(cards.sum()), len(widths), int(max(widths)), list(widths)
        self.mask = torch.full((self.J, self.K), -1e9, device=self.device)
        for j, w in enumerate(widths):
            self.mask[j, :w] = 0.0
        n = len(X)
        S = np.zeros((n, self.J, self.K))
        S[np.arange(n)[:, None], np.arange(self.J)[None, :], np.clip(T, 0, None)] = 1.0
        soft = Q[:, :, :self.K].copy()
        if self.mode != "direct":
            soft = soft[:, :, :self.K] * (np.arange(self.K)[None, None, :] < np.asarray(widths)[None, :, None])
            soft = soft / np.maximum(soft.sum(-1, keepdims=True), 1e-12)
        S = self.lam * S + (1 - self.lam) * soft
        x = self._inputs(X)
        St = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        t = torch.as_tensor(T, device=self.device)
        self.nets = []
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            order = np.random.default_rng(seed).permutation(n)
            hold = max(1, n // 10)
            val, train = torch.as_tensor(order[:hold], device=self.device), torch.as_tensor(order[hold:], device=self.device)
            net = self._build()
            opt = self._optimizer(net)
            best, state, bad = np.inf, None, 0
            for step in range(self.max_steps):
                net.train()
                logp = self._logp(net, x[train])
                valid = (t[train] >= 0).float()
                loss = -((logp * St[train].unsqueeze(1)).sum(-1) * valid.unsqueeze(1)).sum() / (valid.sum() * self.k).clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % self.every == 0:
                    net.eval()
                    with torch.no_grad():
                        score = float(self._score(net, x[val], t[val]))
                    if score < best - 1e-5:
                        best, bad = score, 0
                        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    else:
                        bad += 1
                        if bad > self.patience:
                            break
            net.load_state_dict(state)
            net.eval()
            self.nets.append(net)
        return self


class SubBackoff(Backoff):
    def predict(self, X):
        g, level = self.focus
        inside = (X[:, g] == level)[:, None]
        return [np.where(inside, p, np.nan) for p in super().predict(X)]


class SubTabM(TabMNet):
    def predict(self, X):
        g, level = self.focus
        inside = (X[:, g] == level)[:, None]
        return [np.where(inside, p, np.nan) for p in super().predict(X)]
