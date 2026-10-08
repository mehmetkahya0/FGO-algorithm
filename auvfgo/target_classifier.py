"""Çok modlu (akustik + optik + manyetik) hedef sınıflandırma.

- Her nesnenin sınıfa bağlı gizli (latent) bir imzası vardır; her gözlem bu imzanın
  menzile bağlı gürültülü bir örneğidir.
- Sınıflandırıcı: numpy ile sıfırdan yazılmış çok katmanlı algılayıcı (MLP),
  Adam optimizasyonu, L2 düzenlileştirme ve *modalite bırakma* (modality dropout)
  ile eğitilir; böylece optik/manyetik veri olmadığında da çalışır (geç olmayan,
  öznitelik düzeyinde erken füzyon + eksik-modalite maskeleri).
- Görev sırasında aynı nesnenin ardışık gözlemleri Bayesçi log-olasılık toplamı ile
  birleştirilir (zamansal füzyon).
"""
import numpy as np

N_CLASSES = 3
# sonar: [hedef gücü dB, gölge düzenliliği, boyut m, parlak-bölge düzenliliği]
SONAR_MEAN = np.array([[12.0, 0.55, 2.0, 0.45],
                       [15.0, 0.85, 1.4, 0.75],
                       [16.0, 0.60, 2.6, 0.60]])
SONAR_STD = np.array([[4.0, 0.20, 0.8, 0.18],
                      [3.0, 0.15, 0.4, 0.15],
                      [4.0, 0.20, 1.0, 0.18]])
# optik: [renk kontrastı, kenar doğrusallığı, simetri]
OPT_MEAN = np.array([[0.35, 0.30, 0.40],
                     [0.70, 0.85, 0.80],
                     [0.60, 0.70, 0.35]])
OPT_STD = np.array([[0.15, 0.15, 0.15],
                    [0.15, 0.10, 0.12],
                    [0.18, 0.15, 0.15]])
# manyetik artık [nT] (ölçülen - önsel harita)
MAG_MEAN = np.array([0.0, 28.0, 14.0])
MAG_STD = np.array([3.0, 8.0, 8.0])

N_FEAT = 4 + 3 + 1 + 2 + 1  # sonar, optik, manyetik, 2 maske, menzil
SONAR_COLS = [0, 1, 2, 3, 10]
OPT_COLS = [4, 5, 6, 8]
MAG_COLS = [7, 9]


def sample_latent(cls, rng):
    """Nesneye özgü sabit imza (nesneler arası değişkenlik)."""
    return {
        "sonar": SONAR_MEAN[cls] + rng.normal(0, 0.8 * SONAR_STD[cls]),
        "opt": OPT_MEAN[cls] + rng.normal(0, 0.8 * OPT_STD[cls]),
        "mag": MAG_MEAN[cls] + rng.normal(0, 0.8 * MAG_STD[cls]),
    }


def observe_features(latent, cls, rho, has_opt, has_mag, rng):
    """Tek bir gözlemden öznitelik vektörü üret (eksik modaliteler 0 + maske)."""
    f = np.zeros(N_FEAT)
    k = 0.4 + rho / 70.0  # menzil arttıkça akustik gürültü artar
    f[0:4] = latent["sonar"] + rng.normal(0, 0.6 * k * SONAR_STD[cls])
    if has_opt:
        f[4:7] = latent["opt"] + rng.normal(0, 0.4 * OPT_STD[cls])
        f[8] = 1.0
    if has_mag:
        f[7] = latent["mag"] + rng.normal(0, 3.0)
        f[9] = 1.0
    f[10] = rho / 70.0
    return f


def make_dataset(n, rng, class_probs=(0.6, 0.2, 0.2)):
    y = rng.choice(N_CLASSES, size=n, p=class_probs)
    X = np.zeros((n, N_FEAT))
    for i, c in enumerate(y):
        lat = sample_latent(c, rng)
        rho = rng.uniform(5, 70)
        has_opt = rng.random() < (0.6 if rho < 20 else 0.15)
        has_mag = rng.random() < (0.45 if rho < 20 else 0.1)
        X[i] = observe_features(lat, c, rho, has_opt, has_mag, rng)
    return X, y


class MLPClassifier:
    """İki gizli katmanlı ReLU MLP + softmax (numpy)."""

    def __init__(self, n_in=N_FEAT, hidden=(48, 32), n_out=N_CLASSES, seed=0, cols=None):
        rng = np.random.default_rng(seed)
        self.cols = np.arange(n_in) if cols is None else np.asarray(cols)
        sizes = [len(self.cols), *hidden, n_out]
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mu = None
        self.sd = None

    # -------------------------------------------------------------- yardımcı
    def _prep(self, X):
        X = X[:, self.cols].copy()
        return (X - self.mu) / self.sd

    def _forward(self, A):
        acts = [A]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = acts[-1] @ W + b
            acts.append(np.maximum(Z, 0) if i < len(self.W) - 1 else Z)
        Z = acts[-1]
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        return P, acts

    @staticmethod
    def _modality_dropout(X, rng, p_opt=0.3, p_mag=0.3):
        X = X.copy()
        d_opt = (rng.random(len(X)) < p_opt) & (X[:, 8] > 0)
        X[d_opt, 4:7] = 0.0
        X[d_opt, 8] = 0.0
        d_mag = (rng.random(len(X)) < p_mag) & (X[:, 9] > 0)
        X[d_mag, 7] = 0.0
        X[d_mag, 9] = 0.0
        return X

    # -------------------------------------------------------------- eğitim
    def fit(self, X, y, epochs=60, lr=3e-3, batch=128, l2=1e-4, seed=0, modality_dropout=True,
            class_weight=None):
        rng = np.random.default_rng(seed)
        Xc = X[:, self.cols]
        self.mu = Xc.mean(axis=0)
        self.sd = Xc.std(axis=0) + 1e-6
        # maske ve eksik değer sütunları ölçeklenmesin
        for j, c in enumerate(self.cols):
            if c in (8, 9):
                self.mu[j], self.sd[j] = 0.0, 1.0
        cw = np.ones(N_CLASSES) if class_weight is None else np.asarray(class_weight)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        step = 0
        history = []
        n = len(X)
        for ep in range(epochs):
            perm = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                Xb = self._modality_dropout(X[idx], rng) if modality_dropout else X[idx]
                yb = y[idx]
                A = self._prep(Xb)
                P, acts = self._forward(A)
                wts = cw[yb]
                loss = -np.sum(wts * np.log(P[np.arange(len(yb)), yb] + 1e-12)) / len(yb)
                tot += loss * len(yb)
                G = P.copy()
                G[np.arange(len(yb)), yb] -= 1.0
                G *= wts[:, None] / len(yb)
                gW, gb = [None] * len(self.W), [None] * len(self.b)
                for i in range(len(self.W) - 1, -1, -1):
                    gW[i] = acts[i].T @ G + l2 * self.W[i]
                    gb[i] = G.sum(axis=0)
                    if i > 0:
                        G = (G @ self.W[i].T) * (acts[i] > 0)
                grads = gW + gb
                step += 1
                for j, (p, g) in enumerate(zip(params, grads)):
                    m[j] = 0.9 * m[j] + 0.1 * g
                    v[j] = 0.999 * v[j] + 0.001 * g * g
                    mh = m[j] / (1 - 0.9 ** step)
                    vh = v[j] / (1 - 0.999 ** step)
                    p -= lr * mh / (np.sqrt(vh) + 1e-8)
            history.append(tot / n)
        return history

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        P, _ = self._forward(self._prep(X))
        return P

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def metrics(y_true, y_pred, n=N_CLASSES):
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = np.trace(cm) / max(cm.sum(), 1)
    f1s = []
    for k in range(n):
        tp = cm[k, k]
        prec = tp / max(cm[:, k].sum(), 1)
        rec = tp / max(cm[k, :].sum(), 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    mine_recall = cm[1, 1] / max(cm[1].sum(), 1)
    return {"accuracy": float(acc), "macro_f1": float(np.mean(f1s)),
            "mine_recall": float(mine_recall), "confusion": cm.tolist()}


def train_models(seed=0, n_train=8000, n_test=4000, epochs=60):
    """Modalite ablasyonu: yalnız sonar / sonar+optik / tam füzyon."""
    rng = np.random.default_rng(seed)
    Xtr, ytr = make_dataset(n_train, rng)
    Xte, yte = make_dataset(n_test, rng)
    variants = {
        "Yalnız sonar": SONAR_COLS,
        "Sonar + optik": SONAR_COLS + OPT_COLS,
        "Sonar + manyetik": SONAR_COLS + MAG_COLS,
        "Tam füzyon (sonar+optik+manyetik)": list(range(N_FEAT)),
    }
    results, models, histories = {}, {}, {}
    cw = np.array([1.0, 2.0, 1.5])
    for name, cols in variants.items():
        mdl = MLPClassifier(cols=cols, seed=seed)
        histories[name] = mdl.fit(Xtr, ytr, epochs=epochs, seed=seed, class_weight=cw,
                                  modality_dropout=(len(cols) == N_FEAT))
        res = metrics(yte, mdl.predict(Xte))
        # yalnızca optik+manyetik verinin mevcut olduğu yakın gözlemler
        near = (Xte[:, 8] > 0)
        res["accuracy_optical_available"] = float(np.mean(mdl.predict(Xte[near]) == yte[near]))
        results[name] = res
        models[name] = mdl
    return models, results, histories


class TemporalFusion:
    """Nirengi/nesne başına Bayesçi zamansal sınıf füzyonu."""

    def __init__(self, temperature=0.5, prior=(0.6, 0.2, 0.2)):
        self.T = temperature
        self.logprior = np.log(np.asarray(prior))
        self.logp = {}
        self.count = {}

    def add(self, key, probs):
        lp = self.T * np.log(np.clip(probs, 1e-6, 1.0))
        self.logp[key] = self.logp.get(key, self.logprior.copy()) + lp
        self.count[key] = self.count.get(key, 0) + 1

    def posterior(self, key):
        lp = self.logp[key]
        p = np.exp(lp - lp.max())
        return p / p.sum()
