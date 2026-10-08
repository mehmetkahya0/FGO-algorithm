"""Faktör Grafiği Optimizasyonu (FGO) — sıfırdan, vektörize ve seyrek.

MAP kestirimi:   X* = argmin_X  sum_i rho_i( || W_i r_i(X) ||^2 )

- Her faktör tipi bir `FactorBlock` içinde toplu (vektörize) değerlendirilir.
- Sağlam (robust) çekirdekler (Huber, Cauchy) IRLS ağırlıkları ile uygulanır:
  aykırı sonar eşleşmeleri ve haritada olmayan manyetik anomaliler bastırılır.
- Çözücü: seyrek Jacobian + Levenberg–Marquardt (SuperLU ile normal denklemler).
- Marjinal kovaryans: bilgi matrisinin (J^T J) seyrek LU ile kısmi tersi.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .utils import wrap, landmark_model

POSE_DIM = 5   # [x, y, z, psi, b_g]
LM_DIM = 3


# ----------------------------------------------------------------- çekirdekler
def robust_weight(e, kernel, k):
    if kernel == "huber":
        return np.where(e <= k, 1.0, k / np.maximum(e, 1e-12))
    if kernel == "cauchy":
        return 1.0 / (1.0 + (e / k) ** 2)
    return np.ones_like(e)


def robust_cost(e, kernel, k):
    if kernel == "huber":
        return np.where(e <= k, 0.5 * e * e, k * (e - 0.5 * k))
    if kernel == "cauchy":
        return 0.5 * k * k * np.log1p((e / k) ** 2)
    return 0.5 * e * e


# ----------------------------------------------------------------- faktörler
class FactorBlock:
    dim = 0
    var_dims = ()

    def __init__(self, kernel="none", k=1.0):
        self.kernel, self.k = kernel, k
        self._offs = [[] for _ in self.var_dims]
        self._data = {}
        self._cache = None
        self.count = 0

    def _append(self, offsets, **data):
        for lst, o in zip(self._offs, offsets):
            lst.append(int(o))
        for name, v in data.items():
            self._data.setdefault(name, []).append(np.asarray(v, float))
        self.count += 1
        self._cache = None

    def arrays(self):
        if self._cache is None:
            offs = [np.asarray(o, dtype=np.int64) for o in self._offs]
            data = {k: np.stack(v) for k, v in self._data.items()}
            self._cache = (offs, data)
        return self._cache

    @staticmethod
    def gather(x, off, dv):
        return x[off[:, None] + np.arange(dv)]

    def evaluate(self, x):
        """Beyazlatılmış artık (m, dim) ve değişken başına Jacobian listesi."""
        raise NotImplementedError


class PriorFactor(FactorBlock):
    dim, var_dims = POSE_DIM, (POSE_DIM,)

    def add(self, off, mean, W):
        self._append([off], mean=mean, W=W)

    def evaluate(self, x):
        (off,), d = self.arrays()
        r = self.gather(x, off, POSE_DIM) - d["mean"]
        r[:, 3] = wrap(r[:, 3])
        W = d["W"]
        return np.einsum("mij,mj->mi", W, r), [W.copy()]


class OdometryFactor(FactorBlock):
    """DVL + jiroskop ön-entegrasyonu (preintegration) ile ardışık pozlar arası faktör.

    g    = (1 + s * f_dvl) (dp + J_p (b_i - b_hat))      (s: DVL ölçek hatası, f_dvl: DVL'li örnek oranı)
    r_xy = R(psi_i)^T (p_j - p_i) - g
    r_z  = (z_j - z_i) - dz
    r_psi= wrap(psi_j - psi_i - (dth + J_th (b_i - b_hat)))
    r_b  = b_j - b_i                       (jiroskop sapması rastgele yürüyüş)
    """
    dim, var_dims = POSE_DIM, (POSE_DIM, POSE_DIM, 1)

    def add(self, oi, oj, os_, odom, W):
        self._append([oi, oj, os_], dp=odom["dp"], dz=odom["dz"], dth=odom["dth"],
                     Jp=odom["J_p"], Jth=odom["J_th"], bhat=odom["b_hat"], f=odom["dvl_frac"], W=W)

    def evaluate(self, x):
        (oi, oj, osc), d = self.arrays()
        Xi = self.gather(x, oi, POSE_DIM)
        Xj = self.gather(x, oj, POSE_DIM)
        sc = 1.0 + x[osc] * d["f"]
        m = len(oi)
        c, s = np.cos(Xi[:, 3]), np.sin(Xi[:, 3])
        dx = Xj[:, 0] - Xi[:, 0]
        dy = Xj[:, 1] - Xi[:, 1]
        db = Xi[:, 4] - d["bhat"]
        gx = d["dp"][:, 0] + d["Jp"][:, 0] * db
        gy = d["dp"][:, 1] + d["Jp"][:, 1] * db
        r = np.empty((m, 5))
        r[:, 0] = c * dx + s * dy - sc * gx
        r[:, 1] = -s * dx + c * dy - sc * gy
        r[:, 2] = (Xj[:, 2] - Xi[:, 2]) - d["dz"]
        r[:, 3] = wrap(Xj[:, 3] - Xi[:, 3] - (d["dth"] + d["Jth"] * db))
        r[:, 4] = Xj[:, 4] - Xi[:, 4]
        Ji = np.zeros((m, 5, 5))
        Jj = np.zeros((m, 5, 5))
        Js = np.zeros((m, 5, 1))
        Ji[:, 0, 0], Ji[:, 0, 1] = -c, -s
        Ji[:, 1, 0], Ji[:, 1, 1] = s, -c
        Ji[:, 0, 3] = -s * dx + c * dy
        Ji[:, 1, 3] = -c * dx - s * dy
        Ji[:, 0, 4] = -sc * d["Jp"][:, 0]
        Ji[:, 1, 4] = -sc * d["Jp"][:, 1]
        Ji[:, 2, 2] = -1.0
        Ji[:, 3, 3] = -1.0
        Ji[:, 3, 4] = -d["Jth"]
        Ji[:, 4, 4] = -1.0
        Jj[:, 0, 0], Jj[:, 0, 1] = c, s
        Jj[:, 1, 0], Jj[:, 1, 1] = -s, c
        Jj[:, 2, 2] = 1.0
        Jj[:, 3, 3] = 1.0
        Jj[:, 4, 4] = 1.0
        Js[:, 0, 0] = -d["f"] * gx
        Js[:, 1, 0] = -d["f"] * gy
        W = d["W"]
        return (np.einsum("mij,mj->mi", W, r),
                [np.einsum("mij,mjk->mik", W, Ji), np.einsum("mij,mjk->mik", W, Jj),
                 np.einsum("mij,mjk->mik", W, Js)])


class ScalarPoseFactor(FactorBlock):
    """Pozun tek bir bileşenine doğrudan ölçüm (basınç->z, pusula->psi)."""
    dim, var_dims = 1, (POSE_DIM,)
    index = 2
    angular = False

    def add(self, off, z, sigma):
        self._append([off], z=z, sigma=sigma)

    def evaluate(self, x):
        (off,), d = self.arrays()
        r = x[off + self.index] - d["z"]
        if self.angular:
            r = wrap(r)
        inv = 1.0 / d["sigma"]
        J = np.zeros((len(off), 1, POSE_DIM))
        J[:, 0, self.index] = inv
        return (r * inv)[:, None], [J]


class DepthFactor(ScalarPoseFactor):
    index, angular = 2, False


class HeadingFactor(ScalarPoseFactor):
    index, angular = 3, True


class HeadingCalibFactor(FactorBlock):
    """Pusula ölçümü + kalibrasyon değişkenleri q = [c, k]:

        r = wrap(psi_i + c + k * M(x_i, y_i) / 100 - psi_m)

    c: montaj (sabit) sapması, k: yerel manyetik anomaliye bağlı sapma katsayısı
    [rad / 100 nT]. M, önsel manyetik anomali haritasıdır. Böylece pusulanın
    yer-bağımlı (zamanla ilişkili) hatası da grafın içinde kalibre edilir."""
    dim, var_dims = 1, (POSE_DIM, 2)

    def __init__(self, mag_map, **kw):
        super().__init__(**kw)
        self.map = mag_map

    def add(self, off, off_q, z, sigma):
        self._append([off, off_q], z=z, sigma=sigma)

    def evaluate(self, x):
        (off, oq), d = self.arrays()
        inv = 1.0 / d["sigma"]
        px, py = x[off], x[off + 1]
        c, k = x[oq], x[oq + 1]
        M = self.map.value(px, py) / 100.0
        gx, gy = self.map.grad(px, py)
        r = wrap(x[off + 3] + c + k * M - d["z"]) * inv
        Jp = np.zeros((len(off), 1, POSE_DIM))
        Jp[:, 0, 0] = k * gx / 100.0 * inv
        Jp[:, 0, 1] = k * gy / 100.0 * inv
        Jp[:, 0, 3] = inv
        Jq = np.zeros((len(off), 1, 2))
        Jq[:, 0, 0] = inv
        Jq[:, 0, 1] = M * inv
        return r[:, None], [Jp, Jq]


class ScalarPrior(FactorBlock):
    dim, var_dims = 1, (1,)

    def add(self, off, mean, sigma):
        self._append([off], mean=mean, sigma=sigma)

    def evaluate(self, x):
        (off,), d = self.arrays()
        inv = 1.0 / d["sigma"]
        return ((x[off] - d["mean"]) * inv)[:, None], [inv[:, None, None].copy()]


class MagMapFactor(FactorBlock):
    """Manyetik anomali haritası eşleme (geofiziksel navigasyon):  r = M(x,y) - m."""
    dim, var_dims = 1, (POSE_DIM,)

    def __init__(self, mag_map, **kw):
        super().__init__(**kw)
        self.map = mag_map

    def add(self, off, m, sigma):
        self._append([off], m=m, sigma=sigma)

    def evaluate(self, x):
        (off,), d = self.arrays()
        px, py = x[off], x[off + 1]
        inv = 1.0 / d["sigma"]
        r = (self.map.value(px, py) - d["m"]) * inv
        gx, gy = self.map.grad(px, py)
        J = np.zeros((len(off), 1, POSE_DIM))
        J[:, 0, 0] = gx * inv
        J[:, 0, 1] = gy * inv
        return r[:, None], [J]


class LandmarkFactor(FactorBlock):
    """Sonar / optik kamera ile nirengi (deniz tabanı nesnesi) gözlemi — SLAM."""
    dim, var_dims = 3, (POSE_DIM, LM_DIM)

    def add(self, op, ol, meas, W):
        self._append([op, ol], meas=meas, W=W)

    def evaluate(self, x):
        (op, ol), d = self.arrays()
        P = self.gather(x, op, POSE_DIM)
        L = self.gather(x, ol, LM_DIM)
        h, Hp, Hl = landmark_model(P, L)
        r = h - d["meas"]
        W = d["W"]
        return (np.einsum("mij,mj->mi", W, r),
                [np.einsum("mij,mjk->mik", W, Hp), np.einsum("mij,mjk->mik", W, Hl)])


# ----------------------------------------------------------------- graf
class FactorGraph:
    def __init__(self):
        self.x = np.zeros(0)
        self.angle_idx = []
        self.blocks = []
        self.stats = []

    # değişkenler
    def add_variable(self, init, angle_index=None):
        off = len(self.x)
        self.x = np.concatenate([self.x, np.asarray(init, float)])
        if angle_index is not None:
            self.angle_idx.append(off + angle_index)
        return off

    def add_block(self, block):
        self.blocks.append(block)
        return block

    @property
    def n_factors(self):
        return sum(b.count for b in self.blocks)

    def _retract(self, x, dx):
        xn = x + dx
        if self.angle_idx:
            ai = np.asarray(self.angle_idx)
            xn[ai] = wrap(xn[ai])
        return xn

    # doğrusallaştırma
    def linearize(self, x, jac=True):
        rows, cols, vals, res = [], [], [], []
        cost = 0.0
        row0 = 0
        for blk in self.blocks:
            if blk.count == 0:
                continue
            r, Js = blk.evaluate(x)
            e = np.linalg.norm(r, axis=1)
            cost += float(np.sum(robust_cost(e, blk.kernel, blk.k)))
            if not jac:
                continue
            sw = np.sqrt(robust_weight(e, blk.kernel, blk.k))
            r = r * sw[:, None]
            m, dim = r.shape
            res.append(r.ravel())
            ridx = row0 + np.arange(m * dim).reshape(m, dim)
            offs, _ = blk.arrays()
            for J, off, dv in zip(Js, offs, blk.var_dims):
                J = J * sw[:, None, None]
                rows.append(np.broadcast_to(ridx[:, :, None], (m, dim, dv)).ravel())
                cols.append(np.broadcast_to(off[:, None, None] + np.arange(dv)[None, None, :],
                                            (m, dim, dv)).ravel())
                vals.append(J.ravel())
            row0 += m * dim
        if not jac:
            return None, None, cost
        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(row0, len(x)))
        return J, np.concatenate(res), cost

    def cost(self, x=None):
        return self.linearize(self.x if x is None else x, jac=False)[2]

    # Levenberg–Marquardt
    def optimize(self, max_iters=10, lam=1e-4, tol=1e-7):
        x = self.x.copy()
        J, r, cost = self.linearize(x)
        cost0 = cost
        it_done = 0
        for it in range(max_iters):
            H = (J.T @ J).tocsc()
            g = J.T @ r
            diag = np.maximum(H.diagonal(), 1e-9)
            accepted = False
            for _ in range(10):
                A = (H + sp.diags(lam * diag)).tocsc()
                try:
                    dx = splu(A).solve(-g)
                except RuntimeError:
                    lam *= 10.0
                    continue
                xn = self._retract(x, dx)
                cn = self.cost(xn)
                if cn < cost:
                    accepted = True
                    lam = max(lam / 3.0, 1e-9)
                    break
                lam *= 6.0
            it_done = it + 1
            if not accepted:
                break
            rel = (cost - cn) / max(cost, 1e-12)
            x = xn
            J, r, cost = self.linearize(x)
            if rel < tol or np.max(np.abs(dx)) < 1e-6:
                break
        self.x = x
        self.stats.append({"iters": it_done, "cost0": cost0, "cost": cost})
        return cost

    def information(self):
        J, _, _ = self.linearize(self.x)
        return (J.T @ J).tocsc()

    def marginal_covariances(self, offsets, dim, chunk=200):
        """Seçilen değişkenlerin marjinal kovaryansları (H^-1'in köşegen blokları)."""
        H = self.information()
        n = H.shape[0]
        lu = splu(H + sp.identity(n, format="csc") * 1e-10)
        out = []
        offsets = list(offsets)
        for s in range(0, len(offsets), chunk):
            part = offsets[s:s + chunk]
            E = np.zeros((n, dim * len(part)))
            for k, off in enumerate(part):
                E[off + np.arange(dim), k * dim + np.arange(dim)] = 1.0
            X = lu.solve(E)
            for k, off in enumerate(part):
                out.append(X[off:off + dim, k * dim:(k + 1) * dim])
        return out
