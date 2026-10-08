"""Navigasyon kestiricileri: FGO (ana yöntem), EKF-SLAM ve ölü hesap (DR) referansları."""
import numpy as np
from scipy.stats import chi2

from .factor_graph import (FactorGraph, PriorFactor, OdometryFactor, DepthFactor,
                           HeadingFactor, HeadingCalibFactor, ScalarPrior, MagMapFactor, LandmarkFactor,
                           POSE_DIM, LM_DIM)
from .utils import wrap, rot2, drot2, sqrt_info, landmark_model, body_to_world


# ------------------------------------------------------------ ön-entegrasyon
class OdometryPreintegrator:
    """İki anahtar kare arasındaki DVL hızlarını ve jiroskop açısal hızlarını
    pozun yerel çerçevesinde biriktirir; jiroskop sapmasına göre birinci dereceden
    Jacobian'ları (J_p, J_th) ve gürültü kovaryansını yayar."""

    def __init__(self, b_hat):
        self.b_hat = float(b_hat)
        self.dp = np.zeros(2)
        self.dz = 0.0
        self.th = 0.0
        self.J_p = np.zeros(2)
        self.J_th = 0.0
        self.T = 0.0
        self.P = np.zeros((4, 4))  # [dpx, dpy, dz, dth]
        self.n = 0
        self.n_dvl = 0

    def integrate(self, gyro, v_b, vz, dt, sig_v, sig_vz, sig_g, dvl_ok=True):
        R = rot2(self.th)
        dR = drot2(self.th)
        A = np.eye(4)
        A[0:2, 3] = dR @ v_b * dt
        self.J_p += dR @ v_b * dt * self.J_th
        self.dp += R @ v_b * dt
        self.dz += vz * dt
        Q = np.diag([sig_v ** 2 * dt ** 2, sig_v ** 2 * dt ** 2, sig_vz ** 2 * dt ** 2, sig_g ** 2 * dt ** 2])
        self.P = A @ self.P @ A.T + Q
        self.th += (gyro - self.b_hat) * dt
        self.J_th -= dt
        self.T += dt
        self.n += 1
        self.n_dvl += int(dvl_ok)

    def result(self, bias_rw):
        cov = np.zeros((5, 5))
        cov[:4, :4] = self.P
        cov[4, 4] = bias_rw ** 2 * max(self.T, 1e-3)
        cov += np.eye(5) * 1e-10
        return {"dp": self.dp.copy(), "dz": self.dz, "dth": self.th, "J_p": self.J_p.copy(),
                "J_th": self.J_th, "b_hat": self.b_hat, "cov": cov, "T": self.T,
                "dvl_frac": self.n_dvl / max(self.n, 1)}


def compose(pose, odom, scale=0.0):
    """Pozu ön-entegre odometri ile ilerlet (tahmin / başlangıç değeri)."""
    db = pose[4] - odom["b_hat"]
    dp = (1.0 + scale * odom["dvl_frac"]) * (odom["dp"] + odom["J_p"] * db)
    xy = pose[:2] + rot2(pose[3]) @ dp
    return np.array([xy[0], xy[1], pose[2] + odom["dz"],
                     wrap(pose[3] + odom["dth"] + odom["J_th"] * db), pose[4]])


# ------------------------------------------------------------ FGO
class FGOEstimator:
    """Çok modlu faktör grafiği:  poz düğümleri (x,y,z,psi,b_g) + nirengi düğümleri.

    Faktörler: önsel, odometri (DVL+IMU ön-entegrasyon), basınç (derinlik),
    pusula (Huber), manyetik harita eşleme (Cauchy), sonar/optik nirengi (Cauchy).
    """

    def __init__(self, cfg, mag_map, use_mag=True, use_landmarks=True, use_heading=True,
                 robust=True, estimate_bias=True, use_optical=True, estimate_compass_bias=True,
                 estimate_dvl_scale=True, name="FGO"):
        self.cfg = cfg
        self.name = name
        self.use_mag, self.use_lm, self.use_heading = use_mag, use_landmarks, use_heading
        self.use_optical = use_optical
        self.estimate_bias = estimate_bias
        self.est_cb = estimate_compass_bias
        self.est_scale = estimate_dvl_scale
        kr = (lambda k: k) if robust else (lambda k: "none")
        g = self.graph = FactorGraph()
        self.f_prior = g.add_block(PriorFactor())
        self.f_odom = g.add_block(OdometryFactor())
        self.f_depth = g.add_block(DepthFactor())
        self.f_cprior = g.add_block(ScalarPrior())
        if self.est_cb:
            self.f_head = g.add_block(HeadingCalibFactor(mag_map, kernel=kr("huber"), k=cfg.huber_k))
        else:
            self.f_head = g.add_block(HeadingFactor(kernel=kr("huber"), k=cfg.huber_k))
        self.f_mag = g.add_block(MagMapFactor(mag_map, kernel=kr(cfg.mag_kernel),
                                                   k=cfg.cauchy_k if cfg.mag_kernel == "cauchy" else cfg.huber_k))
        self.f_lm = g.add_block(LandmarkFactor(kernel=kr(cfg.lm_kernel),
                                                   k=cfg.cauchy_k if cfg.lm_kernel == "cauchy" else cfg.huber_k))
        self.pose_off = []
        self.lm_off = {}
        self.cov_xy = np.eye(2) * 4.0
        self.cov_last = np.eye(5)

    def initialize(self, x0, sigmas):
        sig = np.array(sigmas, float)
        if not self.estimate_bias:
            x0 = np.array(x0, float)
            x0[4] = 0.0
            sig[4] = 1e-7
        off = self.graph.add_variable(x0, angle_index=3)
        self.pose_off.append(off)
        self.f_prior.add(off, x0, np.diag(1.0 / sig))
        # DVL ölçek hatası (kalibrasyon) değişkeni
        self.off_s = self.graph.add_variable([0.0])
        self.f_cprior.add(self.off_s, 0.0, 0.02 if self.est_scale else 1e-9)
        if self.est_cb:
            # pusula kalibrasyonu q = [c (montaj sapması), k (anomaliye bağlı sapma)]
            self.off_c = self.graph.add_variable([0.0, 0.0])
            self.f_cprior.add(self.off_c, 0.0, np.deg2rad(5.0))
            self.f_cprior.add(self.off_c + 1, 0.0, 0.05)
        self.cov_xy = np.diag(sig[:2] ** 2)

    def pose(self, i=-1):
        off = self.pose_off[i]
        return self.graph.x[off:off + POSE_DIM].copy()

    def dvl_scale(self):
        return float(self.graph.x[self.off_s])

    def compass_bias(self):
        return float(self.graph.x[self.off_c]) if self.est_cb else 0.0

    def compass_calib(self):
        return self.graph.x[self.off_c:self.off_c + 2].copy() if self.est_cb else np.zeros(2)

    def trajectory(self):
        offs = np.asarray(self.pose_off)
        return self.graph.x[offs[:, None] + np.arange(POSE_DIM)]

    def landmark(self, lid):
        off = self.lm_off[lid]
        return self.graph.x[off:off + LM_DIM].copy()

    def landmarks(self):
        return {k: self.landmark(k) for k in self.lm_off}

    def add_keyframe(self, kf):
        """kf: dict(odom, depth, heading, mag, lm_obs=[(lid, m, cov, src)])"""
        cfg = self.cfg
        prev = self.pose()
        odom = dict(kf["odom"])
        pred = compose(prev, odom, self.dvl_scale())
        off = self.graph.add_variable(pred, angle_index=3)
        cov = odom["cov"].copy()
        if not self.estimate_bias:
            cov[4, 4] = 1e-14
        self.f_odom.add(self.pose_off[-1], off, self.off_s, odom, sqrt_info(cov))
        self.pose_off.append(off)
        self.f_depth.add(off, kf["depth"], cfg.pressure_noise)
        if self.use_heading:
            sh = np.deg2rad(cfg.compass_sigma_model_deg)
            if self.est_cb:
                self.f_head.add(off, self.off_c, kf["heading"], sh)
            else:
                self.f_head.add(off, kf["heading"], sh)
        if self.use_mag:
            self.f_mag.add(off, kf["mag"], cfg.mag_sigma_model)
        if self.use_lm:
            for lid, m, mcov, src in kf["lm_obs"]:
                if src == "optical" and not self.use_optical:
                    continue
                if lid not in self.lm_off:
                    self.lm_off[lid] = self.graph.add_variable(body_to_world(pred, m))
                self.f_lm.add(off, self.lm_off[lid], m, sqrt_info(mcov))

    def optimize(self, iters, with_cov=True):
        self.graph.optimize(max_iters=iters)
        if with_cov:
            try:
                self.cov_last = self.graph.marginal_covariances([self.pose_off[-1]], POSE_DIM)[0]
                self.cov_xy = self.cov_last[:2, :2]
            except Exception:  # pragma: no cover - sayısal güvenlik
                pass


# ------------------------------------------------------------ EKF-SLAM
class EKFSLAM:
    """Aynı ölçüm modellerini kullanan Genişletilmiş Kalman Filtresi (SLAM) referansı.
    Sağlam çekirdek yerine Mahalanobis (ki-kare) kapısı ile aykırı değer reddi yapar."""

    def __init__(self, cfg, mag_map, x0, sigmas):
        self.cfg = cfg
        self.map = mag_map
        # durum: [x, y, z, psi, b_g, c_pusula, k_pusula, s_dvl, nirengiler...]
        self.x = np.concatenate([np.array(x0, float), [0.0, 0.0, 0.0]])
        self.P = np.diag(np.concatenate([np.array(sigmas, float) ** 2,
                                         [np.deg2rad(5.0) ** 2, 0.05 ** 2, 0.02 ** 2]]))
        self.lm_idx = {}
        self.g1 = chi2.ppf(0.999, 1)
        self.g3 = chi2.ppf(0.999, 3)
        self.rejected = 0

    def predict(self, odom):
        x = self.x[:5]
        f = odom["dvl_frac"]
        sc = 1.0 + self.x[7] * f
        db = x[4] - odom["b_hat"]
        g = odom["dp"] + odom["J_p"] * db
        R, dR = rot2(x[3]), drot2(x[3])
        idx = [0, 1, 2, 3, 4, 7]
        F = np.eye(6)
        F[0:2, 3] = dR @ (sc * g)
        F[0:2, 4] = sc * (R @ odom["J_p"])
        F[0:2, 5] = f * (R @ g)
        F[3, 4] = odom["J_th"]
        G = np.eye(5)
        G[0:2, 0:2] = R
        Q = G @ odom["cov"] @ G.T
        self.x[:5] = compose(x, odom, self.x[7])
        P = self.P
        P[idx, :] = F @ P[idx, :]
        P[:, idx] = P[:, idx] @ F.T
        P[:5, :5] += Q

    def _update(self, idx, Hloc, y, Rm, gate):
        n = len(self.x)
        H = np.zeros((len(y), n))
        H[:, idx] = Hloc
        PHt = self.P @ H.T
        S = H @ PHt + Rm
        Si = np.linalg.inv(S)
        if float(y @ Si @ y) > gate:
            self.rejected += 1
            return False
        K = PHt @ Si
        self.x += K @ y
        self.x[3] = wrap(self.x[3])
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def update_depth(self, z):
        self._update([2], np.ones((1, 1)), np.array([z - self.x[2]]),
                     np.array([[self.cfg.pressure_noise ** 2]]), self.g1)

    def update_heading(self, psi):
        s = np.deg2rad(self.cfg.compass_sigma_model_deg)
        x = self.x
        M = float(self.map.value(x[0], x[1])) / 100.0
        gx, gy = self.map.grad(x[0], x[1])
        H = np.array([[x[6] * float(gx) / 100.0, x[6] * float(gy) / 100.0, 1.0, 1.0, M]])
        y = np.array([wrap(psi - x[3] - x[5] - x[6] * M)])
        self._update([0, 1, 3, 5, 6], H, y, np.array([[s * s]]), self.g1)

    def update_mag(self, m):
        gx, gy = self.map.grad(self.x[0], self.x[1])
        y = np.array([m - float(self.map.value(self.x[0], self.x[1]))])
        self._update([0, 1], np.array([[float(gx), float(gy)]]), y,
                     np.array([[self.cfg.mag_sigma_model ** 2]]), self.g1)

    def update_landmark(self, lid, m, cov):
        pose = self.x[:5]
        if lid not in self.lm_idx:
            l = body_to_world(pose, m)
            R, dR = rot2(pose[3]), drot2(pose[3])
            Gx = np.zeros((3, 5))
            Gx[0:2, 0:2] = np.eye(2)
            Gx[0:2, 3] = dR @ m[:2]
            Gx[2, 2] = 1.0
            Gm = np.eye(3)
            Gm[0:2, 0:2] = R
            n = len(self.x)
            Pxl = Gx @ self.P[:5, :]
            Pll = Gx @ self.P[:5, :5] @ Gx.T + Gm @ cov @ Gm.T
            P = np.zeros((n + 3, n + 3))
            P[:n, :n] = self.P
            P[n:, :n] = Pxl
            P[:n, n:] = Pxl.T
            P[n:, n:] = Pll
            self.P = P
            self.x = np.concatenate([self.x, l])
            self.lm_idx[lid] = n
            return
        j = self.lm_idx[lid]
        h, Hp, Hl = landmark_model(pose[None], self.x[j:j + 3][None])
        Hloc = np.hstack([Hp[0], Hl[0]])
        self._update(list(range(5)) + [j, j + 1, j + 2], Hloc, m - h[0], cov, self.g3)

    def landmarks(self):
        return {k: self.x[j:j + 3].copy() for k, j in self.lm_idx.items()}

    def add_keyframe(self, kf, use_optical=True):
        self.predict(kf["odom"])
        self.update_depth(kf["depth"])
        self.update_heading(kf["heading"])
        self.update_mag(kf["mag"])
        for lid, m, cov, src in kf["lm_obs"]:
            if src == "optical" and not use_optical:
                continue
            self.update_landmark(lid, m, cov)


# ------------------------------------------------------------ ölü hesap
class DeadReckoning:
    """Endüstri standardı DVL + AHRS ölü hesap (pusula ile tamamlayıcı filtre)."""

    def __init__(self, x0, compass_gain=0.08):
        self.p = np.array(x0[:2], float)
        self.z = float(x0[2])
        self.psi = float(x0[3])
        self.k = compass_gain

    def step(self, gyro, v_b, dt):
        self.psi = float(wrap(self.psi + gyro * dt))
        self.p = self.p + rot2(self.psi) @ v_b * dt

    def keyframe(self, depth, heading):
        self.z = depth
        self.psi = float(wrap(self.psi + self.k * wrap(heading - self.psi)))

    def pose(self):
        return np.array([self.p[0], self.p[1], self.z, self.psi, 0.0])
