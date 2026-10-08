"""Çok modlu sensör simülasyonu (tamamen yazılımsal).

Ataletsel (jiroskop), DVL (dip-izleme hızı + kesintiler), basınç (derinlik),
manyetometre (pusula yönü + toplam alan şiddeti), ileri bakışlı sonar (akustik),
optik kamera (bulanıklığa bağlı), pasif hidrofon (düşman sonar darbelerini yakalama).
"""
import numpy as np

from .target_classifier import observe_features, sample_latent
from .utils import wrap, rot2


def polar_to_body(rho, beta, eps):
    ce = np.cos(eps)
    return np.array([rho * ce * np.cos(beta), rho * ce * np.sin(beta), rho * np.sin(eps)])


def polar_cov_to_body(rho, beta, eps, sr, sb, se):
    cb, sbt, ce, se_ = np.cos(beta), np.sin(beta), np.cos(eps), np.sin(eps)
    J = np.array([[ce * cb, -rho * ce * sbt, -rho * se_ * cb],
                  [ce * sbt, rho * ce * cb, -rho * se_ * sbt],
                  [se_, 0.0, rho * ce]])
    return J @ np.diag([sr ** 2, sb ** 2, se ** 2]) @ J.T + np.eye(3) * 1e-6


class SensorSuite:
    def __init__(self, cfg, env, rng):
        self.cfg, self.env, self.rng = cfg, env, rng
        self.gyro_bias = cfg.gyro_bias0 * rng.choice([-1.0, 1.0])
        self.compass_misalign = np.deg2rad(cfg.compass_misalign_deg) * rng.choice([-1.0, 1.0])

    # ------------------------------------------------------------ IMU / DVL
    def step_bias(self, dt):
        self.gyro_bias += self.rng.normal(0, self.cfg.gyro_bias_rw * np.sqrt(dt))

    def gyro(self, r_true):
        return r_true + self.gyro_bias + self.rng.normal(0, self.cfg.gyro_noise)

    def dvl(self, p, psi, v_world, vz, u_cmd):
        """Dip kilidi varsa gövde çerçevesinde yer hızı; yoksa pervane modeli (akıntıyı görmez)."""
        c = self.cfg
        if self.env.dvl_dropout(p[0], p[1]):
            v_b = np.array([u_cmd, 0.0]) + self.rng.normal(0, 0.05, 2)
            return v_b, vz + self.rng.normal(0, 0.05), False
        v_b = rot2(psi).T @ v_world * (1.0 + c.dvl_scale_error)
        v_b = v_b + self.rng.normal(0, c.dvl_noise, 2)
        return v_b, vz + self.rng.normal(0, c.dvl_noise), True

    def pressure(self, z):
        return z + self.rng.normal(0, self.cfg.pressure_noise)

    # ------------------------------------------------------------ manyetik
    def compass(self, p, psi):
        anomaly = float(self.env.mag_true(p[0], p[1]))
        err = self.compass_misalign + self.cfg.compass_anomaly_coupling * anomaly
        return float(wrap(psi + err + self.rng.normal(0, np.deg2rad(self.cfg.compass_noise_deg))))

    def magnetometer(self, p):
        return float(self.env.mag_true(p[0], p[1])) + self.rng.normal(0, self.cfg.mag_noise)

    # ------------------------------------------------------------ akustik + optik
    def perceive(self, p, psi):
        """Sonar + optik gözlemler: nesne başına birleşik algılama kaydı."""
        c, env, rng = self.cfg, self.env, self.rng
        out = []
        d = env.obj_pos - p[:3]
        rho = np.linalg.norm(d, axis=1)
        rh = np.hypot(d[:, 0], d[:, 1])
        beta = wrap(np.arctan2(d[:, 1], d[:, 0]) - psi)
        eps = np.arctan2(d[:, 2], rh)
        vis = env.visibility(p[0], p[1])
        fov = np.deg2rad(c.sonar_fov_deg)
        cand = np.where((rho < c.sonar_range) & (np.abs(beta) < fov))[0]
        for i in cand:
            obj = env.objects[i]
            det = {"true_id": int(i), "sonar": None, "optical": None}
            if rng.random() < c.sonar_pd * (1 - 0.3 * rho[i] / c.sonar_range):
                sr = c.sonar_range_sigma + c.sonar_range_sigma_rel * rho[i]
                sb = np.deg2rad(c.sonar_bearing_sigma_deg)
                se = np.deg2rad(c.sonar_elev_sigma_deg)
                rm = rho[i] + rng.normal(0, sr)
                if rng.random() < c.sonar_multipath_prob:
                    rm *= rng.uniform(1.25, 1.8)  # yüzey/taban yansımalı hayalet yankı
                bm = beta[i] + rng.normal(0, sb)
                em = eps[i] + rng.normal(0, se)
                det["sonar"] = (polar_to_body(rm, bm, em), polar_cov_to_body(rm, bm, em, sr, sb, se))
            has_opt = rho[i] < vis and abs(beta[i]) < np.deg2rad(50)
            if has_opt:
                s = c.optical_sigma * (1 + rho[i] / 10.0)
                m = polar_to_body(rho[i], beta[i], eps[i]) + rng.normal(0, s, 3)
                det["optical"] = (m, np.eye(3) * s * s)
            if det["sonar"] is None and det["optical"] is None:
                continue
            has_mag = rh[i] < c.mag_feature_range
            det["features"] = observe_features(obj["latent"], obj["cls"], rho[i], has_opt, has_mag, rng)
            out.append(det)
        # yanlış alarmlar (akustik yankı / balık sürüsü)
        for _ in range(rng.poisson(c.sonar_false_alarm_rate)):
            rm = rng.uniform(15, c.sonar_range)
            bm = rng.uniform(-fov, fov)
            em = rng.uniform(0.05, 0.6)
            sr, sb, se = 0.5, np.deg2rad(1.2), np.deg2rad(2.0)
            lat = sample_latent(0, rng)
            out.append({"true_id": -1, "optical": None,
                        "sonar": (polar_to_body(rm, bm, em), polar_cov_to_body(rm, bm, em, sr, sb, se)),
                        "features": observe_features(lat, 0, rm, False, False, rng)})
        return out

    # ------------------------------------------------------------ pasif dinleme
    def hydrophone(self, p, psi, t0, t1):
        """(t0, t1] aralığında düşman aktif sonar darbelerini yakala -> göreli kerteriz."""
        c, rng = self.cfg, self.rng
        out = []
        for k, th in enumerate(self.env.threats):
            if th.ping_period <= 0:
                continue
            n = int(np.floor((t1 - th.ping_phase) / th.ping_period) -
                    np.floor((t0 - th.ping_phase) / th.ping_period))
            if n <= 0:
                continue
            tp = th.position(t1)
            r = np.linalg.norm(tp - p[:2])
            R = 2.5 * th.radius
            pint = 0.95 if r < R else 0.95 * np.exp(-((r - R) / (0.6 * th.radius)) ** 2)
            if rng.random() < pint:
                b_world = np.arctan2(tp[1] - p[1], tp[0] - p[0])
                b_rel = wrap(b_world - psi + rng.normal(0, np.deg2rad(c.hydrophone_bearing_sigma_deg)))
                out.append({"threat": k, "bearing_rel": float(b_rel), "range_true": float(r)})
        return out
