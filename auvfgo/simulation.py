"""Kapalı çevrim görev simülasyonu.

Araç, FGO'nun (çevrimiçi) poz kestirimi ile güdülür; gerçek hareket akıntılardan
etkilenir. Her anahtar karede: odometri ön-entegrasyonu -> sensör ölçümleri ->
veri ilişkilendirme -> FGO / EKF / DR güncellemesi -> hedef sınıflandırma ->
tehdit inancı güncellemesi -> (gerekirse) yeniden planlama.
"""
import time

import numpy as np

from .environment import OceanEnvironment
from .estimators import OdometryPreintegrator, FGOEstimator, EKFSLAM, DeadReckoning, compose
from .frontend import LandmarkFrontend
from .planner import AStarPlanner
from .sensors import SensorSuite
from .target_classifier import TemporalFusion
from .threat_map import ThreatBelief
from .utils import wrap, rot2


class MissionSimulator:
    def __init__(self, cfg, classifier=None, mode="risk", seed=None, verbose=True):
        """mode: 'risk' (gizlilik/risk farkında) | 'shortest' (yalnız en kısa yol)."""
        self.cfg = cfg
        self.mode = mode
        self.verbose = verbose
        seed = cfg.seed if seed is None else seed
        self.seed = seed
        # ortam aynı tohumdan (karşılaştırmalarda aynı dünya), gürültüler ayrı akıştan
        self.env = OceanEnvironment(cfg, np.random.default_rng(seed))
        self.rng = np.random.default_rng(seed + 1000 + (0 if mode == "risk" else 1))
        self.sensors = SensorSuite(cfg, self.env, self.rng)
        self.classifier = classifier
        self.planner = AStarPlanner(cfg, self.env)
        self.belief = ThreatBelief(cfg, self.env, np.random.default_rng(seed + 7))

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def run(self):
        cfg, env, sen = self.cfg, self.env, self.sensors
        t_wall = time.time()
        start = np.array(cfg.start, float)
        z0 = float(env.seafloor(*start)) - cfg.altitude_cmd
        truth = np.array([start[0], start[1], z0, np.deg2rad(45.0)])
        x0 = np.array([*truth, 0.0])
        sig0 = [0.5, 0.5, 0.1, np.deg2rad(1.0), 0.003]

        fgo = FGOEstimator(cfg, env.mag_prior)
        fgo.initialize(x0, sig0)
        ekf = EKFSLAM(cfg, env.mag_prior, x0, sig0)
        dr = DeadReckoning(x0)
        front = LandmarkFrontend(cfg)
        fusion = TemporalFusion()

        nav = x0.copy()  # kontrol için kullanılan güncel kestirim
        pre = OdometryPreintegrator(nav[4])
        goals = [np.array(g, float) for g in cfg.goals]
        gi = 0
        path = None
        stealth = self.mode == "risk"
        w_risk = cfg.w_risk if stealth else 0.0
        w_info = cfg.w_info if stealth else 0.0
        lam_b, spd_map, haz = self.belief.hazard_maps(stealth)
        kf_count = 0
        need_replan = True
        u = cfg.cruise_speed
        cum_lambda = 0.0
        exposure_time = 0.0

        log = {k: [] for k in ["t", "truth", "fgo_online", "ekf", "dr", "lam_true", "lam_belief",
                               "speed", "dvl_ok", "n_obs", "cum_pdet", "goal_idx",
                               "calib_fgo", "calib_ekf"]}
        kf_inputs = []
        paths = []
        patrol = []
        intercepts = []
        snaps = []
        steps = int(cfg.max_time / cfg.dt)
        t = 0.0
        reached_all = False
        for k in range(steps):
            t = k * cfg.dt
            # ---------------- planlama
            if need_replan:
                path = self.planner.plan(nav[:3], goals[gi], haz, w_risk, w_info)
                paths.append((t, gi, path.copy()))
                need_replan = False
                pidx = 0
            # ---------------- güdüm (kestirime göre!)
            d = np.linalg.norm(path - nav[:2], axis=1)
            pidx = max(pidx, int(np.argmin(d[pidx:pidx + 40])) + pidx)
            j = pidx
            while j < len(path) - 1 and np.linalg.norm(path[j] - nav[:2]) < cfg.lookahead:
                j += 1
            tgt = path[j]
            psi_d = np.arctan2(tgt[1] - nav[1], tgt[0] - nav[0])
            r_cmd = float(np.clip(cfg.yaw_gain * wrap(psi_d - nav[3]), -cfg.max_yaw_rate, cfg.max_yaw_rate))
            ci, cj = self.belief.cell_index(nav[:2])
            u = float(spd_map[ci, cj]) if stealth else cfg.cruise_speed
            # ---------------- gerçek dinamik
            cur = env.current(truth[0], truth[1], t)
            z_cmd = float(env.seafloor(truth[0], truth[1])) - cfg.altitude_cmd
            vz = float(np.clip(0.5 * (z_cmd - truth[2]), -cfg.max_vz, cfg.max_vz))
            r_true = r_cmd + self.rng.normal(0, 0.003)
            v_world = rot2(truth[3]) @ np.array([u, 0.0]) + cur
            truth[:2] += v_world * cfg.dt
            truth[2] += vz * cfg.dt
            truth[3] = wrap(truth[3] + r_true * cfg.dt)
            sen.step_bias(cfg.dt)
            lam = env.true_detection_rate(truth, truth[2], u, t)
            cum_lambda += lam * cfg.dt
            if lam > 1e-3:
                exposure_time += cfg.dt
            # ---------------- IMU + DVL örnekleri
            gyro = sen.gyro(r_true)
            v_b, vz_m, dvl_ok = sen.dvl(truth, truth[3], v_world, vz, u)
            sv = cfg.dvl_sigma_model if dvl_ok else cfg.dvl_model_sigma
            pre.integrate(gyro, v_b, vz_m, cfg.dt, sv, sv, cfg.gyro_noise, dvl_ok)
            dr.step(gyro, v_b, cfg.dt)
            # kontrol kestirimini ara adımlarda ilerlet
            nav[3] = wrap(nav[3] + (gyro - nav[4]) * cfg.dt)
            nav[:2] += rot2(nav[3]) @ v_b * cfg.dt
            nav[2] += vz_m * cfg.dt

            if (k + 1) % cfg.keyframe_every:
                continue
            # ================= ANAHTAR KARE =================
            kf_count += 1
            t1 = t + cfg.dt
            odom = pre.result(cfg.gyro_bias_rw)
            z_m = sen.pressure(truth[2])
            psi_m = sen.compass(truth, truth[3])
            mag_m = sen.magnetometer(truth)
            dets = sen.perceive(truth, truth[3])
            # tahmini poz ile veri ilişkilendirme
            pose_pred = compose(fgo.pose(), odom, fgo.dvl_scale())
            assoc, lm_factors = front.associate(pose_pred, fgo.cov_xy, fgo.landmarks(), dets)
            kf = {"odom": odom, "depth": z_m, "heading": psi_m, "mag": mag_m, "lm_obs": lm_factors}
            kf_inputs.append(kf)
            fgo.add_keyframe(kf)
            ekf.add_keyframe(kf)
            dr.keyframe(z_m, psi_m)
            if kf_count % cfg.opt_every == 0:
                fgo.optimize(cfg.opt_iters)
            nav = fgo.pose()
            pre = OdometryPreintegrator(nav[4])
            # hedef sınıflandırma (zamansal Bayes füzyonu)
            if self.classifier is not None and assoc:
                F = np.array([det["features"] for _, det in assoc])
                P = self.classifier.predict_proba(F)
                for (lid, _), p in zip(assoc, P):
                    fusion.add(lid, p)
            # pasif dinleme -> tehdit inancı
            sig_pos = float(np.sqrt(np.trace(fgo.cov_xy)))
            for ic in sen.hydrophone(truth, truth[3], t1 - cfg.dt * cfg.keyframe_every, t1):
                bw = wrap(ic["bearing_rel"] + nav[3])
                self.belief.intercept(nav[:2], bw, sig_pos, 3.5 * cfg.nominal_threat_radius,
                                     heading_sigma=float(np.sqrt(max(fgo.cov_last[3, 3], 0.0))))
                intercepts.append((t1, ic["threat"], nav[:2].copy(), bw))
            self.belief.decay(cfg.dt * cfg.keyframe_every)
            patrol.append(env.threat_positions(t1))
            # hedefe varış / yeniden planlama
            if np.linalg.norm(nav[:2] - goals[gi]) < cfg.goal_tolerance:
                self._log(f"  [{self.mode}] t={t1:6.0f}s  hedef {gi + 1}/{len(goals)} ulaşıldı")
                gi += 1
                if gi >= len(goals):
                    reached_all = True
                need_replan = True
            if kf_count % cfg.replan_every == 0:
                lam_b, spd_map, haz = self.belief.hazard_maps(stealth)
                need_replan = True
                if kf_count % (cfg.replan_every * 4) == 0:
                    snaps.append((t1, self.belief.p_total().astype(np.float32), lam_b.astype(np.float32),
                                  truth[:2].copy(), env.threat_positions(t1)))
            # kayıt
            ci, cj = self.belief.cell_index(nav[:2])
            log["t"].append(t1)
            log["truth"].append(np.array([*truth, sen.gyro_bias]))
            log["fgo_online"].append(nav.copy())
            log["ekf"].append(ekf.x[:5].copy())
            log["dr"].append(dr.pose())
            log["lam_true"].append(lam)
            log["lam_belief"].append(float(lam_b[ci, cj]))
            log["speed"].append(u)
            log["dvl_ok"].append(odom["dvl_frac"])
            log["n_obs"].append(len(assoc))
            log["cum_pdet"].append(1.0 - np.exp(-cum_lambda))
            log["goal_idx"].append(gi)
            log["calib_fgo"].append(np.r_[fgo.compass_calib(), fgo.dvl_scale()])
            log["calib_ekf"].append(ekf.x[5:8].copy())
            if reached_all:
                break

        self._log(f"  [{self.mode}] görev bitti: t={t:.0f}s, {kf_count} anahtar kare, "
                  f"{fgo.graph.n_factors} faktör, {len(fgo.lm_off)} nirengi — son optimizasyon...")
        t_opt = time.time()
        fgo.optimize(cfg.final_iters, with_cov=False)
        t_opt = time.time() - t_opt
        out = {k: np.array(v) for k, v in log.items()}
        out.update({
            "fgo_smoothed": fgo.trajectory()[1:],
            "fgo": fgo, "ekf_obj": ekf, "frontend": front, "fusion": fusion,
            "kf_inputs": kf_inputs, "paths": paths, "patrol": np.array(patrol),
            "intercepts": intercepts, "snaps": snaps, "belief": self.belief, "env": self.env,
            "reached_all": reached_all, "mission_time": t + cfg.dt,
            "p_detect": 1.0 - np.exp(-cum_lambda), "exposure_time": exposure_time,
            "runtime": time.time() - t_wall, "final_opt_time": t_opt, "mode": self.mode,
            "x0": x0, "sig0": sig0,
            "calib_truth": np.array([sen.compass_misalign, cfg.compass_anomaly_coupling * 100.0,
                                     1.0 / (1.0 + cfg.dvl_scale_error) - 1.0]),
            "calib_final": np.r_[fgo.compass_calib(), fgo.dvl_scale()],
        })
        return out


def replay_fgo(cfg, env, kf_inputs, x0, sig0, opt_every=None, iters=None, **flags):
    """Kaydedilmiş ölçümlerle FGO'yu farklı faktör alt kümeleriyle yeniden çalıştır (ablasyon)."""
    opt_every = cfg.opt_every if opt_every is None else opt_every
    iters = cfg.opt_iters if iters is None else iters
    fgo = FGOEstimator(cfg, env.mag_prior, **flags)
    fgo.initialize(x0, sig0)
    online = []
    for i, kf in enumerate(kf_inputs):
        fgo.add_keyframe(kf)
        if (i + 1) % opt_every == 0:
            fgo.optimize(iters, with_cov=False)
        online.append(fgo.pose())
    fgo.optimize(cfg.final_iters, with_cov=False)
    return fgo, np.array(online), fgo.trajectory()[1:]
