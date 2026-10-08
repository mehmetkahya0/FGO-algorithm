"""Birim testleri: faktör Jacobian'ları, FGO yakınsaması, sağlam çekirdekler, planlayıcı, sınıflandırıcı."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auvfgo.config import Config  # noqa: E402
from auvfgo.environment import OceanEnvironment  # noqa: E402
from auvfgo.estimators import OdometryPreintegrator, compose, FGOEstimator  # noqa: E402
from auvfgo.factor_graph import (FactorGraph, PriorFactor, OdometryFactor, LandmarkFactor, MagMapFactor,  # noqa: E402
                                 HeadingCalibFactor, ScalarPrior, robust_weight)
from auvfgo.planner import AStarPlanner, path_length  # noqa: E402
from auvfgo.target_classifier import MLPClassifier, make_dataset, metrics  # noqa: E402
from auvfgo.utils import wrap, rot2  # noqa: E402


def numeric_jacobians(block, x, eps=1e-6):
    r0, Js = block.evaluate(x)
    offs, _ = block.arrays()
    num = []
    for off, dv in zip(offs, block.var_dims):
        Jn = np.zeros((r0.shape[0], r0.shape[1], dv))
        for k in range(dv):
            xp = x.copy()
            xp[off + k] += eps
            rp, _ = block.evaluate(xp)
            Jn[:, :, k] = (rp - r0) / eps
        num.append(Jn)
    return Js, num


@pytest.fixture(scope="module")
def env():
    return OceanEnvironment(Config(), np.random.default_rng(0))


def _two_poses(rng):
    x = np.concatenate([[10, 20, 50, 0.7, 0.001], [14, 23, 51, 0.8, 0.0012], [30, 25, 80]])
    return x + rng.normal(0, 0.01, len(x))


def test_odometry_jacobian():
    rng = np.random.default_rng(1)
    x = _two_poses(rng)
    blk = OdometryFactor()
    odom = {"dp": np.array([4.0, 1.0]), "dz": 1.0, "dth": 0.1, "J_p": np.array([0.3, -2.0]),
            "J_th": -2.0, "b_hat": 0.0005, "dvl_frac": 0.75}
    x = np.concatenate([x, [0.01]])
    blk.add(0, 5, len(x) - 1, odom, np.diag([1, 2, 3, 4, 5.0]))
    Js, num = numeric_jacobians(blk, x)
    for a, b in zip(Js, num):
        np.testing.assert_allclose(a, b, atol=1e-4)


def test_landmark_jacobian():
    rng = np.random.default_rng(2)
    x = _two_poses(rng)
    blk = LandmarkFactor()
    blk.add(5, 10, np.array([10.0, 2.0, 29.0]), np.eye(3) * 2.0)
    Js, num = numeric_jacobians(blk, x)
    for a, b in zip(Js, num):
        np.testing.assert_allclose(a, b, atol=1e-4)


def test_mag_and_heading_jacobian(env):
    x = np.array([400.0, 300.0, 60.0, 0.3, 0.0, 0.02, 0.04])
    blk = MagMapFactor(env.mag_prior)
    blk.add(0, 12.0, 3.0)
    Js, num = numeric_jacobians(blk, x)
    np.testing.assert_allclose(Js[0], num[0], atol=1e-4)
    hb = HeadingCalibFactor(env.mag_prior)
    hb.add(0, 5, 0.25, 0.05)
    Js, num = numeric_jacobians(hb, x)
    for a, b in zip(Js, num):
        np.testing.assert_allclose(a, b, atol=1e-4)


def test_preintegration_matches_composition():
    """Ön-entegrasyon + compose, adım adım entegrasyonla aynı sonucu vermeli."""
    rng = np.random.default_rng(3)
    pose = np.array([0.0, 0.0, 10.0, 0.4, 0.002])
    pre = OdometryPreintegrator(b_hat=0.001)
    p, psi = pose[:2].copy(), pose[3]
    for _ in range(20):
        g = rng.normal(0.05, 0.02)
        v = np.array([1.5, rng.normal(0, 0.1)])
        pre.integrate(g, v, 0.1, 0.5, 0.02, 0.02, 0.002)
        p = p + rot2(psi) @ v * 0.5
        psi = psi + (g - pose[4]) * 0.5
    odom = pre.result(1e-5)
    out = compose(pose, odom)
    # sapma farkı birinci dereceden düzeltildiği için küçük kalıntı beklenir
    assert np.linalg.norm(out[:2] - p) < 0.05
    assert abs(wrap(out[3] - psi)) < 1e-9


def test_fgo_recovers_trajectory_with_landmarks():
    """Bilinen gerçek değerlerle üretilmiş sentetik ölçümlerden FGO'nun yörüngeyi kurtarması."""
    rng = np.random.default_rng(4)
    g = FactorGraph()
    pri, odo, lmf = g.add_block(PriorFactor()), g.add_block(OdometryFactor()), g.add_block(LandmarkFactor())
    sp_ = g.add_block(ScalarPrior())
    truth = [np.array([0, 0, 20, 0, 0.0])]
    for k in range(30):
        prev = truth[-1]
        truth.append(np.array([*(prev[:2] + rot2(prev[3]) @ [3.0, 0]), 20, prev[3] + 0.05, 0.0]))
    L = np.array([[20, 15, 40.0], [50, 30, 40.0], [60, 60, 40.0]])
    offs = []
    for k, t in enumerate(truth):
        init = t + np.r_[rng.normal(0, 2.0, 2), 0, rng.normal(0, 0.05), 0] * (k > 0)
        offs.append(g.add_variable(init, angle_index=3))
    pri.add(offs[0], truth[0], np.eye(5) * 100)
    o_s = g.add_variable([0.0])
    sp_.add(o_s, 0.0, 1e-6)
    for k in range(1, len(truth)):
        a, b = truth[k - 1], truth[k]
        dp = rot2(a[3]).T @ (b[:2] - a[:2]) + rng.normal(0, 0.05, 2)
        odom = {"dp": dp, "dz": 0.0, "dth": b[3] - a[3] + rng.normal(0, 0.005), "J_p": np.zeros(2),
                "J_th": 0.0, "b_hat": 0.0, "dvl_frac": 1.0}
        odo.add(offs[k - 1], offs[k], o_s, odom, np.diag([20, 20, 20, 200, 1e4]))
    loff = [g.add_variable(l + rng.normal(0, 3.0, 3)) for l in L]
    for k, t in enumerate(truth):
        for j, l in enumerate(L):
            d = l - t[:3]
            if np.linalg.norm(d[:2]) < 50:
                m = np.r_[rot2(t[3]).T @ d[:2], d[2]] + rng.normal(0, 0.1, 3)
                lmf.add(offs[k], loff[j], m, np.eye(3) * 10)
    c0 = g.cost()
    g.optimize(max_iters=20)
    assert g.cost() < c0 * 1e-2
    est = np.array([g.x[o:o + 5] for o in offs])
    err = np.linalg.norm(est[:, :2] - np.array(truth)[:, :2], axis=1)
    assert err.max() < 1.0


def test_robust_weights():
    e = np.array([0.5, 2.0, 10.0])
    w_h = robust_weight(e, "huber", 1.0)
    w_c = robust_weight(e, "cauchy", 1.0)
    assert w_h[0] == 1.0 and w_h[2] == pytest.approx(0.1)
    assert np.all(np.diff(w_c) < 0) and w_c[2] < 0.02


def test_fgo_estimator_runs(env):
    cfg = Config()
    fgo = FGOEstimator(cfg, env.mag_prior)
    fgo.initialize(np.array([100, 100, 60, 0.3, 0.0]), [0.5, 0.5, 0.1, 0.02, 0.003])
    pre = OdometryPreintegrator(0.0)
    for _ in range(4):
        pre.integrate(0.0, np.array([1.5, 0.0]), 0.0, 0.5, 0.04, 0.04, 0.002)
    odom = pre.result(cfg.gyro_bias_rw)
    for _ in range(10):
        p = compose(fgo.pose(), odom)
        fgo.add_keyframe({"odom": odom, "depth": 60.0, "heading": p[3],
                          "mag": float(env.mag_prior.value(p[0], p[1])), "lm_obs": []})
    fgo.optimize(5)
    assert len(fgo.pose_off) == 11
    assert np.all(np.isfinite(fgo.trajectory()))
    assert np.trace(fgo.cov_xy) > 0


def test_planner_avoids_risk(env):
    cfg = Config()
    pl = AStarPlanner(cfg, env)
    n = pl.nx
    haz = np.zeros((n, n))
    haz[40:60, 30:70] = 0.05  # yüksek riskli blok
    start, goal = np.array([100.0, 500.0]), np.array([900.0, 500.0])
    p_short = pl.plan(start, goal, haz, w_risk=0.0, w_info=0.0)
    p_risk = pl.plan(start, goal, haz, w_risk=1000.0, w_info=0.0)
    cells = lambda P: [pl.to_cell(p) for p in P]
    in_block = lambda P: sum(40 <= i < 60 and 30 <= j < 70 for i, j in cells(P))
    assert in_block(p_risk) == 0
    assert in_block(p_short) > 0
    assert path_length(p_risk) > path_length(p_short)


def test_classifier_fusion_beats_sonar_only():
    rng = np.random.default_rng(0)
    Xtr, ytr = make_dataset(3000, rng)
    Xte, yte = make_dataset(2000, rng)
    full = MLPClassifier(seed=0)
    full.fit(Xtr, ytr, epochs=25)
    sonar = MLPClassifier(cols=[0, 1, 2, 3, 10], seed=0)
    sonar.fit(Xtr, ytr, epochs=25, modality_dropout=False)
    a_full = metrics(yte, full.predict(Xte))["accuracy"]
    a_son = metrics(yte, sonar.predict(Xte))["accuracy"]
    assert a_full > 0.75
    assert a_full > a_son
