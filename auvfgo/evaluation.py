"""Başarım metrikleri: navigasyon (ATE), haritalama, hedef tespiti, gizlilik, tutarlılık."""
import numpy as np

from .environment import detection_rate, CLASS_NAMES
from .utils import wrap, body_to_world


def nav_errors(est, truth):
    e = np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)
    return {
        "ate_rmse": float(np.sqrt(np.mean(e ** 2))),
        "max": float(e.max()),
        "final": float(e[-1]),
        "heading_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(wrap(est[:, 3] - truth[:, 3]) ** 2)))),
        "depth_rmse": float(np.sqrt(np.mean((est[:, 2] - truth[:, 2]) ** 2))),
    }


def all_nav(out):
    tr = out["truth"]
    return {
        "Ölü hesap (DVL+AHRS)": nav_errors(out["dr"], tr),
        "EKF-SLAM": nav_errors(out["ekf"], tr),
        "FGO (çevrimiçi)": nav_errors(out["fgo_online"], tr),
        "FGO (düzleştirilmiş)": nav_errors(out["fgo_smoothed"], tr),
    }


def dr_landmarks(out):
    """Ölü hesap pozlarıyla nesne konumlandırma (karşılaştırma için)."""
    acc = {}
    for pose, kf in zip(out["dr"], out["kf_inputs"]):
        for lid, m, _, _ in kf["lm_obs"]:
            acc.setdefault(lid, []).append(body_to_world(pose, m))
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


def mapping_errors(out, min_obs=3):
    """Nesne konum hatası: tüm nirengiler ve doğrulanmış (>= min_obs gözlem) nirengiler."""
    env, front = out["env"], out["frontend"]
    res = {}
    sources = {"FGO": out["fgo"].landmarks(), "EKF-SLAM": out["ekf_obj"].landmarks(),
               "Ölü hesap": dr_landmarks(out)}
    for name, lms in sources.items():
        errs, conf = [], []
        for lid, p in lms.items():
            tid = front.true_id(lid)
            if tid >= 0:
                e = np.linalg.norm(p[:2] - env.obj_pos[tid, :2])
                errs.append(e)
                if front.n_obs(lid) >= min_obs:
                    conf.append(e)
        errs, conf = np.array(errs), np.array(conf)
        rm = lambda a: float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")
        res[name] = {"rmse_all": rm(errs), "median_all": float(np.median(errs)) if len(errs) else float("nan"),
                     "rmse": rm(conf), "median": float(np.median(conf)) if len(conf) else float("nan"),
                     "n_all": int(len(errs)), "n": int(len(conf))}
    return res


def classification_eval(out, p_confirm=0.6, min_obs=2):
    """Görev sırasında zamansal füzyonla elde edilen nesne sınıfları.

    Karar kuralı: bir nesne P(mayın) > p_confirm ve en az `min_obs` gözlem ile
    "doğrulanmış mayın" olarak raporlanır; diğer nesneler en olası sınıfa atanır."""
    env, front, fusion = out["env"], out["frontend"], out["fusion"]
    fgo_lm = out["fgo"].landmarks()
    y_true, y_pred, geo_err, mine_rows = [], [], [], []
    for lid in fusion.logp:
        tid = front.true_id(lid)
        tc = env.objects[tid]["cls"] if tid >= 0 else 0
        post = fusion.posterior(lid)
        is_mine = post[1] > p_confirm and fusion.count[lid] >= min_obs
        pc = 1 if is_mine else int(np.argmax(np.where(np.arange(3) == 1, -1.0, post)))
        y_true.append(tc)
        y_pred.append(pc)
        if is_mine and lid in fgo_lm:
            err = float(np.linalg.norm(fgo_lm[lid][:2] - env.obj_pos[tid, :2])) if tid >= 0 else float("nan")
            mine_rows.append({"lid": int(lid), "true_cls": CLASS_NAMES[tc], "p_mine": float(post[1]),
                              "n_obs": int(fusion.count[lid]), "geo_err_m": err})
            if tc == 1:
                geo_err.append(err)
    cm = np.zeros((3, 3), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    seen = {front.true_id(l) for l in fusion.logp if front.true_id(l) >= 0}
    seen_mines = {i for i in seen if env.objects[i]["cls"] == 1}
    found = {front.true_id(r["lid"]) for r in mine_rows if r["true_cls"] == "mayin"}
    tp_obj = len(found)
    return {
        "confusion": cm.tolist(),
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "mine_reports_n": len(mine_rows),
        "mine_precision": float(sum(r["true_cls"] == "mayin" for r in mine_rows) / max(len(mine_rows), 1)),
        "mine_recall_seen": float(tp_obj / max(len(seen_mines), 1)),
        "mines_total": int(sum(o["cls"] == 1 for o in env.objects)),
        "mines_seen": len(seen_mines),
        "mines_found": tp_obj,
        "mine_geo_err_mean": float(np.mean(geo_err)) if geo_err else float("nan"),
        "mine_reports": mine_rows,
    }


def stealth_eval(out, cfg):
    env, tr = out["env"], out["truth"]
    per = {}
    for i, th in enumerate(env.threats):
        r = np.linalg.norm(tr[:, :2] - out["patrol"][:, i], axis=1)
        lam = detection_rate(r, tr[:, 2], out["speed"], th.kind, th.radius, th.rate_max, cfg.thermocline_depth)
        per[th.name] = float(np.sum(lam) * cfg.dt * cfg.keyframe_every)
    dist = float(np.sum(np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1)))
    return {"p_detect": float(out["p_detect"]), "exposure_time_s": float(out["exposure_time"]),
            "mission_time_s": float(out["mission_time"]), "path_length_m": dist,
            "reached_all_goals": bool(out["reached_all"]), "per_threat_integrated_lambda": per}


def consistency(out, every=10):
    """FGO düzleştirilmiş kestiriminin 3-sigma tutarlılığı (marjinal kovaryanslardan)."""
    fgo, tr = out["fgo"], out["truth"]
    idx = np.arange(1, len(fgo.pose_off), every)
    covs = fgo.graph.marginal_covariances([fgo.pose_off[i] for i in idx], 5)
    est = fgo.trajectory()
    sig = np.array([np.sqrt(np.trace(c[:2, :2])) for c in covs])
    nees = []
    for i, c in zip(idx, covs):
        e = est[i, :2] - tr[i - 1, :2]
        nees.append(float(e @ np.linalg.solve(c[:2, :2], e)))
    nees = np.array(nees)
    return {"t_idx": idx - 1, "sigma_xy": sig, "nees": nees,
            "frac_within_95": float(np.mean(nees < 5.991)), "mean_nees": float(np.mean(nees))}
