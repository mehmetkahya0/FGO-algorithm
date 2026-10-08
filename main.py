"""Tüm deneyleri çalıştırır, şekilleri ve metrikleri results/ altına yazar.

Kullanım:
    python main.py                # tam deney (ana görev + Monte Carlo + ablasyon + zor koşullar)
    python main.py --quick        # hızlı deneme (MC=2, kısa eğitim)
    python main.py --mc 8         # Monte Carlo tohum sayısı
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from auvfgo.config import Config
from auvfgo.evaluation import (all_nav, nav_errors, mapping_errors, classification_eval, stealth_eval,
                               consistency)
from auvfgo.simulation import MissionSimulator, replay_fgo
from auvfgo.target_classifier import train_models
from auvfgo import plotting as P

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ABLATIONS = {
    "FGO (tam model)": {},
    "Sağlam çekirdek yok (L2)": {"robust": False},
    "Manyetik harita faktörü yok": {"use_mag": False},
    "Sonar/optik nirengi (SLAM) yok": {"use_landmarks": False},
    "Optik kamera yok": {"use_optical": False},
    "Pusula kalibrasyonu yok": {"estimate_compass_bias": False},
    "DVL ölçek kalibrasyonu yok": {"estimate_dvl_scale": False},
    "Jiroskop sapma kestirimi yok": {"estimate_bias": False},
    "Pusula faktörü yok": {"use_heading": False},
}


def ablation(cfg, out):
    tr = out["truth"]
    res = {}
    for name, flags in ABLATIONS.items():
        _, _, sm = replay_fgo(cfg, out["env"], out["kf_inputs"], out["x0"], out["sig0"], **flags)
        res[name] = nav_errors(sm, tr)["ate_rmse"]
    return res


def run_seed(args):
    """Bir Monte Carlo denemesi (yalnız metrik döndürür)."""
    seed, cfg, clf = args
    cfg = replace(cfg, seed=seed)
    out_r = MissionSimulator(cfg, clf, mode="risk", verbose=False).run()
    out_s = MissionSimulator(cfg, clf, mode="shortest", verbose=False).run()
    return {
        "seed": seed,
        "nav": all_nav(out_r),
        "mapping": mapping_errors(out_r),
        "classification": {k: v for k, v in classification_eval(out_r).items() if k != "mine_reports"},
        "stealth_risk": stealth_eval(out_r, cfg),
        "stealth_shortest": stealth_eval(out_s, cfg),
        "ablation": ablation(cfg, out_r),
        "runtime_s": out_r["runtime"],
        "calib_truth": out_r["calib_truth"].tolist(),
        "calib_fgo": out_r["calib_final"].tolist(),
        "nees": consistency(out_r)["mean_nees"],
    }


def run_stress(args):
    """Zor koşullar: yoğun çoklu-yol yankısı ve yanlış alarm (aykırı değer dayanıklılığı)."""
    seed, cfg, clf = args
    cfg = replace(cfg, seed=seed, sonar_multipath_prob=0.25, sonar_false_alarm_rate=0.6,
                  compass_anomaly_coupling=8e-4)
    out = MissionSimulator(cfg, clf, mode="risk", verbose=False).run()
    tr = out["truth"]
    _, _, sm_nr = replay_fgo(cfg, out["env"], out["kf_inputs"], out["x0"], out["sig0"], robust=False)
    return {"seed": seed,
            "Ölü hesap (DVL+AHRS)": nav_errors(out["dr"], tr)["ate_rmse"],
            "EKF-SLAM (χ² kapısı)": nav_errors(out["ekf"], tr)["ate_rmse"],
            "FGO, sağlam çekirdek yok": nav_errors(sm_nr, tr)["ate_rmse"],
            "FGO (Huber+Cauchy)": nav_errors(out["fgo_smoothed"], tr)["ate_rmse"],
            "ekf_rejected": out["ekf_obj"].rejected}


def mstd(x):
    x = np.asarray(x, float)
    return f"{np.mean(x):.2f} ± {np.std(x):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mc", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    cfg = Config()
    n_mc = 2 if a.quick else a.mc
    t0 = time.time()

    # 1) YZ hedef sınıflandırıcı
    print("[1/5] Çok modlu MLP hedef sınıflandırıcı eğitiliyor (modalite ablasyonu)...")
    models, cls_results, histories = train_models(seed=cfg.seed, epochs=15 if a.quick else 60)
    clf = models["Tam füzyon (sonar+optik+manyetik)"]
    for n, r in cls_results.items():
        print(f"    {n:38s} doğruluk={r['accuracy']:.3f}  makro-F1={r['macro_f1']:.3f}  "
              f"mayın duyarlılığı={r['mine_recall']:.3f}")

    # 2) ana görev
    print("[2/5] Ana görev (risk-farkında ve en kısa yol) simüle ediliyor...")
    out_r = MissionSimulator(cfg, clf, mode="risk").run()
    out_s = MissionSimulator(cfg, clf, mode="shortest").run()
    nav = all_nav(out_r)
    mapping = mapping_errors(out_r)
    cls_eval = classification_eval(out_r)
    st_r, st_s = stealth_eval(out_r, cfg), stealth_eval(out_s, cfg)
    cons = consistency(out_r)
    for k, v in nav.items():
        print(f"    {k:26s} ATE={v['ate_rmse']:7.2f} m  son={v['final']:7.2f} m  yön={v['heading_rmse_deg']:.2f}°")
    print(f"    Tespit olasılığı: risk-farkında={st_r['p_detect']:.3f}  en kısa={st_s['p_detect']:.3f}")

    # 3) Monte Carlo + ablasyon + zor koşullar (paralel)
    print(f"[3/5] Monte Carlo ({n_mc} tohum) + ablasyon + zor koşullar (paralel {a.workers} işçi)...")
    seeds = [cfg.seed + 100 * (i + 1) for i in range(n_mc)]
    with Pool(a.workers) as pool:
        mc_async = pool.map_async(run_seed, [(s, cfg, clf) for s in seeds])
        st_async = pool.map_async(run_stress, [(s, cfg, clf) for s in seeds[: max(2, n_mc // 2)]])
        mc = mc_async.get()
        stress = st_async.get()
    abl_main = ablation(cfg, out_r)
    abl = {k: [abl_main[k]] + [r["ablation"][k] for r in mc] for k in ABLATIONS}

    # 5) metrikler ve özet
    print("[4/5] Metrikler yazılıyor...")
    g = out_r["fgo"].graph
    metrics = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()},
        "main_run": {
            "navigation": nav, "mapping": mapping,
            "classification": cls_eval, "stealth_risk": st_r, "stealth_shortest": st_s,
            "consistency": {"frac_within_95": cons["frac_within_95"], "mean_nees": cons["mean_nees"]},
            "graph": {"n_variables": int(len(g.x)), "n_factors": int(g.n_factors),
                      "n_poses": len(out_r["fgo"].pose_off), "n_landmarks": len(out_r["fgo"].lm_off),
                      "final_opt_time_s": out_r["final_opt_time"], "mission_runtime_s": out_r["runtime"],
                      "calib_truth_[c_rad,k_rad_per_100nT,s_dvl]": out_r["calib_truth"].tolist(),
                      "calib_fgo": out_r["calib_final"].tolist()},
            "frontend": {"associations": out_r["frontend"].n_assoc, "inconsistent": out_r["frontend"].n_wrong},
        },
        "classifier_ablation": {k: {kk: vv for kk, vv in v.items()} for k, v in cls_results.items()},
        "monte_carlo": mc,
        "ablation_ate": abl,
        "stress_test": stress,
        "total_runtime_s": time.time() - t0,
    }
    with open(f"{RES}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1, default=float)

    allrun = [{"nav": nav, "mapping": mapping, "stealth_risk": st_r, "stealth_shortest": st_s,
               "classification": cls_eval}] + mc
    L = ["# Deney Sonuçları Özeti (otomatik üretildi)\n",
         f"Monte Carlo deneme sayısı: **{len(allrun)}** (ana görev + {len(mc)} farklı tohum). "
         "Değerler ortalama ± standart sapmadır.\n",
         "## 1. Navigasyon doğruluğu (GPS yok)\n",
         "| Yöntem | ATE RMSE [m] | Son konum hatası [m] | Maks. hata [m] | Yön RMSE [°] |",
         "|---|---|---|---|---|"]
    for k in nav:
        L.append(f"| {k} | {mstd([r['nav'][k]['ate_rmse'] for r in allrun])} | "
                 f"{mstd([r['nav'][k]['final'] for r in allrun])} | {mstd([r['nav'][k]['max'] for r in allrun])} | "
                 f"{mstd([r['nav'][k]['heading_rmse_deg'] for r in allrun])} |")
    L += ["\n## 2. FGO ablasyonu (düzleştirilmiş ATE RMSE [m])\n", "| Yapılandırma | ATE RMSE [m] |", "|---|---|"]
    for k in sorted(abl, key=lambda k: np.mean(abl[k])):
        L.append(f"| {k} | {mstd(abl[k])} |")
    L += ["\n## 3. Zor koşullar (çoklu-yol %25, yanlış alarm 0.6/tarama, 2× pusula bozulması)\n",
          "| Yöntem | ATE RMSE [m] |", "|---|---|"]
    for k in ["Ölü hesap (DVL+AHRS)", "EKF-SLAM (χ² kapısı)", "FGO, sağlam çekirdek yok", "FGO (Huber+Cauchy)"]:
        L.append(f"| {k} | {mstd([s[k] for s in stress])} |")
    L += ["\n## 4. Haritalama (nesne konum hatası [m])\n",
          "| Kaynak | RMSE, doğrulanmış (≥3 gözlem) | Medyan, doğrulanmış | RMSE, tüm nirengiler |", "|---|---|---|---|"]
    for k in mapping:
        L.append(f"| {k} | {mstd([r['mapping'][k]['rmse'] for r in allrun])} | "
                 f"{mstd([r['mapping'][k]['median'] for r in allrun])} | {mstd([r['mapping'][k]['rmse_all'] for r in allrun])} |")
    L += ["\n## 5. Gizlilik ve görev\n", "| Planlayıcı | Tespit olasılığı | Maruziyet süresi [s] | Görev süresi [s] | Yol uzunluğu [m] |",
          "|---|---|---|---|---|"]
    for key, nm in [("stealth_shortest", "En kısa yol"), ("stealth_risk", "Risk-farkında (önerilen)")]:
        L.append(f"| {nm} | {mstd([r[key]['p_detect'] for r in allrun])} | "
                 f"{mstd([r[key]['exposure_time_s'] for r in allrun])} | {mstd([r[key]['mission_time_s'] for r in allrun])} | "
                 f"{mstd([r[key]['path_length_m'] for r in allrun])} |")
    L += ["\n## 6. Hedef tespiti ve sınıflandırma\n",
          "Tek gözlem (sentetik test kümesi, modalite ablasyonu):\n",
          "| Model | Doğruluk | Makro-F1 | Mayın duyarlılığı | Doğruluk (optik mevcut) |", "|---|---|---|---|---|"]
    for k, r in cls_results.items():
        L.append(f"| {k} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['mine_recall']:.3f} | "
                 f"{r['accuracy_optical_available']:.3f} |")
    L += ["\nGörev içi (çoklu gözlemin zamansal Bayes füzyonu):\n",
          f"- Nesne sınıflandırma doğruluğu: {mstd([r['classification']['accuracy'] for r in allrun])}",
          f"- Doğrulanmış mayın raporu kuralı: P(mayın) > 0.6 ve ≥ 2 gözlem",
          f"- Mayın kesinliği (precision): {mstd([r['classification']['mine_precision'] for r in allrun])}",
          f"- Mayın duyarlılığı (görülen mayınlar içinde): {mstd([r['classification']['mine_recall_seen'] for r in allrun])}",
          f"- Görülen / toplam mayın: {mstd([r['classification']['mines_seen'] for r in allrun])} / "
          f"{allrun[0]['classification']['mines_total']}",
          f"- Tespit edilen mayınların FGO ile konumlandırma hatası: "
          f"{mstd([r['classification']['mine_geo_err_mean'] for r in allrun if not np.isnan(r['classification']['mine_geo_err_mean'])])} m",
          "\n## 7. Faktör grafiği (ana görev)\n",
          f"- Değişken sayısı: {len(g.x)}, faktör sayısı: {g.n_factors}, poz: {len(out_r['fgo'].pose_off)}, "
          f"nirengi: {len(out_r['fgo'].lm_off)}",
          f"- Son toplu optimizasyon süresi: {out_r['final_opt_time']:.2f} s; tüm görev simülasyonu: {out_r['runtime']:.1f} s",
          f"- Kalibrasyon (gerçek → FGO): pusula montaj sapması {np.rad2deg(out_r['calib_truth'][0]):.2f}° → "
          f"{np.rad2deg(out_r['calib_final'][0]):.2f}°, anomali-sapma katsayısı {out_r['calib_truth'][1]:.4f} → "
          f"{out_r['calib_final'][1]:.4f} rad/100nT, DVL ölçek {100 * out_r['calib_truth'][2]:.3f}% → "
          f"{100 * out_r['calib_final'][2]:.3f}%",
          f"- Ortalama NEES (Monte Carlo): {mstd([cons['mean_nees']] + [r['nees'] for r in mc])} (ideal = 2)",
          f"- 3σ tutarlılık (ana görev): hataların %{100 * cons['frac_within_95']:.0f}'i %95 güven elipsi içinde "
          f"(ortalama NEES={cons['mean_nees']:.2f}, ideal=2)",
          f"\nToplam çalışma süresi: {time.time() - t0:.0f} s"]
    with open(f"{RES}/ozet.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    # 4) şekiller
    print("[5/5] Şekiller çiziliyor...")
    P.fig_scenario(out_r, out_s, cfg, f"{RES}/fig01_senaryo.png")
    P.fig_trajectories(out_r, cfg, f"{RES}/fig02_yorungeler.png")
    P.fig_errors(out_r, cons, cfg, f"{RES}/fig03_hata_ve_sapma.png")
    P.fig_graph(out_r, f"{RES}/fig04_faktor_grafi.png")
    P.fig_ablation(abl, f"{RES}/fig05_ablasyon.png")
    P.fig_threat(out_r, cfg, f"{RES}/fig06_tehdit_haritasi.png")
    P.fig_stealth(out_r, out_s, st_r, st_s, f"{RES}/fig07_gizlilik.png")
    P.fig_classification(cls_results, histories, cls_eval, f"{RES}/fig08_hedef_siniflandirma.png")
    P.fig_calibration(out_r, f"{RES}/fig10_kalibrasyon.png")
    P.fig_monte_carlo([{"nav": nav, "stealth_risk": st_r, "stealth_shortest": st_s}] + mc,
                      f"{RES}/fig09_monte_carlo.png")



if __name__ == "__main__":
    main()
