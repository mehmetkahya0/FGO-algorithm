"""Rapor şekilleri (matplotlib, Agg)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .environment import CLASS_NAMES  # noqa: E402

COL = {"truth": "#111111", "dr": "#c0392b", "ekf": "#e67e22", "fgo_online": "#2e86c1",
       "fgo_smoothed": "#1e8449", "shortest": "#8e44ad"}
LBL = {"truth": "Gerçek", "dr": "Ölü hesap (DVL+AHRS)", "ekf": "EKF-SLAM",
       "fgo_online": "FGO (çevrimiçi)", "fgo_smoothed": "FGO (düzleştirilmiş)"}
CLS_MARK = {0: ("o", "#7f8c8d"), 1: ("X", "#e74c3c"), 2: ("s", "#f39c12")}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _terrain(ax, env, cfg, cbar=True):
    xs = np.linspace(0, cfg.world_size, 220)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    D = env.seafloor(X, Y)
    cs = ax.contourf(X, Y, D, 25, cmap="Blues")
    ax.contour(X, Y, D - cfg.altitude_cmd, [cfg.thermocline_depth], colors="#16a085", linewidths=1.2,
               linestyles="--")
    ax.contourf(X, Y, D, [0, cfg.min_seafloor_depth], colors=["#5d4037"])
    if cbar:
        plt.colorbar(cs, ax=ax, label="Deniz tabanı derinliği [m]", shrink=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.set_xlabel("Doğu [m]")
    ax.set_ylabel("Kuzey [m]")


def _threats(ax, env, out=None):
    for i, th in enumerate(env.threats):
        ls = "-" if th.known else ":"
        c = "#c0392b" if th.kind == "active" else "#8e44ad"
        if th.route is None:
            ax.add_patch(plt.Circle(th.pos, th.radius, fill=False, color=c, ls=ls, lw=1.5))
            ax.plot(*th.pos, marker="^", color=c, ms=9)
            ax.annotate(th.name, th.pos + np.array([8, 8]), fontsize=7, color=c)
        else:
            r = np.vstack([th.route, th.route[:1]])
            ax.plot(r[:, 0], r[:, 1], color=c, ls=ls, lw=1)
            if out is not None:
                p = out["patrol"][:, i]
                ax.plot(p[-1, 0], p[-1, 1], marker="^", color=c, ms=9)
            ax.annotate(th.name, th.route[0] + np.array([-40, -25]), fontsize=7, color=c)


def fig_scenario(out_r, out_s, cfg, path):
    env = out_r["env"]
    fig, ax = plt.subplots(figsize=(9, 8))
    _terrain(ax, env, cfg)
    _threats(ax, env, out_r)
    for c in range(3):
        m, col = CLS_MARK[c]
        P = np.array([o["pos"] for o in env.objects if o["cls"] == c])
        ax.scatter(P[:, 0], P[:, 1], marker=m, c=col, s=14 if c == 0 else 30, label=f"Nesne: {CLASS_NAMES[c]}",
                   edgecolors="k", linewidths=0.3, zorder=3)
    for zx, zy, zr in env.dvl_zones:
        ax.add_patch(plt.Circle((zx, zy), zr, fill=True, color="#f1c40f", alpha=0.18))
    ax.plot(out_s["truth"][:, 0], out_s["truth"][:, 1], color=COL["shortest"], lw=1.6, ls="--",
            label="En kısa yol planlayıcı")
    ax.plot(out_r["truth"][:, 0], out_r["truth"][:, 1], color="#111111", lw=2.0, label="Risk-farkında planlayıcı")
    ax.plot(*cfg.start, "go", ms=10, label="Başlangıç")
    for i, g in enumerate(cfg.goals):
        ax.plot(*g, "*", color="#27ae60", ms=16, mec="k")
        ax.annotate(f"H{i + 1}", np.array(g) + 12, fontsize=9, weight="bold")
    ax.plot([], [], color="#16a085", ls="--", label="Termoklin sınırı (altı: akustik gölge)")
    ax.plot([], [], color="#c0392b", ls=":", label="Bilinmeyen tehdit (önsel yok)")
    ax.add_patch(plt.Circle((-1e3, -1e3), 1, color="#f1c40f", alpha=0.3, label="DVL dip-kilidi kaybı"))
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.set_title("Görev senaryosu: GPS'siz, bilinmeyen ortam ve tehditler")
    _save(fig, path)


def fig_trajectories(out, cfg, path):
    env = out["env"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    ax = axs[0]
    _terrain(ax, env, cfg, cbar=False)
    for k in ["truth", "dr", "ekf", "fgo_smoothed"]:
        ax.plot(out[k][:, 0], out[k][:, 1], color=COL[k], lw=1.6 if k != "truth" else 2.2, label=LBL[k])
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Yörüngeler: gerçek ve kestirimler")
    ax = axs[1]
    tr = out["truth"]
    i0 = len(tr) // 2
    win = slice(max(i0 - 120, 0), i0 + 120)
    ax.plot(tr[win, 0], tr[win, 1], color=COL["truth"], lw=2.2, label="Gerçek")
    for k in ["ekf", "fgo_online", "fgo_smoothed"]:
        ax.plot(out[k][win, 0], out[k][win, 1], color=COL[k], lw=1.4, label=LBL[k])
    lms = out["fgo"].landmarks()
    front = out["frontend"]
    xs, ys = tr[win, 0], tr[win, 1]
    box = (xs.min() - 60, xs.max() + 60, ys.min() - 60, ys.max() + 60)
    first = True
    for lid, p in lms.items():
        tid = front.true_id(lid)
        if not (box[0] < p[0] < box[1] and box[2] < p[1] < box[3]):
            continue
        ax.plot(p[0], p[1], "+", color=COL["fgo_smoothed"], ms=9, mew=2, label="FGO nirengi kestirimi" if first else None)
        if tid >= 0:
            q = env.obj_pos[tid]
            ax.plot(q[0], q[1], "o", mfc="none", color="k", ms=7, label="Gerçek nesne" if first else None)
            ax.plot([p[0], q[0]], [p[1], q[1]], "k-", lw=0.5)
        first = False
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title("Yakınlaştırma: SLAM nirengileri ve yörüngeler")
    _save(fig, path)


def fig_errors(out, cons, cfg, path):
    t = out["t"]
    tr = out["truth"]
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax = axs[0]
    for k in ["dr", "ekf", "fgo_online", "fgo_smoothed"]:
        e = np.linalg.norm(out[k][:, :2] - tr[:, :2], axis=1)
        ax.plot(t, e, color=COL[k], label=LBL[k], lw=1.3)
    tt = t[cons["t_idx"]]
    ax.fill_between(tt, 0, 3 * cons["sigma_xy"], color=COL["fgo_smoothed"], alpha=0.15, label="FGO 3σ sınırı")
    dvl = np.array(out["dvl_ok"]) < 0.5
    ax.fill_between(t, 0, 1, where=dvl, transform=ax.get_xaxis_transform(), color="#f1c40f", alpha=0.3,
                    label="DVL kesintisi")
    ax.set_ylim(0, max(30, 1.1 * np.percentile(np.linalg.norm(out["ekf"][:, :2] - tr[:, :2], axis=1), 99) * 2))
    ax.set_ylabel("Yatay konum hatası [m]")
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Navigasyon hatası (GPS yok)")
    ax = axs[1]
    for k in ["ekf", "fgo_smoothed"]:
        ax.plot(t, np.rad2deg(np.abs(np.angle(np.exp(1j * (out[k][:, 3] - tr[:, 3]))))), color=COL[k],
                label=LBL[k], lw=1.1)
    ax.set_ylabel("|Yön hatası| [°]")
    ax.legend(fontsize=8)
    ax = axs[2]
    fg = out["fgo_smoothed"]
    ax.plot(t, np.rad2deg(tr[:, 4]) * 3600, color="k", lw=2, label="Gerçek jiroskop sapması")
    ax.plot(t, np.rad2deg(fg[:, 4]) * 3600, color=COL["fgo_smoothed"], lw=1.3, label="FGO kestirimi")
    ax.plot(t, np.rad2deg(out["ekf"][:, 4]) * 3600, color=COL["ekf"], lw=1.0, alpha=0.8, label="EKF kestirimi")
    ax.set_ylabel("Jiroskop sapması [°/saat]")
    ax.set_xlabel("Zaman [s]")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_graph(out, path):
    fgo = out["fgo"]
    g = fgo.graph
    H = g.information()
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    n = min(H.shape[0], 900)
    axs[0].spy(H[:n, :n], markersize=0.4, color="#1e8449")
    axs[0].set_title(f"Bilgi matrisi seyreklik deseni (ilk {n} değişken)\n"
                     f"toplam {H.shape[0]} değişken, doluluk %{100 * H.nnz / H.shape[0] ** 2:.2f}")
    names = {"PriorFactor": "Önsel", "OdometryFactor": "Odometri\n(DVL+IMU)", "DepthFactor": "Basınç",
             "HeadingFactor": "Pusula", "HeadingCalibFactor": "Pusula\n(+kalibrasyon)",
             "ScalarPrior": "Kalib.\nönsel", "MagMapFactor": "Manyetik\nharita", "LandmarkFactor": "Sonar/optik\nnirengi"}
    cnt = [(names.get(type(b).__name__, type(b).__name__), b.count) for b in g.blocks if b.count]
    axs[1].bar([c[0] for c in cnt], [c[1] for c in cnt], color="#2e86c1")
    axs[1].set_title(f"Faktör sayıları (toplam {g.n_factors})")
    axs[1].tick_params(axis="x", labelsize=7)
    st = g.stats
    axs[2].semilogy([s["cost0"] for s in st], color="#c0392b", lw=0.8, label="Optimizasyon öncesi")
    axs[2].semilogy([s["cost"] for s in st], color="#1e8449", lw=0.8, label="LM sonrası")
    axs[2].set_xlabel("Artımlı optimizasyon çağrısı")
    axs[2].set_ylabel("Sağlam maliyet  Σρ(‖r‖²)")
    axs[2].set_title("Artımlı Levenberg–Marquardt yakınsaması")
    axs[2].legend(fontsize=8)
    _save(fig, path)


def fig_ablation(abl, path):
    names = list(abl.keys())
    mean = [np.mean(abl[n]) for n in names]
    std = [np.std(abl[n]) for n in names]
    order = np.argsort(mean)
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = ["#1e8449" if names[i].startswith("FGO (tam") else "#5d6d7e" for i in order]
    ax.barh([names[i] for i in order], [mean[i] for i in order], xerr=[std[i] for i in order], color=cols,
            capsize=3)
    for k, i in enumerate(order):
        ax.text(mean[i] + std[i] + 0.3, k, f"{mean[i]:.2f} m", va="center", fontsize=8)
    ax.set_xlabel("Düzleştirilmiş yörünge ATE RMSE [m] (Monte Carlo ort. ± std)")
    ax.set_title("FGO faktör/bileşen ablasyonu: her sensör modalitesinin katkısı")
    _save(fig, path)


def fig_threat(out, cfg, path):
    env = out["env"]
    snaps = out["snaps"]
    sel = [snaps[i] for i in np.linspace(0, len(snaps) - 1, min(4, len(snaps))).astype(int)]
    fig, axs = plt.subplots(1, len(sel), figsize=(4.6 * len(sel), 4.8))
    tr, t = out["truth"], out["t"]
    for ax, (ts, p, lam, pos, thp) in zip(np.atleast_1d(axs), sel):
        im = ax.imshow(lam.T * 1000, origin="lower", extent=[0, cfg.world_size, 0, cfg.world_size],
                       cmap="magma", vmin=0, vmax=cfg.nominal_threat_rate * 1000)
        k = t <= ts
        ax.plot(tr[k, 0], tr[k, 1], color="cyan", lw=1.5)
        for (tt, ti, pp, bw) in out["intercepts"]:
            if ts - 40 < tt <= ts:
                ax.plot([pp[0], pp[0] + 350 * np.cos(bw)], [pp[1], pp[1] + 350 * np.sin(bw)], "w-", lw=0.4)
        for i, th in enumerate(env.threats):
            ax.plot(thp[i, 0], thp[i, 1], marker="x", color="lime" if th.known else "#00ff99", ms=10, mew=2)
        ax.set_xlim(0, cfg.world_size)
        ax.set_ylim(0, cfg.world_size)
        ax.set_title(f"t = {ts:.0f} s")
    cb = fig.colorbar(im, ax=np.atleast_1d(axs).tolist(), shrink=0.8)
    cb.set_label("İnanılan tespit hızı λ [10⁻³/s]")
    fig.suptitle("Bayesçi tehdit inanç haritası: pasif kerterizlerle gizli/hareketli tehditlerin keşfi "
                 "(x: gerçek tehdit konumları, beyaz: son kerterizler)", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_stealth(out_r, out_s, st_r, st_s, path):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axs[0]
    ax.plot(out_r["t"], out_r["cum_pdet"], color="#1e8449", lw=2, label="Risk-farkında (FGO + tehdit inancı)")
    ax.plot(out_s["t"], out_s["cum_pdet"], color=COL["shortest"], lw=2, ls="--", label="En kısa yol")
    ax.set_xlabel("Zaman [s]")
    ax.set_ylabel("Kümülatif tespit edilme olasılığı")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Gizlilik: görev boyunca tespit edilme olasılığı")
    ax = axs[1]
    names = list(st_r["per_threat_integrated_lambda"].keys())
    x = np.arange(len(names))
    ax.bar(x - 0.2, [st_s["per_threat_integrated_lambda"][n] for n in names], 0.4, color=COL["shortest"],
           label="En kısa yol")
    ax.bar(x + 0.2, [st_r["per_threat_integrated_lambda"][n] for n in names], 0.4, color="#1e8449",
           label="Risk-farkında")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=10)
    ax.set_ylabel("∫λ dt  (tehdit başına maruziyet)")
    ax.legend()
    ax.set_title("Tehdit başına maruziyet")
    _save(fig, path)


def fig_classification(results, histories, cls_eval, path):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for n, h in histories.items():
        axs[0].plot(h, label=n)
    axs[0].set_xlabel("Epok")
    axs[0].set_ylabel("Çapraz entropi kaybı")
    axs[0].set_title("MLP eğitim eğrileri")
    axs[0].legend(fontsize=7)
    names = list(results.keys())
    x = np.arange(len(names))
    axs[1].bar(x - 0.27, [results[n]["accuracy"] for n in names], 0.27, label="Doğruluk (tüm gözlemler)")
    axs[1].bar(x, [results[n]["macro_f1"] for n in names], 0.27, label="Makro-F1")
    axs[1].bar(x + 0.27, [results[n]["accuracy_optical_available"] for n in names], 0.27,
               label="Doğruluk (optik mevcut, yakın)")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
    axs[1].set_ylim(0.5, 1.0)
    axs[1].legend(fontsize=7, loc="lower right")
    axs[1].set_title("Modalite ablasyonu (tek gözlem)")
    cm = np.array(cls_eval["confusion"])
    axs[2].imshow(cm, cmap="Greens")
    for i in range(3):
        for j in range(3):
            axs[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    axs[2].set_xticks(range(3))
    axs[2].set_yticks(range(3))
    axs[2].set_xticklabels(CLASS_NAMES)
    axs[2].set_yticklabels(CLASS_NAMES)
    axs[2].set_xlabel("Tahmin")
    axs[2].set_ylabel("Gerçek")
    axs[2].grid(False)
    axs[2].set_title(f"Görev içi zamansal füzyon (doğruluk {cls_eval['accuracy']:.2f})")
    _save(fig, path)


def fig_monte_carlo(mc, path):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    keys = ["Ölü hesap (DVL+AHRS)", "EKF-SLAM", "FGO (çevrimiçi)", "FGO (düzleştirilmiş)"]
    data = [[r["nav"][k]["ate_rmse"] for r in mc] for k in keys]
    bp = axs[0].boxplot(data, patch_artist=True)
    for patch, k in zip(bp["boxes"], ["dr", "ekf", "fgo_online", "fgo_smoothed"]):
        patch.set_facecolor(COL[k])
        patch.set_alpha(0.6)
    axs[0].set_xticks(range(1, 5))
    axs[0].set_xticklabels([k.replace(" (", "\n(") for k in keys], fontsize=8)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("ATE RMSE [m] (log)")
    axs[0].set_title(f"Monte Carlo navigasyon doğruluğu ({len(mc)} farklı dünya/gürültü)")
    pr = [r["stealth_risk"]["p_detect"] for r in mc]
    ps = [r["stealth_shortest"]["p_detect"] for r in mc]
    axs[1].boxplot([ps, pr], patch_artist=True)
    axs[1].set_xticks([1, 2])
    axs[1].set_xticklabels(["En kısa yol", "Risk-farkında"])
    axs[1].set_ylabel("Tespit edilme olasılığı")
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Monte Carlo gizlilik karşılaştırması")
    _save(fig, path)


def fig_calibration(out, path):
    t = out["t"]
    cf, ce, ct = np.array(out["calib_fgo"]), np.array(out["calib_ekf"]), out["calib_truth"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    specs = [(0, np.rad2deg(1), "Pusula montaj sapması c [°]"),
             (1, np.rad2deg(1), "Anomali-bağlı pusula sapması k [°/100 nT]"),
             (2, 100.0, "DVL ölçek hatası s [%]")]
    for ax, (i, sc, lab) in zip(axs, specs):
        ax.axhline(ct[i] * sc, color="k", lw=2, label="Gerçek")
        ax.plot(t, cf[:, i] * sc, color=COL["fgo_smoothed"], lw=1.4, label="FGO (çevrimiçi)")
        ax.plot(t, ce[:, i] * sc, color=COL["ekf"], lw=1.0, ls="--", label="EKF-SLAM")
        ax.set_title(lab)
        ax.set_xlabel("Zaman [s]")
        ax.legend(fontsize=8)
    fig.suptitle("Sensör kalibrasyonunun faktör grafiği içinde çevrimiçi kestirimi", fontsize=11)
    _save(fig, path)
