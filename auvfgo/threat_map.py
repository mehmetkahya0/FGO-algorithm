"""Bayesçi tehdit inanç haritası (gizlilik / tehdit maruziyeti).

İki katman:
 1) İstihbarat katmanı: önceden bilinen tehditler için konum belirsizliğine göre
    Gauss olasılık lekeleri (pasif dinleyiciler dahil).
 2) Aktif yayıcı katmanı: pasif hidrofonla yakalanan sonar darbelerinin kerterizleri
    log-olasılık (log-odds) ızgarasında birikir; farklı noktalardan alınan kerterizler
    kesişerek bilinmeyen/hareketli yayıcıyı nirengi ile ortaya çıkarır. Zamanla önsele
    doğru unutma (decay) uygulanır -> zamanla değişen (devriye) tehditlere uyum.

Risk: her hücrede, hücrenin derinliği (termoklin) ve o hücrede seçilecek hız dikkate
alınarak, inanılan en kötü durum tespit hızı λ [1/s] ve metre başına tehlike λ/u.
"""
import numpy as np

from .environment import detection_rate
from .utils import wrap


def logit(p):
    return np.log(p / (1 - p))


class ThreatBelief:
    def __init__(self, cfg, env, rng):
        self.cfg = cfg
        res = cfg.plan_res
        self.nx = int(round(cfg.world_size / res))
        c = (np.arange(self.nx) + 0.5) * res
        self.X, self.Y = np.meshgrid(c, c, indexing="ij")
        self.cells = np.stack([self.X.ravel(), self.Y.ravel()], axis=1)
        self.depth = env.bathy_prior.value(self.X, self.Y) - cfg.altitude_cmd
        self.l0 = logit(cfg.threat_prior_p)
        self.L = np.full(self.X.shape, self.l0)
        # istihbarat: bilinen tehditler (konumu gürültülü bildirilmiş)
        self.known = []
        for th in env.threats:
            if th.known:
                pos = th.pos + rng.normal(0, th.prior_sigma * 0.5, 2)
                p = 0.95 * np.exp(-((self.X - pos[0]) ** 2 + (self.Y - pos[1]) ** 2) / (2 * th.prior_sigma ** 2))
                self.known.append({"threat": th, "pos": pos, "p": p})
        self.n_intercepts = 0

    # ------------------------------------------------------------ güncelleme
    def intercept(self, p_est, bearing_world, pos_sigma, max_range, heading_sigma=0.0):
        """Tek kerteriz ölçümü ile aktif yayıcı katmanını güncelle.

        Ortamda birden fazla yayıcı bulunduğundan, bir yayıcının darbesi diğer
        yönlerdeki hücreler için *negatif* kanıt değildir; bu yüzden yalnızca
        olabilirlik oranının pozitif kısmı biriktirilir (Hough benzeri ışın oylaması).
        Farklı konumlardan alınan kerterizlerin kesiştiği hücreler hızla yükselir;
        tekil ışınlar ise zamanla unutulur (decay)."""
        sb = np.deg2rad(self.cfg.hydrophone_bearing_sigma_deg)
        dx = self.X - p_est[0]
        dy = self.Y - p_est[1]
        dist = np.hypot(dx, dy)
        delta = wrap(np.arctan2(dy, dx) - bearing_world)
        s_eff = np.sqrt(sb ** 2 + heading_sigma ** 2 + (pos_sigma / np.maximum(dist, 1.0)) ** 2
                        + (0.5 * self.cfg.plan_res / np.maximum(dist, 1.0)) ** 2 + 0.04 ** 2)
        lr = np.log(2 * np.pi) - np.log(np.sqrt(2 * np.pi) * s_eff) - 0.5 * (delta / s_eff) ** 2
        upd = 0.45 * np.clip(lr, 0.0, None)
        mask = (dist < max_range) & (dist > 5.0)
        self.L[mask] += upd[mask]
        np.clip(self.L, self.l0, 8.0, out=self.L)
        self.n_intercepts += 1

    def decay(self, dt):
        a = np.exp(-dt / self.cfg.threat_decay_tau)
        self.L = self.l0 + (self.L - self.l0) * a

    def p_active(self):
        return 1.0 / (1.0 + np.exp(-self.L))

    def p_total(self):
        q = 1.0 - self.p_active()
        for k in self.known:
            q = q * (1.0 - k["p"])
        return 1.0 - q

    # ------------------------------------------------------------ risk
    def _max_conv(self, P, kind, radius, rate, speed, thr=0.02, kmax=400):
        flat = P.ravel()
        idx = np.where(flat > thr)[0]
        if len(idx) == 0:
            return np.zeros(self.X.shape)
        if len(idx) > kmax:
            idx = idx[np.argsort(flat[idx])[-kmax:]]
        cand = self.cells[idx]
        out = np.zeros(len(self.cells))
        depth = self.depth.ravel()
        spd = np.broadcast_to(speed, self.X.shape).ravel()
        for s in range(0, len(idx), 100):
            cc = cand[s:s + 100]
            d = np.linalg.norm(self.cells[None, :, :] - cc[:, None, :], axis=2)
            lam = detection_rate(d, depth[None], spd[None], kind, radius, rate,
                                 self.cfg.thermocline_depth) * flat[idx[s:s + 100]][:, None]
            out = np.maximum(out, lam.max(axis=0))
        return out.reshape(self.X.shape)

    def rate_map(self, speed):
        c = self.cfg
        lam = self._max_conv(self.p_active(), "active", c.nominal_threat_radius, c.nominal_threat_rate, speed)
        for k in self.known:
            th = k["threat"]
            lam = lam + self._max_conv(k["p"], th.kind, th.radius, th.rate_max, speed)
        return lam

    def hazard_maps(self, stealth=True):
        """Dönüş: (λ haritası, hız haritası, metre başına tehlike)."""
        c = self.cfg
        lam_c = self.rate_map(c.cruise_speed)
        if not stealth:
            spd = np.full(self.X.shape, c.cruise_speed)
            return lam_c, spd, lam_c / spd
        lam_s = self.rate_map(c.stealth_speed)
        # her hücrede metre başına tehlikeyi (λ/u) en aza indiren hız seçilir:
        # pasif dinleyiciye karşı yavaş/sessiz, aktif sonara karşı hızlı geçiş daha iyidir.
        slow = (lam_s / c.stealth_speed < lam_c / c.cruise_speed) & (lam_c > c.stealth_hazard_threshold)
        spd = np.where(slow, c.stealth_speed, c.cruise_speed)
        lam = np.where(slow, lam_s, lam_c)
        return lam, spd, lam / spd

    def cell_index(self, p):
        i = int(np.clip(p[0] // self.cfg.plan_res, 0, self.nx - 1))
        j = int(np.clip(p[1] // self.cfg.plan_res, 0, self.nx - 1))
        return i, j
