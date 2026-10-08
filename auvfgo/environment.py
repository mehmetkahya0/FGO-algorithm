"""Bilinmeyen, zamanla değişen, GPS'siz deniz ortamı modeli.

Gerçek (ground-truth) dünya: batimetri, manyetik anomali alanı, okyanus akıntıları,
görüş mesafesi (bulanıklık), DVL dip-kilidi kaybı bölgeleri, deniz tabanı nesneleri
(kaya / mayın / enkaz) ve tehditler (sabit/devriye sonarlar, pasif hidrofonlar).

Aracın erişebildiği önsel bilgiler (kaba batimetri haritası, eski manyetik anomali
haritası, istihbarattan gelen kısmi tehdit listesi) ayrıca, gürültülü olarak üretilir.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RectBivariateSpline

CLASS_NAMES = ["kaya", "mayin", "enkaz"]


class GridMap:
    """Izgara harita + kübik spline enterpolasyonu (değer ve gradyan)."""

    def __init__(self, xs, ys, Z):
        self.xs, self.ys, self.Z = xs, ys, Z
        self.spl = RectBivariateSpline(xs, ys, Z, kx=3, ky=3)

    def value(self, x, y):
        return self.spl.ev(x, y)

    def grad(self, x, y):
        return self.spl.ev(x, y, dx=1), self.spl.ev(x, y, dy=1)


@dataclass
class Threat:
    name: str
    kind: str                 # 'active' | 'passive'
    pos: np.ndarray
    radius: float
    rate_max: float           # en yüksek tespit hızı [1/s]
    ping_period: float = 0.0  # >0 ise aktif sonar darbesi yayar (pasif dinleme ile yakalanabilir)
    known: bool = False       # istihbarat önseli var mı?
    prior_sigma: float = 40.0
    route: np.ndarray = None  # devriye rotası (kapalı döngü)
    speed: float = 0.0
    ping_phase: float = 0.0
    _cum: np.ndarray = field(default=None, repr=False)

    def position(self, t):
        if self.route is None:
            return self.pos
        pts = np.vstack([self.route, self.route[:1]])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if self._cum is None:
            self._cum = np.concatenate([[0.0], np.cumsum(seg)])
        s = (self.speed * t) % self._cum[-1]
        i = int(np.searchsorted(self._cum, s, side="right") - 1)
        a = (s - self._cum[i]) / seg[i]
        return pts[i] + a * (pts[i + 1] - pts[i])


def detection_rate(r, depth, speed, kind, radius, rate_max, thermocline):
    """Bir tehdidin aracı birim zamanda tespit etme hızı λ [1/s].

    - Mesafe ile Gauss biçimli azalma,
    - termoklin altında akustik gölge (etkin menzil %45 azalır),
    - pasif dinleyicilerde hız arttıkça yayılan gürültü (etkin menzil ~ sqrt(u)).
    """
    r = np.asarray(r, float)
    depth = np.asarray(depth, float)
    speed = np.asarray(speed, float)
    if kind == "passive":
        sf = np.sqrt(np.maximum(speed, 0.2) / 1.5)
    else:
        sf = (np.maximum(speed, 0.2) / 1.5) ** 0.15
    reff = radius * sf * np.where(depth > thermocline, 0.55, 1.0)
    return rate_max * np.exp(-(r / reff) ** 2)


class OceanEnvironment:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        S = cfg.world_size
        # --- manyetik anomali kaynakları (jeolojik) ---
        n = 30
        self.mag_c = rng.uniform(-50, S + 50, (n, 2))
        self.mag_a = rng.normal(0.0, 90.0, n)
        self.mag_s = rng.uniform(45.0, 140.0, n)
        self.mag_lin = rng.normal(0.0, 0.03, 2)

        self._make_objects(rng)
        self._make_threats()

        # --- aracın önsel haritaları (gürültülü / kaba) ---
        res = cfg.map_res
        self.xs = np.arange(-100.0, S + 100.0 + 1e-9, res)
        self.ys = self.xs.copy()
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        mag_grid = self.mag_regional(X, Y) + rng.normal(0.0, cfg.mag_map_noise, X.shape)
        self.mag_prior = GridMap(self.xs, self.ys, mag_grid)
        bx = np.arange(-100.0, S + 100.0 + 1e-9, 20.0)
        BX, BY = np.meshgrid(bx, bx, indexing="ij")
        self.bathy_prior = GridMap(bx, bx, self.seafloor(BX, BY) + rng.normal(0, 1.0, BX.shape))

        self.dvl_zones = np.array([[540.0, 450.0, 75.0], [700.0, 760.0, 80.0]])

    # ------------------------------------------------------------------ dünya
    def seafloor(self, x, y):
        """Deniz tabanı derinliği [m] (pozitif aşağı)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        d = 82.0 + 14.0 * np.sin(2 * np.pi * x / 700.0 + 0.6) * np.cos(2 * np.pi * y / 850.0 - 0.3)
        # sığ banka: araç termoklinin üstünde kalır -> tespit edilmeye açık
        d -= 42.0 * np.exp(-((x - 560.0) ** 2 / (2 * 170.0 ** 2) + (y - 660.0) ** 2 / (2 * 110.0 ** 2)))
        # deniz dağı: geçilemez engel
        d -= 70.0 * np.exp(-((x - 640.0) ** 2 + (y - 180.0) ** 2) / (2 * 45.0 ** 2))
        # derin kanal: gizli geçiş koridoru
        a = np.array([80.0, 260.0])
        b = np.array([940.0, 820.0])
        ab = b - a
        tt = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        dist = np.hypot(x - (a[0] + tt * ab[0]), y - (a[1] + tt * ab[1]))
        d += 30.0 * np.exp(-dist ** 2 / (2 * 70.0 ** 2))
        return d

    def mag_regional(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        out = self.mag_lin[0] * (x - 500.0) + self.mag_lin[1] * (y - 500.0)
        for (cx, cy), a, s in zip(self.mag_c, self.mag_a, self.mag_s):
            out = out + a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
        return out

    def mag_true(self, x, y):
        """Gerçek alan = bölgesel anomali + metalik nesnelerin (haritada olmayan) imzası."""
        out = self.mag_regional(x, y)
        for o in self.objects:
            if o["mag_amp"] != 0.0:
                r2 = (x - o["pos"][0]) ** 2 + (y - o["pos"][1]) ** 2
                out = out + o["mag_amp"] * np.exp(-r2 / (2 * o["mag_sigma"] ** 2))
        return out

    def current(self, x, y, t):
        """Zamanla değişen okyanus akıntısı (girdap + gelgit) [m/s]."""
        cx = 0.15 * np.sin(np.pi * y / 500.0 + 0.0005 * t) + 0.06 * np.cos(2 * np.pi * t / 1200.0)
        cy = 0.15 * np.cos(np.pi * x / 600.0 - 0.0004 * t)
        return np.array([cx, cy])

    def visibility(self, x, y):
        """Optik görüş mesafesi [m] (bulanıklık)."""
        return 11.0 + 5.0 * np.sin(x / 140.0) * np.cos(y / 190.0)

    def dvl_dropout(self, x, y):
        d = np.hypot(self.dvl_zones[:, 0] - x, self.dvl_zones[:, 1] - y)
        return bool(np.any(d < self.dvl_zones[:, 2]))

    # --------------------------------------------------------------- nesneler
    def _make_objects(self, rng):
        from .target_classifier import sample_latent

        S = self.cfg.world_size
        counts = {0: 150, 1: 12, 2: 12}
        classes = np.concatenate([np.full(c, k) for k, c in counts.items()])
        rng.shuffle(classes)
        # mayınlar ve enkazın çoğu keşif hedeflerinin çevresinde (mayın tarlası / ilgi alanı)
        survey = np.array(self.cfg.goals[:-1], float)
        pts, cls_out = [], []
        for c in classes:
            for _ in range(500):
                if c != 0 and rng.random() < 0.75:
                    p = survey[rng.integers(len(survey))] + rng.normal(0, 70.0, 2)
                else:
                    p = rng.uniform(30, S - 30, 2)
                if not (30 <= p[0] <= S - 30 and 30 <= p[1] <= S - 30):
                    continue
                if self.seafloor(*p) < self.cfg.min_seafloor_depth + 5:
                    continue
                if pts and np.min(np.linalg.norm(np.array(pts) - p, axis=1)) < 25.0:
                    continue
                pts.append(p)
                cls_out.append(int(c))
                break
        classes = cls_out
        self.objects = []
        for i, (p, c) in enumerate(zip(pts, classes)):
            c = int(c)
            amp = {0: 0.0, 1: 30.0, 2: 15.0}[c]
            self.objects.append({
                "id": i,
                "cls": c,
                "pos": np.array([p[0], p[1], float(self.seafloor(*p))]),
                "latent": sample_latent(c, rng),
                "mag_amp": amp * rng.uniform(0.7, 1.3),
                "mag_sigma": {0: 1.0, 1: 5.0, 2: 7.0}[c],
            })
        self.obj_pos = np.array([o["pos"] for o in self.objects])

    # --------------------------------------------------------------- tehditler
    def _make_threats(self):
        r = self.cfg.nominal_threat_rate
        self.threats = [
            Threat("Sabit aktif sonar A", "active", np.array([370.0, 610.0]), 150.0, r,
                   ping_period=8.0, known=True, prior_sigma=40.0, ping_phase=1.0),
            Threat("Gizli aktif sonar B", "active", np.array([690.0, 330.0]), 140.0, r,
                   ping_period=10.0, known=False, ping_phase=3.0),
            Threat("Pasif hidrofon dizisi", "passive", np.array([590.0, 840.0]), 160.0, r,
                   known=True, prior_sigma=30.0),
            Threat("Devriye gemisi", "active", np.array([300.0, 250.0]), 130.0, r,
                   ping_period=12.0, known=False, ping_phase=5.0,
                   route=np.array([[220.0, 230.0], [500.0, 180.0], [560.0, 430.0], [260.0, 470.0]]),
                   speed=2.5),
        ]

    def threat_positions(self, t):
        return np.array([th.position(t) for th in self.threats])

    def true_detection_rate(self, p, depth, speed, t):
        """Gerçek toplam tespit hızı (tüm tehditler)."""
        lam = 0.0
        for th in self.threats:
            r = np.linalg.norm(th.position(t) - p[:2])
            lam += float(detection_rate(r, depth, speed, th.kind, th.radius, th.rate_max,
                                        self.cfg.thermocline_depth))
        return lam
