"""Risk ve bilgi farkında A* yol planlayıcı.

Kenar maliyeti = uzunluk * (1 + w_risk * tehlike/m + w_info * (1 - bilgi))
 - tehlike/m : tehdit inanç haritasından beklenen tespit hızı / hız  (gizlilik)
 - bilgi     : önsel manyetik haritanın gradyan büyüklüğü (navigasyon için
               bilgilendirici bölgeleri tercih eden "aktif lokalizasyon" terimi)
 - sığ bölgeler (deniz tabanı < min derinlik) engel.
"""
import heapq

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation


class AStarPlanner:
    def __init__(self, cfg, env):
        self.cfg = cfg
        res = cfg.plan_res
        self.res = res
        self.nx = int(round(cfg.world_size / res))
        c = (np.arange(self.nx) + 0.5) * res
        X, Y = np.meshgrid(c, c, indexing="ij")
        self.blocked = binary_dilation(env.bathy_prior.value(X, Y) < cfg.min_seafloor_depth, iterations=2)
        gx, gy = env.mag_prior.grad(X, Y)
        g = gaussian_filter(np.hypot(gx, gy), 2.0)
        self.info = np.clip(g / np.percentile(g, 90), 0, 1)

    def to_cell(self, p):
        return (int(np.clip(p[0] // self.res, 0, self.nx - 1)),
                int(np.clip(p[1] // self.res, 0, self.nx - 1)))

    def to_world(self, ij):
        return (np.asarray(ij, float) + 0.5) * self.res

    def plan(self, start, goal, hazard_per_m, w_risk=None, w_info=None):
        c = self.cfg
        w_risk = c.w_risk if w_risk is None else w_risk
        w_info = c.w_info if w_info is None else w_info
        cost = 1.0 + w_risk * hazard_per_m + w_info * (1.0 - self.info)
        s, g = self.to_cell(start), self.to_cell(goal)
        blocked = self.blocked.copy()
        blocked[s] = False
        blocked[g] = False
        n = self.nx
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        gscore = np.full((n, n), np.inf)
        gscore[s] = 0.0
        parent = {}
        closed = np.zeros((n, n), bool)
        h = lambda a: np.hypot(a[0] - g[0], a[1] - g[1])
        pq = [(h(s), 0.0, s)]
        while pq:
            _, gs, cur = heapq.heappop(pq)
            if closed[cur]:
                continue
            if cur == g:
                break
            closed[cur] = True
            for di, dj, L in nbrs:
                nb = (cur[0] + di, cur[1] + dj)
                if not (0 <= nb[0] < n and 0 <= nb[1] < n) or blocked[nb] or closed[nb]:
                    continue
                ng = gs + L * 0.5 * (cost[cur] + cost[nb])
                if ng < gscore[nb]:
                    gscore[nb] = ng
                    parent[nb] = cur
                    heapq.heappush(pq, (ng + h(nb), ng, nb))
        if g not in parent and g != s:
            return np.array([start[:2], goal[:2]], float)
        path = [g]
        while path[-1] != s:
            path.append(parent[path[-1]])
        path = path[::-1]
        pts = np.array([self.to_world(p) for p in path])
        pts[0] = start[:2]
        pts[-1] = goal[:2]
        return pts


def path_length(pts):
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0
