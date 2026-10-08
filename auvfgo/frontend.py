"""Algılama ön-ucu: sonar/optik gözlemlerin nirengi kimliklerine veri ilişkilendirmesi.

Gerçek kimlikler bilinmez; gözlem FGO'nun güncel poz tahmini ile dünya çerçevesine
taşınır ve en yakın nirengi kestirimine, poz belirsizliğine göre genişleyen bir kapı
içinde açgözlü (greedy) olarak eşlenir. Kapı dışındakiler yeni nirengi başlatır.
Gerçek kimlikler yalnızca *değerlendirme* için kaydedilir.
"""
from collections import Counter, defaultdict

import numpy as np

from .utils import body_to_world


class LandmarkFrontend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.next_id = 0
        self.votes = defaultdict(Counter)
        self.n_assoc = 0
        self.n_wrong = 0

    def associate(self, pose, cov_xy, lm_est, dets):
        """lm_est: {lid: (3,)} güncel nirengi kestirimleri.
        Dönüş: [(lid, det)] ve faktör girdileri [(lid, m, cov, src)]."""
        sig = float(np.sqrt(max(np.trace(cov_xy), 0.0)))
        ids = list(lm_est.keys())
        L = np.array([lm_est[k] for k in ids]) if ids else np.zeros((0, 3))
        pending = []
        for di, det in enumerate(dets):
            m = det["optical"][0] if det["optical"] is not None else det["sonar"][0]
            w = body_to_world(pose, m)
            rng = np.linalg.norm(m)
            gate = self.cfg.assoc_gate + 3.0 * sig + 0.04 * rng
            if len(L):
                d = np.linalg.norm(L[:, :2] - w[:2], axis=1)
                for k in np.where(d < gate)[0]:
                    pending.append((d[k], di, ids[k]))
        pending.sort()
        used_lm, assigned = set(), {}
        for d, di, lid in pending:
            if di in assigned or lid in used_lm:
                continue
            assigned[di] = lid
            used_lm.add(lid)
        out, factors = [], []
        for di, det in enumerate(dets):
            if di in assigned:
                lid = assigned[di]
            else:
                lid = self.next_id
                self.next_id += 1
            out.append((lid, det))
            if det["true_id"] >= 0 or self.votes[lid]:
                if self.votes[lid] and det["true_id"] != self.votes[lid].most_common(1)[0][0]:
                    self.n_wrong += 1
            self.votes[lid][det["true_id"]] += 1
            self.n_assoc += 1
            if det["sonar"] is not None:
                factors.append((lid, det["sonar"][0], det["sonar"][1], "sonar"))
            if det["optical"] is not None:
                factors.append((lid, det["optical"][0], det["optical"][1], "optical"))
        return out, factors

    def n_obs(self, lid):
        return int(sum(self.votes[lid].values()))

    def true_id(self, lid):
        return self.votes[lid].most_common(1)[0][0] if self.votes[lid] else -1
