"""Ortak matematiksel yardımcılar."""
import numpy as np


def wrap(a):
    """Açıyı (-pi, pi] aralığına sar."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rot2(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def drot2(psi):
    """d R(psi) / d psi."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[-s, -c], [c, -s]])


def sqrt_info(cov):
    """Beyazlatma matrisi W:  ||W r||^2 = r^T cov^-1 r."""
    cov = np.asarray(cov, float)
    info = np.linalg.inv(cov)
    info = 0.5 * (info + info.T)
    return np.linalg.cholesky(info).T


def landmark_model(P, L):
    """Vektörize nirengi (landmark) ölçüm modeli.

    P: (m,5) poz [x, y, z, psi, b_g], L: (m,3) nirengi konumu.
    h = [R(psi)^T (l_xy - p_xy), l_z - z]   (gövde-yaw çerçevesinde göreli konum)
    Dönüş: h (m,3), Hp (m,3,5), Hl (m,3,3)
    """
    P = np.atleast_2d(P)
    L = np.atleast_2d(L)
    m = P.shape[0]
    c, s = np.cos(P[:, 3]), np.sin(P[:, 3])
    dx = L[:, 0] - P[:, 0]
    dy = L[:, 1] - P[:, 1]
    dz = L[:, 2] - P[:, 2]
    h = np.stack([c * dx + s * dy, -s * dx + c * dy, dz], axis=1)
    Hp = np.zeros((m, 3, 5))
    Hp[:, 0, 0], Hp[:, 0, 1] = -c, -s
    Hp[:, 1, 0], Hp[:, 1, 1] = s, -c
    Hp[:, 2, 2] = -1.0
    Hp[:, 0, 3] = -s * dx + c * dy
    Hp[:, 1, 3] = -c * dx - s * dy
    Hl = np.zeros((m, 3, 3))
    Hl[:, 0, 0], Hl[:, 0, 1] = c, s
    Hl[:, 1, 0], Hl[:, 1, 1] = -s, c
    Hl[:, 2, 2] = 1.0
    return h, Hp, Hl


def body_to_world(pose, m):
    """Gövde-yaw çerçevesindeki göreli ölçümü dünya konumuna çevir."""
    R = rot2(pose[3])
    xy = pose[:2] + R @ m[:2]
    return np.array([xy[0], xy[1], pose[2] + m[2]])
