from __future__ import annotations
import numpy as np


def wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def angle_error(target: float, actual: float) -> float:
    return float(wrap_pi(target - actual))


def euler_rates_from_body_rates(phi: float, theta: float, p: float, q: float, r: float) -> np.ndarray:
    """ZYX Euler-angle rates from body rates. Avoid theta near +/-90 deg."""
    cth = np.cos(theta)
    cth = np.sign(cth) * max(abs(cth), 1e-6)
    tth = np.sin(theta) / cth
    sphi, cphi = np.sin(phi), np.cos(phi)
    return np.array([
        p + q * sphi * tth + r * cphi * tth,
        q * cphi - r * sphi,
        q * sphi / cth + r * cphi / cth,
    ])


def rot_body_to_ned(phi: float, theta: float, psi: float) -> np.ndarray:
    """Direction cosine matrix from body FRD to NED for ZYX Euler angles."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
        [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
        [-sth,       sphi * cth,                         cphi * cth],
    ])


def nearest_spd(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Symmetrize and floor eigenvalues to keep a covariance positive definite."""
    B = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(B)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals) @ vecs.T
