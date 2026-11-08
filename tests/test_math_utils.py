import numpy as np
from ecqad_uav.utils.math_utils import wrap_pi, rot_body_to_ned


def test_wrap_pi():
    assert np.isclose(wrap_pi(np.pi), -np.pi)
    assert np.isclose(wrap_pi(0.0), 0.0)


def test_rotation_is_orthonormal():
    R = rot_body_to_ned(0.2, -0.1, 0.7)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
