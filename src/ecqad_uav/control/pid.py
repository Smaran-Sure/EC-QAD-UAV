from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    output_min: float = -np.inf
    output_max: float = np.inf
    integral_min: float = -np.inf
    integral_max: float = np.inf

    def __post_init__(self):
        self.integral = 0.0
        self.prev_error = None

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float, derivative_measurement: float | None = None) -> float:
        dt = max(float(dt), 1e-6)
        self.integral = float(np.clip(
            self.integral + error * dt,
            self.integral_min,
            self.integral_max,
        ))
        if derivative_measurement is None:
            deriv = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        else:
            deriv = float(derivative_measurement)
        self.last_derivative = deriv
        u_unsat = self.kp * error + self.ki * self.integral + self.kd * deriv
        u = float(np.clip(u_unsat, self.output_min, self.output_max))
        # Simple anti-windup: do not keep integrating farther into saturation.
        if u != u_unsat and self.ki != 0.0:
            self.integral -= error * dt
        self.prev_error = error
        return u
