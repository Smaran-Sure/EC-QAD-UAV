from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class FaultState:
    gyro_bias_xyz_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gps_available: bool = True
    airspeed_bias_mps: float = 0.0
    aileron_effectiveness: float = 1.0
    active_labels: tuple[str, ...] = ()


class FaultInjector:
    def __init__(self, cfg: dict):
        self.cfg = cfg.get("faults", {})

    @staticmethod
    def _active(spec: dict, t: float) -> bool:
        if not spec.get("enabled", False):
            return False
        if t < float(spec.get("start_s", 0.0)):
            return False
        end = spec.get("end_s", None)
        return end is None or t <= float(end)

    def state(self, t: float) -> FaultState:
        gyro = [0.0, 0.0, 0.0]
        gps_available = True
        airspeed_bias = 0.0
        aileron_eff = 1.0
        labels: list[str] = []

        spec = self.cfg.get("gyro_bias", {})
        if self._active(spec, t):
            axis = {"x": 0, "y": 1, "z": 2}[str(spec.get("axis", "z")).lower()]
            gyro[axis] += math.radians(float(spec.get("magnitude_deg_s", 0.0)))
            labels.append("gyro_bias")

        spec = self.cfg.get("gps_dropout", {})
        if self._active(spec, t):
            gps_available = False
            labels.append("gps_dropout")

        spec = self.cfg.get("airspeed_bias", {})
        if self._active(spec, t):
            airspeed_bias += float(spec.get("magnitude_mps", 0.0))
            labels.append("airspeed_bias")

        spec = self.cfg.get("aileron_loss_of_effectiveness", {})
        if self._active(spec, t):
            aileron_eff = float(spec.get("effectiveness", 1.0))
            labels.append("aileron_loe")

        return FaultState(
            gyro_bias_xyz_rad_s=tuple(gyro),
            gps_available=gps_available,
            airspeed_bias_mps=airspeed_bias,
            aileron_effectiveness=aileron_eff,
            active_labels=tuple(labels),
        )
