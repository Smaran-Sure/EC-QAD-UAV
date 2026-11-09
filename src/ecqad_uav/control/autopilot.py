from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from ecqad_uav.control.pid import PID
from ecqad_uav.plant.jsbsim_plant import ControlCommand, TruthState
from ecqad_uav.utils.math_utils import angle_error


@dataclass
class GuidanceCommand:
    altitude_m: float
    heading_rad: float
    airspeed_mps: float


@dataclass
class NavState:
    altitude_m: float
    airspeed_mps: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    p_rad_s: float
    q_rad_s: float
    r_rad_s: float

    @classmethod
    def from_truth(cls, x: TruthState) -> "NavState":
        return cls(
            altitude_m=x.altitude_m,
            airspeed_mps=x.airspeed_mps,
            roll_rad=x.roll_rad,
            pitch_rad=x.pitch_rad,
            yaw_rad=x.yaw_rad,
            p_rad_s=x.p_rad_s,
            q_rad_s=x.q_rad_s,
            r_rad_s=x.r_rad_s,
        )

    @classmethod
    def from_estimator(cls, xhat: np.ndarray, airspeed_mps: float, altitude0_m: float) -> "NavState":
        return cls(
            altitude_m=altitude0_m - float(xhat[2]),  # pd positive down
            airspeed_mps=float(airspeed_mps),
            roll_rad=float(xhat[6]),
            pitch_rad=float(xhat[7]),
            yaw_rad=float(xhat[8]),
            # body-rate states are not explicitly estimated; bias-corrected latest gyro is supplied separately later
            p_rad_s=0.0,
            q_rad_s=0.0,
            r_rad_s=0.0,
        )


POLICY_INPUT_SCHEMA = "physical_reference_v2"
POLICY_FEATURE_NAMES = (
    "altitude_error", "heading_error", "airspeed_error", "roll", "pitch",
    "body_p", "body_q", "body_r", "altitude", "airspeed",
    "sin_heading", "cos_heading", "reference_altitude", "reference_airspeed",
    "sin_reference_heading", "cos_reference_heading",
)


def policy_features(nav: NavState, ref: GuidanceCommand) -> np.ndarray:
    """Memoryless deployed interface: physical navigation and commands only."""
    return np.asarray([
        (ref.altitude_m-nav.altitude_m)/30,
        angle_error(ref.heading_rad, nav.yaw_rad)/.5,
        (ref.airspeed_mps-nav.airspeed_mps)/5,
        nav.roll_rad/.5, nav.pitch_rad/.2,
        nav.p_rad_s, nav.q_rad_s, nav.r_rad_s/.5,
        nav.altitude_m/300, nav.airspeed_mps/30,
        math.sin(nav.yaw_rad), math.cos(nav.yaw_rad),
        ref.altitude_m/300, ref.airspeed_mps/30,
        math.sin(ref.heading_rad), math.cos(ref.heading_rad),
    ], dtype=np.float32)


class CascadedAutopilot:
    def __init__(self, cfg: dict, trim: ControlCommand):
        self.cfg = cfg
        p = cfg["pid"]
        lim = cfg["limits"]
        self.max_roll = math.radians(float(lim["max_roll_cmd_deg"]))
        self.max_pitch = math.radians(float(lim["max_pitch_cmd_deg"]))
        self.trim = trim
        self.roll = PID(**p["roll"], output_min=-1.0, output_max=1.0, integral_min=-0.5, integral_max=0.5)
        self.pitch = PID(**p["pitch"], output_min=-1.0, output_max=1.0, integral_min=-0.5, integral_max=0.5)
        self.heading = PID(**p["heading"], output_min=-self.max_roll, output_max=self.max_roll,
                           integral_min=-math.radians(20), integral_max=math.radians(20))
        self.altitude = PID(**p["altitude"], output_min=-self.max_pitch, output_max=self.max_pitch,
                            integral_min=-100.0, integral_max=100.0)
        self.airspeed = PID(**p["airspeed"], output_min=-0.5, output_max=0.5,
                            integral_min=-20.0, integral_max=20.0)
        self.kr = float(p["yaw_damper"]["kr"])

    def reset(self) -> None:
        for pid in (self.roll, self.pitch, self.heading, self.altitude, self.airspeed):
            pid.reset()

    def update(self, nav: NavState, ref: GuidanceCommand, dt: float) -> tuple[ControlCommand, dict]:
        h_err = ref.altitude_m - nav.altitude_m
        psi_err = angle_error(ref.heading_rad, nav.yaw_rad)
        v_err = ref.airspeed_mps - nav.airspeed_mps

        phi_cmd = self.heading.update(psi_err, dt)
        theta_cmd = self.altitude.update(h_err, dt)

        phi_err = angle_error(phi_cmd, nav.roll_rad)
        theta_err = angle_error(theta_cmd, nav.pitch_rad)

        # Rate feedback is supplied as derivative measurement, so sign is negative damping.
        ail_delta = self.roll.update(phi_err, dt, derivative_measurement=-nav.p_rad_s)
        ele_delta = self.pitch.update(theta_err, dt, derivative_measurement=-nav.q_rad_s)
        throttle_delta = self.airspeed.update(v_err, dt)
        rudder_delta = self.kr * nav.r_rad_s

        cmd = ControlCommand(
            aileron=self.trim.aileron + ail_delta,
            elevator=self.trim.elevator + ele_delta,
            rudder=self.trim.rudder + rudder_delta,
            throttle=self.trim.throttle + throttle_delta,
        ).clipped()

        dbg = {
            "cmd_altitude_m": ref.altitude_m,
            "cmd_heading_rad": ref.heading_rad,
            "cmd_airspeed_mps": ref.airspeed_mps,
            "cmd_roll_rad": phi_cmd,
            "cmd_pitch_rad": theta_cmd,
            "err_altitude_m": h_err,
            "err_heading_rad": psi_err,
            "err_airspeed_mps": v_err,
        }
        return cmd, dbg


def standard_reference_profile(t: float, altitude0_m: float, heading0_rad: float, airspeed0_mps: float) -> GuidanceCommand:
    """Nominal 180-s benchmark used for all later controller comparisons."""
    h = altitude0_m
    psi = heading0_rad
    v = airspeed0_mps
    if 30.0 <= t < 60.0:
        psi += math.radians(20.0)
    elif 60.0 <= t < 90.0:
        psi += math.radians(20.0)
        h += 20.0
    elif 90.0 <= t < 120.0:
        psi += math.radians(20.0)
        h += 20.0
        v += 3.0
    elif 120.0 <= t < 150.0:
        psi += math.radians(-15.0)
        h += 30.0
        v += 2.0
    elif t >= 150.0:
        pass
    return GuidanceCommand(altitude_m=h, heading_rad=psi, airspeed_mps=v)
