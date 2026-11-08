from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ecqad_uav.plant.jsbsim_plant import TruthState
from ecqad_uav.faults.fault_injector import FaultState
from ecqad_uav.utils.math_utils import rot_body_to_ned, wrap_pi
from ecqad_uav.utils.units import DEG_TO_RAD, G


@dataclass
class IMUMeasurement:
    gyro_rad_s: np.ndarray
    accel_mps2: np.ndarray


@dataclass
class GPSMeasurement:
    position_ned_m: np.ndarray
    velocity_ned_mps: np.ndarray


@dataclass
class SensorPacket:
    t: float
    imu: IMUMeasurement | None = None
    gps: GPSMeasurement | None = None
    heading_rad: float | None = None
    airspeed_mps: float | None = None


class SensorSuite:
    """Asynchronous simulated sensors driven from JSBSim truth.

    Accelerometer is synthesized as specific force from finite-difference NED
    velocity and attitude. For final publication, replace development noise
    values with a cited sensor grade/datasheet and document the model.
    """

    def __init__(self, cfg: dict, plant_hz: float):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.get("seed", 7)))
        self.plant_hz = float(plant_hz)
        self.dt = 1.0 / self.plant_hz
        self.rates = cfg["rates_hz"]
        self.period_steps = {
            k: max(1, int(round(self.plant_hz / float(v)))) for k, v in self.rates.items()
        }
        n = cfg["noise_std"]
        b = cfg["initial_bias_std"]
        self.gyro_noise = float(n["gyro_deg_s"]) * DEG_TO_RAD
        self.accel_noise = float(n["accel_mps2"])
        self.gps_pos_noise = np.asarray(n["gps_position_m"], dtype=float)
        self.gps_vel_noise = np.asarray(n["gps_velocity_mps"], dtype=float)
        self.heading_noise = float(n["heading_deg"]) * DEG_TO_RAD
        self.airspeed_noise = float(n["airspeed_mps"])
        self.gyro_bias0 = self.rng.normal(0.0, float(b["gyro_deg_s"]) * DEG_TO_RAD, 3)
        self.accel_bias0 = self.rng.normal(0.0, float(b["accel_mps2"]), 3)
        self.airspeed_bias0 = float(self.rng.normal(0.0, float(b["airspeed_mps"])))
        self.prev_truth: TruthState | None = None
        self.step_count = 0

    def _specific_force_body(self, truth: TruthState) -> np.ndarray:
        v = np.array([truth.vn_mps, truth.ve_mps, truth.vd_mps])
        if self.prev_truth is None:
            a_ned = np.zeros(3)
        else:
            vp = np.array([self.prev_truth.vn_mps, self.prev_truth.ve_mps, self.prev_truth.vd_mps])
            dt = max(truth.t - self.prev_truth.t, self.dt)
            a_ned = (v - vp) / dt
        g_ned = np.array([0.0, 0.0, G])
        specific_ned = a_ned - g_ned
        R_bn = rot_body_to_ned(truth.roll_rad, truth.pitch_rad, truth.yaw_rad)
        return R_bn.T @ specific_ned

    def sample(self, truth: TruthState, fault: FaultState) -> SensorPacket:
        packet = SensorPacket(t=truth.t)
        f_body = self._specific_force_body(truth)

        if self.step_count % self.period_steps["imu"] == 0:
            gyro_truth = np.array([truth.p_rad_s, truth.q_rad_s, truth.r_rad_s])
            gyro = (
                gyro_truth
                + self.gyro_bias0
                + np.asarray(fault.gyro_bias_xyz_rad_s)
                + self.rng.normal(0.0, self.gyro_noise, 3)
            )
            accel = f_body + self.accel_bias0 + self.rng.normal(0.0, self.accel_noise, 3)
            packet.imu = IMUMeasurement(gyro_rad_s=gyro, accel_mps2=accel)

        if self.step_count % self.period_steps["gps"] == 0 and fault.gps_available:
            pos = np.array([truth.pn_m, truth.pe_m, truth.pd_m])
            vel = np.array([truth.vn_mps, truth.ve_mps, truth.vd_mps])
            packet.gps = GPSMeasurement(
                position_ned_m=pos + self.rng.normal(0.0, self.gps_pos_noise),
                velocity_ned_mps=vel + self.rng.normal(0.0, self.gps_vel_noise),
            )

        if self.step_count % self.period_steps["magnetometer"] == 0:
            packet.heading_rad = float(wrap_pi(truth.yaw_rad + self.rng.normal(0.0, self.heading_noise)))

        if self.step_count % self.period_steps["airspeed"] == 0:
            packet.airspeed_mps = float(
                truth.airspeed_mps
                + self.airspeed_bias0
                + fault.airspeed_bias_mps
                + self.rng.normal(0.0, self.airspeed_noise)
            )

        self.prev_truth = truth
        self.step_count += 1
        return packet
