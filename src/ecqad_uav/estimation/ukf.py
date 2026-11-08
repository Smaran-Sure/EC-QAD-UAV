from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.stats import chi2

from ecqad_uav.sensors.sensor_suite import SensorPacket
from ecqad_uav.utils.math_utils import (
    euler_rates_from_body_rates,
    nearest_spd,
    rot_body_to_ned,
    wrap_pi,
)
from ecqad_uav.utils.units import DEG_TO_RAD, G

# State: [pn, pe, pd, vn, ve, vd, roll, pitch, yaw, bgx, bgy, bgz, bax, bay, baz]
POS = slice(0, 3)
VEL = slice(3, 6)
ATT = slice(6, 9)
BG = slice(9, 12)
BA = slice(12, 15)
ANGLE_IDXS = (6, 7, 8)


@dataclass
class InnovationRecord:
    sensor: str
    t: float
    innovation: np.ndarray
    S: np.ndarray
    nis: float
    dof: int
    threshold95: float


class NavigationUKF:
    """Small self-contained UKF for the conventional EC-QAD baseline.

    This is intentionally transparent rather than highly optimized. It estimates
    NED position/velocity, Euler attitude, gyro bias and accelerometer bias.
    It exposes P, innovations, innovation covariance S, and NIS for later EC-QAD.
    """

    n = 15

    def __init__(self, cfg: dict, sensor_cfg: dict):
        self.cfg = cfg["ukf"]
        self.sensor_cfg = sensor_cfg
        self.alpha = float(self.cfg["alpha"])
        self.beta = float(self.cfg["beta"])
        self.kappa = float(self.cfg["kappa"])
        self.lmbda = self.alpha**2 * (self.n + self.kappa) - self.n
        c = self.n + self.lmbda
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2.0 * c))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lmbda / c
        self.Wc[0] = self.lmbda / c + (1.0 - self.alpha**2 + self.beta)

        self.x = np.zeros(self.n)
        self.P = self._initial_covariance(cfg["initial_std"])
        self.Q_per_s = self._process_covariance(cfg["process_std"])
        self.initialized = False
        self.last_predict_t: float | None = None
        self.innovations: list[InnovationRecord] = []
        self.last_imu_gyro = np.zeros(3)
        self.last_imu_accel = np.array([0.0, 0.0, -G])

        noise = sensor_cfg["noise_std"]
        self.R_gps = np.diag(
            np.r_[np.asarray(noise["gps_position_m"], float) ** 2,
                  np.asarray(noise["gps_velocity_mps"], float) ** 2]
        )
        self.R_heading = np.array([[(float(noise["heading_deg"]) * DEG_TO_RAD) ** 2]])
        self.R_airspeed = np.array([[float(noise["airspeed_mps"]) ** 2]])
        self.nis_alpha = float(cfg.get("nis_alpha", 0.95))
        self.thresholds = {m: float(chi2.ppf(self.nis_alpha, m)) for m in (1, 6)}

    @staticmethod
    def _initial_covariance(std: dict) -> np.ndarray:
        s = np.r_[
            np.full(3, float(std["position_m"])),
            np.full(3, float(std["velocity_mps"])),
            np.full(3, float(std["attitude_deg"]) * DEG_TO_RAD),
            np.full(3, float(std["gyro_bias_deg_s"]) * DEG_TO_RAD),
            np.full(3, float(std["accel_bias_mps2"])),
        ]
        return np.diag(s**2)

    @staticmethod
    def _process_covariance(std: dict) -> np.ndarray:
        s = np.r_[
            np.full(3, float(std["position_m_sqrt_s"])),
            np.full(3, float(std["velocity_mps_sqrt_s"])),
            np.full(3, float(std["attitude_deg_sqrt_s"]) * DEG_TO_RAD),
            np.full(3, float(std["gyro_bias_deg_s_sqrt_s"]) * DEG_TO_RAD),
            np.full(3, float(std["accel_bias_mps2_sqrt_s"])),
        ]
        return np.diag(s**2)

    def initialize(self, packet: SensorPacket) -> bool:
        if packet.gps is None or packet.heading_rad is None:
            return False
        self.x[:] = 0.0
        self.x[POS] = packet.gps.position_ned_m
        self.x[VEL] = packet.gps.velocity_ned_mps
        # Baseline starts close to wings-level trim; roll/pitch are later corrected by IMU dynamics.
        self.x[6] = 0.0
        self.x[7] = 0.0
        self.x[8] = packet.heading_rad
        self.initialized = True
        self.last_predict_t = packet.t
        return True

    def _state_residual(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        y = np.asarray(a) - np.asarray(b)
        for i in ANGLE_IDXS:
            y[i] = wrap_pi(y[i])
        return y

    def _state_add(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        out = np.asarray(x) + np.asarray(dx)
        for i in ANGLE_IDXS:
            out[i] = wrap_pi(out[i])
        return out

    def _state_mean(self, sigmas: np.ndarray) -> np.ndarray:
        mean = np.dot(self.Wm, sigmas)
        for i in ANGLE_IDXS:
            s = np.dot(self.Wm, np.sin(sigmas[:, i]))
            c = np.dot(self.Wm, np.cos(sigmas[:, i]))
            mean[i] = np.arctan2(s, c)
        return mean

    def _sigma_points(self) -> np.ndarray:
        P = nearest_spd(self.P)
        scale = self.n + self.lmbda
        jitter = 1e-10
        for _ in range(8):
            try:
                U = np.linalg.cholesky(scale * (P + jitter * np.eye(self.n)))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            raise np.linalg.LinAlgError("Could not factor UKF covariance.")
        sigmas = np.vstack((self.x, self.x + U.T, self.x - U.T))
        sigmas[:, 6:9] = wrap_pi(sigmas[:, 6:9])
        return sigmas

    @staticmethod
    def _fx_batch(x, dt, gyro, accel):
        """Same strapdown equations as _fx, vectorized over sigma points."""
        y = x.copy()
        phi, theta, psi = x[:, 6:9].T
        sp, cp = np.sin(phi), np.cos(phi)
        st, ct = np.sin(theta), np.cos(theta)
        ss, cs = np.sin(psi), np.cos(psi)
        ct = np.where(ct >= 0, 1., -1.) * np.maximum(abs(ct), 1e-6)
        p, q, r = (gyro - x[:, BG]).T
        rates = np.column_stack((p + (q*sp+r*cp)*st/ct,
                                 q*cp-r*sp, (q*sp+r*cp)/ct))
        fx, fy, fz = (accel - x[:, BA]).T
        acc = np.column_stack((ct*cs*fx+(sp*st*cs-cp*ss)*fy+(cp*st*cs+sp*ss)*fz,
                               ct*ss*fx+(sp*st*ss+cp*cs)*fy+(cp*st*ss-sp*cs)*fz,
                               -st*fx+sp*ct*fy+cp*ct*fz+G))
        y[:, POS] += x[:, VEL]*dt + .5*acc*dt*dt
        y[:, VEL] += acc*dt
        y[:, ATT] = wrap_pi(x[:, ATT] + rates*dt)
        return y

    @staticmethod
    def _fx(x: np.ndarray, dt: float, gyro: np.ndarray, accel: np.ndarray) -> np.ndarray:
        y = x.copy()
        phi, theta, psi = x[ATT]
        omega = np.asarray(gyro) - x[BG]
        f_b = np.asarray(accel) - x[BA]

        eul_dot = euler_rates_from_body_rates(phi, theta, *omega)
        R_bn = rot_body_to_ned(phi, theta, psi)
        a_ned = R_bn @ f_b + np.array([0.0, 0.0, G])

        y[POS] = x[POS] + x[VEL] * dt + 0.5 * a_ned * dt * dt
        y[VEL] = x[VEL] + a_ned * dt
        y[ATT] = x[ATT] + eul_dot * dt
        y[ATT] = wrap_pi(y[ATT])
        # Bias states are random walks represented by Q.
        return y

    def predict(self, t: float, gyro: np.ndarray, accel: np.ndarray) -> None:
        if not self.initialized:
            return
        if self.last_predict_t is None:
            self.last_predict_t = t
            return
        dt = max(float(t - self.last_predict_t), 1e-4)
        self.last_predict_t = t
        self.last_imu_gyro = np.asarray(gyro, float)
        self.last_imu_accel = np.asarray(accel, float)

        sigmas = self._sigma_points()
        prop = self._fx_batch(sigmas, dt, gyro, accel)
        xbar = self._state_mean(prop)
        dx = prop - xbar
        dx[:, ATT] = wrap_pi(dx[:, ATT])
        Pbar = dx.T @ (self.Wc[:, None] * dx)
        self.x = xbar
        self.P = nearest_spd(Pbar + self.Q_per_s * dt)

    def _update(
        self,
        sensor: str,
        t: float,
        z: np.ndarray,
        hx: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray,
        angle_measurement_indices: tuple[int, ...] = (),
    ) -> InnovationRecord:
        z = np.atleast_1d(np.asarray(z, float))
        sigmas = self._sigma_points()
        if sensor == "gps":
            Z = sigmas[:, :6]
        elif sensor == "heading":
            Z = sigmas[:, 8:9]
        elif sensor == "airspeed":
            Z = np.linalg.norm(sigmas[:, VEL], axis=1)[:, None]
        else:
            Z = np.asarray([np.atleast_1d(hx(s)) for s in sigmas])
        m = z.size

        zbar = np.dot(self.Wm, Z)
        for j in angle_measurement_indices:
            sj = np.dot(self.Wm, np.sin(Z[:, j]))
            cj = np.dot(self.Wm, np.cos(Z[:, j]))
            zbar[j] = np.arctan2(sj, cj)

        dz = Z - zbar
        for j in angle_measurement_indices:
            dz[:, j] = wrap_pi(dz[:, j])
        dx = sigmas - self.x
        dx[:, ATT] = wrap_pi(dx[:, ATT])
        S = dz.T @ (self.Wc[:, None] * dz)
        Pxz = dx.T @ (self.Wc[:, None] * dz)
        S = nearest_spd(S + R)
        innovation = z - zbar
        for j in angle_measurement_indices:
            innovation[j] = wrap_pi(innovation[j])

        Sinv = np.linalg.inv(S)
        K = Pxz @ Sinv
        self.x = self._state_add(self.x, K @ innovation)
        self.P = nearest_spd(self.P - K @ S @ K.T)
        nis = float(innovation.T @ Sinv @ innovation)
        rec = InnovationRecord(
            sensor=sensor,
            t=float(t),
            innovation=innovation.copy(),
            S=S.copy(),
            nis=nis,
            dof=m,
            threshold95=self.thresholds[m],
        )
        self.innovations.append(rec)
        return rec

    def process(self, packet: SensorPacket) -> list[InnovationRecord]:
        if not self.initialized:
            self.initialize(packet)
            if not self.initialized:
                return []

        records: list[InnovationRecord] = []
        if packet.imu is not None:
            self.predict(packet.t, packet.imu.gyro_rad_s, packet.imu.accel_mps2)

        if packet.gps is not None:
            z = np.r_[packet.gps.position_ned_m, packet.gps.velocity_ned_mps]
            records.append(self._update(
                "gps", packet.t, z,
                lambda x: np.r_[x[POS], x[VEL]],
                self.R_gps,
            ))

        if packet.heading_rad is not None:
            records.append(self._update(
                "heading", packet.t, np.array([packet.heading_rad]),
                lambda x: np.array([x[8]]),
                self.R_heading,
                angle_measurement_indices=(0,),
            ))

        if packet.airspeed_mps is not None:
            # Baseline assumes still air. If wind is introduced later, add wind states
            # and a proper pitot/air-relative measurement model before publication.
            records.append(self._update(
                "airspeed", packet.t, np.array([packet.airspeed_mps]),
                lambda x: np.array([np.linalg.norm(x[VEL])]),
                self.R_airspeed,
            ))
        return records

    @property
    def covariance_trace(self) -> float:
        return float(np.trace(self.P))

    def state_dict(self) -> dict:
        names = [
            "est_pn_m", "est_pe_m", "est_pd_m",
            "est_vn_mps", "est_ve_mps", "est_vd_mps",
            "est_roll_rad", "est_pitch_rad", "est_yaw_rad",
            "est_bgx_rad_s", "est_bgy_rad_s", "est_bgz_rad_s",
            "est_bax_mps2", "est_bay_mps2", "est_baz_mps2",
        ]
        out = dict(zip(names, self.x.tolist()))
        for i in range(self.n):
            out[f"P_{i:02d}_{i:02d}"] = float(self.P[i, i])
        out["P_trace"] = self.covariance_trace
        return out
