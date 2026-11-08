from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import math
import numpy as np

from ecqad_uav.control.autopilot import CascadedAutopilot, NavState, standard_reference_profile
from ecqad_uav.estimation.ukf import NavigationUKF, BG
from ecqad_uav.faults.fault_injector import FaultInjector
from ecqad_uav.plant.jsbsim_plant import JSBSimPlant, ControlCommand
from ecqad_uav.sensors.sensor_suite import SensorSuite
from ecqad_uav.utils.logger import RunLogger


class SimulationRunner:
    def __init__(
        self,
        aircraft_cfg: dict,
        sensor_cfg: dict,
        estimator_cfg: dict,
        controller_cfg: dict,
        fault_cfg: dict,
        *,
        control_mode: str = "open_loop",
        estimator_enabled: bool = False,
        sensor_enabled: bool = True,
        duration_s: float | None = None,
    ):
        if control_mode not in {"trim_hold", "open_loop", "truth_pid", "estimated_pid"}:
            raise ValueError(f"Unsupported control_mode={control_mode}")
        self.aircraft_cfg = aircraft_cfg
        self.sensor_cfg = sensor_cfg
        self.estimator_cfg = estimator_cfg
        self.controller_cfg = controller_cfg
        self.fault_cfg = fault_cfg
        self.control_mode = control_mode
        self.estimator_enabled = estimator_enabled or control_mode == "estimated_pid"
        self.sensor_enabled = sensor_enabled or self.estimator_enabled
        self.duration_s = float(duration_s or aircraft_cfg["simulation"]["duration_s"])

    @staticmethod
    def _open_loop_profile(t: float, trim: ControlCommand) -> ControlCommand:
        # Small pulses for sign/response validation; hold trim otherwise.
        cmd = ControlCommand(**trim.__dict__)
        if 10.0 <= t < 11.0:
            cmd.aileron += 0.15
        elif 20.0 <= t < 21.0:
            cmd.elevator += 0.10
        elif 30.0 <= t < 33.0:
            cmd.throttle += 0.10
        return cmd.clipped()

    def run(self) -> tuple[RunLogger, dict]:
        plant = JSBSimPlant(self.aircraft_cfg)
        init_info = plant.initialize()
        truth = plant.read_truth()
        trim = ControlCommand(**init_info["trim_command"])
        altitude0 = truth.altitude_m
        heading0 = truth.yaw_rad
        airspeed0 = truth.airspeed_mps

        faults = FaultInjector(self.fault_cfg)
        sensors = SensorSuite(self.sensor_cfg, self.aircraft_cfg["simulation"]["plant_hz"])
        estimator = NavigationUKF(self.estimator_cfg, self.sensor_cfg) if self.estimator_enabled else None
        autopilot = CascadedAutopilot(self.controller_cfg, trim)
        logger = RunLogger()

        plant_hz = float(self.aircraft_cfg["simulation"]["plant_hz"])
        control_hz = float(self.controller_cfg["controller_hz"])
        control_period = max(1, int(round(plant_hz / control_hz)))
        n_steps = int(round(self.duration_s * plant_hz))
        current_cmd = trim
        latest_airspeed = airspeed0
        latest_debug: dict = {}

        for k in range(n_steps + 1):
            t = truth.t
            fstate = faults.state(t)
            packet = sensors.sample(truth, fstate) if self.sensor_enabled else None

            if packet is not None and packet.airspeed_mps is not None:
                latest_airspeed = packet.airspeed_mps

            nis_step = {"gps": np.nan, "heading": np.nan, "airspeed": np.nan}
            nis_threshold_step = {"gps": np.nan, "heading": np.nan, "airspeed": np.nan}
            if estimator is not None and packet is not None:
                recs = estimator.process(packet)
                for rec in recs:
                    nis_step[rec.sensor] = rec.nis
                    nis_threshold_step[rec.sensor] = rec.threshold95

            if k % control_period == 0:
                if self.control_mode == "trim_hold":
                    current_cmd = trim
                    latest_debug = {}
                elif self.control_mode == "open_loop":
                    current_cmd = self._open_loop_profile(t, trim)
                    latest_debug = {}
                else:
                    ref = standard_reference_profile(t, altitude0, heading0, airspeed0)
                    if self.control_mode == "truth_pid":
                        nav = NavState.from_truth(truth)
                    else:
                        if estimator is None or not estimator.initialized:
                            nav = NavState.from_truth(truth)  # only startup fallback before first GPS+heading
                        else:
                            nav = NavState.from_estimator(estimator.x, latest_airspeed, altitude0)
                            rates = estimator.last_imu_gyro - estimator.x[BG]
                            nav.p_rad_s, nav.q_rad_s, nav.r_rad_s = map(float, rates)
                    current_cmd, latest_debug = autopilot.update(nav, ref, 1.0 / control_hz)

            row = {
                **{f"truth_{k0}": v for k0, v in truth.as_dict().items() if k0 != "t"},
                "time_s": t,
                "u_aileron": current_cmd.aileron,
                "u_elevator": current_cmd.elevator,
                "u_rudder": current_cmd.rudder,
                "u_throttle": current_cmd.throttle,
                "actual_aileron_effectiveness": fstate.aileron_effectiveness,
                "gps_available": int(fstate.gps_available),
                "fault_gyro_bias_x_rad_s": fstate.gyro_bias_xyz_rad_s[0],
                "fault_gyro_bias_y_rad_s": fstate.gyro_bias_xyz_rad_s[1],
                "fault_gyro_bias_z_rad_s": fstate.gyro_bias_xyz_rad_s[2],
                "fault_airspeed_bias_mps": fstate.airspeed_bias_mps,
                "fault_labels": ";".join(fstate.active_labels),
                "nis_gps": nis_step["gps"],
                "nis_heading": nis_step["heading"],
                "nis_airspeed": nis_step["airspeed"],
                "nis95_gps": nis_threshold_step["gps"],
                "nis95_heading": nis_threshold_step["heading"],
                "nis95_airspeed": nis_threshold_step["airspeed"],
                **latest_debug,
            }

            if packet is not None:
                if packet.imu is not None:
                    for i, a in enumerate("xyz"):
                        row[f"meas_gyro_{a}_rad_s"] = float(packet.imu.gyro_rad_s[i])
                        row[f"meas_accel_{a}_mps2"] = float(packet.imu.accel_mps2[i])
                if packet.gps is not None:
                    for i, a in enumerate("ned"):
                        row[f"meas_gps_p{a}_m"] = float(packet.gps.position_ned_m[i])
                        row[f"meas_gps_v{a}_mps"] = float(packet.gps.velocity_ned_mps[i])
                if packet.heading_rad is not None:
                    row["meas_heading_rad"] = packet.heading_rad
                if packet.airspeed_mps is not None:
                    row["meas_airspeed_mps"] = packet.airspeed_mps

            if estimator is not None and estimator.initialized:
                row.update(estimator.state_dict())

            logger.append(row)
            if k == n_steps:
                break
            truth = plant.step(current_cmd, aileron_effectiveness=fstate.aileron_effectiveness)

        metadata = {
            "control_mode": self.control_mode,
            "estimator_enabled": self.estimator_enabled,
            "duration_s": self.duration_s,
            "plant_hz": plant_hz,
            "aircraft_model": plant.model,
            "initialization": init_info,
            "trim": asdict(trim),
        }
        return logger, metadata
