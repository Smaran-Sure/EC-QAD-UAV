from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import os
from pathlib import Path
import warnings
import numpy as np
import jsbsim

from ecqad_uav.utils.config import project_root
from ecqad_uav.utils.units import FT_TO_M, M_TO_FT, KNOT_TO_MPS, MPS_TO_KNOT

EARTH_RADIUS_M = 6378137.0


@dataclass
class TruthState:
    t: float
    pn_m: float
    pe_m: float
    pd_m: float
    vn_mps: float
    ve_mps: float
    vd_mps: float
    u_mps: float
    v_mps: float
    w_mps: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    p_rad_s: float
    q_rad_s: float
    r_rad_s: float
    altitude_m: float
    airspeed_mps: float
    lat_rad: float
    lon_rad: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ControlCommand:
    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.5

    def clipped(self) -> "ControlCommand":
        return ControlCommand(
            aileron=float(np.clip(self.aileron, -1.0, 1.0)),
            elevator=float(np.clip(self.elevator, -1.0, 1.0)),
            rudder=float(np.clip(self.rudder, -1.0, 1.0)),
            throttle=float(np.clip(self.throttle, 0.0, 1.0)),
        )


class JSBSimPlant:
    """Thin, SI-unit wrapper around JSBSim.

    The final paper should record the exact JSBSim version and aircraft-model commit.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        sim_cfg = cfg["simulation"]
        self.dt = 1.0 / float(sim_cfg["plant_hz"])

        try:
            jsbsim.FGJSBBase().debug_lvl = 0
        except Exception:
            pass

        # Custom roots supplement the wheel's data; they must never hide stock models.
        roots = []
        if os.getenv("JSBSIM_ROOT"):
            roots.append(Path(os.environ["JSBSIM_ROOT"]).expanduser().resolve())
        roots.extend([project_root() / "models" / "jsbsim",
                      Path(jsbsim.get_default_root_dir()).resolve()])
        self._model_roots = list(dict.fromkeys(roots))

        acfg = cfg["aircraft"]
        preferred = acfg["model"]
        self.model = preferred
        loaded = self._load_model(preferred)
        if not loaded and acfg.get("allow_fallback_for_smoke_test", False):
            fallback = acfg.get("fallback_model", "c172p")
            warnings.warn(
                f"Could not load '{preferred}'. Falling back to '{fallback}' for a smoke test only. "
                "Do not use fallback results in the final UAV paper.", RuntimeWarning
            )
            loaded = self._load_model(fallback)
            self.model = fallback
        if not loaded:
            raise RuntimeError(
                f"Could not load JSBSim model '{preferred}' or an enabled fallback. "
                f"Searched data roots: {self._model_roots}"
            )

        self._lat0 = None
        self._lon0 = None
        self._alt0_m = None
        self.last_command = ControlCommand()

    def _load_model(self, model: str) -> bool:
        for root in self._model_roots:
            if not (root / "aircraft" / model / f"{model}.xml").is_file():
                continue
            # A failed XML load may leave partial state; always start afresh.
            fdm = jsbsim.FGFDMExec(str(root))
            fdm.set_dt(self.dt)
            try:
                loaded = fdm.load_model(model)
            except Exception as exc:
                warnings.warn(f"Could not load '{model}' from {root}: {exc}", RuntimeWarning)
                continue
            if loaded:
                self.fdm = fdm
                self.model_root = str(root)
                return True
        return False




    def initialize(self, trim: bool | None = None) -> dict:
        sim = self.cfg["simulation"]
        initial = sim["initial"]

        # -----------------------------
        # Initial flight condition
        # -----------------------------
        self.fdm["ic/h-sl-ft"] = (
            float(initial["altitude_m"]) * M_TO_FT
        )

        self.fdm["ic/vc-kts"] = (
            float(initial["calibrated_airspeed_mps"]) * MPS_TO_KNOT
        )

        self.fdm["ic/psi-true-deg"] = float(
            initial.get("heading_deg", 0.0)
        )

        # Straight-and-level initial condition.
        self.fdm["ic/gamma-deg"] = float(
            initial.get("flight_path_deg", 0.0)
        )

        self.fdm["ic/beta-deg"] = 0.0
        self.fdm["ic/phi-deg"] = 0.0

        # Start all engines before applying the ICs.
        if sim.get("engine_running", True):
            self.fdm["propulsion/set-running"] = -1

        # Apply initial conditions.
        self.fdm.run_ic()

        # run_ic() can reset propulsion state in some models,
        # so explicitly ensure the engine is running again.
        if sim.get("engine_running", True):
            self.fdm["propulsion/set-running"] = -1

        self.set_controls(ControlCommand(**self.cfg["controls"]["trim"]))

        # -----------------------------
        # Trim
        # -----------------------------
        do_trim = (
            sim.get("trim_before_run", True)
            if trim is None
            else trim
        )

        trim_ok = False
        trim_error = None

        if do_trim:
            try:
                # JSBSim trim modes:
                #
                # 0 = longitudinal
                # 1 = full
                # 2 = ground
                # 3 = pull-up
                # 4 = custom
                # 5 = turn
                #
                # We are establishing a straight-and-level
                # fixed-wing operating point, so longitudinal
                # trim is the appropriate first trim mode.
                self.fdm["simulation/do_simple_trim"] = 0

                trim_ok = True

            except Exception as exc:
                # Handles TrimFailureError across JSBSim versions.
                trim_error = repr(exc)

                warnings.warn(
                    f"JSBSim longitudinal trim failed: {exc}",
                    RuntimeWarning,
                )

        # Ensure propulsion remains active after trimming.
        if sim.get("engine_running", True):
            self.fdm["propulsion/set-running"] = -1

        # -----------------------------
        # Save initialized state
        # -----------------------------
        truth = self.read_truth()

        self._lat0 = truth.lat_rad
        self._lon0 = truth.lon_rad
        self._alt0_m = truth.altitude_m

        trim_cmd = (self.read_control_command() if trim_ok else
                    ControlCommand(**self.cfg["controls"]["trim"]))
        self.last_command = trim_cmd

        return {
            "trim_ok": trim_ok,
            "trim_error": trim_error,
            "model": self.model,
            "model_root": self.model_root,
            "trim_command": trim_cmd.__dict__,
            "initial_truth": truth.as_dict(),
        }

    def _get(self, *names: str, default: float | None = None) -> float:
        for name in names:
            try:
                value = float(self.fdm[name])
                if math.isfinite(value):
                    return value
            except Exception:
                continue
        if default is not None:
            return float(default)
        raise KeyError(f"None of these JSBSim properties was available: {names}")

    def read_control_command(self) -> ControlCommand:
        return ControlCommand(
            aileron=self._get("fcs/aileron-cmd-norm", default=0.0),
            elevator=self._get("fcs/elevator-cmd-norm", default=0.0),
            rudder=self._get("fcs/rudder-cmd-norm", default=0.0),
            throttle=self._get("fcs/throttle-cmd-norm[0]", "fcs/throttle-cmd-norm", default=0.5),
        )

    def set_controls(self, cmd: ControlCommand, aileron_effectiveness: float = 1.0) -> None:
        cmd = cmd.clipped()
        actual = ControlCommand(
            aileron=cmd.aileron * float(aileron_effectiveness),
            elevator=cmd.elevator,
            rudder=cmd.rudder,
            throttle=cmd.throttle,
        ).clipped()
        ramp = self.cfg["simulation"].get("motor_startup_ramp_s", 0.0)
        if ramp > 0:
            # Avoid the documented electric-motor zero-RPM torque singularity.
            t = float(self.fdm["simulation/sim-time-sec"])
            actual.throttle = min(actual.throttle, 1e-5 + max(t, 0.0) / ramp)
        self.fdm["fcs/aileron-cmd-norm"] = actual.aileron
        self.fdm["fcs/elevator-cmd-norm"] = actual.elevator
        self.fdm["fcs/rudder-cmd-norm"] = actual.rudder
        try:
            self.fdm["fcs/throttle-cmd-norm[0]"] = actual.throttle
        except Exception:
            self.fdm["fcs/throttle-cmd-norm"] = actual.throttle
        self.last_command = actual

    def step(self, cmd: ControlCommand | None = None, aileron_effectiveness: float = 1.0) -> TruthState:
        if cmd is not None:
            self.set_controls(cmd, aileron_effectiveness=aileron_effectiveness)
        ok = self.fdm.run()
        if not ok:
            raise RuntimeError("JSBSim terminated before the requested simulation completed.")
        return self.read_truth()

    def read_truth(self) -> TruthState:
        lat = self._get("position/lat-geod-rad", "position/lat-gc-rad")
        lon = self._get("position/long-gc-rad", "position/long-geod-rad")
        alt_m = self._get("position/h-sl-ft", "position/geod-alt-ft") * FT_TO_M
        if self._lat0 is None:
            lat0, lon0, alt0 = lat, lon, alt_m
        else:
            lat0, lon0, alt0 = self._lat0, self._lon0, self._alt0_m

        # Geodetic latitude requires the WGS84 meridional radius.
        e2 = 6.6943799901413165e-3
        denom = 1.0 - e2 * math.sin(lat0)**2
        meridional = EARTH_RADIUS_M * (1.0-e2) / denom**1.5
        prime_vertical = EARTH_RADIUS_M / math.sqrt(denom)
        pn = (lat - lat0) * (meridional + alt0)
        pe = (lon - lon0) * (prime_vertical + alt0) * math.cos(lat0)
        pd = -(alt_m - alt0)

        return TruthState(
            t=self._get("simulation/sim-time-sec", default=0.0),
            pn_m=pn,
            pe_m=pe,
            pd_m=pd,
            vn_mps=self._get("velocities/v-north-fps") * FT_TO_M,
            ve_mps=self._get("velocities/v-east-fps") * FT_TO_M,
            vd_mps=self._get("velocities/v-down-fps") * FT_TO_M,
            u_mps=self._get("velocities/u-fps", "velocities/u-aero-fps") * FT_TO_M,
            v_mps=self._get("velocities/v-fps", "velocities/v-aero-fps") * FT_TO_M,
            w_mps=self._get("velocities/w-fps", "velocities/w-aero-fps") * FT_TO_M,
            roll_rad=self._get("attitude/phi-rad"),
            pitch_rad=self._get("attitude/theta-rad"),
            yaw_rad=self._get("attitude/psi-rad"),
            p_rad_s=self._get("velocities/p-rad_sec", "velocities/p-aero-rad_sec"),
            q_rad_s=self._get("velocities/q-rad_sec", "velocities/q-aero-rad_sec"),
            r_rad_s=self._get("velocities/r-rad_sec", "velocities/r-aero-rad_sec"),
            altitude_m=alt_m,
            airspeed_mps=self._get("velocities/vtrue-kts", "velocities/vc-kts") * KNOT_TO_MPS,
            lat_rad=lat,
            lon_rad=lon,
        )
