from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ecqad_uav.plant.jsbsim_plant import (
    JSBSimPlant,
    ControlCommand,
)
from ecqad_uav.utils.config import load_yaml
from ecqad_uav.utils.units import M_TO_FT, MPS_TO_KNOT


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

TARGET_SPEED_MPS = 25.0
TARGET_ALTITUDE_M = 300.0

# Let engine / FCS settle first, then evaluate drift.
SETTLE_TIME_S = 0.75
EVAL_TIME_S = 0.75


def build_candidate(alpha_deg, elevator, throttle):
    cfg = load_yaml("aircraft.yaml")

    plant = JSBSimPlant(cfg)
    fdm = plant.fdm

    # Desired flight condition
    fdm["ic/h-sl-ft"] = TARGET_ALTITUDE_M * M_TO_FT
    fdm["ic/vc-kts"] = TARGET_SPEED_MPS * MPS_TO_KNOT

    # Straight flight
    fdm["ic/psi-true-deg"] = 0.0
    fdm["ic/gamma-deg"] = 0.0
    fdm["ic/alpha-deg"] = float(alpha_deg)
    fdm["ic/beta-deg"] = 0.0
    fdm["ic/phi-deg"] = 0.0

    # Zero initial body rates
    try:
        fdm["ic/p-rad_sec"] = 0.0
        fdm["ic/q-rad_sec"] = 0.0
        fdm["ic/r-rad_sec"] = 0.0
    except Exception:
        pass

    fdm.run_ic()

    # Start engine AFTER applying initial conditions.
    fdm["propulsion/set-running"] = -1

    cmd = ControlCommand(
        aileron=0.0,
        elevator=float(elevator),
        rudder=0.0,
        throttle=float(throttle),
    )

    plant.set_controls(cmd)

    return plant, cmd


def evaluate_candidate(x, verbose=False):
    """
    x = [alpha_deg, elevator_norm, throttle_norm]

    Trim objective:
        airspeed drift -> 0
        altitude drift -> 0
        pitch rate     -> 0
    """

    alpha_deg, elevator, throttle = x

    try:
        plant, cmd = build_candidate(
            alpha_deg,
            elevator,
            throttle,
        )

        # ---------------------------------
        # Settling phase
        # ---------------------------------
        settle_steps = max(
            1,
            int(SETTLE_TIME_S / plant.dt),
        )

        for _ in range(settle_steps):
            plant.step(cmd)

        start = plant.read_truth()

        # ---------------------------------
        # Evaluation phase
        # ---------------------------------
        eval_steps = max(
            1,
            int(EVAL_TIME_S / plant.dt),
        )

        for _ in range(eval_steps):
            plant.step(cmd)

        end = plant.read_truth()

        dt_eval = max(
            end.t - start.t,
            plant.dt,
        )

        # Average drift rates over evaluation window.
        speed_drift = (
            end.airspeed_mps - start.airspeed_mps
        ) / dt_eval

        altitude_drift = (
            end.altitude_m - start.altitude_m
        ) / dt_eval

        q_final = end.q_rad_s

        residual = np.array(
            [
                speed_drift,
                altitude_drift,
                5.0 * q_final,
            ],
            dtype=float,
        )

        if verbose:
            fdm = plant.fdm

            rpm = plant._get(
                "propulsion/engine[0]/engine-rpm",
                "propulsion/engine[0]/rpm",
                default=np.nan,
            )

            thrust = plant._get(
                "propulsion/engine[0]/thrust-lbs",
                default=np.nan,
            )

            alpha_actual = plant._get(
                "aero/alpha-deg",
                default=np.nan,
            )

            print()
            print("Candidate")
            print("------------------------------")
            print(f"alpha command : {alpha_deg:.4f} deg")
            print(f"alpha actual  : {alpha_actual:.4f} deg")
            print(f"elevator      : {elevator:.6f}")
            print(f"throttle      : {throttle:.6f}")
            print(f"RPM           : {rpm:.2f}")
            print(f"thrust        : {thrust:.4f} lb")

            print()
            print("Evaluation-window drift")
            print("------------------------------")
            print(
                f"airspeed rate : "
                f"{speed_drift:.8f} m/s^2"
            )
            print(
                f"altitude rate : "
                f"{altitude_drift:.8f} m/s"
            )
            print(
                f"pitch rate q  : "
                f"{q_final:.8f} rad/s"
            )

            print()
            print("Final state")
            print("------------------------------")
            print(
                f"airspeed      : "
                f"{end.airspeed_mps:.4f} m/s"
            )
            print(
                f"altitude      : "
                f"{end.altitude_m:.4f} m"
            )
            print(
                f"pitch         : "
                f"{np.degrees(end.pitch_rad):.4f} deg"
            )

        if not np.all(np.isfinite(residual)):
            return np.array(
                [100.0, 100.0, 100.0],
                dtype=float,
            )

        return residual

    except Exception as exc:
        if verbose:
            print("Candidate failed:", repr(exc))

        return np.array(
            [100.0, 100.0, 100.0],
            dtype=float,
        )


def main():
    print()
    print("====================================")
    print(" CUSTOM RASCAL LONGITUDINAL TRIM")
    print("====================================")
    print()

    print(
        f"Target altitude : "
        f"{TARGET_ALTITUDE_M:.1f} m"
    )
    print(
        f"Target airspeed : "
        f"{TARGET_SPEED_MPS:.1f} m/s"
    )

    # Based partly on the propulsion sweep:
    # around 0.7-0.8 throttle appears much more
    # plausible near 25 m/s than 0.5.
    x0 = np.array(
        [
            6.0,    # alpha deg
            0.0,    # elevator
            0.75,   # throttle
        ],
        dtype=float,
    )

    lower = np.array(
        [
            -2.0,
            -1.0,
            0.0,
        ]
    )

    upper = np.array(
        [
            18.0,
            1.0,
            1.0,
        ]
    )

    print()
    print("Initial candidate:")
    evaluate_candidate(
        x0,
        verbose=True,
    )

    print()
    print("Running numerical solver...")
    print()

    result = least_squares(
        evaluate_candidate,
        x0=x0,
        bounds=(lower, upper),
        xtol=1e-7,
        ftol=1e-7,
        gtol=1e-7,
        max_nfev=100,
        diff_step=1e-3,
        verbose=1,
    )

    alpha_deg, elevator, throttle = result.x

    print()
    print("====================================")
    print(" OPTIMIZER RESULT")
    print("====================================")
    print(f"success     : {result.success}")
    print(f"message     : {result.message}")
    print(f"cost        : {result.cost}")
    print(f"evaluations : {result.nfev}")

    residual = evaluate_candidate(
        result.x,
        verbose=True,
    )

    residual_norm = float(
        np.linalg.norm(residual)
    )

    # This is only the numerical-trim gate.
    # A later 60-second free-flight validation
    # is still required before accepting the trim.
    trim_ok = (
        result.success
        and abs(residual[0]) < 0.15
        and abs(residual[1]) < 0.15
        and abs(residual[2] / 5.0) < 0.02
    )

    output = {
        "trim_ok": bool(trim_ok),
        "optimizer_success": bool(
            result.success
        ),
        "optimizer_message": str(
            result.message
        ),
        "target_airspeed_mps": (
            TARGET_SPEED_MPS
        ),
        "target_altitude_m": (
            TARGET_ALTITUDE_M
        ),
        "alpha_deg": float(alpha_deg),
        "elevator": float(elevator),
        "throttle": float(throttle),
        "aileron": 0.0,
        "rudder": 0.0,
        "residuals": residual.tolist(),
        "residual_norm": residual_norm,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
    }

    out_file = (
        RESULTS
        / "rascal_custom_trim.json"
    )

    with out_file.open("w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("====================================")

    if trim_ok:
        print("CUSTOM TRIM: PASS")
    else:
        print(
            "CUSTOM TRIM: "
            "NOT YET ACCEPTABLE"
        )

    print("====================================")
    print()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
