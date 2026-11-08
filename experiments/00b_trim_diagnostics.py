from __future__ import annotations

import copy
import pandas as pd

from ecqad_uav.plant.jsbsim_plant import (
    JSBSimPlant,
    ControlCommand,
)
from ecqad_uav.utils.config import load_yaml


def trim_sweep():
    print("\n=== TRIM ENVELOPE SWEEP ===\n")

    base_cfg = load_yaml("aircraft.yaml")

    speeds_mps = [
        16.0,
        18.0,
        20.0,
        22.0,
        24.0,
        25.0,
        26.0,
        28.0,
        30.0,
        32.0,
        35.0,
    ]

    rows = []

    for speed in speeds_mps:
        cfg = copy.deepcopy(base_cfg)

        cfg["simulation"]["initial"][
            "calibrated_airspeed_mps"
        ] = speed

        try:
            plant = JSBSimPlant(cfg)

            info = plant.initialize(trim=True)

            truth = info["initial_truth"]
            cmd = info["trim_command"]

            rows.append(
                {
                    "speed_mps": speed,
                    "trim_ok": info["trim_ok"],
                    "trim_error": info["trim_error"],
                    "throttle": cmd["throttle"],
                    "elevator": cmd["elevator"],
                    "pitch_deg": (
                        truth["pitch_rad"] * 180.0 / 3.141592653589793
                    ),
                    "airspeed_mps": truth["airspeed_mps"],
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "speed_mps": speed,
                    "trim_ok": False,
                    "trim_error": repr(exc),
                    "throttle": None,
                    "elevator": None,
                    "pitch_deg": None,
                    "airspeed_mps": None,
                }
            )

    df = pd.DataFrame(rows)

    print(df.to_string(index=False))

    df.to_csv(
        "results/trim_envelope.csv",
        index=False,
    )

    return df


def propulsion_sweep():
    print("\n=== PROPULSION / THROTTLE SWEEP ===\n")

    base_cfg = load_yaml("aircraft.yaml")

    throttles = [
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
    ]

    rows = []

    for throttle in throttles:

        cfg = copy.deepcopy(base_cfg)

        plant = JSBSimPlant(cfg)

        # Deliberately skip trim.
        plant.initialize(trim=False)

        cmd = ControlCommand(
            aileron=0.0,
            elevator=0.0,
            rudder=0.0,
            throttle=throttle,
        )

        # Allow propulsion state to settle briefly.
        n_steps = int(2.0 / plant.dt)

        for _ in range(n_steps):
            plant.step(cmd)

        throttle_cmd = plant._get(
            "fcs/throttle-cmd-norm[0]",
            "fcs/throttle-cmd-norm",
            default=float("nan"),
        )

        throttle_pos = plant._get(
            "fcs/throttle-pos-norm[0]",
            "fcs/throttle-pos-norm",
            default=float("nan"),
        )

        thrust_lbs = plant._get(
            "propulsion/engine[0]/thrust-lbs",
            default=float("nan"),
        )

        rpm = plant._get(
            "propulsion/engine[0]/engine-rpm",
            "propulsion/engine[0]/rpm",
            default=float("nan"),
        )

        udot_fps2 = plant._get(
            "accelerations/udot-ft_sec2",
            default=float("nan"),
        )

        truth = plant.read_truth()

        rows.append(
            {
                "requested_throttle": throttle,
                "throttle_cmd": throttle_cmd,
                "throttle_pos": throttle_pos,
                "thrust_lbs": thrust_lbs,
                "rpm": rpm,
                "udot_ft_s2": udot_fps2,
                "airspeed_mps_after_2s": truth.airspeed_mps,
            }
        )

    df = pd.DataFrame(rows)

    print(df.to_string(index=False))

    df.to_csv(
        "results/propulsion_sweep.csv",
        index=False,
    )

    return df


if __name__ == "__main__":

    trim_df = trim_sweep()

    propulsion_df = propulsion_sweep()

    print("\n=== DIAGNOSTIC SUMMARY ===")

    successful = trim_df[
        trim_df["trim_ok"] == True
    ]

    if len(successful) > 0:
        print(
            "\nSuccessful trim speeds:"
        )

        print(
            successful[
                [
                    "speed_mps",
                    "throttle",
                    "elevator",
                    "pitch_deg",
                ]
            ].to_string(index=False)
        )

    else:
        print(
            "\nNo speeds trimmed successfully."
        )

        print(
            "Inspect the propulsion sweep next."
        )

    print(
        "\nSaved:\n"
        "  results/trim_envelope.csv\n"
        "  results/propulsion_sweep.csv"
    )