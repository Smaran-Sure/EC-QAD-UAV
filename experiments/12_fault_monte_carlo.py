"""Phase 17+: randomized fault Monte Carlo for the conventional system."""
from __future__ import annotations
import argparse
from copy import deepcopy
import numpy as np
import pandas as pd

from ecqad_uav.sim.runner import SimulationRunner
from ecqad_uav.sim.metrics import summarize
from ecqad_uav.utils.config import project_root
from ecqad_uav.utils.logger import save_json
from _common import configs, disabled_faults

FAULTS = ("gyro_bias", "gps_dropout", "airspeed_bias", "aileron_loss_of_effectiveness")


def configure_fault(base: dict, name: str, rng: np.random.Generator) -> dict:
    c = disabled_faults(base)
    s = c["faults"][name]
    s["enabled"] = True
    s["start_s"] = float(rng.uniform(45.0, 100.0))
    if name == "gyro_bias":
        s["axis"] = str(rng.choice(["x", "y", "z"]))
        s["magnitude_deg_s"] = float(rng.uniform(0.2, 1.0) * rng.choice([-1, 1]))
    elif name == "gps_dropout":
        s["end_s"] = float(s["start_s"] + rng.uniform(10.0, 35.0))
    elif name == "airspeed_bias":
        s["magnitude_mps"] = float(rng.uniform(1.0, 5.0) * rng.choice([-1, 1]))
    elif name == "aileron_loss_of_effectiveness":
        s["effectiveness"] = float(rng.uniform(0.4, 0.8))
    return c


def main(runs: int):
    aircraft, sensors, est, ctrl, fault_cfg = configs()
    rng = np.random.default_rng(20261109)
    out = project_root() / "results" / "12_fault_monte_carlo"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(runs):
        name = FAULTS[i % len(FAULTS)]
        sc = deepcopy(sensors)
        sc["seed"] = 5000 + i
        fc = configure_fault(fault_cfg, name, rng)
        runner = SimulationRunner(aircraft, sc, est, ctrl, fc,
                                  control_mode="estimated_pid", estimator_enabled=True,
                                  duration_s=180.0)
        logger, _ = runner.run()
        metrics = summarize(logger.dataframe())
        rows.append({"run": i, "fault": name, **metrics})
        logger.save_csv(out / f"run_{i:03d}_{name}.csv")
        print(f"Completed fault MC {i+1}/{runs}: {name}")

    table = pd.DataFrame(rows)
    table.to_csv(out / "summary_per_run.csv", index=False)
    grouped = {}
    for name, grp in table.groupby("fault"):
        grouped[name] = {}
        for c in grp.columns:
            if c in {"run", "fault"}:
                continue
            vals = grp[c].dropna().to_numpy(float)
            if len(vals):
                grouped[name][c] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
    save_json(grouped, out / "summary_by_fault.json")
    print(table.groupby("fault").mean(numeric_only=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=20)
    args = p.parse_args()
    main(args.runs)
