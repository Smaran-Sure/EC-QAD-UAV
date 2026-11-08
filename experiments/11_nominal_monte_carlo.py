"""Phase 13: nominal Monte Carlo over sensor seed and small initial-condition perturbations."""
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


def main(runs: int):
    aircraft, sensors, est, ctrl, fault_cfg = configs()
    fault_cfg = disabled_faults(fault_cfg)
    rng = np.random.default_rng(20261108)
    rows = []
    out = project_root() / "results" / "11_nominal_monte_carlo"
    out.mkdir(parents=True, exist_ok=True)

    for i in range(runs):
        ac = deepcopy(aircraft)
        sc = deepcopy(sensors)
        sc["seed"] = 1000 + i
        ac["simulation"]["initial"]["altitude_m"] += float(rng.normal(0.0, 3.0))
        ac["simulation"]["initial"]["calibrated_airspeed_mps"] += float(rng.normal(0.0, 0.5))
        ac["simulation"]["initial"]["heading_deg"] += float(rng.normal(0.0, 3.0))

        runner = SimulationRunner(ac, sc, est, ctrl, fault_cfg,
                                  control_mode="estimated_pid", estimator_enabled=True,
                                  duration_s=180.0)
        logger, meta = runner.run()
        metrics = summarize(logger.dataframe())
        rows.append({"run": i, **metrics})
        logger.save_csv(out / f"run_{i:03d}.csv")
        print(f"Completed nominal MC {i+1}/{runs}")

    table = pd.DataFrame(rows)
    table.to_csv(out / "summary_per_run.csv", index=False)
    summary = {}
    for c in table.columns:
        if c == "run":
            continue
        vals = table[c].dropna().to_numpy(float)
        if len(vals):
            summary[c] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "p05": float(np.quantile(vals, 0.05)),
                "p95": float(np.quantile(vals, 0.95)),
            }
    save_json(summary, out / "summary_stats.json")
    print(table.describe(include="all"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=20)
    args = p.parse_args()
    main(args.runs)
