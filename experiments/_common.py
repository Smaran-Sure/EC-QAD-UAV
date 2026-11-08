from __future__ import annotations
from copy import deepcopy
from pathlib import Path

from ecqad_uav.utils.config import load_yaml, project_root
from ecqad_uav.utils.logger import save_json
from ecqad_uav.sim.runner import SimulationRunner
from ecqad_uav.sim.metrics import summarize
from ecqad_uav.plots.standard_plots import plot_truth_overview, plot_estimator, plot_nis, plot_tracking


def configs():
    return (
        load_yaml("aircraft.yaml"),
        load_yaml("sensors.yaml"),
        load_yaml("estimator.yaml"),
        load_yaml("controller.yaml"),
        load_yaml("faults.yaml"),
    )


def disabled_faults(fault_cfg: dict) -> dict:
    c = deepcopy(fault_cfg)
    for spec in c.get("faults", {}).values():
        spec["enabled"] = False
    return c


def only_fault(fault_cfg: dict, name: str) -> dict:
    c = disabled_faults(fault_cfg)
    if name not in c["faults"]:
        raise KeyError(name)
    c["faults"][name]["enabled"] = True
    return c


def execute(name: str, *, control_mode: str, estimator: bool, faults: dict,
            duration_s: float | None = None, make_plots: bool = True):
    aircraft, sensors, est, ctrl, _ = configs()
    runner = SimulationRunner(
        aircraft, sensors, est, ctrl, faults,
        control_mode=control_mode,
        estimator_enabled=estimator,
        sensor_enabled=True,
        duration_s=duration_s,
    )
    logger, meta = runner.run()
    df = logger.dataframe()
    out = project_root() / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    logger.save_csv(out / "run.csv")
    metrics = summarize(df)
    save_json(meta, out / "metadata.json")
    save_json(metrics, out / "metrics.json")
    if make_plots:
        plot_truth_overview(df, out / "plots")
        plot_tracking(df, out / "plots")
        if estimator:
            plot_estimator(df, out / "plots")
            plot_nis(df, out / "plots")
    print(f"Saved: {out}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6g}")
    return df, metrics, meta, out
