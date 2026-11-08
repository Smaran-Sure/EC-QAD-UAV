"""Phase 20 gate: summarize whether the conventional baseline is ready for neural-teacher work."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from ecqad_uav.utils.config import project_root


def exists(name: str) -> bool:
    return (project_root() / "results" / name / "run.csv").exists()


def main():
    root = project_root() / "results"
    gates = []

    required = [
        "01_open_loop", "02_trim", "03_sensor_validation", "04_ukf_validation",
        "05_truth_control", "06_estimated_control", "07_fault_gyro_bias",
        "08_fault_gps_dropout", "09_fault_airspeed_bias", "10_fault_actuator_loe",
    ]
    for name in required:
        gates.append((f"result exists: {name}", exists(name)))

    ukf_path = root / "04_ukf_validation" / "run.csv"
    if ukf_path.exists():
        df = pd.read_csv(ukf_path)
        pcols = [c for c in df.columns if c.startswith("P_") and c.count("_") == 2]
        finite_cov = bool(np.isfinite(df[pcols].to_numpy()).all()) if pcols else False
        positive_diag = bool((df[pcols].dropna().to_numpy() >= 0).all()) if pcols else False
        gates += [("UKF covariance finite", finite_cov), ("UKF covariance diagonal nonnegative", positive_diag)]
        for sensor in ("gps", "heading", "airspeed"):
            n, th = f"nis_{sensor}", f"nis95_{sensor}"
            if {n, th}.issubset(df.columns):
                v = df[[n, th]].dropna()
                frac = float((v[n] <= v[th]).mean()) if len(v) else float("nan")
                gates.append((f"nominal {sensor} NIS fraction below 95% threshold is 0.80-0.99",
                              bool(np.isfinite(frac) and 0.80 <= frac <= 0.99)))

    gps_fault = root / "08_fault_gps_dropout" / "run.csv"
    if gps_fault.exists():
        df = pd.read_csv(gps_fault)
        # Position covariance indices 0..2. Compare pre-dropout to late-dropout median.
        cols = [f"P_{i:02d}_{i:02d}" for i in range(3)]
        if set(cols).issubset(df.columns):
            pre = df[(df.time_s >= 45) & (df.time_s < 58)][cols].sum(axis=1).median()
            during = df[(df.time_s >= 72) & (df.time_s < 79)][cols].sum(axis=1).median()
            gates.append(("GPS dropout increases position covariance", bool(during > pre)))

    print("\nCONVENTIONAL GNC BASELINE GATE\n" + "=" * 34)
    for label, ok in gates:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    passed = sum(ok for _, ok in gates)
    print(f"\nPassed {passed}/{len(gates)} checks.")
    print("Do not start EC-QAD/neural compression until failed checks are investigated, not merely bypassed.")


if __name__ == "__main__":
    main()
