from __future__ import annotations
import numpy as np
import pandas as pd
from ecqad_uav.utils.math_utils import wrap_pi


def rmse(x) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def summarize(df: pd.DataFrame) -> dict:
    out: dict[str, float] = {}
    if {"cmd_altitude_m", "truth_altitude_m"}.issubset(df.columns):
        out["altitude_rmse_m"] = rmse(df["cmd_altitude_m"] - df["truth_altitude_m"])
        out["altitude_peak_abs_m"] = float(np.nanmax(np.abs(df["cmd_altitude_m"] - df["truth_altitude_m"])))
    if {"cmd_heading_rad", "truth_yaw_rad"}.issubset(df.columns):
        e = wrap_pi(df["cmd_heading_rad"].to_numpy() - df["truth_yaw_rad"].to_numpy())
        out["heading_rmse_deg"] = np.degrees(rmse(e))
        out["heading_peak_abs_deg"] = float(np.degrees(np.nanmax(np.abs(e))))
    if {"cmd_airspeed_mps", "truth_airspeed_mps"}.issubset(df.columns):
        e = df["cmd_airspeed_mps"] - df["truth_airspeed_mps"]
        out["airspeed_rmse_mps"] = rmse(e)
        out["airspeed_peak_abs_mps"] = float(np.nanmax(np.abs(e)))
    if {"est_pn_m", "truth_pn_m"}.issubset(df.columns):
        for axis in ("pn", "pe", "pd"):
            out[f"est_{axis}_rmse_m"] = rmse(df[f"est_{axis}_m"] - df[f"truth_{axis}_m"])
    if {"est_roll_rad", "truth_roll_rad"}.issubset(df.columns):
        for name in ("roll", "pitch", "yaw"):
            e = wrap_pi(df[f"est_{name}_rad"].to_numpy() - df[f"truth_{name}_rad"].to_numpy())
            out[f"est_{name}_rmse_deg"] = np.degrees(rmse(e))
    for sensor in ("gps", "heading", "airspeed"):
        c = f"nis_{sensor}"
        th = f"nis95_{sensor}"
        if c in df and th in df:
            valid = df[[c, th]].dropna()
            if len(valid):
                out[f"nis_{sensor}_fraction_below_95"] = float((valid[c] <= valid[th]).mean())
    return out
