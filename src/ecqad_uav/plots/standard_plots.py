from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_truth_overview(df: pd.DataFrame, out: str | Path):
    out = Path(out)
    t = df["time_s"]
    series = [
        ("truth_altitude_m", "Altitude [m]"),
        ("truth_airspeed_mps", "Airspeed [m/s]"),
        ("truth_roll_rad", "Roll [deg]"),
        ("truth_pitch_rad", "Pitch [deg]"),
        ("truth_yaw_rad", "Yaw [deg]"),
    ]
    for col, label in series:
        if col not in df:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        y = np.degrees(df[col]) if col.endswith("_rad") else df[col]
        ax.plot(t, y)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        _save(fig, out / f"{col}.png")


def plot_estimator(df: pd.DataFrame, out: str | Path):
    out = Path(out)
    t = df["time_s"]
    pairs = [
        ("roll", "rad", 6), ("pitch", "rad", 7), ("yaw", "rad", 8),
        ("pn", "m", 0), ("pe", "m", 1), ("pd", "m", 2),
    ]
    for name, unit, idx in pairs:
        tc = f"truth_{name}_{unit}"
        ec = f"est_{name}_{unit}"
        pc = f"P_{idx:02d}_{idx:02d}"
        if not {tc, ec, pc}.issubset(df.columns):
            continue
        truth, est = df[tc], df[ec]
        sigma = 3.0 * np.sqrt(np.maximum(df[pc], 0.0))
        if unit == "rad":
            truth, est, sigma = np.degrees(truth), np.degrees(est), np.degrees(sigma)
            ylabel = f"{name} [deg]"
        else:
            ylabel = f"{name} [m]"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, truth, label="truth")
        ax.plot(t, est, label="estimate")
        ax.fill_between(t, est - sigma, est + sigma, alpha=0.2, label="±3σ")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, out / f"estimator_{name}.png")


def plot_nis(df: pd.DataFrame, out: str | Path):
    out = Path(out)
    for sensor in ("gps", "heading", "airspeed"):
        n, th = f"nis_{sensor}", f"nis95_{sensor}"
        if not {n, th}.issubset(df.columns):
            continue
        valid = df[["time_s", n, th]].dropna()
        if not len(valid):
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(valid["time_s"], valid[n], label="NIS")
        ax.plot(valid["time_s"], valid[th], linestyle="--", label="95% χ² threshold")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("NIS")
        ax.set_title(sensor)
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, out / f"nis_{sensor}.png")


def plot_tracking(df: pd.DataFrame, out: str | Path):
    out = Path(out)
    specs = [
        ("cmd_altitude_m", "truth_altitude_m", "Altitude [m]", 1.0),
        ("cmd_heading_rad", "truth_yaw_rad", "Heading [deg]", 180.0 / np.pi),
        ("cmd_airspeed_mps", "truth_airspeed_mps", "Airspeed [m/s]", 1.0),
    ]
    for cmd, actual, label, scale in specs:
        if not {cmd, actual}.issubset(df.columns):
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["time_s"], df[cmd] * scale, label="command")
        ax.plot(df["time_s"], df[actual] * scale, label="actual")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, out / f"tracking_{actual}.png")
