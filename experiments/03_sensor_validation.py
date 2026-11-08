"""Phase 4: validate measurement means/std against truth under healthy flight."""
import numpy as np
from ecqad_uav.utils.math_utils import wrap_pi
from _common import configs, disabled_faults, execute


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.mean(x)), float(np.std(x, ddof=1))) if len(x) > 1 else (float("nan"), float("nan"))


if __name__ == "__main__":
    *_, fault_cfg = configs()
    df, _, _, _ = execute("03_sensor_validation", control_mode="trim_hold", estimator=False,
                           faults=disabled_faults(fault_cfg), duration_s=120.0)
    checks = {}
    for meas, truth in [
        ("meas_gps_pn_m", "truth_pn_m"), ("meas_gps_pe_m", "truth_pe_m"),
        ("meas_gps_pd_m", "truth_pd_m"), ("meas_gps_vn_mps", "truth_vn_mps"),
        ("meas_gps_ve_mps", "truth_ve_mps"), ("meas_gps_vd_mps", "truth_vd_mps"),
        ("meas_airspeed_mps", "truth_airspeed_mps"),
        ("meas_gyro_x_rad_s", "truth_p_rad_s"), ("meas_gyro_y_rad_s", "truth_q_rad_s"),
        ("meas_gyro_z_rad_s", "truth_r_rad_s"),
    ]:
        if meas in df:
            valid = df[[meas, truth]].dropna()
            checks[meas] = stats(valid[meas] - valid[truth])
    if "meas_heading_rad" in df:
        valid = df[["meas_heading_rad", "truth_yaw_rad"]].dropna()
        checks["meas_heading_rad"] = stats(wrap_pi(valid["meas_heading_rad"] - valid["truth_yaw_rad"]))
    print("\nMeasurement error mean, std:")
    for k, (mu, sd) in checks.items():
        print(f"  {k:24s} mean={mu:+.6g}  std={sd:.6g}")
