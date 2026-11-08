"""Phases 5-8: nominal UKF, covariance, innovation and NIS validation."""
from _common import configs, disabled_faults, execute

if __name__ == "__main__":
    *_, fault_cfg = configs()
    execute("04_ukf_validation", control_mode="trim_hold", estimator=True,
            faults=disabled_faults(fault_cfg), duration_s=180.0)
