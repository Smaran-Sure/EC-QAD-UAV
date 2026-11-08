"""Phases 11-13: close the PID loop on UKF estimated state; healthy baseline."""
from _common import configs, disabled_faults, execute

if __name__ == "__main__":
    *_, fault_cfg = configs()
    execute("06_estimated_control", control_mode="estimated_pid", estimator=True,
            faults=disabled_faults(fault_cfg), duration_s=180.0)
