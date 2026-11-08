"""Phases 9-10: tune/validate cascaded PID using perfect truth state."""
from _common import configs, disabled_faults, execute

if __name__ == "__main__":
    *_, fault_cfg = configs()
    execute("05_truth_control", control_mode="truth_pid", estimator=False,
            faults=disabled_faults(fault_cfg), duration_s=180.0)
