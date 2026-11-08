"""Phase 3: request JSBSim simple trim and hold the trimmed controls."""
from _common import configs, disabled_faults, execute

if __name__ == "__main__":
    *_, fault_cfg = configs()
    execute("02_trim", control_mode="trim_hold", estimator=False,
            faults=disabled_faults(fault_cfg), duration_s=60.0)
