"""Phase 14: inject persistent gyro-z bias."""
from _common import configs, only_fault, execute

if __name__ == "__main__":
    *_, fault_cfg = configs()
    execute("07_fault_gyro_bias", control_mode="estimated_pid", estimator=True,
            faults=only_fault(fault_cfg, "gyro_bias"), duration_s=180.0)
