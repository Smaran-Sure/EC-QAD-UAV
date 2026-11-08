"""Phase 15: GPS dropout; position covariance should grow then recover."""
from _common import configs, only_fault, execute

if __name__ == "__main__":
    *_, fault_cfg = configs()
    execute("08_fault_gps_dropout", control_mode="estimated_pid", estimator=True,
            faults=only_fault(fault_cfg, "gps_dropout"), duration_s=180.0)
