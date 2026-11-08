"""Phase 16: pitot/airspeed bias and associated innovation/control response."""
from _common import configs, only_fault, execute

if __name__ == "__main__":
    *_, fault_cfg = configs()
    execute("09_fault_airspeed_bias", control_mode="estimated_pid", estimator=True,
            faults=only_fault(fault_cfg, "airspeed_bias"), duration_s=180.0)
