"""Phase 17: aileron loss-of-effectiveness; useful negative-control for estimator-health claims."""
from _common import configs, only_fault, execute

if __name__ == "__main__":
    *_, fault_cfg = configs()
    execute("10_fault_actuator_loe", control_mode="estimated_pid", estimator=True,
            faults=only_fault(fault_cfg, "aileron_loss_of_effectiveness"), duration_s=180.0)
