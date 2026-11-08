"""Phase 2: plant/sign sanity check with small aileron/elevator/throttle pulses."""
from _common import configs, disabled_faults, execute

if __name__ == "__main__":
    *_, fault_cfg = configs()
    execute("01_open_loop", control_mode="open_loop", estimator=False,
            faults=disabled_faults(fault_cfg), duration_s=45.0)
