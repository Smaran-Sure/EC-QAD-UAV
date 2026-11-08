import math
from ecqad_uav.faults.fault_injector import FaultInjector


def test_gyro_fault_window():
    cfg = {"faults": {"gyro_bias": {"enabled": True, "start_s": 10.0, "end_s": 20.0,
                                      "axis": "z", "magnitude_deg_s": 0.5}}}
    inj = FaultInjector(cfg)
    assert inj.state(5.0).gyro_bias_xyz_rad_s[2] == 0.0
    assert math.isclose(inj.state(15.0).gyro_bias_xyz_rad_s[2], math.radians(0.5))
    assert inj.state(25.0).gyro_bias_xyz_rad_s[2] == 0.0
