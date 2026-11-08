from ecqad_uav.control.pid import PID


def test_pid_saturates():
    pid = PID(10.0, 0.0, 0.0, output_min=-1.0, output_max=1.0)
    assert pid.update(1.0, 0.02) == 1.0
    assert pid.update(-1.0, 0.02) == -1.0
