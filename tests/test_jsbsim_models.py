from pathlib import Path

import jsbsim
import pytest

from ecqad_uav.plant.jsbsim_plant import JSBSimPlant
from ecqad_uav.utils.config import load_yaml, project_root


def test_repo_rascal_without_environment_override(monkeypatch):
    monkeypatch.delenv('JSBSIM_ROOT', raising=False)
    plant = JSBSimPlant(load_yaml('aircraft.yaml'))
    assert plant.model == 'Rascal110'
    assert Path(plant.model_root) == project_root() / 'models' / 'jsbsim'


def test_invalid_custom_model_can_use_stock_fallback(monkeypatch, tmp_path):
    aircraft = tmp_path / 'aircraft' / 'missing_test_model'
    aircraft.mkdir(parents=True)
    (aircraft / 'missing_test_model.xml').write_text('')
    monkeypatch.setenv('JSBSIM_ROOT', str(tmp_path))
    cfg = load_yaml('aircraft.yaml')
    cfg['aircraft']['model'] = 'missing_test_model'
    cfg['aircraft']['allow_fallback_for_smoke_test'] = True
    with pytest.warns(RuntimeWarning, match='Falling back'):
        plant = JSBSimPlant(cfg)
    assert plant.model == 'c172p'
    assert Path(plant.model_root) == Path(jsbsim.get_default_root_dir()).resolve()


def test_electric_motor_startup_has_no_impulsive_roll():
    import math
    from ecqad_uav.plant.jsbsim_plant import ControlCommand
    cfg=load_yaml('aircraft.yaml')
    plant=JSBSimPlant(cfg); plant.initialize()
    for _ in range(400):
        state=plant.step(ControlCommand(throttle=.36))
        assert all(math.isfinite(v) for v in state.as_dict().values())
        assert abs(state.p_rad_s)<2.0
        assert 12<state.airspeed_mps<40


def test_geodetic_position_rate_matches_navigation_velocity():
    from ecqad_uav.plant.jsbsim_plant import ControlCommand
    cfg=load_yaml('aircraft.yaml')
    plant=JSBSimPlant(cfg); plant.initialize()
    previous=plant.read_truth()
    for _ in range(200):
        state=plant.step(ControlCommand(throttle=.36))
        finite_difference=(state.pn_m-previous.pn_m)/plant.dt
        # A spherical conversion previously caused about 0.17 m/s bias here.
        assert abs(finite_difference-.5*(state.vn_mps+previous.vn_mps))<.03
        previous=state
