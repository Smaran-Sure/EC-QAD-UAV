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
    with pytest.warns(RuntimeWarning, match='Falling back'):
        plant = JSBSimPlant(cfg)
    assert plant.model == 'c172p'
    assert Path(plant.model_root) == Path(jsbsim.get_default_root_dir()).resolve()
