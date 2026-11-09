import numpy as np
import torch
from ecqad_uav.learning.policy import Policy, conditioning_weights, fake_quant
from ecqad_uav.estimation.ukf import NavigationUKF


def test_numpy_quantized_rollout_matches_training_forward():
    torch.manual_seed(2)
    x=torch.randn(40,16)
    for bits in (32,8,4,2):
        model=Policy(bits)
        expected=np.clip(model(x).detach().numpy(),[-1,-1,-1,0],[1,1,1,1])
        actual=model.numpy_policy()(x.numpy())
        assert np.allclose(actual,expected,atol=2e-6)


def test_conditioning_bounds_normalization_and_shuffle():
    rng=np.random.default_rng(2); cov=rng.exponential(2,1000); nis=rng.chisquare(3,1000)
    for mode in ('covariance','innovation','ecqad','shuffled'):
        w,_=conditioning_weights(cov,nis,mode,7)
        assert w.min()>=.5 and w.max()<=2
        assert abs(w.mean()-1)<1e-6
    a,_=conditioning_weights(cov,nis,'ecqad',7)
    b,_=conditioning_weights(cov,nis,'shuffled',7)
    assert np.array_equal(np.sort(a),np.sort(b))
    assert not np.array_equal(a,b)


def test_two_bit_has_four_weight_levels_and_gradient():
    x=torch.linspace(-1,1,101,requires_grad=True)
    q=fake_quant(x,2,torch.tensor(-1.),torch.tensor(1.))
    assert len(torch.unique(q.detach().round(decimals=5)))==4
    q.sum().backward(); assert torch.all(x.grad==1)


def test_vectorized_ukf_matches_scalar_dynamics():
    rng=np.random.default_rng(3); x=rng.normal(0,.2,(31,15))
    gyro=np.array([.3,-.2,.1]); accel=np.array([1,2,-9.7])
    actual=NavigationUKF._fx_batch(x,.01,gyro,accel)
    expected=np.array([NavigationUKF._fx(s,.01,gyro,accel) for s in x])
    assert np.allclose(actual,expected,atol=1e-12)


def test_fault_balancing_has_equal_class_weight_mass():
    labels=np.array(['nominal']*80+['gyro']*10+['gps']*5+['airspeed']*5)
    w,_=conditioning_weights(np.zeros(100),np.ones(100),'fault_balanced',labels=labels)
    assert np.isclose(w.mean(),1.)
    for name in np.unique(labels): assert np.isclose(w[labels==name].sum(),25.)


def test_policy_interface_is_physical_and_pid_free(monkeypatch):
    from ecqad_uav.control.autopilot import NavState, GuidanceCommand, policy_features, POLICY_FEATURE_NAMES
    import ecqad_uav.sim.runner as runner
    from ecqad_uav.utils.config import load_yaml
    assert POLICY_FEATURE_NAMES == ('altitude_error','heading_error','airspeed_error','roll','pitch',
        'body_p','body_q','body_r','altitude','airspeed','sin_heading','cos_heading',
        'reference_altitude','reference_airspeed','sin_reference_heading','cos_reference_heading')
    nav=NavState(300,25,.1,.02,.3,.01,.02,.03)
    ref=GuidanceCommand(320,.4,28)
    expected=np.array([20/30,.1/.5,3/5,.1/.5,.02/.2,.01,.02,.03/.5,1,25/30,
        np.sin(.3),np.cos(.3),320/300,28/30,np.sin(.4),np.cos(.4)],dtype=np.float32)
    np.testing.assert_allclose(policy_features(nav,ref),expected,atol=1e-7)
    def forbidden(*args,**kwargs):
        raise AssertionError('PID construction forbidden during neural deployment')
    monkeypatch.setattr(runner,'CascadedAutopilot',forbidden)
    configs=[load_yaml(n+'.yaml') for n in ['aircraft','sensors','estimator','controller','faults']]
    for bits in (32,8,4,2):
        policy=Policy(bits).numpy_policy()
        log,meta=runner.SimulationRunner(*configs,control_mode='estimated_pid',policy=policy,duration_s=.1).run()
        assert meta['pid_executed'] is False
        assert log.dataframe()[[f'pid_{i}' for i in range(4)]].isna().all().all()


def test_rejects_unversioned_exploratory_checkpoint(tmp_path):
    path=tmp_path/'old.pt'; model=Policy()
    torch.save(dict(state=model.state_dict(),bits=32,width=32),path)
    import pytest
    with pytest.raises(ValueError,match='schema mismatch'):
        Policy.load(path)
