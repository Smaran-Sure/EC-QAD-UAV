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
