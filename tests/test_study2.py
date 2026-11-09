import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from ecqad_uav.learning.study2 import BoundedPolicy,action_transform,feature_jacobian,teacher_jacobian,propagate_covariance
from ecqad_uav.control.autopilot import NavState,GuidanceCommand,policy_features,POLICY_FEATURE_NAMES
from ecqad_uav.learning.policy import conditioning_weights


def experiment():
    folder=Path(__file__).resolve().parents[1]/'experiments'
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))
    import run_study2
    return run_study2


def test_torch_numpy_action_semantics_match():
    torch.manual_seed(17);x=torch.randn(1000,16)*4
    for bits in (32,8,4,2):
        m=BoundedPolicy(bits);actual=m.numpy_policy()(x.numpy());expected=m(x).detach().numpy()
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)
        assert (actual[:,:3]>=-1).all() and (actual[:,:3]<=1).all()
        assert (actual[:,3]>=0).all() and (actual[:,3]<=1).all()
    v=torch.tensor([[-1000.,1000.,1000.,-1000.],[1000.,-1000.,0.,1000.]])
    np.testing.assert_array_equal(action_transform(v).numpy(),action_transform(v.numpy()))
    assert np.isfinite(action_transform(v).numpy()).all()


def features_from_state(x):
    nav=NavState.from_estimator(x,25.,300.)
    nav.p_rad_s,nav.q_rad_s,nav.r_rad_s=np.array([.01,.02,.03])-x[9:12]
    return policy_features(nav,GuidanceCommand(320.,.5,28.))


def test_feature_jacobian_matches_physical_interface_finite_difference():
    x=np.zeros(15);x[2]=-8;x[6:9]=[.05,-.02,.4]
    for yaw in (.4,np.pi-1e-3,-np.pi+1e-3):
        x[8]=yaw;g=feature_jacobian(features_from_state(x));fd=np.zeros_like(g)
        eps=np.r_[np.full(6,.01),np.full(3,1e-4),np.full(6,1e-4)]
        for j in range(15):
            a=x.copy();b=x.copy();a[j]+=eps[j];b[j]-=eps[j]
            fd[:,j]=(features_from_state(a).astype(float)-features_from_state(b).astype(float))/(2*eps[j])
        np.testing.assert_allclose(g,fd,atol=.002,rtol=.003)


def test_covariance_propagation_psd_zero_and_insensitivity():
    rng=np.random.default_rng(11);a=rng.normal(size=(15,15));p=a@a.T
    g=feature_jacobian(features_from_state(np.zeros(15)));j=rng.normal(size=(4,16))
    z,u,q=propagate_covariance(p,g,j)
    np.testing.assert_allclose(z,z.T,atol=1e-12);np.testing.assert_allclose(u,u.T,atol=1e-12)
    assert np.linalg.eigvalsh(z).min()>-1e-10 and np.linalg.eigvalsh(u).min()>-1e-10
    assert np.isfinite(q) and q>=0
    assert propagate_covariance(p*0,g,j)[2]==0
    assert propagate_covariance(p,g,j*0)[2]==0
    direction=np.zeros((15,15));direction[0,0]=10 # position north unused by phi
    assert propagate_covariance(direction,g,j)[2]==0
    np.testing.assert_allclose(u,(j@g)@p@(j@g).T,rtol=1e-12,atol=1e-10)


def test_local_empirical_actuator_covariance():
    torch.manual_seed(5);m=BoundedPolicy();x=np.zeros(15);x[6:9]=[.1,.05,.4]
    f=features_from_state(x);g=feature_jacobian(f);j=teacher_jacobian(m,f[None])[0]
    rng=np.random.default_rng(22);a=rng.normal(size=(15,15));p=a@a.T*1e-8
    _,theory,q=propagate_covariance(p,g,j)
    samples=rng.multivariate_normal(np.zeros(15),p,size=12000)
    features=np.array([features_from_state(x+dx) for dx in samples]);actions=m.numpy_policy()(features)
    empirical=np.cov(actions,rowvar=False)
    assert np.linalg.norm(empirical-theory)/np.linalg.norm(theory)<.08
    assert abs(np.trace(empirical)/q-1)<.08


def test_study2_split_ids_and_test_selection_isolation(monkeypatch):
    e=experiment();names=['train','validation_select','validation_gate','validation_signal','test']
    sets=[e.scenario_ids(s) for s in names]
    for i in range(len(sets)):
        for j in range(i):assert sets[i].isdisjoint(sets[j])
    jobs=e.selection_jobs('dummy.pt','uniform',7,25)
    assert {j['split'] for j in jobs}=={'validation_select'}
    assert not e.scenario_ids('test').intersection({e.scenario(j['split'],j['condition'],j['episode'])[2]['episode_seed'] for j in jobs})
    with pytest.raises(ValueError):e.dataset('test')
    with pytest.raises(TypeError):e.selection_jobs('dummy.pt','uniform',7,25,split='test')


def test_study2_deterministic_and_manifest(tmp_path,monkeypatch):
    e=experiment();a=e.scenario('train','gyro_bias',0);b=e.scenario('train','gyro_bias',0);assert a==b
    torch.manual_seed(9);first=BoundedPolicy().numpy_policy()(np.ones(16,dtype=np.float32))
    torch.manual_seed(9);second=BoundedPolicy().numpy_policy()(np.ones(16,dtype=np.float32));np.testing.assert_array_equal(first,second)
    monkeypatch.setattr(e,'OUT',tmp_path);e.freeze_manifest();e.freeze_manifest()
    import json
    p=tmp_path/'protocol_manifest.json';m=json.loads(p.read_text());m['sha256']['configs/study2.yaml']='incorrect';p.write_text(json.dumps(m))
    with pytest.raises(AssertionError,match='frozen'):e.freeze_manifest()


def test_study2_neural_inputs_remain_pid_and_confidence_free(monkeypatch):
    import ecqad_uav.sim.runner as runner
    from ecqad_uav.utils.config import load_yaml
    assert len(POLICY_FEATURE_NAMES)==16
    assert not any(s in ' '.join(POLICY_FEATURE_NAMES) for s in ('covariance','nis','integral','jacobian'))
    def forbidden(*a,**kw):raise AssertionError('PID forbidden')
    monkeypatch.setattr(runner,'CascadedAutopilot',forbidden)
    configs=[load_yaml(n+'.yaml') for n in ['aircraft','sensors','estimator','controller','faults']]
    for bits in (32,4):
        m=BoundedPolicy(bits);_,meta=runner.SimulationRunner(*configs,control_mode='estimated_pid',policy=m.numpy_policy(),duration_s=.1).run()
        assert not meta['pid_executed']


def test_exact_covariance_distribution_shuffle():
    c=np.random.default_rng(3).lognormal(size=1000);n=np.ones(len(c))
    a,_=conditioning_weights(c,n,'covariance',61);b,_=conditioning_weights(c,n,'covariance_shuffled',61)
    np.testing.assert_array_equal(np.sort(a),np.sort(b));assert not np.array_equal(a,b)


def test_gate_cannot_pass_on_action_discrepancy_only():
    from ecqad_uav.plots.study2_report import gate_decision
    import pandas as pd
    d=pd.DataFrame([dict(kind='incremental_prediction',baseline='covariance+sensitivity',target=t,reduction=.1,ci_low=.01 if t=='action_error' else -.01,ci_high=.2,positive_seeds=5) for t in ['action_error','future_tracking','excess_future_tracking']])
    assert not gate_decision(d)['passed']
