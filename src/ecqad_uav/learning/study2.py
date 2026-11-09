"""Study 2 bounded policies and training-only covariance propagation.

Study 1 Policy and all runtime physical features remain unchanged.
"""
from pathlib import Path
import numpy as np
import torch
from ecqad_uav.learning.policy import Policy, fake_quant
from ecqad_uav.control.autopilot import POLICY_INPUT_SCHEMA

ACTION_SCHEMA = 'tanh_actuators_v1'


def action_transform(latent):
    """One canonical backend-neutral bounded action definition."""
    if isinstance(latent, torch.Tensor):
        tanh = torch.tanh
        offset = latent.new_tensor([0., 0., 0., 1.])
        scale = latent.new_tensor([1., 1., 1., .5])
    else:
        tanh = np.tanh
        offset = np.array([0., 0., 0., 1.], dtype=latent.dtype)
        scale = np.array([1., 1., 1., .5], dtype=latent.dtype)
    return (tanh(latent) + offset) * scale


class BoundedPolicy(Policy):
    def forward(self, features):
        return action_transform(super().forward(features))

    def numpy_policy(self):
        with torch.no_grad():
            layers=[(fake_quant(l.weight,self.bits,l.weight.min(),l.weight.max()).numpy().copy(),l.bias.numpy().copy()) for l in self.layers]
            mean,std,cap,om,os=[t.detach().numpy().copy() for t in (self.mean,self.std,self.act_max,self.out_mean,self.out_scale)]
        bits=self.bits
        def predict(features):
            x=(np.asarray(features,dtype=np.float32)-mean)/std
            for i,(w,b) in enumerate(layers):
                x=x@w.T+b
                if i<2:
                    x=np.clip(x,0,cap[i])
                    if bits!=32:
                        step=cap[i]/(2**bits-1);x=np.round(x/step)*step
            return action_transform(x*os+om)
        return predict

    def save(self,path,**metadata):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(input_schema=POLICY_INPUT_SCHEMA,action_schema=ACTION_SCHEMA,state=self.state_dict(),bits=self.bits,width=self.width,metadata=metadata),path)

    @classmethod
    def load(cls,path):
        d=torch.load(path,map_location='cpu',weights_only=True)
        if d.get('input_schema')!=POLICY_INPUT_SCHEMA or d.get('action_schema')!=ACTION_SCHEMA:
            raise ValueError('Study 2 requires physical_reference_v2 and bounded action semantics')
        m=cls(d['bits'],d['width']);m.load_state_dict(d['state']);m.eval();return m


def feature_jacobian(features):
    """Analytic d phi/d UKF state, conditional on latest gyro/pitot readings.

    x=[pn,pe,pd,vn,ve,vd,roll,pitch,yaw,bgx,bgy,bgz,bax,bay,baz].
    Altitude=h0-pd; rates=gyro-bg. Reference and raw measurements fixed.
    Heading-error derivative is local away from the +/-pi discontinuity.
    """
    f=np.asarray(features,dtype=np.float64)
    g=np.zeros(f.shape[:-1]+(16,15),dtype=np.float64)
    g[...,0,2]=1/30;g[...,1,8]=-2
    g[...,3,6]=2;g[...,4,7]=5
    g[...,5,9]=-1;g[...,6,10]=-1;g[...,7,11]=-2
    g[...,8,2]=-1/300
    g[...,10,8]=f[...,11];g[...,11,8]=-f[...,10]
    return g


def teacher_jacobian(teacher,features):
    x=torch.as_tensor(features,dtype=torch.float32).detach().requires_grad_(True)
    u=teacher(x)
    return torch.stack([torch.autograd.grad(u[...,j].sum(),x,retain_graph=j<3)[0] for j in range(4)],dim=-2).detach().numpy().astype(np.float64)


def propagate_covariance(P,G,J):
    """Full correlated covariance; no diagonal approximation or score search."""
    P,G,J=map(lambda a:np.asarray(a,dtype=np.float64),(P,G,J))
    sz=G@P@np.swapaxes(G,-1,-2)
    su=J@sz@np.swapaxes(J,-1,-2)
    # Roundoff symmetrization, no covariance/eigenvalue tuning.
    sz=(sz+np.swapaxes(sz,-1,-2))/2
    su=(su+np.swapaxes(su,-1,-2))/2
    q=np.trace(su,axis1=-2,axis2=-1)
    if not np.isfinite(q).all() or (q < -1e-10).any():
        raise ValueError('Invalid propagated actuator covariance')
    return sz,su,np.maximum(q,0.)


def uncertainty_diagnostics(teacher,features,P):
    G=feature_jacobian(features);J=teacher_jacobian(teacher,features)
    sz,su,q=propagate_covariance(P,G,J)
    return dict(q=q,trace_P=np.trace(P,axis1=-2,axis2=-1),sensitivity=np.linalg.norm(J,axis=(-2,-1)),
                action_variance=np.diagonal(su,axis1=-2,axis2=-1),
                min_feature_eigenvalue=np.linalg.eigvalsh(sz)[...,0],min_action_eigenvalue=np.linalg.eigvalsh(su)[...,0])
