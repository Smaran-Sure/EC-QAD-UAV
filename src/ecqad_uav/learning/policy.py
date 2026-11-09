"""Small FP32 / fake-quantized policies and training-only importance weights."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fake_quant(x, bits, lo, hi):
    if bits == 32:
        return x
    step = (hi-lo).clamp(min=1e-8) / (2**bits-1)
    quant = ((x.clamp(lo, hi)-lo)/step).round()*step+lo
    return x + (quant-x).detach()  # straight-through estimator


class Policy(nn.Module):
    def __init__(self, bits=32, width=32):
        super().__init__()
        self.bits, self.width = bits, width
        self.layers = nn.ModuleList([nn.Linear(16, width), nn.Linear(width, width), nn.Linear(width, 4)])
        self.register_buffer('mean', torch.zeros(16))
        self.register_buffer('std', torch.ones(16))
        self.register_buffer('act_max', torch.ones(2)*6)
        self.register_buffer('out_mean', torch.tensor([0.,0.,0.,.4]))
        self.register_buffer('out_scale', torch.tensor([.2,.2,.1,.2]))

    def forward(self, features):
        x = (features-self.mean)/self.std
        for i, layer in enumerate(self.layers):
            w = layer.weight
            if self.bits != 32:
                w = fake_quant(w, self.bits, w.detach().min(), w.detach().max())
            x = F.linear(x, w, layer.bias)
            if i < 2:
                x = F.relu(x).clamp(max=self.act_max[i])
                x = fake_quant(x, self.bits, x.new_tensor(0.), self.act_max[i])
        return x*self.out_scale+self.out_mean

    def numpy_policy(self):
        """Export identical fake-quantized arithmetic to NumPy for fast rollouts.

        This is not a hardware integer implementation; no latency/energy claim.
        """
        with torch.no_grad():
            weights = []
            for layer in self.layers:
                w = layer.weight
                w = fake_quant(w, self.bits, w.min(), w.max())
                weights.append((w.numpy().copy(), layer.bias.numpy().copy()))
        mean, std = self.mean.numpy(), self.std.numpy()
        cap = self.act_max.numpy()
        om, os = self.out_mean.numpy(), self.out_scale.numpy()
        bits = self.bits
        def predict(features):
            x = (np.asarray(features, dtype=np.float32)-mean)/std
            for i,(w,b) in enumerate(weights):
                x = x @ w.T + b
                if i < 2:
                    x = np.clip(x, 0, cap[i])
                    if bits != 32:
                        step = cap[i]/(2**bits-1)
                        x = np.round(x/step)*step
            return np.clip(x*os+om, [-1,-1,-1,0], [1,1,1,1])
        return predict

    def save(self, path, **metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=self.state_dict(), bits=self.bits, width=self.width, metadata=metadata), path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location='cpu', weights_only=True)
        m = cls(d['bits'], d['width']); m.load_state_dict(d['state']); m.eval()
        return m

    def storage_bytes(self):
        nw = sum(l.weight.numel() for l in self.layers)
        nb = sum(l.bias.numel() for l in self.layers)
        # Ideal packed weights + FP32 biases, feature/output normalization,
        # activation ranges, affine weight scales/zero points. Not a measured binary.
        return int(np.ceil(nw*self.bits/8)+4*(nb+16*2+4*2+2+6))


def conditioning_weights(cov, nis, mode, seed=0, labels=None, saliency=None):
    """Training-population mean exactly one, bounds [0.5,2]."""
    signals = np.column_stack((cov, np.log1p(nis)))
    center, scale = signals.mean(0), np.maximum(signals.std(0), 1e-6)
    z = np.clip((signals-center)/scale, -3, 3)
    if mode in ('uniform', 'ptq'):
        return np.ones(len(cov), dtype=np.float32), dict(center=center.tolist(), scale=scale.tolist())
    if mode == 'fault_balanced':
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        weights=(len(labels)/len(counts)/counts[inverse]).astype(np.float32)
        return weights, dict(mean=float(weights.mean()),minimum=float(weights.min()),maximum=float(weights.max()))
    if mode == 'saliency':
        value=np.log1p(saliency)
        score=.5*np.clip((value-value.mean())/max(value.std(),1e-6),-3,3)
    elif mode == 'balanced':
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        score = np.log(len(labels)/len(counts)/counts[inverse])
    else:
        score = z[:,0] if mode=='covariance' else z[:,1] if mode=='innovation' else z.mean(1)
        score = .5*score
    lo, hi = -20., 20.
    for _ in range(70):
        shift=(lo+hi)/2
        if np.clip(np.exp(score+shift), .5, 2).mean()>1: hi=shift
        else: lo=shift
    weights=np.clip(np.exp(score+(lo+hi)/2), .5, 2).astype(np.float32)
    if mode=='shuffled':
        weights=np.random.default_rng(seed).permutation(weights)
    return weights, dict(center=center.tolist(), scale=scale.tolist(), mean=float(weights.mean()), minimum=float(weights.min()), maximum=float(weights.max()), log_shift=(lo+hi)/2)


def policy_saliency(model, features):
    """Frobenius norm of teacher action Jacobian wrt standardized features.

    A simple sensitivity-inspired comparator, not a reproduction of SQIL.
    """
    x=torch.as_tensor(features,dtype=torch.float32).detach().requires_grad_(True)
    out=model(x); squared=torch.zeros(len(x))
    for j in range(4):
        grad=torch.autograd.grad(out[:,j].sum(),x,retain_graph=j<3)[0]
        squared+=(grad*model.std).square().sum(1)
    return squared.sqrt().detach().numpy()
