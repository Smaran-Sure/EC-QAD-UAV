"""Reproducible study: collect, teacher, students, evaluate, report."""
from __future__ import annotations
import os
os.environ.setdefault('JSBSIM_DEBUG','0')
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ecqad-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from ecqad_uav.utils.config import load_yaml, project_root
from ecqad_uav.sim.runner import SimulationRunner
from ecqad_uav.sim.metrics import summarize
from ecqad_uav.utils.math_utils import wrap_pi

ROOT=project_root(); OUT=ROOT/'results'; CFG=load_yaml('study.yaml')
CONDITIONS=CFG['conditions']


def scenario(split, condition, episode):
    spec=CFG[split]
    seed=spec['seed_base']+1000*CONDITIONS.index(condition)+episode
    rng=np.random.default_rng(seed)
    ac,sc,ec,cc,fc=[load_yaml(n+'.yaml') for n in ['aircraft','sensors','estimator','controller','faults']]
    initial=ac['simulation']['initial']
    initial['altitude_m']+=float(rng.uniform(-10,10))
    initial['calibrated_airspeed_mps']+=float(rng.uniform(-1,1))
    initial['heading_deg']+=float(rng.uniform(-5,5))
    sc['seed']=seed
    for f in fc['faults'].values(): f['enabled']=False
    onset=float(rng.uniform(*spec['onset'])); severity=0.; duration=0.
    if condition!='nominal':
        f=fc['faults'][condition]; f['enabled']=True; f['start_s']=onset
        if condition=='gyro_bias':
            severity=float(rng.uniform(*spec['gyro_deg_s'])*rng.choice([-1,1])); f['magnitude_deg_s']=severity
            f['axis']=str(rng.choice(['x','y','z'])); f['end_s']=145.
        elif condition=='gps_dropout':
            duration=float(rng.uniform(*spec['dropout_s'])); severity=duration; f['end_s']=onset+duration
        else:
            severity=float(rng.uniform(*spec['airspeed_mps'])*rng.choice([-1,1])); f['magnitude_mps']=severity; f['end_s']=145.
    if condition!='nominal': duration=float(fc['faults'][condition]['end_s']-onset)
    params=dict(reference_scale=float(rng.uniform(.8,1.2)),reference_time_scale=float(rng.uniform(.93,1.07)))
    meta=dict(split=split,condition=condition,episode=episode,episode_seed=seed,onset_s=onset,severity=severity,duration_s=duration,episode_duration_s=CFG['duration_s'],fault_end_s=(onset+duration if condition!='nominal' else None),gyro_axis=(fc['faults']['gyro_bias']['axis'] if condition=='gyro_bias' else ''),**params)
    return [ac,sc,ec,cc,fc],params,meta


def save_trace(path,d,logger=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    numeric=d.select_dtypes(include='number')
    arrays=dict(data=numeric.to_numpy(),columns=np.array(numeric.columns, dtype=str))
    if logger is not None and logger.covariances:
        arrays['P']=np.array(logger.covariances)
        # Ragged innovations packed into fixed maximum dimension; sensor and dof preserve meaning.
        recs=logger.innovation_records
        nu=np.full((len(recs),6),np.nan); S=np.full((len(recs),6,6),np.nan)
        for i,r in enumerate(recs): nu[i,:r.dof]=r.innovation; S[i,:r.dof,:r.dof]=r.S
        arrays.update(innovation=nu,innovation_covariance=S,innovation_t=np.array([r.t for r in recs]),innovation_sensor=np.array([r.sensor for r in recs]),innovation_dof=np.array([r.dof for r in recs]),innovation_nis=np.array([r.nis for r in recs]))
    np.savez_compressed(path,**arrays)


def rollout(job):
    split,condition,episode,model_path,teacher_path,method,bits,train_seed,save=job
    configs,params,meta=scenario(split.replace('_policy',''),condition,episode)
    policy=observer=None
    if model_path:
        from ecqad_uav.learning.policy import Policy
        policy=Policy.load(model_path).numpy_policy()
    if teacher_path:
        from ecqad_uav.learning.policy import Policy
        observer=Policy.load(teacher_path).numpy_policy()
    logger,runmeta=SimulationRunner(*configs,control_mode='estimated_pid',duration_s=CFG['duration_s'],policy=policy,observer=observer,log_hz=CFG['log_hz'],record_arrays=save,**params).run()
    d=logger.dataframe()
    metrics=summarize(d)
    eh=(d.cmd_altitude_m-d.truth_altitude_m).to_numpy()/20
    ey=wrap_pi(d.cmd_heading_rad.to_numpy()-d.truth_yaw_rad.to_numpy())/np.radians(20)
    ev=(d.cmd_airspeed_mps-d.truth_airspeed_mps).to_numpy()/3
    score=(eh**2+ey**2+ev**2)/3
    fraction=min(runmeta['actual_duration_s']/CFG['duration_s'],1)
    metrics.update(tracking_score=float(np.sqrt(fraction*np.mean(score)+(1-fraction)*100)),
                   constraint_fraction=float(d.constraint_violation.mean()),failed=int(not runmeta['completed']),
                   completed_s=runmeta['actual_duration_s'],control_effort=float(np.square(d[['u_aileron','u_elevator','u_rudder']]).mean().mean()))
    action=d[['u_aileron','u_elevator','u_rudder','u_throttle']].to_numpy()
    target=d[[f'teacher_{i}' for i in range(4)]].to_numpy()
    discrepancy=np.linalg.norm(action-target,axis=1)
    saturation=(np.abs(action[:,:3])>=.99).any(axis=1)|(action[:,3]<=.01)|(action[:,3]>=.99)
    edges=np.diff(np.r_[False,saturation,False].astype(int))
    lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    metrics['saturation_fraction']=float(saturation.mean())
    metrics['max_saturation_duration_s']=float(lengths.max()/CFG['log_hz']) if len(lengths) else 0.
    metrics['action_rmse']=float(np.sqrt(np.mean(np.square(action-target))))
    result={**meta,**metrics,'policy_input_schema':runmeta['policy_input_schema'],'pid_executed':runmeta['pid_executed'],'method':method,'bits':bits,'training_seed':train_seed}
    ident=f'{method}_{bits}_seed{train_seed}_{condition}_{episode:03d}'
    if split in ['train','validation']:
        path=OUT/'data'/split/f'{condition}_{episode:03d}.npz'
        save_trace(path,d,logger if save else None)
        path.with_suffix('.json').write_text(json.dumps({**result,'scenario_configs':configs},indent=2))
    elif save:
        save_trace(OUT/'traces'/f'{ident}.npz',d,logger)
    if split=='evaluation' and method in ('uniform','teacher'):
        # One row per second; futures never cross trajectory boundaries.
        future=pd.Series(score).rolling(int(5*CFG['log_hz']),min_periods=1).mean().shift(-int(5*CFG['log_hz'])+1).to_numpy()
        signal=pd.DataFrame(dict(time_s=d.time_s,covariance=d.covariance_score,nis=np.log1p(d.nis_normalized_ewma),action_error=discrepancy,future_tracking=future,tracking=score,
                                 fault_active=(d.fault_labels!='').astype(int),roll_abs=abs(d.est_roll_rad),pitch_abs=abs(d.est_pitch_rad)))
        for j in range(16): signal[f'feature_{j}']=d[f'feature_{j}']
        signal=signal.iloc[::int(CFG['log_hz'])].copy()
        signal['method']=method
        for k in ['condition','episode','episode_seed','training_seed','bits']: signal[k]=result[k]
        folder=OUT/'signals'; folder.mkdir(exist_ok=True)
        signal.to_csv(folder/f'{ident}.csv',index=False)
    return result


def collect(workers):
    jobs=[]
    for split,key in [('train','train_episodes_per_condition'),('validation','validation_episodes_per_condition')]:
        for condition in CONDITIONS:
            for ep in range(CFG[key]):
                if (OUT/'data'/split/f'{condition}_{ep:03d}.npz').exists(): continue
                jobs.append((split,condition,ep,None,None,'pid',32,0,ep==0))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(rollout,jobs):
            print('collect',r['split'],r['condition'],r['episode'],'score',round(r['tracking_score'],3),'failed',r['failed'],flush=True)
    rows=[json.loads(p.read_text()) for p in (OUT/'data').glob('*/*.json')]
    for r in rows: r.pop('scenario_configs',None)
    pd.DataFrame(rows).to_csv(OUT/'conventional_metrics.csv',index=False)


def dataset(split):
    xs=[]; ys=[]; cs=[]; ns=[]; labels=[]
    for p in sorted((OUT/'data'/split).glob('*.npz')):
        a=np.load(p); d=pd.DataFrame(a['data'],columns=a['columns'])
        xs.append(d[[f'feature_{i}' for i in range(16)]].to_numpy(np.float32))
        ys.append(d[[f'pid_{i}' for i in range(4)]].to_numpy(np.float32))
        cs.append(d.covariance_score.to_numpy()); ns.append(d.nis_normalized_ewma.to_numpy())
        meta=json.loads(p.with_suffix('.json').read_text())
        active=(d.time_s>=meta['onset_s']) & (d.time_s< (meta['onset_s']+meta['duration_s'] if meta['condition']=='gps_dropout' else 145))
        labels.extend(np.where(active,meta['condition'],'nominal'))
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(cs),np.concatenate(ns),np.array(labels)


def train(seed,students=False,ablations=False):
    import torch
    from ecqad_uav.learning.policy import Policy,conditioning_weights,policy_saliency
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    X,Y,C,N,L=dataset('train'); VX,VY,*_=dataset('validation')
    x,y,vx,vy=map(torch.from_numpy,(X,Y,VX,VY))
    ck=OUT/'checkpoints'/f'seed{seed}'; ck.mkdir(parents=True,exist_ok=True)
    teacher_path=(OUT/'checkpoints'/f"seed{CFG['teacher_seed']}"/'teacher.pt') if students else ck/'teacher.pt'
    runs=[]
    if students:
        teacher=Policy.load(teacher_path)
        with torch.no_grad(): y=teacher(x); vy=teacher(vx)
        modes=CFG['methods'] if ablations else ['uniform','ecqad']
        tasks=[(b,m) for b in CFG['bits'] for m in modes if m in ['uniform','ecqad'] or b==4]
    else: tasks=[(32,'teacher')]
    for bits,method in tasks:
        path=teacher_path if method=='teacher' else ck/f'{method}_{bits}.pt'
        if path.exists(): print('already trained',path,flush=True); continue
        torch.manual_seed(seed)  # matched initialization and minibatch order across QAD methods
        model=Policy(bits,CFG['width'])
        if students: model.load_state_dict(teacher.state_dict())
        else:
            model.mean.copy_(x.mean(0)); model.std.copy_(x.std(0).clamp(min=.05))
            model.out_mean.copy_(y.mean(0)); model.out_scale.copy_(y.std(0).clamp(min=.05))
        saliency=policy_saliency(teacher,X) if method=='saliency' else None
        weights,wmeta=conditioning_weights(C,N,'uniform' if method=='teacher' else method,seed,L,saliency)
        w=torch.from_numpy(weights)
        optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
        epochs=CFG['student_epochs'] if students else CFG['teacher_epochs']
        start=time.monotonic(); best=float('inf'); best_state=None
        for epoch in range(epochs):
            model.train(); order=torch.randperm(len(x)); total=0.
            for idx in order.split(CFG['batch_size']):
                optimizer.zero_grad()
                # Common actuator-space MSE for all QAD variants, mean weight 1 over training population.
                pred=model(x[idx]); loss=((pred-y[idx]).square().mean(1)*w[idx]).mean()
                loss.backward(); optimizer.step(); total+=float(loss.detach())*len(idx)
            model.eval()
            with torch.no_grad(): val=float((model(vx)-vy).square().mean())
            if val<best:
                best=val; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if epoch%30==0 or epoch==epochs-1: print(method,bits,'seed',seed,'epoch',epoch,'train',total/len(x),'validation',val,flush=True)
        model.load_state_dict(best_state)
        model.save(path,seed=seed,teacher_seed=CFG['teacher_seed'],method=method,weighting=wmeta,validation_mse=best,epochs=epochs)
        runs.append(dict(training_seed=seed,method=method,bits=bits,epochs=epochs,training_samples=len(x),validation_samples=len(vx),validation_action_mse=best,elapsed_s=time.monotonic()-start,storage_bytes=model.storage_bytes(),**{f'weight_{k}':v for k,v in wmeta.items() if isinstance(v,(int,float))}))
        tablepath=OUT/'training_runs.csv'
        old=pd.read_csv(tablepath) if tablepath.exists() else pd.DataFrame()
        pd.concat([old,pd.DataFrame([runs[-1]])],ignore_index=True).to_csv(tablepath,index=False)
    if students:
        # PTQ is a zero-optimization comparator with identical ranges/architecture.
        for bits in CFG['bits']:
            m=Policy(bits,CFG['width']); m.load_state_dict(teacher.state_dict()); m.save(ck/f'ptq_{bits}.pt',seed=seed,method='ptq')


def evaluate(seed,workers,episodes,ablations,tag="",methods=None):
    ck=OUT/'checkpoints'/f'seed{seed}'; teacher=OUT/'checkpoints'/f"seed{CFG['teacher_seed']}"/'teacher.pt'
    models=[('teacher',32,teacher)] if seed==CFG['teacher_seed'] else []
    if seed==7: models.append(('pid',32,None))
    for bits in CFG['bits']:
        for method in (CFG['methods']+['ptq'] if ablations else ['uniform','ecqad']):
            if method not in ['uniform','ecqad','ptq'] and bits!=4: continue
            path=ck/f'{method}_{bits}.pt'
            if path.exists(): models.append((method,bits,path))
    if methods: models=[item for item in models if item[0] in methods.split(',')]
    path=OUT/f'{tag}per_episode_metrics.csv'
    old=pd.read_csv(path) if path.exists() else pd.DataFrame()
    done=set(zip(old.training_seed,old.method,old.bits,old.condition,old.episode)) if len(old) else set()
    jobs=[]
    for method,bits,model in models:
        for condition in CONDITIONS:
            for ep in range(episodes):
                if (seed,method,bits,condition,ep) in done: continue
                jobs.append(('evaluation',condition,ep,str(model) if model else None,str(teacher),method,bits,seed,ep==0))
    jobs.sort(key=lambda j: (j[2], CONDITIONS.index(j[1]), j[5], j[6]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,r in enumerate(pool.map(rollout,jobs)):
            old=pd.concat([old,pd.DataFrame([r])],ignore_index=True)
            old.to_csv(path,index=False)
            print('evaluate',i+1,'/',len(jobs),r['method'],r['bits'],r['condition'],r['episode'],'score',round(r['tracking_score'],3),'failed',r['failed'],flush=True)
    numeric=['altitude_rmse_m','heading_rmse_deg','airspeed_rmse_mps','tracking_score','constraint_fraction','failed','action_rmse']
    old.groupby(['training_seed','method','bits','condition'])[numeric].mean().reset_index().to_csv(OUT/f'{tag}metrics.csv',index=False)


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['collect','teacher','teacher_gate','deployment_gate','students','evaluate','report']); p.add_argument('--seed',type=int,default=7); p.add_argument('--workers',type=int,default=4); p.add_argument('--episodes',type=int,default=20); p.add_argument('--ablations',action='store_true'); p.add_argument('--tag',default=''); p.add_argument('--methods',default=None); args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage=='collect': collect(args.workers)
    elif args.stage=='teacher': train(args.seed)
    elif args.stage=='teacher_gate':
        teacher=OUT/'checkpoints'/f'seed{args.seed}'/'teacher.pt'
        jobs=[('validation_policy',c,e,str(teacher),None,'teacher',32,args.seed,False) for c in CONDITIONS for e in range(CFG['validation_episodes_per_condition'])]
        with ProcessPoolExecutor(max_workers=args.workers) as pool: rows=list(pool.map(rollout,jobs))
        result=pd.DataFrame(rows); result.to_csv(OUT/f'teacher_validation_seed{args.seed}.csv',index=False)
        base=pd.read_csv(OUT/'conventional_metrics.csv'); base=base[base.split=='validation']
        if result.pid_executed.any() or set(result.policy_input_schema)!={'physical_reference_v2'}:
            raise RuntimeError('Neural deployment interface gate failed')
        ratio=result.tracking_score.mean()/base.tracking_score.mean()
        print('teacher validation score ratio to PID',ratio,'failures',result.failed.sum(),flush=True)
        if result.failed.sum() or ratio>1.5: raise RuntimeError('Teacher closed-loop validation gate failed')
    elif args.stage=='deployment_gate':
        ck=OUT/'checkpoints'/f'seed{args.seed}'
        jobs=[('validation_policy','nominal',0,str(ck/f'{m}_{b}.pt'),str(OUT/'checkpoints'/f"seed{CFG['teacher_seed']}"/'teacher.pt'),m,b,args.seed,False)
              for m in ('uniform','ecqad') for b in CFG['bits']]
        with ProcessPoolExecutor(max_workers=args.workers) as pool: rows=list(pool.map(rollout,jobs))
        result=pd.DataFrame(rows)
        result.to_csv(OUT/f'deployment_validation_seed{args.seed}.csv',index=False)
        if result.pid_executed.any() or set(result.policy_input_schema)!={'physical_reference_v2'}:
            raise RuntimeError('Student deployment interface gate failed')
        print(result[['method','bits','completed_s','failed','tracking_score']].to_string(index=False),flush=True)
    elif args.stage=='students': train(args.seed,True,args.ablations)
    elif args.stage=='evaluate': evaluate(args.seed,args.workers,args.episodes,args.ablations,args.tag,args.methods)
    else:
        from ecqad_uav.plots.study_report import report
        report()

if __name__=='__main__': main()
