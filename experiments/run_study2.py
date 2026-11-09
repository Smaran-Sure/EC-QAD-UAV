"""Isolated, resumable Study 2. Never writes Study 1 artifacts."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('JSBSIM_DEBUG','0')
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ecqad-mpl')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, hashlib, json, time, shutil
import numpy as np
import pandas as pd
import torch
from ecqad_uav.utils.config import project_root,load_yaml
from ecqad_uav.utils.math_utils import wrap_pi
from ecqad_uav.sim.runner import SimulationRunner
from ecqad_uav.sim.metrics import summarize
from ecqad_uav.learning.study2 import BoundedPolicy,uncertainty_diagnostics
from ecqad_uav.learning.policy import conditioning_weights
from run_study import save_trace
ROOT=project_root(); OUT=ROOT/'results/study2'; CFG=load_yaml('study2.yaml')
CONDITIONS=CFG['conditions']


def scenario_ids(split):
    spec=CFG[split]
    return {spec['seed_base']+1000*i+e for i in range(len(CONDITIONS)) for e in range(spec['episodes_per_condition'])}


def scenario(split,condition,episode):
    if split not in ('train','validation_select','validation_gate','validation_signal','test'):raise ValueError(split)
    spec=CFG[split]
    if not 0<=episode<spec['episodes_per_condition']:raise ValueError('Scenario outside preregistered set')
    seed=spec['seed_base']+1000*CONDITIONS.index(condition)+episode;rng=np.random.default_rng(seed)
    ac,sc,ec,cc,fc=[load_yaml(n+'.yaml') for n in ['aircraft','sensors','estimator','controller','faults']]
    initial=ac['simulation']['initial'];initial['altitude_m']+=float(rng.uniform(-10,10));initial['calibrated_airspeed_mps']+=float(rng.uniform(-1,1));initial['heading_deg']+=float(rng.uniform(-5,5))
    sc['seed']=seed
    for f in fc['faults'].values():f['enabled']=False
    onset=float(rng.uniform(*spec['onset']));severity=duration=0.;axis=''
    if condition!='nominal':
        f=fc['faults'][condition];f['enabled']=True;f['start_s']=onset
        if condition=='gyro_bias':
            severity=float(rng.uniform(*spec['gyro_deg_s'])*rng.choice([-1,1]));f['magnitude_deg_s']=severity
            axis=str(rng.choice(['x','y','z']));f['axis']=axis;f['end_s']=145.
        elif condition=='gps_dropout':
            severity=float(rng.uniform(*spec['dropout_s']));f['end_s']=onset+severity
        else:
            severity=float(rng.uniform(*spec['airspeed_mps'])*rng.choice([-1,1]));f['magnitude_mps']=severity;f['end_s']=145.
        duration=f['end_s']-onset
    params=dict(reference_scale=float(rng.uniform(.8,1.2)),reference_time_scale=float(rng.uniform(.93,1.07)))
    meta=dict(split=split,condition=condition,episode=episode,episode_seed=seed,onset_s=onset,severity=severity,duration_s=duration,gyro_axis=axis,**params)
    return [ac,sc,ec,cc,fc],params,meta


def selection_jobs(path,method,seed,epoch):
    # No split argument: callers cannot request TEST for checkpoint selection.
    return [dict(split='validation_select',condition=c,episode=e,path=str(path),method=method,seed=seed,epoch=epoch,save=False)
            for c in CONDITIONS for e in range(CFG['validation_select']['episodes_per_condition'])]


def primary_teacher():
    p=OUT/'primary_teacher.json'
    if not p.exists():raise RuntimeError('Teacher gate must finish before student training')
    return Path(json.loads(p.read_text())['path'])


def rollout(job):
    torch.set_num_threads(1)
    split,c,ep=job['split'],job['condition'],job['episode']
    if split=='test' and not (OUT/'test_release.json').exists():raise RuntimeError('TEST locked until signal/method decisions frozen')
    configs,params,meta=scenario(split,c,ep)
    policy=BoundedPolicy.load(job['path']).numpy_policy() if job.get('path') else None
    observer=BoundedPolicy.load(job['teacher']).numpy_policy() if job.get('teacher') else None
    save=job.get('save',False)
    log,run=SimulationRunner(*configs,control_mode='estimated_pid',duration_s=CFG['duration_s'],policy=policy,observer=observer,log_hz=CFG['log_hz'],record_arrays=save,**params).run()
    d=log.dataframe();m=summarize(d)
    err=np.column_stack(((d.cmd_altitude_m-d.truth_altitude_m)/20,wrap_pi(d.cmd_heading_rad.to_numpy()-d.truth_yaw_rad.to_numpy())/np.radians(20),(d.cmd_airspeed_mps-d.truth_airspeed_mps)/3))
    track=(err**2).mean(1);fraction=min(run['actual_duration_s']/CFG['duration_s'],1)
    action=d[['u_aileron','u_elevator','u_rudder','u_throttle']].to_numpy();target=d[[f'teacher_{i}' for i in range(4)]].to_numpy()
    m.update(tracking_score=float(np.sqrt(fraction*track.mean()+(1-fraction)*100)),constraint_fraction=float(d.constraint_violation.mean()),failed=int(not run['completed']),completed_s=run['actual_duration_s'],control_effort=float(np.square(action[:,:3]).mean()),action_rmse=float(np.sqrt(np.square(action-target).mean())))
    result={**meta,**m,'method':job['method'],'training_seed':job['seed'],'epoch':job.get('epoch',0),'bits':32 if job['method'] in ('teacher','pid') else 4,'pid_executed':run['pid_executed'],'policy_input_schema':run['policy_input_schema']}
    ident=f"{job['method']}_seed{job['seed']}_epoch{job.get('epoch',0)}_{c}_{ep:03d}"
    if save:
        folder=OUT/'data'/split if job['method']=='pid' else OUT/'traces'/split
        name=f'{c}_{ep:03d}' if job['method']=='pid' else ident
        save_trace(folder/f'{name}.npz',d,log)
        (folder/f'{name}.json').write_text(json.dumps(result,indent=2))
    if job.get('signal'):
        # Full five-second future only. No fabricated post-termination futures.
        n=int(CFG['signal']['horizon_s']*CFG['log_hz'])
        future=pd.Series(track).rolling(n,min_periods=n).mean().shift(-n+1).to_numpy()
        idx=np.arange(0,len(d),int(CFG['log_hz']))
        f=d[[f'feature_{i}' for i in range(16)]].to_numpy(np.float32)[idx]
        p=np.asarray(log.covariances)[idx]
        teacher=BoundedPolicy.load(job['teacher']);diag=uncertainty_diagnostics(teacher,f,p)
        sig=pd.DataFrame(dict(time_s=d.time_s.to_numpy()[idx],covariance=d.covariance_score.to_numpy()[idx],nis=np.log1p(d.nis_normalized_ewma.to_numpy()[idx]),action_error=np.linalg.norm(action-target,axis=1)[idx],tracking=track[idx],future_tracking=future[idx],q=diag['q'],trace_P=diag['trace_P'],sensitivity=diag['sensitivity'],constraint=d.constraint_violation.to_numpy()[idx]))
        for j in range(4):sig[f'action_variance_{j}']=diag['action_variance'][:,j]
        for j in range(16):sig[f'feature_{j}']=f[:,j]
        for k in ['condition','episode','episode_seed','split','method','training_seed','severity','onset_s','duration_s','gyro_axis']:sig[k]=result[k]
        folder=OUT/'signals'/split;folder.mkdir(parents=True,exist_ok=True);sig.to_csv(folder/f'{ident}.csv',index=False)
        result['minimum_propagated_eigenvalue']=float(min(diag['min_feature_eigenvalue'].min(),diag['min_action_eigenvalue'].min()))
    return result


def run_jobs(jobs,workers,output):
    old=pd.read_csv(output) if output.exists() else pd.DataFrame()
    keys=['split','method','training_seed','epoch','condition','episode']
    done=set(old[keys].itertuples(index=False,name=None)) if len(old) else set()
    jobs=[j for j in jobs if (j['split'],j['method'],j['seed'],j.get('epoch',0),j['condition'],j['episode']) not in done]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,r in enumerate(pool.map(rollout,jobs)):
            old=pd.concat([old,pd.DataFrame([r])],ignore_index=True);output.parent.mkdir(parents=True,exist_ok=True);old.to_csv(output,index=False)
            print('rollout',i+1,'/',len(jobs),r['split'],r['method'],r['training_seed'],r['epoch'],r['condition'],r['episode'],round(r['tracking_score'],4),'failed',r['failed'],flush=True)
    return old


def collect(workers):
    jobs=[dict(split=s,condition=c,episode=e,path=None,method='pid',seed=0,save=True) for s in ('train','validation_select','validation_gate') for c in CONDITIONS for e in range(CFG[s]['episodes_per_condition'])]
    run_jobs(jobs,workers,OUT/'expert_metrics.csv')


def dataset(split):
    if split not in ('train','validation_select'):raise ValueError('Training data access limited to TRAIN and offline VALIDATION_SELECT')
    X=[];Y=[];C=[];N=[];L=[]
    for p in sorted((OUT/'data'/split).glob('*.npz')):
        a=np.load(p);d=pd.DataFrame(a['data'],columns=a['columns']);meta=json.loads(p.with_suffix('.json').read_text())
        X.append(d[[f'feature_{j}' for j in range(16)]].to_numpy(np.float32));Y.append(d[[f'pid_{j}' for j in range(4)]].to_numpy(np.float32));C.append(d.covariance_score.to_numpy());N.append(d.nis_normalized_ewma.to_numpy())
        active=(d.time_s>=meta['onset_s'])&(d.time_s<meta['onset_s']+meta['duration_s'])
        L.extend(np.where(active,meta['condition'],'nominal'))
    if not X:raise RuntimeError('Run collect first')
    return np.concatenate(X),np.concatenate(Y),np.concatenate(C),np.concatenate(N),np.array(L)


def train(seed,method,workers):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    if method not in ('teacher','uniform'):raise ValueError('Only teacher and uniform QAD allowed before the signal gate')
    students=method!='teacher';bits=4 if students else 32
    folder=OUT/'checkpoints'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    selected=folder/f'{method}.pt'
    if selected.exists():print('Already selected',selected,flush=True);return
    X,Y,C,N,L=dataset('train');VX,VY,*_=dataset('validation_select')
    x,y,vx,vy=map(torch.from_numpy,(X,Y,VX,VY));model=BoundedPolicy(bits,CFG['width'])
    if students:
        teacher=BoundedPolicy.load(primary_teacher());model.load_state_dict(teacher.state_dict())
        with torch.no_grad():y=teacher(x);vy=teacher(vx)
    else:
        model.mean.copy_(x.mean(0));model.std.copy_(x.std(0).clamp(min=.05))
        latent=y.clone();latent[:,3]=2*latent[:,3]-1;latent=torch.atanh(latent.clamp(-.999,.999))
        model.out_mean.copy_(latent.mean(0));model.out_scale.copy_(latent.std(0).clamp(min=.05))
    optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
    epochs=CFG['student_epochs' if students else 'teacher_epochs'];candidates=CFG['student_candidates' if students else 'teacher_candidates']
    records=[]
    # A completed candidate restores model AND optimizer AND minibatch RNG exactly.
    resume=folder/f'{method}_resume.pt';start=0
    if resume.exists():
        state=torch.load(resume,weights_only=True);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']);start=state['epoch'];records=state['records']
    for epoch in range(start+1,epochs+1):
        model.train();order=torch.randperm(len(x));loss_sum=0.
        for idx in order.split(CFG['batch_size']):
            optimizer.zero_grad();loss=(model(x[idx])-y[idx]).square().mean();loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(idx)
        if epoch%25==0:print('train',method,seed,epoch,'loss',loss_sum/len(x),flush=True)
        if epoch in candidates:
            model.eval()
            with torch.no_grad():mse=float((model(vx)-vy).square().mean())
            path=folder/f'{method}_epoch{epoch}.pt';model.save(path,seed=seed,method=method,epoch=epoch,offline_action_mse=mse)
            r=run_jobs(selection_jobs(path,method,seed,epoch),workers,OUT/'selection_rollouts.csv')
            r=r[(r.method==method)&(r.training_seed==seed)&(r.epoch==epoch)]
            record=dict(method=method,training_seed=seed,epoch=epoch,offline_action_mse=mse,closed_loop_validation_score=float(r.tracking_score.mean()),failures=int(r.failed.sum()),path=str(path))
            records.append(record)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),epoch=epoch,records=records),resume)
            table=ROOT/'results/study2_checkpoint_selection.csv';old=pd.read_csv(table) if table.exists() else pd.DataFrame();old=pd.concat([old,pd.DataFrame([record])],ignore_index=True).drop_duplicates(['method','training_seed','epoch'],keep='last');old.to_csv(table,index=False)
    best=min(records,key=lambda r:(r['closed_loop_validation_score'],r['epoch']))
    shutil.copy2(best['path'],selected)
    print('SELECTED',method,seed,best,flush=True)


def teacher_gate(workers):
    jobs=[dict(split='validation_gate',condition=c,episode=e,path=str(OUT/'checkpoints'/f'seed{s}'/'teacher.pt'),method='teacher',seed=s,save=False) for s in CFG['teacher_seeds'] for c in CONDITIONS for e in range(CFG['validation_gate']['episodes_per_condition'])]
    d=run_jobs(jobs,workers,OUT/'teacher_gate_rollouts.csv');expert=pd.read_csv(OUT/'expert_metrics.csv');expert=expert[expert.split=='validation_gate'];rows=[]
    for s in CFG['teacher_seeds']:
        g=d[d.training_seed==s];r=dict(training_seed=s,failures=int(g.failed.sum()))
        for regime,mask in [('nominal',g.condition=='nominal'),('off_nominal',g.condition!='nominal')]:
            score=g.loc[mask,'tracking_score'].mean();base=expert.loc[(expert.condition=='nominal') if regime=='nominal' else (expert.condition!='nominal'),'tracking_score'].mean()
            r[regime+'_score']=score;r[regime+'_expert_score']=base;r[regime+'_ratio']=score/base
        r['passed']=bool(r['failures']==0 and r['nominal_ratio']<=CFG['teacher_gate_ratio'] and r['off_nominal_ratio']<=CFG['teacher_gate_ratio'] and not g.pid_executed.any());rows.append(r)
    table=pd.DataFrame(rows);table.to_csv(ROOT/'results/study2_teacher_gate.csv',index=False);print(table.to_string(index=False),flush=True)
    passed=[r['training_seed'] for r in rows if r['passed']]
    if not passed:raise RuntimeError('No Study 2 teacher passed predefined gate; do not train students')
    (OUT/'primary_teacher.json').write_text(json.dumps(dict(seed=passed[0],path=str(OUT/'checkpoints'/f'seed{passed[0]}'/'teacher.pt'),rule=CFG['primary_teacher']),indent=2))


def signals(workers):
    teacher=primary_teacher();ts=json.loads((OUT/'primary_teacher.json').read_text())['seed'];models=[('teacher',ts,teacher)]+[('uniform',s,OUT/'checkpoints'/f'seed{s}'/'uniform.pt') for s in CFG['student_seeds']]
    jobs=[dict(split='validation_signal',condition=c,episode=e,path=str(p),teacher=str(teacher),method=m,seed=s,save=True,signal=True) for e in range(CFG['validation_signal']['episodes_per_condition']) for c in CONDITIONS for m,s,p in models]
    run_jobs(jobs,workers,ROOT/'results/study2_evaluation.csv')


def freeze_manifest():
    paths=[Path('configs/study2.yaml'),Path('docs/STUDY2_PROTOCOL.md'),Path('experiments/run_study2.py'),Path('src/ecqad_uav/learning/study2.py')]
    paths += [p for base in ['src','models','configs'] for p in Path(base).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    m={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    path=OUT/'protocol_manifest.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['sha256']==m,'Executed code/config differs from frozen Study 2 protocol'
    else:path.write_text(json.dumps(dict(sha256=m,study1_tag='study1-complete-20261109'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','collect','teachers','teacher_gate','students','signals','analyze']);p.add_argument('--workers',type=int,default=6);p.add_argument('--seed',type=int);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    freeze_manifest()
    if a.stage=='collect':collect(a.workers)
    elif a.stage=='teachers':
        for seed in ([a.seed] if a.seed is not None else CFG['teacher_seeds']):train(seed,'teacher',a.workers)
    elif a.stage=='teacher_gate':teacher_gate(a.workers)
    elif a.stage=='students':
        for seed in ([a.seed] if a.seed is not None else CFG['student_seeds']):train(seed,'uniform',a.workers)
    elif a.stage=='signals':signals(a.workers)
    elif a.stage=='analyze':
        from ecqad_uav.plots.study2_report import analyze
        analyze()

if __name__=='__main__':main()
