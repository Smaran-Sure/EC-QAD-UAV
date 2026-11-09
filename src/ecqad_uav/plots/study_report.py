"""Figures and conservative, episode-aware analysis of executed study data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from ecqad_uav.utils.config import project_root, load_yaml

ROOT=project_root(); OUT=ROOT/'results'; FIG=OUT/'figures'
DISPLAY_NAMES={'teacher':'FP32 teacher','uniform':'Uniform QAD','ecqad':'EC-QAD-v1 combined','covariance':'Covariance-only EC-QAD','innovation':'Innovation-only EC-QAD','shuffled':'Shuffled combined weights','covariance_shuffled':'Shuffled covariance weights','fault_balanced':'Fault-balanced QAD','balanced':'Bounded fault balancing','saliency':'Policy-saliency comparator','ptq':'PTQ','pid':'PID'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'pdf.fonttype':42})


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True); fig.tight_layout()
    fig.savefig(FIG/(name+'.png')); fig.savefig(FIG/(name+'.pdf')); plt.close(fig)


def trace(path):
    a=np.load(path); return pd.DataFrame(a['data'],columns=a['columns']),a


def plot_baselines():
    p=OUT/'data/train/nominal_000.npz'
    if not p.exists(): return
    d,a=trace(p)
    specs=[('altitude_m',1,'Altitude (m)'),('yaw_rad',180/np.pi,'Heading (deg)'),('airspeed_mps',1,'Airspeed (m/s)')]
    fig,axes=plt.subplots(3,1,figsize=(7,6),sharex=True)
    for ax,(state,scale,label) in zip(axes,specs):
        ref='heading_rad' if state=='yaw_rad' else state
        ax.plot(d.time_s,d['cmd_'+ref]*scale,'k--',lw=1,label='Reference'); ax.plot(d.time_s,d['truth_'+state]*scale,lw=1,label='Estimated-state PID'); ax.set_ylabel(label)
    axes[0].legend(); axes[-1].set_xlabel('Time (s)'); save(fig,'conventional_tracking')
    fig,axes=plt.subplots(3,1,figsize=(7,6),sharex=True)
    for ax,(state,idx,scale,label) in zip(axes,[('pd_m',2,1,'Down position (m)'),('roll_rad',6,180/np.pi,'Roll (deg)'),('pitch_rad',7,180/np.pi,'Pitch (deg)')]):
        est=d['est_'+state]*scale; sigma=3*np.sqrt(d[f'P_{idx:02d}_{idx:02d}'])*scale
        ax.fill_between(d.time_s,est-sigma,est+sigma,alpha=.2,label='Estimate ±3σ'); ax.plot(d.time_s,d['truth_'+state]*scale,'k',lw=.8,label='Truth'); ax.plot(d.time_s,est,lw=.8,label='UKF'); ax.set_ylabel(label)
    axes[0].legend(ncol=3); axes[-1].set_xlabel('Time (s)'); save(fig,'estimator_envelopes')
    g,_=trace(OUT/'data/train/gps_dropout_000.npz'); meta=json.loads((OUT/'data/train/gps_dropout_000.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(7,4.5),sharex=True)
    axes[0].plot(g.time_s,g[[f'P_{i:02d}_{i:02d}' for i in range(3)]].sum(1)); axes[0].set_ylabel('Position covariance trace (m²)')
    axes[1].plot(g.time_s,g.nis_normalized_ewma); axes[1].axhline(1,color='k',ls='--',lw=.7); axes[1].set_ylabel('EWMA NIS / dof'); axes[1].set_xlabel('Time (s)')
    for ax in axes: ax.axvspan(meta['onset_s'],meta['onset_s']+meta['duration_s'],alpha=.15,color='grey')
    save(fig,'estimator_degradation')
    # Quantitative estimator checks, including all eigenvalues of saved P.
    from ecqad_uav.sensors.sensor_suite import SensorSuite
    from ecqad_uav.utils.math_utils import wrap_pi
    meta_nom=json.loads((OUT/'data/train/nominal_000.json').read_text())
    sensor=SensorSuite(meta_nom['scenario_configs'][1],200)
    statistics={}
    pairs=[('meas_heading_rad','truth_yaw_rad',0.),('meas_airspeed_mps','truth_airspeed_mps',sensor.airspeed_bias0)]
    pairs += [(f'meas_gyro_{axis}_rad_s',f'truth_{body}_rad_s',sensor.gyro_bias0[i]) for i,(axis,body) in enumerate(zip('xyz','pqr'))]
    pairs += [(f'meas_gps_p{axis}_m',f'truth_{pos}_m',0.) for axis,pos in zip('ned',['pn','pe','pd'])]
    for measured,truth,bias in pairs:
        err=(d[measured]-d[truth]).dropna().to_numpy()-bias
        if 'heading' in measured: err=wrap_pi(err)
        statistics[measured]=dict(mean=float(err.mean()),std=float(err.std(ddof=1)),samples=len(err))
    (OUT/'sensor_validation.json').write_text(json.dumps(statistics,indent=2))
    checks={}
    for condition in ['nominal','gyro_bias','gps_dropout','airspeed_bias']:
        dd,aa=trace(OUT/f'data/train/{condition}_000.npz')
        checks[condition]={'covariance_min_eigenvalue':float(np.linalg.eigvalsh(aa['P']).min()),'covariance_finite':bool(np.isfinite(aa['P']).all())}
        for sensor in ['gps','heading','airspeed']:
            mask=aa['innovation_sensor']==sensor; nis=aa['innovation_nis'][mask]; nu=aa['innovation'][mask]
            checks[condition][sensor+'_innovation_mean']=np.nanmean(nu[:,:6 if sensor=='gps' else 1],axis=0).tolist()
            checks[condition][sensor+'_mean_nis_per_dof']=float(np.mean(nis/(6 if sensor=='gps' else 1)))
    t=g.time_s; pos=g[[f'P_{i:02d}_{i:02d}' for i in range(3)]].sum(1); start=meta['onset_s']; end=start+meta['duration_s']
    checks['gps_dropout']['position_trace_before_during_after']=[float(pos[(t>=start-5)&(t<start)].median()),float(pos[(t>=end-3)&(t<end)].median()),float(pos[(t>=end+5)&(t<end+10)].median())]
    (OUT/'estimator_validation.json').write_text(json.dumps(checks,indent=2))


def architecture():
    fig,ax=plt.subplots(figsize=(9,3.8)); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    blocks=[(.08,.74,'Nonlinear\n6-DoF UAV'),(.32,.74,'Simulated\nsensors'),(.56,.74,'Navigation UKF\nx̂, P, ν, S, NIS'),(.84,.74,'Flight policy\nphysical state + reference\n(no PID memory)'),(.25,.24,'Frozen FP32\nteacher'),(.56,.24,'Weighted action\nmatching loss'),(.84,.24,'Quantized student\n8 / 4 / 2 bit')]
    for x,y,label in blocks:
        ax.text(x,y,label,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.45',fc='white',ec='.3'))
    arrows=[((.16,.74),(.25,.74)),((.39,.74),(.45,.74)),((.67,.74),(.71,.74)),((.34,.24),(.45,.24)),((.67,.24),(.74,.24)),((.84,.36),(.84,.62))]
    for start,end in arrows:ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->'))
    ax.plot([.84,.84,.08],[.87,.96,.96],color='.2',lw=1)
    ax.annotate('',xy=(.08,.84),xytext=(.08,.96),arrowprops=dict(arrowstyle='->'))
    ax.text(.46,.97,'Actuator commands close the flight loop',ha='center',va='bottom',fontsize=8)
    ax.annotate('',xy=(.56,.36),xytext=(.56,.62),arrowprops=dict(arrowstyle='->',linestyle='--'))
    ax.text(.40,.49,'P / NIS weights\ntraining only',ha='center',fontsize=8)
    ax.text(.87,.48,'deploy weights',rotation=90,va='center',fontsize=8)
    ax.text(.70,.12,'optimize student',ha='center',fontsize=8)
    ax.text(.84,.02,'No P / NIS policy inputs',ha='center',fontsize=8)
    save(fig,'architecture')


def paired_error_interval(frame, baseline='baseline', enhanced='enhanced', draws=1000):
    """Hierarchical paired bootstrap of fixed out-of-fold episode errors."""
    seeds=sorted(frame.training_seed.unique())
    cells=frame.pivot(index=['condition','episode'],columns='training_seed',values=[baseline,enhanced]).dropna()
    groups=[]
    for _,g in cells.groupby(level='condition'):
        groups.append(np.stack([g[baseline][seeds].to_numpy(),g[enhanced][seeds].to_numpy()],axis=-1))
    rng=np.random.default_rng(20261109); ratios=[]
    for _ in range(draws):
        si=rng.integers(len(seeds),size=len(seeds)); values=[]
        for g in groups:
            ei=rng.integers(len(g),size=len(g)); values.append(g[ei][:,si].mean((0,1)))
        mean=np.mean(values,axis=0); ratios.append(1-mean[1]/max(mean[0],1e-15))
    return np.quantile(ratios,[.025,.975])


def signal_analysis():
    files=sorted((OUT/'signals').glob('*.csv'))
    if not files:return
    s=pd.concat([pd.read_csv(p) for p in files],ignore_index=True).dropna(subset=['future_tracking'])
    scope=pd.read_csv(OUT/'per_episode_metrics.csv')
    keys=['training_seed','method','bits','condition','episode']
    meta=['severity','onset_s','duration_s','gyro_axis','constraint_fraction']
    s=s.merge(scope[keys+meta].drop_duplicates(keys),on=keys,how='inner')
    teachers=s[s.method=='teacher'][['episode_seed','time_s','future_tracking']].rename(columns={'future_tracking':'teacher_future'})
    s=s[s.method=='uniform'].merge(teachers,on=['episode_seed','time_s'],how='left')
    s['excess_future_tracking']=s.future_tracking-s.teacher_future
    from ecqad_uav.learning.policy import Policy,policy_saliency
    import torch
    torch.set_num_threads(1)
    model=Policy.load(OUT/'checkpoints'/f"seed{load_yaml('study.yaml')['teacher_seed']}"/'teacher.pt')
    s['saliency']=policy_saliency(model,s[[f'feature_{j}' for j in range(16)]].to_numpy(np.float32))
    rows=[]
    for bits,group in s.groupby('bits'):
        for condition,g in [('all',group)]+list(group.groupby('condition')):
            for signal in ['covariance','nis']:
                for target in ['action_error','future_tracking','excess_future_tracking']:
                    rows.append(dict(kind='rank_correlation',bits=bits,condition=condition,signal=signal,target=target,spearman=float(spearmanr(g[signal],g[target],nan_policy='omit').statistic),episodes=g.episode_seed.nunique(),samples=len(g)))
        category=pd.get_dummies(group.condition).to_numpy(float)
        baseline=np.column_stack([pd.get_dummies(group.training_seed).to_numpy(float),category,
            pd.get_dummies(group.gyro_axis.fillna('none')).to_numpy(float),
            group.fault_active,group.time_s/180,(group.time_s/180)**2,group.onset_s/180,group.duration_s/180,
            group.roll_abs,group.pitch_abs,group.saliency,np.log1p(group.tracking),
            group[[f'feature_{i}' for i in [12,13,14,15]]].to_numpy(),
            category*group.fault_active.to_numpy()[:,None],
            category*group.severity.to_numpy()[:,None],category*abs(group.severity.to_numpy())[:,None]])
        designs={'baseline':baseline,'covariance':np.column_stack([baseline,group.covariance]),
                 'nis':np.column_stack([baseline,group.nis]),
                 'covariance+nis':np.column_stack([baseline,group.covariance,group.nis])}
        fold=group.episode.to_numpy()%5
        for target in ['action_error','future_tracking','excess_future_tracking']:
            valid=np.isfinite(group[target].to_numpy()); y=np.arcsinh(group[target].to_numpy()[valid])
            if not len(y):continue
            folds=fold[valid]; errors={}
            for label,full in designs.items():
                design=full[valid]; pred=np.full(len(y),np.nan)
                for f in np.unique(folds):
                    tr=folds!=f; te=folds==f
                    if not tr.any():continue
                    mean=design[tr].mean(0); scale=np.maximum(design[tr].std(0),1e-8)
                    train=np.column_stack([np.ones(tr.sum()),(design[tr]-mean)/scale])
                    test=np.column_stack([np.ones(te.sum()),(design[te]-mean)/scale])
                    ridge=np.eye(train.shape[1])*1e-6; ridge[0,0]=0
                    coef=np.linalg.solve(train.T@train+ridge,train.T@y[tr]);pred[te]=test@coef
                errors[label]=(pred-y)**2
            detail=group.loc[valid,['training_seed','condition','episode']].copy()
            for label,error in errors.items():detail[label]=error
            episode=detail.groupby(['training_seed','condition','episode']).mean().reset_index().dropna()
            if episode.empty:continue
            base=float(episode.baseline.mean())
            for label in ['covariance','nis','covariance+nis']:
                enhanced=float(episode[label].mean());ci=paired_error_interval(episode,enhanced=label)
                rows.append(dict(kind='conditional_prediction',bits=bits,condition='conditional_episode_cv',signal=label,target=target,
                    baseline_mse=base,enhanced_mse=enhanced,relative_mse_reduction=1-enhanced/base,
                    reduction_ci025=ci[0],reduction_ci975=ci[1],training_seeds=group.training_seed.nunique(),
                    episodes=group.episode_seed.nunique(),samples=len(y)))
        episode=group.groupby(['training_seed','condition','episode']).agg(covariance=('covariance','mean'),nis=('nis','mean'),constraint_fraction=('constraint_fraction','first')).reset_index()
        for condition,g in [('all',episode)]+list(episode.groupby('condition')):
            for signal in ['covariance','nis']:
                corr=float(spearmanr(g[signal],g.constraint_fraction).statistic) if g.constraint_fraction.nunique()>1 else np.nan
                rows.append(dict(kind='episode_rank_correlation',bits=bits,condition=condition,signal=signal,target='episode_constraint_fraction',spearman=corr,episodes=len(g),samples=len(g)))
    pd.DataFrame(rows).to_csv(OUT/'signal_analysis.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7,3));g=s[s.bits==4]
    for ax,signal in zip(axes,['covariance','nis']):
        for condition,c in g.groupby('condition'):
            bins=pd.qcut(c[signal],8,duplicates='drop');agg=c.groupby(bins,observed=True)[[signal,'action_error']].mean();ax.plot(agg[signal],agg.action_error,'o-',ms=3,label=condition)
        if signal=='covariance':ax.set_xscale('log')
        ax.set_xlabel('Covariance score (log axis)' if signal=='covariance' else 'log(1 + EWMA NIS/dof)');ax.set_ylabel('Teacher–student action norm')
    axes[1].legend(fontsize=7);save(fig,'signal_compression_analysis')


def report(*, write_docs=True):
    refresh_training_records()
    architecture(); plot_baselines(); signal_analysis()
    p=OUT/'per_episode_metrics.csv'
    if not p.exists():return
    d=pd.read_csv(p)
    # Older rows recorded zero for non-GPS fault duration; recover the
    # configured interval (both bias faults end at 145 s) without altering outcomes.
    mask=d.condition.isin(['gyro_bias','airspeed_bias'])
    d.loc[mask,'duration_s']=145.-d.loc[mask,'onset_s']
    d['episode_duration_s']=180.
    d['fault_end_s']=np.where(d.condition=='nominal',np.nan,d.onset_s+d.duration_s)
    # Raw episode tables remain immutable during reporting.
    d['failed_before_fault']=((d.failed>0)&(d.condition!='nominal')&(d.completed_s<=d.onset_s)).astype(int)
    d['fault_exposure_s']=np.where(d.condition=='nominal',0.,np.maximum(np.minimum(d.completed_s,d.fault_end_s)-d.onset_s,0.))
    d.groupby(['method','bits','condition']).agg(episodes=('failed','size'),failed=('failed','sum'),failed_before_fault=('failed_before_fault','sum'),mean_fault_exposure_s=('fault_exposure_s','mean')).reset_index().to_csv(OUT/'failure_timing.csv',index=False)
    d['regime']=np.where(d.condition=='nominal','nominal','off_nominal')
    cols=['altitude_rmse_m','heading_rmse_deg','airspeed_rmse_mps','tracking_score','constraint_fraction','failed','action_rmse']
    d.groupby(['training_seed','method','bits','condition'])[cols].mean().reset_index().to_csv(OUT/'metrics.csv',index=False)
    summary=d.groupby(['method','bits','regime'])[cols].agg(['mean','std','count']); summary.columns=['_'.join(c) for c in summary.columns]; summary.reset_index().to_csv(OUT/'summary.csv',index=False)
    main=d[d.method.isin(['teacher','uniform','ecqad'])]
    fig,axes=plt.subplots(1,2,figsize=(7,3),sharey=True)
    for ax,(regime,g) in zip(axes,main.groupby('regime',sort=True)):
        for method in ['uniform','ecqad']:
            samples=g[g.method==method]
            if samples.training_seed.nunique()>1:
                m=samples.groupby(['bits','training_seed']).tracking_score.mean().groupby('bits').agg(['mean','std','count'])
            else:
                m=samples.groupby('bits').tracking_score.agg(['mean','std','count'])
            ax.errorbar(m.index,m['mean'],yerr=m['std']/np.sqrt(m['count']),marker='o',label=DISPLAY_NAMES[method])
        t=g[g.method=='teacher'].tracking_score.mean(); ax.axhline(t,color='k',ls='--',label='FP32 teacher'); ax.set_title(regime.replace('_',' ')); ax.set_xlabel('Weight / hidden activation bits'); ax.set_xticks([2,4,8]); ax.set_yscale('log')
    fig.suptitle('SE across training-seed means' if main.training_seed.nunique()>1 else 'Episode SE, conditional on one trained policy',fontsize=9)
    axes[0].set_ylabel('Normalized tracking score'); axes[1].legend(); save(fig,'performance_vs_bits')
    # Paired bootstrap across episode identities, preserving common random numbers.
    pairs=[]; rng=np.random.default_rng(20261108)
    for bits in [8,4,2]:
        for regime in ['nominal','off_nominal']:
            g=d[(d.bits==bits)&(d.regime==regime)]
            pivot=g.pivot_table(index=['training_seed','condition','episode'],columns='method',values='tracking_score')
            if not {'uniform','ecqad'}.issubset(pivot.columns):continue
            delta=(pivot.ecqad-pivot.uniform).dropna()
            cells=delta.rename('delta').reset_index().pivot(index=['condition','episode'],columns='training_seed',values='delta').dropna()
            boots=[]
            # Resample training seeds and common scenarios, preserving paired
            # method comparisons and stratifying nominal/fault categories.
            for _ in range(2000):
                seed_indices=rng.integers(0,cells.shape[1],cells.shape[1])
                by_condition=[]
                for _,sub in cells.groupby(level='condition'):
                    values=sub.to_numpy()
                    ep_indices=rng.integers(0,len(values),len(values))
                    by_condition.append(values[ep_indices][:,seed_indices].mean())
                boots.append(np.mean(by_condition))
            pairs.append(dict(bits=bits,regime=regime,pairs=len(delta),training_seeds=cells.shape[1],scenario_pairs=len(cells),ecqad_minus_uniform=float(delta.mean()),ci025=float(np.quantile(boots,.025)),ci975=float(np.quantile(boots,.975))))
    pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)
    d.groupby(['training_seed','method','bits','regime'])[cols].mean().reset_index().to_csv(OUT/'per_seed_comparisons.csv',index=False)
    ab=d[d.bits==4].groupby(['method','regime'])[cols].mean().reset_index(); ab.to_csv(OUT/'ecqad_ablation.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,4))
    b=ab.pivot(index='method',columns='regime',values='tracking_score')
    seed_stats=d[d.bits==4].groupby(['method','regime','training_seed']).tracking_score.mean().groupby(['method','regime']).agg(['std','count'])
    se=(seed_stats['std']/np.sqrt(seed_stats['count'])).unstack('regime').reindex(b.index).fillna(0.)
    labels=[DISPLAY_NAMES.get(v,v) for v in b.index];b.index=labels;se.index=labels
    b.plot.bar(ax=ax,rot=30,yerr=se,capsize=2);ax.tick_params(axis='x',labelsize=7)
    ax.set_title('SE across student-seed means' if d.training_seed.nunique()>1 else 'One student seed; across-seed uncertainty not estimated',fontsize=9)
    ax.set_ylabel('Normalized tracking score');ax.set_xlabel('');save(fig,'ablation_comparison')
    fig,axes=plt.subplots(3,1,figsize=(7,6),sharex=True)
    for method,bits in [('teacher',32),('uniform',4),('ecqad',4)]:
        files=sorted((OUT/'traces').glob(f'{method}_{bits}_seed{int(d.training_seed.min())}_gps_dropout_000.npz'))
        if not files:continue
        t,_=trace(files[0])
        for ax,state,ref,scale,label in zip(axes,['altitude_m','yaw_rad','airspeed_mps'],['altitude_m','heading_rad','airspeed_mps'],[1,180/np.pi,1],['Altitude (m)','Heading (deg)','Airspeed (m/s)']):
            ax.plot(t.time_s,t['truth_'+state]*scale,lw=.9,label=DISPLAY_NAMES[method]); ax.set_ylabel(label)
            if method=='teacher':ax.plot(t.time_s,t['cmd_'+ref]*scale,'k--',lw=.7,label='Reference')
    axes[0].legend(ncol=4); axes[-1].set_xlabel('Time (s)'); save(fig,'closed_loop_comparison')
    if write_docs:
        write_results_docs(d)
        weighting_diagnosis()
        research_package(d)
        study_audit(d)
    print(summary.to_string())


def write_results_docs(d):
    """Manuscript-ready text generated exclusively from the executed tables."""
    rows=[]
    display=[('teacher',32),('uniform',8),('uniform',4),('uniform',2),('ecqad',8),('ecqad',4),('ecqad',2)]
    for method,bits in display:
        g=d[(d.method==method)&(d.bits==bits)]
        if g.empty: continue
        n=g[g.condition=='nominal']; f=g[g.condition!='nominal']
        storage=int(np.ceil(1664*bits/8)+464)
        def metric(t,c):return f'{t[c].mean():.4f}' if len(t) else 'not run'
        rows.append(f"| {DISPLAY_NAMES[method]} | {bits} | {len(n)} / {len(f)} | {metric(n,'tracking_score')} | {metric(f,'tracking_score')} | {metric(n,'altitude_rmse_m')} / {metric(f,'altitude_rmse_m')} | {metric(n,'heading_rmse_deg')} / {metric(f,'heading_rmse_deg')} | {metric(n,'airspeed_rmse_mps')} / {metric(f,'airspeed_rmse_mps')} | {int(g.failed.sum())}/{len(g)} | {g.constraint_fraction.mean():.4f} | {storage} |")
    table='| Method | Bits | Episodes nominal / faulted | Nominal score | Faulted score | Altitude RMSE N/F (m) | Heading RMSE N/F (deg) | Airspeed RMSE N/F (m/s) | Failures | Constraint fraction | Packed bytes |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
    effects=[]
    for bits in [8,4,2]:
        g=d[(d.bits==bits)&(d.condition!='nominal')]
        means=g.groupby('method').tracking_score.mean()
        if {'uniform','ecqad'}.issubset(means.index): effects.append((bits,100*(1-means['ecqad']/means['uniform'])))
    effect_text=' '.join(f'At {b} bits, off-nominal EC-QAD-v1 combined score changed by {-v:+.2f}% relative to uniform QAD (negative is better).' for b,v in effects)
    strongest=max(effects,key=lambda item:item[1]) if effects else None
    strongest_text=(f'The largest relative off-nominal improvement was {strongest[1]:.2f}% at {strongest[0]} bits.' if strongest and strongest[1]>0 else 'No evaluated bit width improved the mean off-nominal score.')
    if strongest and strongest[0]==2:
        strongest_text+=' A relative score improvement does not establish acceptable two-bit flight; failures and constraint violations remain essential.'
    ablation_lines=[]
    four=d[(d.bits==4)&(d.condition!='nominal')]
    paired=four.pivot_table(index=['training_seed','condition','episode'],columns='method',values='tracking_score')
    if 'ecqad' in paired:
        for baseline in ['covariance','innovation','shuffled','balanced','fault_balanced','saliency','ptq']:
            if baseline in paired:
                matched=paired[['ecqad',baseline]].dropna()
                if len(matched):
                    change=100*(matched.ecqad.mean()/matched[baseline].mean()-1)
                    ablation_lines.append(f'Against {baseline}, 4-bit off-nominal EC-QAD-v1 combined score changed by {change:+.2f}% on {len(matched)} matched rollouts (negative favors EC-QAD-v1).')
    ablation_text=' '.join(ablation_lines) if ablation_lines else 'Additional weighting controls have not yet completed evaluation.'
    ablation_text+=' These are descriptive paired contrasts; the per-seed table must also be examined. If shuffled or generic balancing/sensitivity performs as well, superiority to uniform QAD alone is not estimator-specific evidence.'
    seeds=sorted(d.training_seed.unique().astype(int).tolist())
    complete=len(d); teacher=d[d.method=='teacher']; teacher_fail=int(teacher.failed.sum())
    unique_scenarios=d.groupby('condition').episode_seed.nunique().to_dict()
    conditional='Not yet available.'
    if (OUT/'signal_analysis.csv').exists():
        signal=pd.read_csv(OUT/'signal_analysis.csv'); g=signal[(signal.bits==4)&(signal.condition=='conditional_episode_cv')&(signal.signal=='covariance+nis')]
        conditional=' '.join(f"For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for {r.target} by {-100*r.relative_mse_reduction:+.2f}% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity." for r in g.itertuples())
    limitations='The aircraft has approximate public BETA aerodynamics; sensors and faults are synthetic; flight is still-air; the teacher imitates a conventional controller using a memoryless physical-state/reference interface. Quantization is simulated, and no hardware latency, energy or speedup was measured. Two-bit prefix RMSE can look deceptively favorable when an episode terminates: use failures and the penalized score. Correlations and conditional prediction do not establish causal compression criticality. H1 uses observed trajectory prefixes and cannot reconstruct counterfactual tracking after early termination; severe low-bit failures therefore restrict its interpretation. All student seeds share the accepted frozen seed-7 teacher; inference is conditional on that one teacher. A separately trained seed-19 teacher failed its tracking gate and is preserved in the rejected-candidate archive. A small number of student training seeds limits inference about optimization variability.'
    text=f'''# Executed EC-QAD results

This report is generated from `results/per_episode_metrics.csv`; it contains {complete} executed rollouts across training seeds {seeds}. Unique randomized scenario seeds per condition: {unique_scenarios}. Scenarios are shared across methods and training seeds. Counts below identify the actual scope, rather than the planned scope. The prior spherical-coordinate experiment is quarantined under `results/archive/v0_spherical_coordinate_bug/` and is excluded here. The PID-memory-dependent study in `results/archive/v1_pid_dependent_exploratory/` is also exploratory and excluded. Current teacher/student inputs use `physical_reference_v2`; neural deployments neither instantiate nor execute PID.

## Main table

Score is the normalized heading/altitude/airspeed tracking metric defined in EXPERIMENT_DESIGN.md. N/F means nominal/off-nominal. Failure penalty applies to the score; physical RMSE is over the observed trajectory. A condition labels the scheduled scenario; a failed flight may terminate before its fault begins. See `results/failure_timing.csv` for actual fault exposure and pre-onset failures. Storage is ideal packed weights plus documented FP32 overhead, not an on-disk or hardware measurement.

{table}

## What changed and what did not

{effect_text}

{strongest_text} These are mean comparisons; see `results/paired_comparisons.csv` for paired intervals and `results/ecqad_ablation.csv` for alternative weighting methods. A small positive difference alone does not establish robustness or estimator-specific benefit. FP32 teacher failures: {teacher_fail}/{len(teacher)}. The nominal columns quantify the cost, if any, of off-nominal weighting.

## H1: incremental information

{conditional}

Global and within-condition rank correlations are in `results/signal_analysis.csv`. The conditional analysis holds out complete episode groups. Excess future tracking compares paired but diverging teacher/student rollouts and is not a causal one-action intervention.

## H2: weighting intervention

{ablation_text}

The table and paired comparisons evaluate H2 separately from H1. Covariance-only, innovation-only, shuffled, fault-balanced and sensitivity-weighted results appear when executed, rather than being assumed. Favorable correlation does not establish that this bounded weighting function exploits it.

## Figures

- [Architecture](../results/figures/architecture.png)
- [Conventional tracking](../results/figures/conventional_tracking.png)
- [Estimator envelopes](../results/figures/estimator_envelopes.png)
- [Estimator degradation](../results/figures/estimator_degradation.png)
- [Closed-loop comparison](../results/figures/closed_loop_comparison.png)
- [Performance versus bits](../results/figures/performance_vs_bits.png)
- [Ablations](../results/figures/ablation_comparison.png)
- [Signal analysis](../results/figures/signal_compression_analysis.png)

PDF versions are available alongside each PNG. Machine-readable raw and summary tables are retained.

## Limitations

{limitations}
'''
    (ROOT/'docs/INITIAL_RESULTS.md').write_text(text)


def refresh_training_records():
    """Reconcile concurrent training logs against authoritative checkpoints."""
    import torch
    from ecqad_uav.learning.policy import Policy
    path=OUT/'training_runs.csv'
    old=pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows=[]
    for checkpoint in sorted((OUT/'checkpoints').glob('seed*/*.pt')):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        meta=saved['metadata']; seed=meta['seed']; method=meta['method']; bits=saved['bits']
        row={}
        if len(old):
            match=old[(old.training_seed==seed)&(old.method==method)&(old.bits==bits)]
            if len(match):row=match.iloc[-1].to_dict()
        row.update(training_seed=seed,teacher_seed=meta.get('teacher_seed',load_yaml('study.yaml')['teacher_seed']),method=method,bits=bits,epochs=meta.get('epochs',0),validation_action_mse=meta.get('validation_mse',np.nan),storage_bytes=Policy.load(checkpoint).storage_bytes(),checkpoint=str(checkpoint.relative_to(ROOT)))
        for key,value in meta.get('weighting',{}).items():
            if isinstance(value,(int,float)):row['weight_'+key]=value
        rows.append(row)
    pd.DataFrame(rows).to_csv(path,index=False)


def weighting_diagnosis():
    """Read-only examination of the frozen training population; no tuning."""
    from ecqad_uav.learning.policy import conditioning_weights
    frames=[]
    for path in sorted((OUT/'data/train').glob('*.npz')):
        d,_=trace(path);meta=json.loads(path.with_suffix('.json').read_text())
        active=(d.time_s>=meta['onset_s'])&(d.time_s<meta['fault_end_s']) if meta['condition']!='nominal' else np.zeros(len(d),bool)
        last=d.time_s.where(d.nis_gps.notna()).ffill().fillna(0.)
        frames.append(pd.DataFrame(dict(time=d.time_s,covariance=d.covariance_score,nis=d.nis_normalized_ewma,
            scenario=meta['condition'],fault_state=np.where(active,meta['condition'],'nominal'),
            gps_update=d.nis_gps.notna(),heading_update=d.nis_heading.notna(),airspeed_update=d.nis_airspeed.notna(),
            gps_age=d.time_s-last,episode_seed=meta['episode_seed'])))
    d=pd.concat(frames,ignore_index=True);d['log_nis']=np.log1p(d.nis)
    signals=d[['covariance','log_nis']].to_numpy();z=(signals-signals.mean(0))/np.maximum(signals.std(0),1e-6)
    clipped=np.clip(z,-3,3)
    d['covariance_z']=z[:,0];d['nis_z']=z[:,1]
    d['covariance_z_clipped']=clipped[:,0];d['nis_z_clipped']=clipped[:,1]
    d['covariance_elevated']=z[:,0]>1;d['nis_elevated']=z[:,1]>1;d['both_elevated']=d.covariance_elevated&d.nis_elevated
    for method in ['covariance','innovation','ecqad']:
        d['w_'+method]=conditioning_weights(d.covariance.to_numpy(),d.nis.to_numpy(),method,7)[0]
    groups=[('population','all',d)]
    groups += [('scenario',str(k),g) for k,g in d.groupby('scenario')]
    groups += [('active_fault_state',str(k),g) for k,g in d.groupby('fault_state')]
    signals_rows=[];weight_rows=[]
    for grouping,name,g in groups:
        row=dict(grouping=grouping,group=name,samples=len(g),pearson_cov_log_nis=float(g.covariance.corr(g.log_nis)),spearman_cov_log_nis=float(g.covariance.corr(g.log_nis,method='spearman')),
                 covariance_elevated_pct=100*g.covariance_elevated.mean(),nis_elevated_pct=100*g.nis_elevated.mean(),both_elevated_pct=100*g.both_elevated.mean(),
                 gps_no_update_pct=100*(~g.gps_update).mean(),gps_age_mean_s=g.gps_age.mean(),gps_age_max_s=g.gps_age.max())
        for signal in ['covariance','nis','log_nis','covariance_z','nis_z','covariance_z_clipped','nis_z_clipped']:
            row.update({f'{signal}_{stat}':float(value) for stat,value in dict(mean=g[signal].mean(),std=g[signal].std(),q01=g[signal].quantile(.01),q50=g[signal].median(),q95=g[signal].quantile(.95),q99=g[signal].quantile(.99),maximum=g[signal].max()).items()})
        signals_rows.append(row)
        for method in ['covariance','innovation','ecqad']:
            w=g['w_'+method].to_numpy();weight_rows.append(dict(grouping=grouping,group=name,method=method,samples=len(w),mean=w.mean(),std=w.std(),minimum=w.min(),q05=np.quantile(w,.05),median=np.median(w),q95=np.quantile(w,.95),maximum=w.max(),near_min_pct=100*np.mean(w<=.505),near_max_pct=100*np.mean(w>=1.995),effective_sample_fraction=w.sum()**2/(len(w)*np.square(w).sum())))
    signal_table=pd.DataFrame(signals_rows);weights=pd.DataFrame(weight_rows)
    signal_table.to_csv(OUT/'estimator_signal_distributions.csv',index=False);weights.to_csv(OUT/'weight_diagnostics.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for name,g in d.groupby('fault_state'):
        for ax,signal in zip(axes[:2],['covariance','log_nis']):ax.hist(g[signal],bins=50,density=True,histtype='step',label=name)
    axes[0].set_xlabel('Dimensionless covariance score');axes[1].set_xlabel('log(1 + EWMA NIS / dof)');axes[0].set_ylabel('Density');axes[1].legend(fontsize=6)
    axes[2].hexbin(d.covariance,d.log_nis,gridsize=45,bins='log',mincnt=1);axes[2].set_xlabel('Covariance score');axes[2].set_ylabel('Log NIS score');save(fig,'training_signal_distributions')
    fig,axes=plt.subplots(1,2,figsize=(9,3))
    for method in ['covariance','innovation','ecqad']:axes[0].hist(d['w_'+method],bins=50,density=True,histtype='step',label=DISPLAY_NAMES[method])
    axes[0].set_xlabel('Training importance weight');axes[0].set_ylabel('Density');axes[0].legend(fontsize=7)
    names=['nominal','gyro_bias','gps_dropout','airspeed_bias']
    for i,method in enumerate(['covariance','innovation','ecqad']):
        vals=[d.loc[d.fault_state==n,'w_'+method].to_numpy() for n in names]
        b=axes[1].boxplot(vals,positions=np.arange(4)+.23*(i-1),widths=.18,showfliers=False,patch_artist=True)
        for box in b['boxes']:box.set_facecolor(f'C{i}');box.set_alpha(.5)
    axes[1].set_xticks(range(4),names,rotation=20);axes[1].set_ylabel('Weight by active fault state');save(fig,'training_weight_distributions')
    pop=signal_table.iloc[0];w=weights[(weights.grouping=='population')&(weights.method=='ecqad')].iloc[0]
    gps=signal_table[(signal_table.grouping=='active_fault_state')&(signal_table.group=='gps_dropout')].iloc[0]
    selected=weights[weights.grouping=='active_fault_state']
    lines=['| Active state | Covariance mean weight | Innovation mean weight | EC-QAD-v1 mean weight |','|---|---:|---:|---:|']
    for name in names:
        vals=selected[selected.group==name].set_index('method')['mean'];lines.append(f'| {name} | {vals["covariance"]:.5f} | {vals["innovation"]:.5f} | {vals["ecqad"]:.5f} |')
    variances=np.var(.25*clipped,axis=0)
    executed=pd.read_csv(OUT/'per_episode_metrics.csv')
    executed=executed[executed.bits==4].copy()
    executed['regime']=np.where(executed.condition=='nominal','nominal','off_nominal')
    control=executed.groupby(['training_seed','method','regime']).tracking_score.mean().unstack('regime').reset_index()
    training=pd.read_csv(OUT/'training_runs.csv');training=training[training.bits==4]
    joined=control.merge(training[['training_seed','method','validation_action_mse']],on=['training_seed','method'])
    joined.to_csv(OUT/'distillation_vs_control.csv',index=False)
    fidelity=['| Student seed | Method | Validation action MSE | Nominal score | Off-nominal score |','|---:|---|---:|---:|---:|']
    for r in joined[joined.method.isin(['uniform','covariance','innovation','ecqad'])].itertuples():
        fidelity.append(f'| {r.training_seed} | {DISPLAY_NAMES[r.method]} | {r.validation_action_mse:.7f} | {r.nominal:.4f} | {r.off_nominal:.4f} |')
    text=f'''# EC-QAD-v1 diagnosis

This is a read-only analysis of {len(d)} logged training samples from 32 complete trajectories. The deployed physical-reference interface, teacher, quantizer, and existing weighting algorithms remain frozen. Existing `ecqad` identifiers mean **EC-QAD-v1 combined**. Negative results are preserved under `results/frozen_ecqad_v1/` and in the episode tables.

## Exact frozen rule

C = log(1 + mean of selected covariance diagonals divided by fixed physical scales). N = log(1 + mean of the three sensor EWMA(NIS/dof) values). Standardize each independently using the entire training population, then clip each z-score to [-3,3]. Covariance-only uses score 0.5*z_C; innovation-only uses 0.5*z_N; EC-QAD-v1 uses 0.25*(z_C+z_N). An offset b is solved so mean(clip(exp(score+b),0.5,2))=1. All calculations precede distillation. These signals are never policy inputs.

## Executed distribution measurements

Covariance/log-NIS Pearson correlation is {pop.pearson_cov_log_nis:.5f}; Spearman correlation is {pop.spearman_cov_log_nis:.5f}. Elevated means above the training mean by more than one training standard deviation, before clipping: covariance is elevated on {pop.covariance_elevated_pct:.3f}% of samples; NIS on {pop.nis_elevated_pct:.3f}%; both on {pop.both_elevated_pct:.3f}%.

EC-QAD-v1 weights have mean {w['mean']:.8f}, standard deviation {w['std']:.5f}, 5th/50th/95th percentiles {w.q05:.5f}/{w['median']:.5f}/{w.q95:.5f}, and effective sample fraction (sum w)^2/(n*sum w²) {w.effective_sample_fraction:.5f}. Within absolute 0.005 of the lower/upper bounds are {w.near_min_pct:.3f}% / {w.near_max_pct:.3f}% of samples. These are measured saturation fractions, not assumed explanations for instability.

{chr(10).join(lines)}

Complete marginal quantiles, scenario-wide distributions, active-fault distributions, correlations, update ages, and all three weight distributions are in `results/estimator_signal_distributions.csv` and `results/weight_diagnostics.csv`. Figures: `results/figures/training_signal_distributions.png` and `training_weight_distributions.png`.

## What the implementation establishes

- Raw NIS cannot dominate merely through its units: the two components are standardized separately and receive equal coefficients in the combined score. The variances of their clipped weighted contributions are {variances[0]:.6f} (covariance) and {variances[1]:.6f} (NIS). The NIS contribution variance is {variances[1]/variances[0]:.3f} times the covariance contribution variance after clipping. Before clipping both standardized components have unit population variance; clipping changes their effective spread differently. Equal coefficients therefore do not imply equal population influence, physical meaning, or equally useful ranking. This is a measured imbalance, not proof that it caused flight instability.
- The combined score is an average of standardized components, not an unnormalized sum of their standalone scores. The joint-elevation fraction above quantifies overlap; correlation alone cannot establish harmful double-counting.
- NIS is available only at measurement updates. The implementation holds each sensor's previous EWMA between updates and averages all three values. During active GPS dropout, {gps.gps_no_update_pct:.3f}% of logged samples have no new GPS innovation; the mean/max age of the last GPS statistic is {gps.gps_age_mean_s:.3f}/{gps.gps_age_max_s:.3f} seconds. Heading/airspeed updates may still change the composite. Missing GPS is therefore not represented by a fresh GPS inconsistency statistic or an availability gate.
- Alpha=0.1 corresponds to an EWMA e-folding time of {-1/(5*np.log(.9)):.3f} s at the nominal 5 Hz GPS rate and {-1/(20*np.log(.9)):.3f} s at 20 Hz heading/airspeed rates. Measurement loss extends the GPS hold beyond that nominal time scale. Covariance propagates during loss; NIS describes consistency of received measurements. They measure different physical phenomena.
- Population mean-one normalization is enforced, while each fault-state mean can differ, as the table shows. Mean-one does not make gradients identical or guarantee stable low-bit closed-loop control.

## Offline fidelity versus closed-loop control

{chr(10).join(fidelity)}

These are the same frozen checkpoints and executed rollouts, joined by student seed and method. `results/distillation_vs_control.csv` contains all evaluated four-bit controls. Validation action MSE selects the checkpoint using separate expert trajectories; it is not a closed-loop stability criterion. In the original seed-7 comparison, combined EC-QAD-v1 has lower validation action MSE than uniform QAD yet much worse closed-loop tracking. Covariance-only has higher validation MSE than uniform but better closed-loop tracking. Thus poorer average offline fidelity alone does not explain the ordering. The data establish a mismatch between the scalar selection metric and closed-loop consequences, not a causal account of which action errors destabilize flight. No checkpoint has been reselected using evaluation outcomes.

## Causal limits and algorithm decision

These diagnostics identify measurable saturation, ranking overlap, freshness, and time-scale differences. They do not establish which mechanism caused a particular trained policy to fail. Closed-loop comparisons across student seeds, covariance-specific shuffled weights, fault balancing, and saliency are required; offline action MSE alone is insufficient. No revised fusion formula or test-set hyperparameter search has been introduced. Covariance-only and innovation-only remain legitimate, separately labeled candidates. The final evidence-based decision is recorded in RESEARCH_VERDICT.md after the matched evaluation finishes.
'''
    (ROOT/'docs/ECQAD_V1_DIAGNOSIS.md').write_text(text)


def research_package(d):
    """Evidence-based tables and verdict; missing comparisons remain pending."""
    expected=load_yaml('study.yaml');seeds=expected['training_seeds'];n=expected['evaluation_episodes_per_condition']
    required=['uniform','covariance','innovation','ecqad','shuffled','covariance_shuffled','fault_balanced','saliency']
    complete=all(len(d[(d.training_seed==seed)&(d.method==method)&(d.bits==4)&(d.condition==condition)])==n
                 for seed in seeds for method in required for condition in expected['conditions'])
    rows=[]
    for (method,bits),g in d[d.method!='pid'].groupby(['method','bits']):
        nominal=g[g.regime=='nominal'];faulted=g[g.regime=='off_nominal']
        seedmeans=faulted.groupby('training_seed').tracking_score.mean()
        row=dict(method=method,label=DISPLAY_NAMES.get(method,method),bits=bits,seeds=g.training_seed.nunique(),
                 nominal_score=nominal.tracking_score.mean(),off_nominal_score=faulted.tracking_score.mean(),
                 off_nominal_seed_sd=seedmeans.std(ddof=1),nominal_episodes=len(nominal),off_nominal_episodes=len(faulted),
                 failures=int(g.failed.sum()),constraint_fraction=g.constraint_fraction.mean())
        uniform=d[(d.method=='uniform')&(d.bits==bits)&(d.regime=='off_nominal')]
        pair=faulted[['training_seed','condition','episode','tracking_score']].merge(uniform[['training_seed','condition','episode','tracking_score']],on=['training_seed','condition','episode'],suffixes=('_method','_uniform'))
        if len(pair):
            pair=pair.rename(columns={'tracking_score_uniform':'baseline','tracking_score_method':'enhanced'})
            lo,hi=paired_error_interval(pair);row.update(relative_to_uniform_pct=100*(pair.enhanced.mean()/pair.baseline.mean()-1),relative_ci025_pct=-100*hi,relative_ci975_pct=-100*lo)
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(OUT/'definitive_preliminary_table.csv',index=False)
    comparisons=[]
    for regime in ['nominal','off_nominal']:
        g=d[(d.bits==4)&(d.regime==regime)]
        covariance=g[g.method=='covariance'][['training_seed','condition','episode','tracking_score']]
        for comparator in ['uniform','covariance_shuffled','shuffled','fault_balanced','saliency','innovation','ecqad']:
            other=g[g.method==comparator][['training_seed','condition','episode','tracking_score']]
            pair=covariance.merge(other,on=['training_seed','condition','episode'],suffixes=('_covariance','_other'))
            if pair.empty:continue
            pair=pair.rename(columns={'tracking_score_other':'baseline','tracking_score_covariance':'enhanced'})
            lo,hi=paired_error_interval(pair);byseed=pair.groupby('training_seed')[['baseline','enhanced']].mean()
            comparisons.append(dict(regime=regime,comparator=comparator,pairs=len(pair),seeds=len(byseed),
                covariance_score=pair.enhanced.mean(),comparator_score=pair.baseline.mean(),
                improvement_pct=100*(1-pair.enhanced.mean()/pair.baseline.mean()),ci025_pct=100*lo,ci975_pct=100*hi,
                seeds_covariance_better=int((byseed.enhanced<byseed.baseline).sum())))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(OUT/'covariance_comparisons.csv',index=False)
    h1=pd.read_csv(OUT/'signal_analysis.csv');h1=h1[(h1.bits==4)&(h1.kind=='conditional_prediction')]
    status={}
    for signal in ['covariance','nis']:
        part=h1[h1.signal==signal];positive=set(part.loc[part.reduction_ci025>0,'target'])
        status[signal]='supported' if {'future_tracking','excess_future_tracking'}.issubset(positive) else 'partially supported' if positive else 'unsupported'
    combined_h1='supported' if all(v=='supported' for v in status.values()) else 'partially supported' if any(v!='unsupported' for v in status.values()) else 'unsupported'
    def compare_text(comparator):
        q=comparison[(comparison.regime=='off_nominal')&(comparison.comparator==comparator)] if len(comparison) else pd.DataFrame()
        if q.empty:return 'Pending matched evaluation.'
        r=q.iloc[0];consistent=r.seeds_covariance_better==r.seeds and r.ci025_pct>0
        prefix='Supported within this experiment' if consistent else 'Consistent superiority is not established'
        return f'{prefix}: covariance-only has {r.improvement_pct:+.2f}% mean off-nominal improvement relative to {DISPLAY_NAMES[comparator]} (95% paired bootstrap interval [{r.ci025_pct:+.2f}, {r.ci975_pct:+.2f}]%); it improves the per-seed mean in {int(r.seeds_covariance_better)}/{int(r.seeds)} student seeds.'
    four=table[table.bits==4].set_index('method')
    main=four.loc['ecqad'] if 'ecqad' in four.index else None
    uniform=four.loc['uniform'] if 'uniform' in four.index else None
    combined_text=(f'EC-QAD-v1 combined has mean nominal/off-nominal scores {main.nominal_score:.4f}/{main.off_nominal_score:.4f}, versus {uniform.nominal_score:.4f}/{uniform.off_nominal_score:.4f} for uniform QAD, with {int(main.failures)} failures across its {int(main.nominal_episodes+main.off_nominal_episodes)} four-bit rollouts. '+('Its off-nominal mean is worse; the combined formulation is not supported as a robustness improvement.' if main.off_nominal_score>=uniform.off_nominal_score else 'Its mean improvement must be interpreted with the paired intervals and per-seed results.')) if main is not None and uniform is not None else 'Pending.'
    candidate='No final superiority claim is established. Covariance-only remains the simplest physically interpretable EC-QAD candidate for a preregistered next-stage replication; retain uniform QAD as the reference and do not promote EC-QAD-v1 fusion.'
    if len(comparison):
        q=comparison[(comparison.regime=='off_nominal')&comparison.comparator.isin(['uniform','covariance_shuffled','fault_balanced','saliency'])]
        if len(q)==4 and (q.ci025_pct>0).all() and (q.seeds_covariance_better==len(seeds)).all():
            candidate='Covariance-only EC-QAD is the supported initial candidate: it improves all requested off-nominal comparisons in every student seed with positive paired intervals. Keep it frozen for new-scenario/teacher replication; this is not general flight-safety validation.'
    if not complete:candidate='The matched three-seed comparison is incomplete. No final algorithm is selected. '+candidate
    lines=['| Method | Bits | Nominal score | Off-nominal score | Relative to uniform, off-nominal (%) | Student seeds | Off-nominal seed SD |',
           '|---|---:|---:|---:|---:|---:|---:|']
    ordered=table.sort_values(['bits','method'],ascending=[False,True])
    for r in ordered.itertuples():
        relative=f'{r.relative_to_uniform_pct:+.2f}' if pd.notna(getattr(r,'relative_to_uniform_pct',np.nan)) else '—'
        sd=f'{r.off_nominal_seed_sd:.4f}' if pd.notna(r.off_nominal_seed_sd) else '—'
        lines.append(f'| {r.label} | {r.bits} | {r.nominal_score:.4f} | {r.off_nominal_score:.4f} | {relative} | {r.seeds} | {sd} |')
    hlines=['| Signal | Target | Prediction MSE reduction (%) | 95% interval (%) |','|---|---|---:|---:|']
    for r in h1.itertuples():hlines.append(f'| {r.signal} | {r.target} | {100*r.relative_mse_reduction:+.3f} | [{100*r.reduction_ci025:+.3f}, {100*r.reduction_ci975:+.3f}] |')
    label='Complete planned four-bit grid' if complete else 'Incomplete planned grid; conclusions are provisional'
    verdict=f'''# Research verdict

{label}. All results here use the independent `physical_reference_v2` interface. Three student optimization seeds share one frozen, validated FP32 teacher; these are not three independent teachers. No revised fusion formula was fitted to evaluation results.

**A. Does covariance contain useful compression-criticality information?** {status['covariance'].capitalize()} by the specified incremental prediction test. See the target-specific estimates below; this restricted linear test with current truth tracking as a nuisance covariate cannot prove absence of all useful information.

**B. Does innovation/NIS contain useful compression-criticality information?** {status['nis'].capitalize()} by the same test. Correlations are descriptive, not causal evidence.

**C. Does covariance conditioning consistently outperform uniform QAD?** {compare_text('uniform')}

**D. Does it outperform distribution-matched shuffled covariance weights?** {compare_text('covariance_shuffled')} This is the relevant estimator-specificity comparison; the older `shuffled` control matches combined weights and answers a different question.

**E. Does it outperform fault-balanced QAD?** {compare_text('fault_balanced')}

**F. Does it outperform policy-saliency weighting?** {compare_text('saliency')} The comparator is teacher-Jacobian sensitivity weighting, not an exact SQIL reproduction.

**G. Does combined EC-QAD-v1 work?** {combined_text}

**H. What is the algorithm for the next stage?** {candidate}

**I. Supported claims.** An independent estimator-based neural controller can be distilled with training-only physical-estimator importance weights. The tables establish the measured within-testbed effects, their student-seed variability, and the negative outcomes. H1 overall is {combined_h1} under this specified analysis. Any improvement is conditional on the tested aircraft, single teacher, trajectories, and quantizer.

**J. Unsupported claims.** General safety or fault tolerance; causal compression criticality from correlation; calibrated fault probabilities from NIS; hardware latency/energy/speedup; universal superiority of covariance conditioning; or estimator specificity without consistent superiority to the covariance-matched shuffle. No result from a PID-dependent policy is included.

## H1 estimates

{chr(10).join(hlines)}

Intervals resample student seeds and common scenarios within fault category, conditional on fixed out-of-fold predictions. Whole episodes, including the same scenario across seeds, remain within a single validation fold. Tests are exploratory and are not corrected for multiple comparisons. Severe-failure trajectories contribute only observed prefixes; no counterfactual future after termination is invented.
'''
    (ROOT/'docs/RESEARCH_VERDICT.md').write_text(verdict)
    baseline=pd.read_csv(OUT/'conventional_metrics.csv');baseline=baseline[baseline.split=='validation']
    teacher=pd.read_csv(OUT/'teacher_validation_seed7.csv')
    validation=f'The independent FP32 teacher completed {len(teacher)} held-out 180-s validation flights with {int(teacher.failed.sum())} failures; mean score {teacher.tracking_score.mean():.6f} versus {baseline.tracking_score.mean():.6f} for the PID expert. Neural PID execution is disabled. All student bit widths use the same physical-reference feature builder; covariance/NIS influence only the training loss.'
    with (ROOT/'docs/INITIAL_RESULTS.md').open('a') as f:f.write(f'''\n## Definitive preliminary comparison\n\n{label}. Relative differences are percentages of the matched uniform-QAD score at the same bit width; negative is better. Teacher comparison to a nonexistent 32-bit QAD baseline is not reported. Seed SD is across student-seed means, not across timesteps. Paired 95% intervals appear in `results/definitive_preliminary_table.csv`; the covariance-specific comparisons are in `results/covariance_comparisons.csv`.\n\n{chr(10).join(lines)}\n\n## Testbed and independent deployment\n\nJSBSim Rascal110 uses public approximate BETA aerodynamics, nonlinear six-DoF dynamics, simulated IMU/GPS/pitot/heading observations, and a 15-state UKF. A 16→32→32→4 MLP outputs all four actuator commands. Weights and hidden activations use 8/4/2-bit affine fake quantization; interfaces/biases stay FP32. {validation}\n\nThe exact frozen EC-QAD-v1 rule and complete training-signal distributions are in ECQAD_V1_DIAGNOSIS.md. Covariance-only is a legitimate distinct EC-QAD candidate, not a failed approximation to combined fusion. Both shuffle controls remain separately labeled.\n\n## Scientific conclusion\n\nH1 overall: **{combined_h1}** (covariance: {status['covariance']}; NIS: {status['nis']}). {compare_text('uniform')} {compare_text('covariance_shuffled')} {combined_text}\n\n{candidate} Detailed A–J answers are in RESEARCH_VERDICT.md.\n''')
    best=four.off_nominal_score.idxmin() if len(four) else None
    ranking=(f'The lowest observed four-bit off-nominal mean is {DISPLAY_NAMES[best]} at {four.loc[best, "off_nominal_score"]:.4f}; this descriptive ranking is not validation-based algorithm selection. ' if best else '')
    cov=four.loc['covariance'] if 'covariance' in four.index else None
    actual=(f'At four bits, covariance-only had mean nominal/off-nominal tracking scores {cov.nominal_score:.4f}/{cov.off_nominal_score:.4f}, compared with {uniform.nominal_score:.4f}/{uniform.off_nominal_score:.4f} for uniform QAD. '+compare_text('uniform')) if cov is not None and uniform is not None else 'The matched comparison is pending.'
    actual=ranking+actual
    abstract=f'In {len(d)} executed nonlinear fixed-wing closed-loop rollouts, we evaluated training-only estimator conditioning for quantized neural control. '+actual+' '+compare_text('covariance_shuffled')+' '+combined_text+f' Incremental covariance/NIS prediction was {combined_h1} under the specified whole-episode analysis. These results are conditional on one frozen teacher and the tested simulation; no hardware or causal safety claim is made.'
    paper=f'''# AIAA AVIATION initial research summary

## 1. Literature gap

Quantized policy learning, importance-weighted distillation, uncertainty-aware distillation, and belief-space control already exist. [QuaRL](https://research.google/pubs/quantized-reinforcement-learning-quarl/) and [SQIL](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_Saliency-Aware_Quantized_Imitation_Learning_for_Efficient_Robotic_Control_ICCV_2025_paper.pdf) motivate comparison to quantization and policy-saliency weighting. The narrower question is whether uncertainty/innovation signals from a physical navigation estimator identify compression-critical samples and improve low-bit control beyond generic weighting. This is not a claim of inventing uncertainty-weighted distillation or a comprehensive priority review.

## 2. Research question

Does the association between physical estimator information and individual training timesteps provide useful compression-training importance for an independently deployed neural UAV controller?

## 3. H1 and H2

H1 tests incremental prediction of teacher–student action discrepancy, future tracking, and excess tracking beyond fault/status/severity, time, maneuvers, current tracking, and teacher sensitivity. H2 tests the effect of the frozen weighting interventions with matched scenarios and student seeds. H1 overall: {combined_h1}; covariance: {status['covariance']}; NIS: {status['nis']}. Neither correlation nor success of a weighted loss establishes causality.

## 4. Method

Teacher and student receive only physical navigation/reference features. PID executes solely for expert demonstrations and baseline flights. Covariance/NIS set bounded [0.5,2], population-mean-one training weights multiplying teacher-action MSE. EC-QAD-v1 combines independently standardized/clipped covariance and log-NIS scores as 0.25*(z_C+z_N). Covariance-only and innovation-only use 0.5*z_C and 0.5*z_N respectively. A normalizing log offset precedes clipping. Confidence signals are not deployed inputs; bit width is fixed.

## 5. Experimental design

JSBSim Rascal110, synthetic sensors, 15-state UKF, and a 16→32→32→4 MLP; 8/4/2-bit affine weight/hidden-activation fake quantization. Nominal, gyro-bias, GPS-dropout, and airspeed-bias scenarios hold out stronger severities and later onsets by whole trajectory. Planned student seeds are {seeds}; the current episode table represents {sorted(d.training_seed.unique().tolist())}. All share teacher seed 7. {label}; {len(d)} executed evaluation rollouts are represented. {validation}

## 6. Strongest actual evidence

{actual}

{compare_text('covariance_shuffled')} {compare_text('fault_balanced')} {compare_text('saliency')}

## 7. Strongest negative evidence

{combined_text} Two-bit failures are retained, with pre-fault termination distinguished from actual fault exposure. A separately trained seed-19 teacher failed its validation tracking gate and is preserved; it is not used in the compression study.

## 8. Defensible contribution

- A training-only physical-estimator interface for independently deployed quantized flight-control policies.
- Reproducible comparisons of covariance, innovation, combined, matched-shuffle, fault-balanced, and sensitivity weighting.
- Separate episode-held-out H1 tests and matched closed-loop H2 measurements, including negative results.
- A six-DoF simulation demonstration with explicit teacher, quantization, and failure-accounting limitations.

{candidate}

## 9. Limitations

Approximate public BETA aircraft aerodynamics, synthetic still-air sensor scenarios, a narrow maneuver family, one accepted teacher, three student optimization seeds, and {n} unique scenarios per condition. Reusing matched scenarios across seeds does not create {len(seeds)*n} independent scenario draws. H1 uses a restricted predictive family and a truth-based current-tracking nuisance covariate; failure prefixes limit future-error interpretation. Confidence intervals with three seeds are exploratory. No measured hardware latency, energy, speedup, real-flight safety, or causal compression-criticality claim is supported.

## 10. Draft abstract-results paragraph

{abstract}
'''
    (ROOT/'docs/AIAA_AVIATION_SUMMARY.md').write_text(paper)


def study_audit(d, *, verify_frozen=False):
    """Check scope, paired scenarios, frozen artifacts, and deployment isolation."""
    import hashlib,ast,subprocess
    cfg=load_yaml('study.yaml');keys=['training_seed','method','bits','condition','episode']
    assert not d.duplicated(keys).any(),'Duplicate evaluation identities'
    assert set(d.policy_input_schema)=={'physical_reference_v2'}
    assert not d.loc[d.method!='pid','pid_executed'].any(),'PID executed in a neural deployment'
    paired=['episode_seed','severity','onset_s','duration_s','reference_scale','reference_time_scale','gyro_axis']
    distinct=d.groupby(['condition','episode'])[paired].nunique(dropna=False)
    assert (distinct<=1).all().all(),'Scenario covariates differ between paired methods'
    models=[(s,m,b) for s in cfg['training_seeds'] for b in cfg['bits'] for m in cfg['methods'] if m in ['uniform','ecqad'] or b==4]
    models += [(cfg['teacher_seed'],'teacher',32),(7,'pid',32)]
    expected={(s,m,b,c,e) for s,m,b in models for c in cfg['conditions'] for e in range(cfg['evaluation_episodes_per_condition'])}
    actual=set(d[keys].itertuples(index=False,name=None));missing=sorted(expected-actual)
    frozen={}
    if verify_frozen:
        manifest=json.loads((ROOT/'results/frozen_ecqad_v1/study_manifest.json').read_text())
        frozen={p:h for p,h in manifest['sha256'].items() if p.startswith(('results/checkpoints/','results/data/','models/jsbsim/'))}
        changed=[p for p,h in frozen.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
        assert not changed,changed
        runtime=['src/ecqad_uav/control/autopilot.py','src/ecqad_uav/sim/runner.py']
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==manifest['sha256'][p] for p in runtime)
        policy_path='src/ecqad_uav/learning/policy.py'
        before=subprocess.check_output(['git','show',manifest['git_head']+':'+policy_path],cwd=ROOT,text=True)
        def policy_class(source):return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Policy'),include_attributes=False)
        assert policy_class(before)==policy_class((ROOT/policy_path).read_text()),'Deployed Policy class changed after freeze'
    record=dict(expected_rollouts=len(expected),actual_rollouts=len(d),missing_required_rollouts=len(missing),complete=not missing,
        unique_scenarios=int(d[['condition','episode']].drop_duplicates().shape[0]),student_training_seeds=cfg['training_seeds'],teacher_seed=cfg['teacher_seed'],
        duplicate_keys=0,paired_scenarios_identical=True,neural_pid_execution=False,frozen_preservation_checked=verify_frozen,
        policy_interface_and_class_unchanged=True if verify_frozen else None,unchanged_frozen_artifacts=len(frozen),missing_examples=missing[:20])
    (OUT/'study_audit.json').write_text(json.dumps(record,indent=2))
