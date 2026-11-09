"""Figures and conservative, episode-aware analysis of executed study data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from ecqad_uav.utils.config import project_root

ROOT=project_root(); OUT=ROOT/'results'; FIG=OUT/'figures'
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


def signal_analysis():
    files=sorted((OUT/'signals').glob('*.csv'))
    if not files:return
    s=pd.concat([pd.read_csv(p) for p in files],ignore_index=True).dropna(subset=['future_tracking'])
    # Parallel evaluations may have separate tagged metrics files. Include only
    # episodes admitted to the canonical table used by this report.
    scope_path=OUT/'per_episode_metrics.csv'
    if scope_path.exists():
        keys=['training_seed','method','bits','condition','episode']
        scope=pd.read_csv(scope_path)[keys].drop_duplicates()
        s=s.merge(scope,on=keys,how='inner')
    teachers=s[s.method=='teacher'][['training_seed','episode_seed','time_s','future_tracking']].rename(columns={'future_tracking':'teacher_future'})
    s=s[s.method=='uniform'].merge(teachers,on=['training_seed','episode_seed','time_s'],how='left')
    s['excess_future_tracking']=s.future_tracking-s.teacher_future
    from ecqad_uav.learning.policy import Policy,policy_saliency
    import torch
    torch.set_num_threads(1)
    s['saliency']=0.
    for seed,g in s.groupby('training_seed'):
        model=Policy.load(OUT/'checkpoints'/f'seed{seed}'/'teacher.pt')
        s.loc[g.index,'saliency']=policy_saliency(model,g[[f'feature_{j}' for j in range(16)]].to_numpy(np.float32))
    rows=[]
    for bits,group in s.groupby('bits'):
        for condition,g in [('all',group)]+list(group.groupby('condition')):
            for signal in ['covariance','nis']:
                for target in ['action_error','future_tracking','excess_future_tracking']:
                    rows.append(dict(bits=bits,condition=condition,signal=signal,target=target,spearman=float(spearmanr(g[signal],g[target],nan_policy='omit').statistic),episodes=g.episode_seed.nunique(),samples=len(g)))
        # Leave-one-episode-group-out folds (same scenario across training seeds stays together).
        baseline=np.column_stack([np.ones(len(group)),pd.get_dummies(group.condition).to_numpy(float),pd.get_dummies(group.training_seed).to_numpy(float),group.fault_active,group.time_s/180,(group.time_s/180)**2,group.roll_abs,group.pitch_abs,group.saliency,np.log1p(group.tracking),pd.get_dummies(group.condition).to_numpy(float)*group.fault_active.to_numpy()[:,None]])
        enhanced=np.column_stack([baseline,group.covariance,group.nis])
        fold=group.episode.to_numpy()%5
        for target in ['action_error','future_tracking','excess_future_tracking']:
            y=group[target].to_numpy(); y=np.arcsinh(y)
            if not np.isfinite(y).all(): continue
            errors=[]
            for design in [baseline,enhanced]:
                pred=np.zeros(len(y))
                for f in range(5):
                    tr=fold!=f; te=fold==f
                    if not te.any(): continue
                    coef=np.linalg.lstsq(design[tr],y[tr],rcond=None)[0]; pred[te]=design[te]@coef
                errors.append(float(np.mean((pred-y)**2)))
            rows.append(dict(bits=bits,condition='conditional_episode_cv',signal='covariance+nis',target=target,baseline_mse=errors[0],enhanced_mse=errors[1],relative_mse_reduction=1-errors[1]/errors[0],episodes=group.episode_seed.nunique(),samples=len(group)))
    pd.DataFrame(rows).to_csv(OUT/'signal_analysis.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7,3))
    g=s[s.bits==4]
    for ax,signal in zip(axes,['covariance','nis']):
        for condition,c in g.groupby('condition'):
            bins=pd.qcut(c[signal],8,duplicates='drop'); agg=c.groupby(bins,observed=True)[[signal,'action_error']].mean(); ax.plot(agg[signal],agg.action_error,'o-',ms=3,label=condition)
        if signal=='covariance': ax.set_xscale('log')
        ax.set_xlabel('Covariance score (log axis)' if signal=='covariance' else 'log(1 + EWMA NIS/dof)'); ax.set_ylabel('Teacher–student action norm')
    axes[1].legend(fontsize=7); save(fig,'signal_compression_analysis')


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
    d.to_csv(p,index=False)
    d['regime']=np.where(d.condition=='nominal','nominal','off_nominal')
    cols=['altitude_rmse_m','heading_rmse_deg','airspeed_rmse_mps','tracking_score','constraint_fraction','failed','action_rmse']
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
            ax.errorbar(m.index,m['mean'],yerr=m['std']/np.sqrt(m['count']),marker='o',label=method)
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
    fig,ax=plt.subplots(figsize=(8,3)); b=ab.pivot(index='method',columns='regime',values='tracking_score'); b.plot.bar(ax=ax,rot=25); ax.set_ylabel('Normalized tracking score'); ax.set_xlabel(''); save(fig,'ablation_comparison')
    fig,axes=plt.subplots(3,1,figsize=(7,6),sharex=True)
    for method,bits in [('teacher',32),('uniform',4),('ecqad',4)]:
        files=sorted((OUT/'traces').glob(f'{method}_{bits}_seed{int(d.training_seed.min())}_gps_dropout_000.npz'))
        if not files:continue
        t,_=trace(files[0])
        for ax,state,ref,scale,label in zip(axes,['altitude_m','yaw_rad','airspeed_mps'],['altitude_m','heading_rad','airspeed_mps'],[1,180/np.pi,1],['Altitude (m)','Heading (deg)','Airspeed (m/s)']):
            ax.plot(t.time_s,t['truth_'+state]*scale,lw=.9,label=method); ax.set_ylabel(label)
            if method=='teacher':ax.plot(t.time_s,t['cmd_'+ref]*scale,'k--',lw=.7,label='Reference')
    axes[0].legend(ncol=4); axes[-1].set_xlabel('Time (s)'); save(fig,'closed_loop_comparison')
    if write_docs: write_results_docs(d)
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
        rows.append(f"| {method} | {bits} | {len(n)} / {len(f)} | {metric(n,'tracking_score')} | {metric(f,'tracking_score')} | {metric(n,'altitude_rmse_m')} / {metric(f,'altitude_rmse_m')} | {metric(n,'heading_rmse_deg')} / {metric(f,'heading_rmse_deg')} | {metric(n,'airspeed_rmse_mps')} / {metric(f,'airspeed_rmse_mps')} | {int(g.failed.sum())}/{len(g)} | {g.constraint_fraction.mean():.4f} | {storage} |")
    table='| Method | Bits | Episodes nominal / faulted | Nominal score | Faulted score | Altitude RMSE N/F (m) | Heading RMSE N/F (deg) | Airspeed RMSE N/F (m/s) | Failures | Constraint fraction | Packed bytes |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
    effects=[]
    for bits in [8,4,2]:
        g=d[(d.bits==bits)&(d.condition!='nominal')]
        means=g.groupby('method').tracking_score.mean()
        if {'uniform','ecqad'}.issubset(means.index): effects.append((bits,100*(1-means['ecqad']/means['uniform'])))
    effect_text=' '.join(f'At {b} bits, off-nominal EC-QAD score changed by {-v:+.2f}% relative to uniform QAD (negative is better).' for b,v in effects)
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
                    ablation_lines.append(f'Against {baseline}, 4-bit off-nominal EC-QAD score changed by {change:+.2f}% on {len(matched)} matched rollouts (negative favors EC-QAD).')
    ablation_text=' '.join(ablation_lines) if ablation_lines else 'Additional weighting controls have not yet completed evaluation.'
    ablation_text+=' These are descriptive paired contrasts; the per-seed table must also be examined. If shuffled or generic balancing/sensitivity performs as well, superiority to uniform QAD alone is not estimator-specific evidence.'
    seeds=sorted(d.training_seed.unique().astype(int).tolist())
    complete=len(d); teacher=d[d.method=='teacher']; teacher_fail=int(teacher.failed.sum())
    unique_scenarios=d.groupby('condition').episode_seed.nunique().to_dict()
    conditional='Not yet available.'
    if (OUT/'signal_analysis.csv').exists():
        signal=pd.read_csv(OUT/'signal_analysis.csv'); g=signal[(signal.bits==4)&(signal.condition=='conditional_episode_cv')]
        conditional=' '.join(f"For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for {r.target} by {-100*r.relative_mse_reduction:+.2f}% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity." for r in g.itertuples())
    limitations='The aircraft has approximate public BETA aerodynamics; sensors and faults are synthetic; flight is still-air; the teacher imitates a conventional controller using a memoryless physical-state/reference interface. Quantization is simulated, and no hardware latency, energy or speedup was measured. Two-bit prefix RMSE can look deceptively favorable when an episode terminates: use failures and the penalized score. Correlations and conditional prediction do not establish causal compression criticality. A small number of training seeds limits inference about optimization variability.'
    text=f'''# Executed EC-QAD results

This report is generated from `results/per_episode_metrics.csv`; it contains {complete} executed rollouts across training seeds {seeds}. Unique randomized scenario seeds per condition: {unique_scenarios}. Scenarios are shared across methods and training seeds. Counts below identify the actual scope, rather than the planned scope. The prior spherical-coordinate experiment is quarantined under `results/archive/v0_spherical_coordinate_bug/` and is excluded here.

## Main table

Score is the normalized heading/altitude/airspeed tracking metric defined in EXPERIMENT_DESIGN.md. N/F means nominal/off-nominal. Failure penalty applies to the score; physical RMSE is over the observed trajectory. Storage is ideal packed weights plus documented FP32 overhead, not an on-disk or hardware measurement.

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
    abstract=f'We evaluated estimator-conditioned quantization-aware distillation in a nonlinear six-degree-of-freedom fixed-wing simulation with simulated sensors and a navigation UKF. Across {complete} executed closed-loop rollouts and training seeds {seeds}, '+effect_text+' '+strongest_text+' These results characterize this bounded weighting formulation; they do not establish general robustness or hardware efficiency.'
    paper=f'''# AIAA AVIATION research summary

## 1. Problem

Low-bit policy approximation can alter closed-loop UAV tracking. A state estimator supplies physical uncertainty and consistency information that could identify where preserving teacher fidelity matters most.

## 2. Literature gap and scope

Quantized policies, quantization-aware policy distillation, uncertainty-aware distillation and importance-weighted losses already exist. [QuaRL](https://research.google/pubs/quantized-reinforcement-learning-quarl/) studies policy quantization. [SQIL (Park et al., ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_Saliency-Aware_Quantized_Imitation_Learning_for_Efficient_Robotic_Control_ICCV_2025_paper.pdf) combines quantization-aware imitation and saliency-weighted action distillation. [Logit-uncertainty distillation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11682123/) is another existing uncertainty-weighting example. These sources support a narrow positioning, not a universal priority claim: this study tests covariance and innovation statistics from a physical flight-state estimator as compression-training importance signals. The literature review is targeted, not exhaustive. The implemented Jacobian comparator is sensitivity-inspired and is not presented as a reproduction of SQIL.

## 3. Research question

H1: do estimator signals predict compression-related action error or subsequent tracking degradation beyond fault labels and policy sensitivity? H2: does using those signals as bounded distillation weights improve a low-bit policy? These hypotheses are tested separately.

## 4. EC-QAD algorithm

Dimensionless selected-covariance and normalized-NIS summaries are standardized using training data, combined, exponentiated with a population-normalizing offset, and clipped to [0.5,2]. Their training-population mean is one. The student minimizes weighted teacher-action MSE. Covariance/NIS do not become deployed student inputs. Fixed bit width redistributes training emphasis; it does not allocate bits dynamically.

## 5. Experimental setup

JSBSim Rascal110, simulated IMU/GPS/heading/airspeed, 15-state UKF, cascaded PID expert, 16→32→32→4 MLP teacher/student, 8/4/2-bit affine weight and hidden-activation fake quantization. Whole trajectory splits hold out stronger faults and later onsets. Nominal, gyro-bias, GPS-dropout and airspeed-bias episodes use common random numbers across methods. {complete} rollouts and training seeds {seeds} are represented in the current results. Exact parameters are in METHOD.md and EXPERIMENT_DESIGN.md.

## 6. Initial quantitative findings

{effect_text}

{strongest_text}

{conditional}

{ablation_text}

The full physical-metric table, failure counts, ablations and paired comparisons are in INITIAL_RESULTS.md and machine-readable results tables.

## 7. Defensible contribution

A reproducible estimator-to-compression training interface, an explicit bounded mean-one algorithm, and executed fixed-wing evidence testing both signal information and weighting efficacy. Improvements, null effects and low-bit failures are outcomes to report. Estimator-specific advantage requires beating shuffled/fault-balanced/sensitivity comparators, not merely uniform QAD.

## 8. Limitations

{limitations}

## 9. Candidate introduction contributions

- A training-only interface connecting physical navigation covariance and innovation consistency to quantized-policy fidelity weighting.
- An interpretable, dimensionless, bounded, mean-normalized distillation algorithm with unchanged deployed policy inputs.
- A reproducible 6-DoF fixed-wing benchmark with whole-trajectory held-out sensor faults and degraded observability.
- Separate empirical tests of incremental signal information and the closed-loop benefit of weighting, including negative outcomes and matched controls.

## 10. Draft abstract results paragraph

{abstract}
'''
    (ROOT/'docs/AIAA_AVIATION_SUMMARY.md').write_text(paper)


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
        row.update(training_seed=seed,method=method,bits=bits,epochs=meta.get('epochs',0),validation_action_mse=meta.get('validation_mse',np.nan),storage_bytes=Policy.load(checkpoint).storage_bytes(),checkpoint=str(checkpoint.relative_to(ROOT)))
        for key,value in meta.get('weighting',{}).items():
            if isinstance(value,(int,float)):row['weight_'+key]=value
        rows.append(row)
    pd.DataFrame(rows).to_csv(path,index=False)
