"""Preregistered Study 2 signal gate. TEST outcomes are never read here."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ecqad_uav.utils.config import project_root,load_yaml
ROOT=project_root();OUT=ROOT/'results/study2';CFG=load_yaml('study2.yaml')


def interval(frame,base,other):
    seeds=sorted(frame.training_seed.unique())
    cells=frame.pivot(index=['condition','episode'],columns='training_seed',values=[base,other]).dropna()
    groups=[np.stack([g[base][seeds].to_numpy(),g[other][seeds].to_numpy()],axis=-1) for _,g in cells.groupby(level='condition')]
    rng=np.random.default_rng(CFG['signal']['bootstrap_seed']);values=[]
    for _ in range(CFG['signal']['bootstrap_draws']):
        si=rng.integers(len(seeds),size=len(seeds));means=[]
        for g in groups:means.append(g[rng.integers(len(g),size=len(g))][:,si].mean((0,1)))
        a,b=np.mean(means,axis=0);values.append(1-b/max(a,1e-15))
    return np.quantile(values,CFG['signal']['interval_quantiles'])


def episode_cv(group):
    category=pd.get_dummies(group.condition).to_numpy(float)
    active=((group.time_s>=group.onset_s)&(group.time_s<group.onset_s+group.duration_s)).to_numpy(float)
    base=np.column_stack([pd.get_dummies(group.training_seed).to_numpy(float),category,pd.get_dummies(group.gyro_axis.fillna('none')).to_numpy(float),active,group.time_s/180,(group.time_s/180)**2,group.onset_s/180,group.duration_s/180,np.abs(group.feature_3),np.abs(group.feature_4),np.log1p(group.tracking),group[[f'feature_{j}' for j in (12,13,14,15)]],category*active[:,None],category*group.severity.to_numpy()[:,None],category*np.abs(group.severity.to_numpy())[:,None]])
    c=group.covariance.to_numpy();j=group.sensitivity.to_numpy();q=np.log1p(group.q.to_numpy());n=group.nis.to_numpy()
    designs={'baseline':base,'covariance':np.column_stack([base,c]),'trace_P':np.column_stack([base,group.trace_P]),'sensitivity':np.column_stack([base,j]),'q':np.column_stack([base,q]),'nis':np.column_stack([base,n]),'covariance+sensitivity':np.column_stack([base,c,j]),'covariance+sensitivity+q':np.column_stack([base,c,j,q])}
    contrasts=[('baseline',k) for k in ['covariance','trace_P','sensitivity','q','nis']]+[('covariance+sensitivity','covariance+sensitivity+q')]
    rows=[];details=[]
    for target in ['action_error','future_tracking','excess_future_tracking']:
        valid=np.isfinite(group[target]);g=group.loc[valid];y=np.arcsinh(g[target].to_numpy());folds=g.episode.to_numpy()%CFG['signal']['folds'];detail=g[['training_seed','condition','episode']].copy()
        for label,full in designs.items():
            design=full[valid];pred=np.full(len(y),np.nan)
            for f in range(CFG['signal']['folds']):
                tr=folds!=f;te=folds==f;mean=design[tr].mean(0);scale=np.maximum(design[tr].std(0),1e-12)
                train=np.column_stack([np.ones(tr.sum()),(design[tr]-mean)/scale]);test=np.column_stack([np.ones(te.sum()),(design[te]-mean)/scale])
                ridge=np.eye(train.shape[1])*CFG['signal']['ridge'];ridge[0,0]=0
                pred[te]=test@np.linalg.solve(train.T@train+ridge,train.T@y[tr])
            detail[label]=(pred-y)**2
        ep=detail.groupby(['training_seed','condition','episode']).mean().reset_index();ep['target']=target;details.append(ep)
        for b,a in contrasts:
            lo,hi=interval(ep,b,a);byseed=ep.groupby('training_seed')[[b,a]].mean();directions=1-byseed[a]/byseed[b]
            rows.append(dict(kind='incremental_prediction',target=target,baseline=b,predictor=a,reduction=1-ep[a].mean()/ep[b].mean(),ci_low=lo,ci_high=hi,positive_seeds=int((directions>0).sum()),seeds=len(directions),episode_groups=g.episode_seed.nunique(),samples=len(g),seed_reductions=json.dumps(directions.to_dict())))
        for cond,part in [('all',g)]+list(g.groupby('condition')):
            for sig in ['covariance','trace_P','sensitivity','q','nis']:
                rows.append(dict(kind='rank_correlation',target=target,predictor=sig,condition=cond,spearman=float(spearmanr(part[sig],part[target]).statistic)))
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True)


def gate_decision(table):
    primary=table[(table.kind=='incremental_prediction')&(table.baseline=='covariance+sensitivity')&table.target.isin(CFG['signal']['primary_targets'])]
    assert len(primary)==2
    passed=bool(((primary.reduction>0)&(primary.ci_low>0)&(primary.positive_seeds>=CFG['signal']['required_positive_seeds'])).any())
    return dict(passed=passed,criterion=CFG['signal']['gate'],required_positive_seeds=CFG['signal']['required_positive_seeds'],interval='97.5% two-sided per outcome; two prespecified primary outcomes',outcomes=primary.to_dict('records'),test_accessed=False)


def analyze():
    files=sorted((OUT/'signals/validation_signal').glob('*.csv'))
    s=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    assert set(s.split)=={'validation_signal'}
    scope=pd.read_csv(ROOT/'results/study2_evaluation.csv');scope=scope[scope.split=='validation_signal']
    expected=(len(CFG['student_seeds'])+1)*len(CFG['conditions'])*CFG['validation_signal']['episodes_per_condition']
    assert len(scope)==expected,f'Incomplete signal grid {len(scope)}/{expected}'
    assert not scope.pid_executed.any()
    teacher=s[s.method=='teacher'][['episode_seed','time_s','future_tracking']].rename(columns={'future_tracking':'teacher_future'})
    g=s[s.method=='uniform'].merge(teacher,on=['episode_seed','time_s'],validate='many_to_one');g['excess_future_tracking']=g.future_tracking-g.teacher_future
    table,detail=episode_cv(g);table.to_csv(ROOT/'results/study2_signal_analysis.csv',index=False);detail.to_csv(OUT/'signal_episode_errors.csv',index=False)
    gate=gate_decision(table);(OUT/'signal_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    pred=table[table.kind=='incremental_prediction'];lines=['| Predictor | Baseline | Outcome | MSE reduction (%) | 97.5% interval (%) | Positive seeds |','|---|---|---|---:|---:|---:|']
    for r in pred.itertuples():lines.append(f'| {r.predictor} | {r.baseline} | {r.target} | {100*r.reduction:+.4f} | [{100*r.ci_low:+.4f}, {100*r.ci_high:+.4f}] | {int(r.positive_seeds)}/{int(r.seeds)} |')
    text='\n'.join(lines)
    conclusion='PASS: preregistered evidence permits a separately frozen CR-EC-QAD training stage.' if gate['passed'] else 'NO-GO: the propagated signal does not satisfy the predefined evidence gate. No CR-EC-QAD or revised covariance formula will be trained.'
    answers=[]
    for label,key,base in [('A. Raw estimator covariance','covariance','baseline'),('B. Teacher sensitivity','sensitivity','baseline'),('C. Propagated uncertainty beyond covariance and sensitivity','covariance+sensitivity+q','covariance+sensitivity')]:
        t=pred[(pred.predictor==key)&(pred.baseline==base)&pred.target.isin(CFG['signal']['primary_targets'])];positive=t[(t.ci_low>0)&(t.positive_seeds>=4)]
        answers.append(f'**{label}:** '+('Positive incremental evidence on '+', '.join(positive.target)+'.' if len(positive) else 'Not supported on the prespecified future-tracking outcomes under this analysis.'))
    doc=f'''# Study 2 signal analysis

{conclusion}

This analysis uses only VALIDATION_SIGNAL trajectories, disjoint from TRAIN, checkpoint-selection validation, teacher-gate validation, and unopened TEST. It includes five independently optimized uniform four-bit students sharing the predefined accepted bounded teacher. No evaluation outcome selected a checkpoint. These are exploratory simulation findings, not causal effects.

{chr(10).join(answers)}

**D. Uncertainty:** the table reports hierarchical paired bootstrap intervals over student seeds and common scenario groups, conditional on fixed episode-held-out predictions. The two primary outcomes use 97.5% intervals (Bonferroni for two 95%-family tests). A lower bound above zero is required, not merely a positive point estimate.

**E. Seed consistency:** each row records positive-seed counts; exact per-seed reductions are in the CSV. The gate requires at least four of five directions positive for the same primary outcome whose interval excludes zero.

{text}

Whole scenario episodes and all their student-seed copies share a fold (episode index modulo five). Baseline covariates include fault category/active status/severity, time, onset/duration, gyro axis, current truth tracking severity, attitude magnitude, command references, and seed indicators. Fault labels and current truth tracking are analysis-only nuisance covariates, not controller or estimator inputs. Fold-local standardization and fixed ridge 1e-6 predict asinh-transformed outcomes. Episode-average errors receive equal weight; severe failures contribute only observed prefixes with a full five-second future. Missing futures are not imputed. Covariance/NIS rank correlations are descriptive. The primary q test adds log1p(q) beyond both old scalar covariance and unscaled-feature teacher-Jacobian Frobenius norm; raw trace(P) is a unit-dependent diagnostic only.

The score propagates UKF uncertainty conditional on fixed pitot and latest gyro readings. It does not include independent sensor-readout uncertainty outside P, or capture nonlinear finite excursions. A negative finding under this restricted predictive model cannot prove absence of all estimator information. TEST remains unopened; no held-out TEST performance claim follows from these validation results.
'''
    (ROOT/'docs/STUDY2_SIGNAL_ANALYSIS.md').write_text(doc)
    # Descriptive paired uniform-vs-teacher comparison, explicitly validation-only.
    t=scope[scope.method=='teacher'][['condition','episode','tracking_score']]
    u=scope[scope.method=='uniform'].merge(t,on=['condition','episode'],suffixes=('_student','_teacher'))
    rows=[]
    for seed,part in u.groupby('training_seed'):
        rows.append(dict(split='validation_signal',training_seed=seed,comparison='uniform_vs_teacher',student_score=part.tracking_score_student.mean(),teacher_score=part.tracking_score_teacher.mean(),relative_change_pct=100*(part.tracking_score_student.mean()/part.tracking_score_teacher.mean()-1),failed=int(part.failed.sum())))
    pd.DataFrame(rows).to_csv(ROOT/'results/study2_paired_comparisons.csv',index=False)
    metrics=['tracking_score','altitude_rmse_m','heading_rmse_deg','airspeed_rmse_mps','altitude_peak_abs_m','heading_peak_abs_deg','airspeed_peak_abs_mps','failed','constraint_fraction','control_effort','action_rmse']
    summary=scope.groupby(['method','training_seed','condition'])[metrics].mean().reset_index();summary.to_csv(OUT/'validation_summary.csv',index=False)
    (ROOT/'docs/STUDY2_RESULTS.md').write_text(f'''# Study 2 executed results

{conclusion}

There are {len(scope)} completed VALIDATION_SIGNAL rollouts, with {CFG['validation_signal']['episodes_per_condition']} unique scenarios per condition shared across five student seeds and one frozen teacher. These are validation results, not final TEST results. Every episode's physical tracking RMSE, peak error, failure status, control effort, constraint fraction, and teacher–student action discrepancy is in `results/study2_evaluation.csv`; grouped metrics are in `results/study2/validation_summary.csv`. Teacher-seed gates and every candidate checkpoint's offline MSE/closed-loop score are separately recorded in the required prefixed CSVs.

See STUDY2_SIGNAL_ANALYSIS.md for the executed incremental prediction table and STUDY2_RESEARCH_VERDICT.md for the stopping decision. Study 1 is unchanged. No fake-quantization hardware benefit is claimed.
''')
    verdict=f'''# Study 2 research verdict

{conclusion}

1. Raw covariance: {answers[0]}
2. Teacher sensitivity: {answers[1]}
3. Propagated q beyond both: {answers[2]}
4. CR-EC-QAD versus uniform: not tested at this stage; the gate is a prerequisite, not evidence of training benefit.
5. CR-EC-QAD versus exact shuffled control: not tested. Estimator specificity cannot be claimed.
6. Seed consistency: five student optimization seeds were analyzed; target-specific direction counts and intervals appear in STUDY2_SIGNAL_ANALYSIS.md.
7. Held-out severity/onset: validation episodes use stronger/later faults than TRAIN. Final TEST severities/onsets remain unopened; no final-test generalization claim is made.
8. Mechanism: neither estimator-specific improvement nor exclusion of generic nonuniform weighting follows from this signal test.
9. Supported claims: bounded training/deployment action parity, independent physical/reference deployment, full-covariance first-order propagation, disjoint whole-episode selection/analysis, and the executed target-specific results.
10. Unsupported claims: causal criticality, real-flight safety, hardware speed/energy, universal absence of covariance information, or superiority of any untrained CR-EC-QAD method.

Teacher acceptance and failed seeds are in `results/study2_teacher_gate.csv`. All student seeds share the first passing teacher in the preregistered order; teacher robustness is measured separately. No uncertainty, NIS, or Jacobian is a deployed input. No alternate score, fusion, threshold, or nonlinear weighting search was conducted.
'''
    (ROOT/'docs/STUDY2_RESEARCH_VERDICT.md').write_text(verdict)
    if not gate['passed']:(ROOT/'docs/STUDY2_NEGATIVE_VERDICT.md').write_text('# Study 2 negative verdict\n\n'+conclusion+'\n\nThe proposed control-relevant estimator signal did not support proceeding to estimator-specific QAD under this testbed and preregistered analysis. The stronger statement that no useful covariance information exists is not established. TEST remains unopened; weighted-method training and final TEST comparisons are deliberately not run, as required by the no-go rule.\n')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));primary=pred[(pred.baseline=='covariance+sensitivity')];pos=np.arange(len(primary));v=100*primary.reduction.to_numpy();lo=100*primary.ci_low.to_numpy();hi=100*primary.ci_high.to_numpy()
    ax.errorbar(v,pos,xerr=np.maximum(np.array([v-lo,hi-v]),0),fmt='o');ax.axvline(0,color='k',ls='--',lw=.8);ax.set_yticks(pos,primary.target);ax.set_xlabel('Incremental prediction MSE reduction (%)\nq beyond covariance and sensitivity; 97.5% intervals');fig.tight_layout();folder=OUT/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'signal_gate.png',dpi=220);fig.savefig(folder/'signal_gate.pdf');plt.close(fig)
    print(conclusion,flush=True)
