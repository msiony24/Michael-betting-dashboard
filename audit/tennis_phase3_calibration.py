from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Support running inside an audit-only package or copied into the live repo.
CANDIDATES = [
    ROOT / 'audit' / 'results_phase2' / 'tennis_phase2_predictions.csv',
    ROOT / 'audit' / 'phase2_predictions_input.csv',
]
IN = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])
OUT = ROOT / 'audit' / 'results_phase3'
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ['serve_return','form','opp_strength','surface_win','fatigue','transition','pressure','deciding','experience']

def metrics(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-9, 1-1e-9)
    return {
        'n': int(len(y)),
        'accuracy': float(np.mean((p >= .5) == y)),
        'log_loss': float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
        'brier': float(np.mean((p-y)**2)),
        'mean_confidence': float(np.mean(np.maximum(p,1-p))),
    }

def apply_feature(g, f, mult):
    return np.clip(g.p_core.to_numpy(float) + mult*g['adj_'+f].to_numpy(float), .05, .95)

def best_multiplier(g, f, lo=-2.0, hi=3.0, step=.05):
    base = metrics(g.y, g.p_core)
    best = None
    for mult in np.arange(lo, hi + step/2, step):
        p = apply_feature(g, f, mult)
        m = metrics(g.y, p)
        if best is None or m['log_loss'] < best['log_loss']:
            best = {'multiplier': float(mult), **m}
    best['delta_log_loss_vs_core'] = best['log_loss'] - base['log_loss']
    return best

def best_shrink(g, lo=.5, hi=1.30, step=.01):
    base = metrics(g.y, g.p_core)
    best = None
    core = g.p_core.to_numpy(float)
    for s in np.arange(lo, hi + step/2, step):
        p = np.clip(.5 + (core-.5)*s, .05, .95)
        m = metrics(g.y, p)
        if best is None or m['log_loss'] < best['log_loss']:
            best = {'shrink_strength': float(s), **m}
    best['delta_log_loss_vs_core'] = best['log_loss'] - base['log_loss']
    return best

def best_platt(g, slope_grid=np.arange(.7,1.41,.02), intercept_grid=np.arange(-.12,.121,.01)):
    p0 = np.clip(g.p_core.to_numpy(float), 1e-6, 1-1e-6)
    z = np.log(p0/(1-p0))
    y = g.y.to_numpy(float)
    base = metrics(y,p0)
    best = None
    for a in slope_grid:
        for b in intercept_grid:
            p = 1/(1+np.exp(-(a*z+b)))
            m = metrics(y,p)
            if best is None or m['log_loss'] < best['log_loss']:
                best = {'slope':float(a),'intercept':float(b),**m}
    best['delta_log_loss_vs_core'] = best['log_loss']-base['log_loss']
    return best

def evaluate_fixed(g, f, mult):
    return metrics(g.y, apply_feature(g,f,mult))

def forward_select(train, candidates, max_steps=6):
    cur = train.p_core.to_numpy(float).copy()
    remaining = list(candidates)
    selected = []
    current_ll = metrics(train.y, cur)['log_loss']
    for step in range(max_steps):
        best = None
        for f in remaining:
            adj = train['adj_'+f].to_numpy(float)
            for mult in np.arange(-1.5, 2.51, .1):
                p = np.clip(cur + mult*adj, .05,.95)
                ll = metrics(train.y,p)['log_loss']
                if best is None or ll < best[0]: best=(ll,f,float(mult),p)
        if best is None or best[0] >= current_ll - 1e-6:
            break
        current_ll,f,mult,cur=best
        selected.append((f,mult))
        remaining.remove(f)
    return selected

def apply_sequence(g, seq):
    p=g.p_core.to_numpy(float).copy()
    for f,m in seq:
        p=np.clip(p+m*g['adj_'+f].to_numpy(float),.05,.95)
    return p

df = pd.read_csv(IN, parse_dates=['date'])
df['year'] = df.date.dt.year
train24 = df[df.year==2024].copy()
val25 = df[df.year==2025].copy()
train2425 = df[df.year<=2025].copy()
test26 = df[df.year==2026].copy()

# 1) Temporal multiplier stability: fit on 2024, validate 2025; then fit 2024-25, validate 2026.
rows=[]
for f in FEATURES:
    b24=best_multiplier(train24,f)
    m25=evaluate_fixed(val25,f,b24['multiplier'])
    base25=metrics(val25.y,val25.p_core)
    b2425=best_multiplier(train2425,f)
    m26=evaluate_fixed(test26,f,b2425['multiplier'])
    base26=metrics(test26.y,test26.p_core)
    rows.append({
        'feature':f,
        'fit_2024_multiplier':b24['multiplier'],
        'fit_2024_delta_log_loss':b24['delta_log_loss_vs_core'],
        'validate_2025_delta_log_loss':m25['log_loss']-base25['log_loss'],
        'validate_2025_delta_brier':m25['brier']-base25['brier'],
        'fit_2024_25_multiplier':b2425['multiplier'],
        'fit_2024_25_delta_log_loss':b2425['delta_log_loss_vs_core'],
        'validate_2026_delta_log_loss':m26['log_loss']-base26['log_loss'],
        'validate_2026_delta_brier':m26['brier']-base26['brier'],
        'validate_2026_delta_accuracy':m26['accuracy']-base26['accuracy'],
    })
stability=pd.DataFrame(rows)
stability['stable_positive']=(stability.validate_2025_delta_log_loss<0)&(stability.validate_2026_delta_log_loss<0)
stability.to_csv(OUT/'tennis_phase3_multiplier_stability.csv',index=False)

# 2) Calibration: direct shrink and Platt-style slope/intercept.
cal_rows=[]
for train_name,train,test_name,test in [('2024',train24,'2025',val25),('2024-25',train2425,'2026',test26)]:
    bs=best_shrink(train); bp=best_platt(train)
    core_test=metrics(test.y,test.p_core)
    ps=np.clip(.5+(test.p_core.to_numpy(float)-.5)*bs['shrink_strength'],.05,.95)
    ms=metrics(test.y,ps)
    p0=np.clip(test.p_core.to_numpy(float),1e-6,1-1e-6); z=np.log(p0/(1-p0)); pp=1/(1+np.exp(-(bp['slope']*z+bp['intercept']))); mp=metrics(test.y,pp)
    cal_rows += [
        {'fit':train_name,'validate':test_name,'method':'core','parameter_1':1.0,'parameter_2':0.0,**core_test,'delta_log_loss_vs_core':0.0},
        {'fit':train_name,'validate':test_name,'method':'linear_shrink','parameter_1':bs['shrink_strength'],'parameter_2':0.0,**ms,'delta_log_loss_vs_core':ms['log_loss']-core_test['log_loss']},
        {'fit':train_name,'validate':test_name,'method':'platt','parameter_1':bp['slope'],'parameter_2':bp['intercept'],**mp,'delta_log_loss_vs_core':mp['log_loss']-core_test['log_loss']},
    ]
pd.DataFrame(cal_rows).to_csv(OUT/'tennis_phase3_calibration_validation.csv',index=False)

# 3) Forward selection with temporal validation.
seq24=forward_select(train24,FEATURES); seq2425=forward_select(train2425,FEATURES)
fs=[]
for fit_name,seq,val_name,val in [('2024',seq24,'2025',val25),('2024-25',seq2425,'2026',test26)]:
    base=metrics(val.y,val.p_core); p=val.p_core.to_numpy(float).copy()
    fs.append({'fit':fit_name,'validate':val_name,'step':0,'feature':'CORE','multiplier':0.0,**base,'delta_log_loss_vs_core':0.0})
    for i,(f,m) in enumerate(seq,1):
        p=np.clip(p+m*val['adj_'+f].to_numpy(float),.05,.95); mm=metrics(val.y,p)
        fs.append({'fit':fit_name,'validate':val_name,'step':i,'feature':f,'multiplier':m,**mm,'delta_log_loss_vs_core':mm['log_loss']-base['log_loss']})
pd.DataFrame(fs).to_csv(OUT/'tennis_phase3_forward_temporal.csv',index=False)

# 4) Serve/return diagnostics: coverage and performance only where available.
sr=[]
for year,g in df.groupby('year'):
    for surf,gg in list(g.groupby('surface'))+[('ALL',g)]:
        avail=gg[gg.sr_available.astype(bool)]
        if len(avail)==0:
            sr.append({'year':year,'surface':surf,'n_total':len(gg),'n_available':0,'coverage':0.0})
            continue
        b=best_multiplier(avail,'serve_return')
        core=metrics(avail.y,avail.p_core); current=metrics(avail.y,avail.p_serve_return)
        sr.append({'year':year,'surface':surf,'n_total':len(gg),'n_available':len(avail),'coverage':len(avail)/len(gg),
                   'core_log_loss':core['log_loss'],'current_sr_log_loss':current['log_loss'],'current_delta_log_loss':current['log_loss']-core['log_loss'],
                   'best_in_sample_multiplier':b['multiplier'],'best_in_sample_delta_log_loss':b['delta_log_loss_vs_core']})
pd.DataFrame(sr).to_csv(OUT/'tennis_phase3_serve_return_diagnostics.csv',index=False)

# 5) Fatigue diagnostics: sign, magnitude buckets, and temporal optimal multiplier.
fat=[]
for year,g in df.groupby('year'):
    # Compare residual win rate to core expectation by adjustment sign/magnitude.
    bins=[-1,-.03,-.015,-.005,.005,.015,.03,1]
    labels=['<=-3pp','-3:-1.5pp','-1.5:-0.5pp','near_zero','0.5:1.5pp','1.5:3pp','>=3pp']
    gg=g.copy(); gg['bucket']=pd.cut(gg.adj_fatigue,bins=bins,labels=labels,include_lowest=True)
    for bucket,x in gg.groupby('bucket',observed=False):
        if len(x)==0: continue
        fat.append({'year':year,'bucket':str(bucket),'n':len(x),'mean_adj':x.adj_fatigue.mean(),'actual_win_rate':x.y.mean(),'core_mean_p':x.p_core.mean(),'residual_actual_minus_core':x.y.mean()-x.p_core.mean()})
pd.DataFrame(fat).to_csv(OUT/'tennis_phase3_fatigue_diagnostics.csv',index=False)

# 6) Candidate v0.98 spec: only factors positive in BOTH temporal validations, with conservative multipliers.
# Conservative = median of the two independently fitted multipliers, clipped to avoid extrapolation.
stable=stability[stability.stable_positive].copy()
stable['recommended_multiplier']=stable[['fit_2024_multiplier','fit_2024_25_multiplier']].median(axis=1).clip(-1.5,2.0)
# Require 2026 gain as primary and no 2025 harm.
stable=stable.sort_values('validate_2026_delta_log_loss')
seq=[(r.feature,float(r.recommended_multiplier)) for r in stable.itertuples()]
base26=metrics(test26.y,test26.p_core); p26=apply_sequence(test26,seq); cand26=metrics(test26.y,p26)
base25=metrics(val25.y,val25.p_core); p25=apply_sequence(val25,seq); cand25=metrics(val25.y,p25)

spec_rows=[]
for r in stability.sort_values('validate_2026_delta_log_loss').itertuples():
    action='KEEP_CURRENT_SHAPE_RECALIBRATE' if bool(r.stable_positive) else 'DO_NOT_ADD_UNTIL_REDESIGNED'
    if r.feature=='fatigue' and r.fit_2024_25_multiplier<0: action='REDESIGN_SIGN_OR_CONSTRUCTION'
    if r.feature=='serve_return': action='REDESIGN_DATA_COVERAGE_AND_BASELINE'
    spec_rows.append({'feature':r.feature,'action':action,'fit_2024_multiplier':r.fit_2024_multiplier,'fit_2024_25_multiplier':r.fit_2024_25_multiplier,
                      'validate_2025_delta_log_loss':r.validate_2025_delta_log_loss,'validate_2026_delta_log_loss':r.validate_2026_delta_log_loss,
                      'recommended_multiplier_if_kept':float(np.median([r.fit_2024_multiplier,r.fit_2024_25_multiplier])) if bool(r.stable_positive) else 0.0})
pd.DataFrame(spec_rows).to_csv(OUT/'tennis_phase3_v098_feature_spec.csv',index=False)

summary=pd.DataFrame([
    {'evaluation':'2025_core','n':base25['n'],'accuracy':base25['accuracy'],'log_loss':base25['log_loss'],'brier':base25['brier']},
    {'evaluation':'2025_conservative_candidate','n':cand25['n'],'accuracy':cand25['accuracy'],'log_loss':cand25['log_loss'],'brier':cand25['brier']},
    {'evaluation':'2026_core','n':base26['n'],'accuracy':base26['accuracy'],'log_loss':base26['log_loss'],'brier':base26['brier']},
    {'evaluation':'2026_conservative_candidate','n':cand26['n'],'accuracy':cand26['accuracy'],'log_loss':cand26['log_loss'],'brier':cand26['brier']},
])
summary.to_csv(OUT/'tennis_phase3_candidate_summary.csv',index=False)

print('INPUT',IN)
print('\nMULTIPLIER STABILITY')
print(stability.sort_values('validate_2026_delta_log_loss').to_string(index=False,float_format=lambda x:f'{x:.6f}'))
print('\nCALIBRATION')
print(pd.DataFrame(cal_rows).to_string(index=False,float_format=lambda x:f'{x:.6f}'))
print('\nTEMPORAL FORWARD')
print(pd.DataFrame(fs).to_string(index=False,float_format=lambda x:f'{x:.6f}'))
print('\nCONSERVATIVE STABLE SEQUENCE',seq)
print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
