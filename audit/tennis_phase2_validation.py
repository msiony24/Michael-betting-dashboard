from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; IN=ROOT/'audit'/'results_phase2'/'tennis_phase2_predictions.csv'; OUT=ROOT/'audit'/'results_phase2'
df=pd.read_csv(IN,parse_dates=['date'])
features=['serve_return','form','opp_strength','surface_win','fatigue','transition','pressure','deciding','experience']

def metrics(g,p):
    y=g.y.to_numpy(float); p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    return {'n':len(g),'accuracy':float(np.mean((p>=.5)==y)),'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),'brier':float(np.mean((p-y)**2))}
train=df[df.date.dt.year<=2025]; test=df[df.date.dt.year==2026]
base_tr=metrics(train,train.p_core); base_te=metrics(test,test.p_core)
rows=[]; grid=np.arange(-0.5,2.01,.05)
for f in features:
    best=None
    for mult in grid:
        p=np.clip(train.p_core+mult*train['adj_'+f],.05,.95); m=metrics(train,p)
        if best is None or m['log_loss']<best[0]:best=(m['log_loss'],mult,m)
    mult=best[1]; mt=metrics(test,np.clip(test.p_core+mult*test['adj_'+f],.05,.95))
    rows.append({'feature':f,'train_best_multiplier':mult,'train_delta_log_loss':best[2]['log_loss']-base_tr['log_loss'],'test_delta_log_loss':mt['log_loss']-base_te['log_loss'],'test_delta_brier':mt['brier']-base_te['brier'],'test_delta_accuracy':mt['accuracy']-base_te['accuracy']})
res=pd.DataFrame(rows).sort_values('test_delta_log_loss');res.to_csv(OUT/'tennis_phase2_holdout_feature_validation.csv',index=False)
# forward selection trained on 2024-25, tested 2026 using fixed current factor shapes and multipliers
selected=[]; cur_train=train.p_core.to_numpy(float); cur_test=test.p_core.to_numpy(float); remaining=features.copy(); fs=[]
for step in range(len(features)):
    best=None
    for f in remaining:
        for mult in np.arange(0,1.51,.1):
            p=np.clip(cur_train+mult*train['adj_'+f].to_numpy(float),.05,.95); ll=metrics(train,p)['log_loss']
            if best is None or ll<best[0]:best=(ll,f,mult,p)
    if best[0] >= metrics(train,cur_train)['log_loss']-1e-6:break
    _,f,mult,p=best; selected.append((f,mult));cur_train=p;cur_test=np.clip(cur_test+mult*test['adj_'+f].to_numpy(float),.05,.95);remaining.remove(f)
    mt=metrics(test,cur_test); fs.append({'step':len(selected),'feature':f,'multiplier':mult,'train_log_loss':metrics(train,cur_train)['log_loss'],'test_log_loss':mt['log_loss'],'test_accuracy':mt['accuracy'],'test_brier':mt['brier']})
pd.DataFrame(fs).to_csv(OUT/'tennis_phase2_forward_selection.csv',index=False)
print('BASE TRAIN',base_tr); print('BASE TEST 2026',base_te);print('\nHOLDOUT VALIDATION\n',res.to_string(index=False,float_format=lambda x:f'{x:.6f}'));print('\nFORWARD\n',pd.DataFrame(fs).to_string(index=False,float_format=lambda x:f'{x:.6f}'))
