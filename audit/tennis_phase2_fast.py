from __future__ import annotations
import math,re
from collections import defaultdict
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'audit'/'results_phase2'; P1=ROOT/'audit'/'results'/'tennis_core_predictions.csv'
TS=pd.Timedelta

def key(x): return ' '.join(str(x or '').strip().casefold().split())
def elo_prob(a,b): return 1/(1+10**((b-a)/400))
def logistic(x): return 1/(1+math.exp(-x))
def sets(score): return len(re.findall(r'\d+\s*-\s*\d+',str(score or '')))
def num(x):
    try:
        v=float(x); return v if math.isfinite(v) else None
    except: return None

def load():
    fs=[]
    for p in sorted(DATA.glob('atp_matches_20*.csv')):
        f=pd.read_csv(p,low_memory=False); fs.append(f)
    d=pd.concat(fs,ignore_index=True,sort=False)
    d['date']=pd.to_datetime(d.tourney_date.astype(str),format='%Y%m%d',errors='coerce')
    return d.dropna(subset=['date','winner_name','loser_name']).sort_values(['date','tourney_name','round']).reset_index(drop=True)

def rec(r,pk):
    won=key(r.winner_name)==pk; s='w' if won else 'l'; o='l' if won else 'w'
    return {'date':r.date,'won':won,'surface':str(r.surface),'level':str(r.tourney_level),'round':str(r.round),'score':str(r.score),
            'rank':num(getattr(r,'winner_rank' if won else 'loser_rank',None)),
            'opp':str(r.loser_name if won else r.winner_name),'opp_rank':num(getattr(r,'loser_rank' if won else 'winner_rank',None)),
            'svpt':num(getattr(r,f'{s}_svpt',None)),'fw':num(getattr(r,f'{s}_1stWon',None)),'sw':num(getattr(r,f'{s}_2ndWon',None)),
            'osvpt':num(getattr(r,f'{o}_svpt',None)),'ofw':num(getattr(r,f'{o}_1stWon',None)),'osw':num(getattr(r,f'{o}_2ndWon',None))}

def since(h,date,days):
    c=date-TS(days=days); return [x for x in h if x['date']>=c]
def bprof(h,date,surface):
    two=since(h,date,730); recent=h[-10:]; surf=[x for x in two if x['surface'].casefold()==str(surface).casefold()]
    adv=[x for x in two if x['round'] in ('QF','SF','F')]; big=[x for x in two if x['level'] in ('G','M','F')]; dec=[x for x in two if sets(x['score'])>=3]
    mean=lambda xs: sum(x['won'] for x in xs)/len(xs) if xs else .5
    return {'surface':mean(surf) if surf else mean(recent),'adv':mean(adv) if len(adv)>=4 else .5,'big':mean(big) if len(big)>=5 else .5,'dec':mean(dec) if len(dec)>=4 else .5,'sample':len(two)}
def oppprof(h,elo):
    x=h[-10:]
    if not x:return {'q':.5,'s':.5,'raw':.5}
    rs=[]; qs=[]; es=[]; ranks=[]
    for z in x:
        oe=elo.get(key(z['opp']),1500.0); es.append(oe)
        if z['opp_rank'] is not None:ranks.append(z['opp_rank'])
        oq=logistic((oe-1500)/170); rs.append(1.0 if z['won'] else 0.0); qs.append(.55+.45*oq if z['won'] else .45*oq)
    raw=sum(rs)/len(rs); q=.6*raw+.4*(sum(qs)/len(qs)); ae=sum(es)/len(es); ar=sum(ranks)/len(ranks) if ranks else None
    ec=logistic((ae-1500)/160); rc=logistic(-(ar-75)/35) if ar is not None else .5
    return {'q':q,'s':.6*ec+.4*rc,'raw':raw}
def srprof(h,date,surface):
    groups=[(since(h,date,365),.5),(since(h,date,90),.3),([x for x in since(h,date,730) if x['surface'].casefold()==str(surface).casefold()],.2)]
    def agg(xs):
        ss=[x for x in xs if None not in (x['svpt'],x['fw'],x['sw'])]; rr=[x for x in xs if None not in (x['osvpt'],x['ofw'],x['osw'])]
        sp=sum(x['svpt'] for x in ss); sw=sum(x['fw']+x['sw'] for x in ss); rp=sum(x['osvpt'] for x in rr); rw=sum(x['osvpt']-x['ofw']-x['osw'] for x in rr)
        return {'spw':sw/sp if sp else None,'rpw':rw/rp if rp else None,'sp':sp,'rp':rp,'cov':max(len(ss),len(rr))}
    ag=[(agg(g),w) for g,w in groups]
    def blend(m,base,points):
        vals=[(a[m],w) for a,w in ag if a[m] is not None]
        if not vals:return None
        raw=sum(v*w for v,w in vals)/sum(w for _,w in vals); total=sum(a[points] for a,_ in ag); sh=total/(total+350)
        return base+(raw-base)*sh
    spw=blend('spw',.635,'sp'); rpw=blend('rpw',.365,'rp'); cov=ag[0][0]['cov']
    return {'ok':spw is not None and rpw is not None and cov>=3,'spw':spw,'rpw':rpw,'cov':cov}
def sradj(a,b):
    if not a['ok'] or not b['ok']:return 0.0
    ea=np.clip(.635+(a['spw']-.635)-(b['rpw']-.365),.50,.78); eb=np.clip(.635+(b['spw']-.635)-(a['rpw']-.365),.50,.78); scale=np.clip(min(a['cov'],b['cov'])/12,.35,1)
    return float(np.clip((ea-eb)*.42*scale,-.032,.032))
def fatigue(h,date,q,raw):
    r3=since(h,date,3);r7=since(h,date,7);r14=since(h,date,14)
    if not h:return 0.0
    weeks=min(len(set(x['date'].to_period('W') for x in r14)),3);rest=max(0,(date-h[-1]['date']).days)
    sc=len(r3)*1.2+len(r7)*.65+len(r14)*.18+sum(sets(x['score']) for x in r3)*.22+sum(sets(x['score']) for x in r7)*.10+sum(sets(x['score'])>=3 for x in r7)*.75+max(0,weeks-1)*.7-min(rest,7)*.25
    if len(r7)>=3:sc*=1-min(max(0,.65*q+.35*raw-.58)*2,.30)
    return sc
def trans(h,date,surface):
    if not h:return .5
    ch=h[-1]['surface'].casefold()!=str(surface).casefold(); same=[x for x in h if x['surface'].casefold()==str(surface).casefold()]; recent=[x for x in same if x['date']>=date-TS(days=30)]
    ds=None if not same else (date-same[-1]['date']).days; a=.5+(0 if ch else .2)+min(len(recent),4)*.075
    if ch and not recent:a-=.18
    if ds is not None and ds>60:a-=.1
    return float(np.clip(a,.1,.9))
def exscore(h,surface,rankok):
    if not h:return 0.0
    surf=sum(x['surface'].casefold()==str(surface).casefold() for x in h); gm=sum(x['level'] in ('G','M') for x in h); t50=sum(x['opp_rank'] is not None and x['opp_rank']<=50 for x in h)
    rel=.55*min(len(h)/120,1)+.25*min(max(sum(x['surface'].casefold()==s for x in h) for s in ('hard','clay','grass','carpet'))/60,1)+.2*(1 if rankok else .45)
    return .32*math.log1p(len(h))+.24*math.log1p(surf)+.18*math.log1p(gm)+.16*math.log1p(t50)+.1*rel
def pmult(level,r):
    return {'G':1.35,'M':1.15,'F':1.25,'D':1.2,'C':.65}.get(level,.8)*{'Q':.25,'R128':.3,'R64':.4,'R32':.55,'R16':.75,'QF':1,'SF':1.2,'F':1.35}.get(r,.55)
def met(df,c):
    p=np.clip(df[c].to_numpy(float),1e-9,1-1e-9);y=df.y.to_numpy(float)
    return {'n':len(df),'accuracy':float(np.mean((p>=.5)==y)),'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),'brier':float(np.mean((p-y)**2)),'mean_confidence':float(np.mean(np.maximum(p,1-p)))}

def main():
    m=load(); p1=pd.read_csv(P1,parse_dates=['date']); start=p1.date.min(); histories=defaultdict(list); elo={}
    def eu(w,l):
        rw=elo.get(w,1500);rl=elo.get(l,1500);e=elo_prob(rw,rl);d=24*(1-e);elo[w]=rw+d;elo[l]=rl-d
    rows_by_date=defaultdict(list); lookup={}
    for r in m.itertuples(index=False): rows_by_date[r.date].append(r); lookup[(r.date.date(),str(r.tourney_name),frozenset((key(r.winner_name),key(r.loser_name))))]=r
    for r in m[m.date<start].itertuples(index=False):
        w,l=key(r.winner_name),key(r.loser_name);histories[w].append(rec(r,w));histories[l].append(rec(r,l));eu(w,l)
    out=[]
    for date,day in p1.groupby('date',sort=True):
        for pr in day.itertuples(index=False):
            a,b=key(pr.player_a),key(pr.player_b); r=lookup.get((date.date(),str(pr.tournament),frozenset((a,b))))
            if r is None:continue
            ha,hb=histories[a],histories[b]; pa,pb=bprof(ha,date,pr.surface),bprof(hb,date,pr.surface); oa,ob=oppprof(ha,elo),oppprof(hb,elo)
            fmult=1.12 if str(pr.round) in ('Q','R128','R64','R32') else .95
            adjs={}
            adjs['form']=float(np.clip((oa['q']-ob['q'])*.04*fmult,-.045,.045))*.65
            adjs['opp_strength']=float(np.clip((oa['s']-ob['s'])*.045,-.025,.025))*.65
            adjs['surface_win']=float(np.clip((pa['surface']-pb['surface'])*.045,-.04,.04))*.65
            sa,sb=srprof(ha,date,pr.surface),srprof(hb,date,pr.surface);adjs['serve_return']=sradj(sa,sb)
            fm=1.18 if str(pr.round) in ('QF','SF','F') else 1.0;adjs['fatigue']=float(np.clip((fatigue(hb,date,ob['q'],ob['raw'])-fatigue(ha,date,oa['q'],oa['raw']))*.007*fm,-.055,.055))
            adjs['transition']=float(np.clip((trans(ha,date,pr.surface)-trans(hb,date,pr.surface))*.045,-.035,.035))
            adjs['pressure']=float(np.clip(((pa['adv']-pb['adv'])*.035+(pa['big']-pb['big'])*.025)*pmult(str(r.tourney_level),str(r.round)),-.05,.05))
            adjs['deciding']=float(np.clip((pa['dec']-pb['dec'])*.02,-.03,.03))
            ea,eb=exscore(ha,pr.surface,True),exscore(hb,pr.surface,True);adjs['experience']=.04*math.tanh((ea-eb)/2.5)
            core=float(pr.p_v097_core); d={'date':date,'surface':pr.surface,'round':pr.round,'player_a':pr.player_a,'player_b':pr.player_b,'y':pr.y,'p_core':core,'sr_available':sa['ok'] and sb['ok'],'sample_min':min(pa['sample'],pb['sample'])}
            for n,v in adjs.items():d['adj_'+n]=v;d['p_'+n]=float(np.clip(core+v,.05,.95))
            sec=np.clip(sum(adjs.values()),-.12,.12);d['p_all_tested']=float(np.clip(core+sec,.05,.95));strength=.88 if d['sample_min']>=40 and d['sr_available'] else (.82 if d['sample_min']>=20 else .76);d['p_core_shrunk']=float(np.clip(.5+(core-.5)*strength,.08,.92));d['p_all_tested_shrunk']=float(np.clip(.5+(d['p_all_tested']-.5)*strength,.08,.92));out.append(d)
        for r in rows_by_date.get(date,[]):
            w,l=key(r.winner_name),key(r.loser_name);histories[w].append(rec(r,w));histories[l].append(rec(r,l));eu(w,l)
    df=pd.DataFrame(out);OUT.mkdir(parents=True,exist_ok=True);df.to_csv(OUT/'tennis_phase2_predictions.csv',index=False)
    cols=['p_core']+[f'p_{n}' for n in ['serve_return','form','opp_strength','surface_win','fatigue','transition','pressure','deciding','experience']]+['p_all_tested','p_core_shrunk','p_all_tested_shrunk']
    sm=pd.DataFrame([{'model':c,**met(df,c)} for c in cols]).sort_values('log_loss');sm.to_csv(OUT/'tennis_phase2_summary.csv',index=False)
    base=met(df,'p_core'); eff=[]
    for c in cols[1:]:
        x=met(df,c);eff.append({'model':c,'delta_accuracy':x['accuracy']-base['accuracy'],'delta_log_loss':x['log_loss']-base['log_loss'],'delta_brier':x['brier']-base['brier']})
    ef=pd.DataFrame(eff).sort_values('delta_log_loss');ef.to_csv(OUT/'tennis_phase2_feature_effects.csv',index=False)
    print(sm.to_string(index=False,float_format=lambda x:f'{x:.6f}'));print('\n',ef.to_string(index=False,float_format=lambda x:f'{x:.6f}'));print('N',len(df))
if __name__=='__main__':main()
