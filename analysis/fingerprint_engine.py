"""Reliability- and robustness-constrained research-fingerprint engine. All fitting is restricted to supplied reference indices."""
from pathlib import Path
from collections import Counter
import hashlib,json,math
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy import stats

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CONFIG=json.loads((ROOT/'configs/analysis_protocol.json').read_text(encoding='utf-8'))
Z=np.array([-.8416,-.2533,.2533,.8416])
NOMINAL={'T1','T34','T40','T41','T42'}
STATE_GRID={'T1':list(range(1,6)),'T2':list(range(1,10)),'T3':[1,2,3],
 'T6':list(range(1,10)),'T7':list(range(1,10)),'T9':list(range(1,10)),
 'T10':list(range(1,10)),'T33':[1,2,3],'T34':[1,2,3],
 'T40':list(range(1,11)),'T41':list(range(1,11)),'T42':list(range(1,6)),
 'T43':list(range(1,10))}

def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value

def write_json(path,value):
    Path(path).write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def load_data(input_dir):
    folder=Path(input_dir).resolve()
    source=folder/'variety_summary.csv'
    materials_frame=pd.read_csv(folder/'materials.csv',dtype=str)
    if materials_frame.Variety.duplicated().any():raise ValueError('Duplicate variety IDs')
    meta=json.loads((ROOT/'data_schema/trait_metadata.json').read_text(encoding='utf-8'))
    frame=pd.read_csv(source,float_precision='round_trip')
    if frame.duplicated(['Variety','Trait']).any():raise ValueError('Duplicate variety-trait rows')
    ids=materials_frame.Variety.tolist();ts=[r['id'] for r in meta]
    expected=pd.MultiIndex.from_product([ids,ts],names=['Variety','Trait'])
    observed=frame.set_index(['Variety','Trait'])
    if set(observed.index)!=set(expected):raise ValueError('Input must contain every variety-trait row, retaining missing values')
    ordered=observed.reindex(expected)
    ref=ordered.Reference.to_numpy(float).reshape(len(ids),len(ts))
    image=ordered.Image.to_numpy(float).reshape(len(ids),len(ts))
    kinds=['nominal' if t in NOMINAL else 'continuous' if meta[j]['source_data_type'] in ['continuous','ratio'] else 'count' if t=='T32' else 'ordered' for j,t in enumerate(ts)]
    return dict(source=source,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),traits=ts,names=[r['name'] for r in meta],
      kinds=kinds,ids=ids,reference=ref,image=image,materials={r.Variety:r for r in materials_frame.itertuples()})

def fit_codebook(reference,traits,kinds):
    params=[]
    for j,(t,kind) in enumerate(zip(traits,kinds)):
        x=reference[:,j];x=x[np.isfinite(x)]
        if kind=='continuous':
            mu=x.mean();sd=x.std(ddof=0)
            params.append(dict(Trait=t,Kind=kind,Mean=mu,SD_population=sd,Edges=mu+sd*Z,Distance_range=4.0))
        else:
            span=np.ptp(np.rint(x)) if kind=='count' else np.ptp(STATE_GRID[t])
            params.append(dict(Trait=t,Kind=kind,Distance_range=max(1,float(span))))
    return params

def encode(values,params):
    c=np.full(values.shape,-999,dtype=np.int16)
    for j,p in enumerate(params):
        a=values[...,j];valid=np.isfinite(a)
        if p['Kind']=='continuous':c[...,j][valid]=np.searchsorted(p['Edges'],a[valid],side='right')+1 if p['SD_population']>0 else 3
        else:c[...,j][valid]=np.rint(a[valid]).astype(np.int16)
    return c

def agreement(x,y,kind,trait):
    good=np.isfinite(x)&np.isfinite(y);a=x[good];b=y[good]
    if len(a)==0:return np.nan,np.nan,0,np.nan
    if kind in ['continuous','count']:
        den=a.var()+b.var()+(a.mean()-b.mean())**2
        rel=2*np.mean((a-a.mean())*(b-b.mean()))/den if den>0 else np.nan
        mape=100*np.mean(abs((b-a)/a)) if np.all(a!=0) else np.nan
    else:
        grid=np.array(STATE_GRID[trait]);idx={int(z):k for k,z in enumerate(grid)}
        cm=np.zeros((len(grid),len(grid)))
        for ar,br in zip(a,b):cm[idx[int(ar)],idx[int(br)]]+=1
        weight=(grid[:,None]!=grid[None,:]).astype(float) if kind=='nominal' else abs(grid[:,None]-grid[None,:]).astype(float)
        ex=np.outer(cm.sum(1),cm.sum(0))/len(a);den=np.sum(ex*weight)
        rel=1-np.sum(cm*weight)/den if den>0 else np.nan;mape=np.nan
    return rel,mape,len(a),float(np.mean(a==b))

def screen(data,train,params,config=CONFIG,exclude_extra=()):
    r=data['reference'][train];im=data['image'][train];rc=encode(r,params);ic=encode(im,params)
    pool=[];rows=[]
    for j,(t,kind) in enumerate(zip(data['traits'],data['kinds'])):
        rel,mape,n,exact=agreement(r[:,j],im[:,j],kind,t)
        dom=max(Counter(rc[:,j]).most_common(1)[0][1],Counter(ic[:,j]).most_common(1)[0][1])/len(train)
        reasons=[]
        if t in config['exclude_independent_traits']:reasons.append('deterministic_derived_trait')
        if t in exclude_extra:reasons.append('predefined_sensitivity_exclusion')
        if n<len(train)*config['required_pair_coverage']:reasons.append('incomplete_reference_pairs')
        if not np.isfinite(rel):reasons.append('undefined_primary_agreement')
        elif kind in ['continuous','count']:
            if rel<config['numeric_ccc_min']:reasons.append('CCC_below_threshold')
            if not np.isfinite(mape) or mape>=config['numeric_mape_max_exclusive']:reasons.append('MAPE_not_below_threshold')
        elif rel<(config['nominal_unweighted_kappa_min'] if kind=='nominal' else config['ordered_linear_kappa_min']):reasons.append('kappa_below_threshold')
        if dom>=config['dominant_code_fraction_max_exclusive']:reasons.append('dominant_state')
        eligible=not reasons
        if eligible:pool.append(j)
        rows.append(dict(Trait=t,Name=data['names'][j],Kind=kind,Paired_reference_n=n,Primary_agreement=rel,MAPE_percent=mape,Exact_raw_state_fraction=exact if kind not in ['continuous','count'] else np.nan,Reference_program_code_agreement=float(np.mean((rc[:,j]==ic[:,j])[np.isfinite(r[:,j])&np.isfinite(im[:,j])])) if n else np.nan,Dominant_code_fraction=dom,Eligible=eligible,Reasons=';'.join(reasons),Study_scope='Research descriptor across panel; official applicability separately mapped'))
    return pool,pd.DataFrame(rows),rc,ic

def fit_model(data,train,pool,params):
    ref=data['reference'][train][:,pool];im=data['image'][train][:,pool]
    kinds=[data['kinds'][j] for j in pool];ts=[data['traits'][j] for j in pool]
    numeric=[k for k,z in enumerate(kinds) if z in ['continuous','count']]
    absolute=im[:,numeric]-ref[:,numeric]
    relative=absolute.copy()
    for col,k in enumerate(numeric):
        if kinds[k]=='continuous':relative[:,col]/=ref[:,k]
    cat={}
    for k,kind in enumerate(kinds):
        if kind in ['continuous','count']:continue
        grid=np.array(STATE_GRID[ts[k]],int);index={int(z):v for v,z in enumerate(grid)}
        counts=np.zeros((len(grid),len(grid)))
        for a,b in zip(ref[:,k],im[:,k]):counts[index[int(a)],index[int(b)]]+=1
        pooled=(counts.sum(0)+1e-6)/(counts.sum()+len(grid)*1e-6)
        alpha=counts+CONFIG['categorical_prior_strength']*pooled[None,:]
        cat[k]=dict(grid=grid,counts=counts,alpha=alpha,pooled=pooled)
    return dict(traits=ts,kinds=kinds,params=[params[j] for j in pool],reference=ref,image=im,numeric=numeric,absolute=absolute,relative=relative,categorical=cat,donor_ids=[data['ids'][z] for z in train])

def simulate_codes(model,runs,rng,mode='relative',shared_categorical=False):
    ref=model['reference'];n,p=ref.shape
    values=np.broadcast_to(ref,(runs,n,p)).copy()
    donors=rng.integers(0,n,size=(runs,n))
    residual=model['relative'] if mode=='relative' else model['absolute']
    errors=residual[donors]
    truncations=0
    for c,k in enumerate(model['numeric']):
        if mode=='relative' and model['kinds'][k]=='continuous':values[:,:,k]=ref[None,:,k]*(1+errors[:,:,c])
        else:values[:,:,k]=ref[None,:,k]+errors[:,:,c]
        truncations+=int(np.sum(values[:,:,k]<0))
        values[:,:,k]=np.maximum(values[:,:,k],0)
        if model['kinds'][k]=='count':
            resolution=CONFIG['count_simulated_mean_resolution'];values[:,:,k]=np.rint(values[:,:,k]/resolution)*resolution
    shared=rng.random((runs,n)) if shared_categorical else None
    for k,cat in model['categorical'].items():
        probs=np.empty((runs,len(cat['grid']),len(cat['grid'])))
        for st in range(len(cat['grid'])):probs[:,st,:]=rng.dirichlet(cat['alpha'][st],size=runs)
        for st,a in enumerate(cat['grid']):
            rows=np.flatnonzero(ref[:,k]==a)
            if len(rows)==0:continue
            u=shared[:,rows] if shared_categorical else rng.random((runs,len(rows)))
            chosen=(u[:,:,None]>np.cumsum(probs[:,st,:],axis=1)[:,None,:]).sum(2)
            values[:,rows,k]=cat['grid'][np.minimum(chosen,len(cat['grid'])-1)]
    return encode(values,model['params']),truncations

def zeta_counts(patterns,p):
    """Number of patterns disjoint with each subset; handles independent panels in axis0."""
    if patterns.ndim==1:patterns=patterns[None,:]
    m=patterns.shape[0];size=1<<p
    offsets=np.arange(m,dtype=np.int64)[:,None]*size
    f=np.bincount((patterns.astype(np.int64)+offsets).ravel(),minlength=m*size).reshape(m,size)
    for bit in range(p):
        step=1<<bit;v=f.reshape(m,-1,2*step);v[:,:,step:]+=v[:,:,:step]
    return f[:,(size-1)^np.arange(size)]

def all_collision_counts(codes):
    if codes.ndim==2:codes=codes[None,:,:]
    m,n,p=codes.shape;ia,ib=np.triu_indices(n,1)
    patterns=np.zeros((m,len(ia)),np.uint32)
    for k in range(p):patterns|=(codes[:,ia,k]!=codes[:,ib,k]).astype(np.uint32)<<k
    return zeta_counts(patterns,p)

def all_code_match_counts(sim,reference_codes):
    m,n,p=sim.shape;patterns=np.zeros((m,n),np.uint32)
    for k in range(p):patterns|=(sim[:,:,k]!=reference_codes[None,:,k]).astype(np.uint32)<<k
    return zeta_counts(patterns,p)

def wilson_lower(success,n):
    x=np.asarray(success,dtype=float)/n;z=stats.norm.ppf(.975)
    return (x+z*z/(2*n)-z*np.sqrt(x*(1-x)/n+z*z/(4*n*n)))/(1+z*z/n)

def simulate_all_subsets(model,runs,seed,mode='relative',shared_categorical=False,batch=None):
    p=len(model['traits']);size=1<<p;n=len(model['reference']);rng=np.random.default_rng(seed)
    rc=encode(model['reference'],model['params'])
    unique=np.zeros(size,np.int64);collisions=np.zeros(size,float);aq=np.zeros(size,float);aq2=np.zeros(size,float);trunc=0
    for start in range(0,runs,batch or CONFIG['simulation_batch']):
        m=min(batch or CONFIG['simulation_batch'],runs-start)
        c,nt=simulate_codes(model,m,rng,mode,shared_categorical);trunc+=nt
        cc=all_collision_counts(c);agree=all_code_match_counts(c,rc)/n
        unique+=(cc==0).sum(0);collisions+=cc.sum(0);aq+=agree.sum(0);aq2+=(agree*agree).sum(0)
    q=aq/runs;variance=np.maximum(0,(aq2-runs*q*q)/max(1,runs-1))
    return dict(Unique_successes=unique,Unique_probability=unique/runs,Unique_MC_lower=wilson_lower(unique,runs),Mean_collision_pairs=collisions/runs,Full_code_agreement=q,Full_code_agreement_MCSE=np.sqrt(variance/runs),Truncations=trunc,Runs=runs)

def subset_columns(mask,p):return [k for k in range(p) if mask&(1<<k)]

def min_distance(codes,cols,params):
    n=len(codes);ia,ib=np.triu_indices(n,1);d=np.zeros(len(ia))
    for k in cols:
        a=codes[ia,k];b=codes[ib,k]
        d+=((a!=b).astype(float) if params[k]['Kind']=='nominal' else abs(a.astype(float)-b)/params[k]['Distance_range'])/len(cols)
    return float(d.min()) if len(d) else np.nan

def build_search(model,summary,screen):
    p=len(model['traits']);size=1<<p
    rc=encode(model['reference'],model['params']);ic=encode(model['image'],model['params'])
    rcc=all_collision_counts(rc)[0];icc=all_collision_counts(ic)[0]
    masks=np.arange(size,dtype=np.uint32);sizes=np.bitwise_count(masks)
    baseline=(rcc==0)&(icc==0)&(masks>0)
    feasible=baseline&(summary['Unique_MC_lower']>=CONFIG['complete_uniqueness_target'])
    strict=feasible&((summary['Full_code_agreement']-1.96*summary['Full_code_agreement_MCSE'])>=CONFIG['stricter_full_code_agreement_benchmark'])
    rows=[]
    for mask in range(1,size):
        cols=subset_columns(mask,p)
        rows.append(dict(Mask=mask,Size=int(sizes[mask]),Traits=';'.join(model['traits'][k] for k in cols),Reference_collision_pairs=int(rcc[mask]),Observed_image_collision_pairs=int(icc[mask]),Baseline_dual_unique=bool(baseline[mask]),Unique_probability=summary['Unique_probability'][mask],Unique_MC_lower=summary['Unique_MC_lower'][mask],Mean_simulated_collision_pairs=summary['Mean_collision_pairs'][mask],Full_code_agreement=summary['Full_code_agreement'][mask],Full_code_agreement_MCSE=summary['Full_code_agreement_MCSE'][mask],Development_feasible=bool(feasible[mask]),Stricter_agreement_feasible=bool(strict[mask])))
    search=pd.DataFrame(rows)
    def choose(candidates):
        if len(candidates)==0:return None,[]
        k=min(int(sizes[z]) for z in candidates);minimum=[int(z) for z in candidates if sizes[z]==k]
        def rank(mask):
            cols=subset_columns(mask,p)
            distance=min(min_distance(rc,cols,model['params']),min_distance(ic,cols,model['params']))
            reli=screen.set_index('Trait').loc[[model['traits'][k] for k in cols],'Primary_agreement'].mean()
            return (-summary['Full_code_agreement'][mask],-summary['Unique_probability'][mask],-distance,-reli,tuple(int(model['traits'][k][1:]) for k in cols))
        return sorted(minimum,key=rank)[0],minimum
    winner,minimum=choose(np.flatnonzero(feasible));strict_winner,strict_min=choose(np.flatnonzero(strict));base_winner,base_min=choose(np.flatnonzero(baseline))
    # Diagnostic frontier representative is explicitly not an accepted constrained-selection result.
    diagnostic=None
    if baseline.any():
        diagnostic=int(sorted(np.flatnonzero(baseline),key=lambda z:(-summary['Unique_probability'][z],-summary['Full_code_agreement'][z],sizes[z],z))[0])
    return search,dict(winner=winner,minimum_solutions=minimum,strict_winner=strict_winner,strict_minimum_solutions=strict_min,baseline_winner=base_winner,baseline_minimum_solutions=base_min,diagnostic=diagnostic)

def audit_selected(model,mask,runs,seed,mode='relative',shared_categorical=False):
    cols=subset_columns(mask,len(model['traits']));rng=np.random.default_rng(seed)
    rc=encode(model['reference'],model['params'])[:,cols];n=len(rc)
    successes=0;agreements=[];collisions=[];trunc=0
    ia,ib=np.triu_indices(n,1)
    for start in range(0,runs,CONFIG['simulation_batch']):
        m=min(CONFIG['simulation_batch'],runs-start);c,nt=simulate_codes(model,m,rng,mode,shared_categorical);trunc+=nt;c=c[:,:,cols]
        separated=np.zeros((m,len(ia)),bool)
        for k in range(len(cols)):separated|=c[:,ia,k]!=c[:,ib,k]
        cc=(~separated).sum(1);collisions.extend(cc.tolist());successes+=int((cc==0).sum())
        agreements.extend(np.all(c==rc[None,:,:],axis=2).mean(1).tolist())
    q=np.array(agreements);mcse=q.std(ddof=1)/np.sqrt(runs)
    lower=float(wilson_lower(successes,runs))
    return dict(Runs=runs,Seed=seed,Error_mode=mode,Shared_categorical=shared_categorical,Unique_successes=successes,Unique_probability=successes/runs,Unique_MC_lower=lower,Mean_collision_pairs=float(np.mean(collisions)),Full_code_agreement=float(q.mean()),Full_code_agreement_MCSE=float(mcse),Full_code_agreement_MC_lower=float(max(0,q.mean()-1.96*mcse)),Primary_audit_pass=bool(lower>=CONFIG['complete_uniqueness_target']),Stricter_audit_pass=bool(lower>=CONFIG['complete_uniqueness_target'] and q.mean()-1.96*mcse>=CONFIG['stricter_full_code_agreement_benchmark']),Truncations=trunc)

def profile(c):
    _,counts=np.unique(c,axis=0,return_counts=True)
    return dict(UPP=len(counts)/len(c),Collision_pairs=int(np.sum(counts*(counts-1)//2)),Complete_unique=bool(len(counts)==len(c)))

def evaluate(data,train,test,params,pool,mask):
    cols=subset_columns(mask,len(pool));selected=[pool[k] for k in cols]
    r=encode(data['reference'],params)[:,selected];im=encode(data['image'],params)[:,selected]
    rr=r[train];ee=im[test];cross=int(np.all(rr[:,None,:]==ee[None,:,:],axis=2).sum())
    rp=profile(rr);ep=profile(ee);mixed=profile(np.vstack([rr,ee]))
    return dict(Reference_n=len(train),Evaluation_n=len(test),Reference_pair_denominator=len(train)*(len(train)-1)//2,Evaluation_pair_denominator=len(test)*(len(test)-1)//2,Cross_pair_denominator=len(train)*len(test),Mixed_pair_denominator=(len(train)+len(test))*(len(train)+len(test)-1)//2,Reference_collisions=rp['Collision_pairs'],Evaluation_collisions=ep['Collision_pairs'],Cross_collisions=cross,Mixed_collisions=mixed['Collision_pairs'],Evaluation_UPP=ep['UPP'],Evaluation_complete_unique=ep['Complete_unique'],Strict_all_library_collision_free=bool(rp['Complete_unique'] and ep['Complete_unique'] and cross==0),Same_variety_full_code_agreement=float(np.mean(np.all(r[test]==im[test],axis=1))),Same_variety_denominator=len(test),Traits=';'.join(data['traits'][k] for k in selected))

def model_json(model):
    return dict(Traits=model['traits'],Kinds=model['kinds'],Codebook=model['params'],Calibration_donor_IDs=model['donor_ids'],Absolute_discrepancy=model['absolute'],Relative_discrepancy=model['relative'],Numeric_positions=model['numeric'],Categorical={str(k):v for k,v in model['categorical'].items()},Target='Paired reference-image discrepancy at variety level; not independent repeat-imaging error')
