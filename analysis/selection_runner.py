"""Frozen selection/confirmation/audit implementation; I/O paths are portable."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from fingerprint_engine import *
RESULTS=ROOT/'outputs'
def fit_run(data,train,test,label,seed,exclude_extra=(),audit=True):
    dest=RESULTS/label;dest.mkdir(exist_ok=True)
    resultfile=dest/'result.json'
    if resultfile.exists():
        print(label,'resuming completed result',flush=True)
        return json.loads(resultfile.read_text(encoding='utf-8'))
    started=time.time()
    params=fit_codebook(data['reference'][train],data['traits'],data['kinds'])
    pool,scr,rc,ic=screen(data,train,params,exclude_extra=exclude_extra)
    scr.to_csv(dest/'trait_eligibility.csv',index=False,encoding='utf-8-sig')
    result=dict(Label=label,Seed=seed,Reference_n=len(train),Evaluation_n=len(test),Reference_IDs=[data['ids'][i] for i in train],Evaluation_IDs=[data['ids'][i] for i in test],Pool=[data['traits'][j] for j in pool],Primary_status='no_eligible_traits',Selected=None,Primary_evaluation=None,Strict_status='not_run',Strict_evaluation=None)
    if not pool:
        write_json(resultfile,result);return result
    if len(pool)>19:raise ValueError('Pool too large for enumerator; do not silently truncate')
    model=fit_model(data,train,pool,params)
    write_json(dest/'reference_fitted_model.json',model_json(model))
    print(label,'pool',result['Pool'],'development simulation start',flush=True)
    development=simulate_all_subsets(model,CONFIG['simulation_development_runs'],seed)
    print(label,'all-subset confirmation start',flush=True)
    summary=simulate_all_subsets(model,CONFIG['simulation_confirmation_runs'],seed+500_000)
    search,choice=build_search(model,summary,scr)
    search['Development1000_unique_probability']=[development['Unique_probability'][m] for m in search.Mask]
    search['Development1000_full_code_agreement']=[development['Full_code_agreement'][m] for m in search.Mask]
    search.to_csv(dest/'all_subsets.csv',index=False,encoding='utf-8-sig')
    counts=search.groupby('Size').agg(Combinations=('Mask','size'),Baseline_dual_unique=('Baseline_dual_unique','sum'),Development_robust_feasible=('Development_feasible','sum'),Stricter_feasible=('Stricter_agreement_feasible','sum')).reset_index()
    counts.to_csv(dest/'subset_size_counts.csv',index=False,encoding='utf-8-sig')
    def describe(mask):
        if mask is None:return None
        row=search[search.Mask==mask].iloc[0].to_dict()
        row['Traits_list']=row['Traits'].split(';')
        return row
    result.update(Search_combinations=len(search),Size_counts=counts.to_dict('records'),Development_feasible=int(search.Development_feasible.sum()),Stricter_development_feasible=int(search.Stricter_agreement_feasible.sum()),Minimum_development_solutions=[describe(z) for z in choice['minimum_solutions']],Baseline_representative=describe(choice['baseline_winner']),Diagnostic_frontier=describe(choice['diagnostic']),Development_truncations=summary['Truncations'])
    audited={}
    for rule,key in [('Primary','winner'),('Strict','strict_winner')]:
        mask=choice[key]
        if mask is None:
            result[rule+'_status']='no_development_feasible_subset'
            continue
        chosen=describe(mask)
        if audit:
            if mask not in audited:
                print(label,rule,'audit',chosen['Traits'],'start',flush=True)
                audited[mask]=audit_selected(model,mask,CONFIG['simulation_audit_runs'],seed+1_000_000)
            auditrow=audited[mask];passed=auditrow['Primary_audit_pass'] if rule=='Primary' else auditrow['Stricter_audit_pass']
            result[rule+'_audit']=auditrow
            result[rule+'_status']='qualified' if passed else 'independent_MC_audit_failed'
        else:passed=True;result[rule+'_status']='development_only_no_audit'
        result[rule+'_development_selected']=chosen
        if passed:
            result['Selected' if rule=='Primary' else 'Strict_selected']=chosen
            if len(test):result[rule+'_evaluation']=evaluate(data,train,test,params,pool,mask)
            else:
                cols=[pool[k] for k in subset_columns(mask,len(pool))]
                result[rule+'_full_panel_observed']=dict(Reference=profile(rc[:,cols]),Image=profile(ic[:,cols]),Full_code_agreement=float(np.all(rc[:,cols]==ic[:,cols],axis=1).mean()))
    # For a failed constrained-selection run, diagnostic results never enter Primary evaluation summaries.
    if result['Primary_status']!='qualified' and choice['diagnostic'] is not None and len(test):
        result['Nonqualified_diagnostic_evaluation']=evaluate(data,train,test,params,pool,choice['diagnostic'])
    result['Elapsed_seconds']=time.time()-started
    write_json(resultfile,result)
    print(label,'finished',result['Primary_status'],'strict',result['Strict_status'],'seconds',round(result['Elapsed_seconds'],1),flush=True)
    return clean_json(result)
