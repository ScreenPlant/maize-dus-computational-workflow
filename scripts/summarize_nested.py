"""Summarize all requested runs and report comparison-specific pair denominators."""
from pathlib import Path
import argparse,json,csv
import numpy as np
from scipy.stats import norm
def interval(k,n):
    z=norm.ppf(.975);p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(c-h),float(c+h)]
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--runs',type=int,default=1000);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[];missing=[];evaluations=[];all_runs=[]
    for rep in range(1,a.runs+1):
        f=a.results/f'nested_{rep:03d}'/'result.json'
        if not f.exists():missing.append(rep);continue
        r=json.loads(f.read_text(encoding='utf-8'));qualified=r['Primary_status']=='qualified';e=r.get('Primary_evaluation') or {}
        all_runs.append(r)
        if qualified:
            if not e:raise ValueError('Qualified run lacks held-out evaluation')
            evaluations.append(e)
        rows.append(dict(Replicate=rep,Qualified=qualified,Qualified_and_evaluation_unique=qualified and bool(e.get('Evaluation_complete_unique')),Qualified_and_strict_transfer=qualified and bool(e.get('Strict_all_library_collision_free')),Traits=(r.get('Selected') or {}).get('Traits','')))
    if missing:raise ValueError(f'{len(missing)} requested runs are missing; do not report a complete-run denominator')
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'run_outcomes.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={k:sum(bool(r[k]) for r in rows) for k in ['Qualified','Qualified_and_evaluation_unique','Qualified_and_strict_transfer']};summary['Denominator']=a.runs
    summary['Run_level_MC_Wilson95']={k:interval(summary[k],a.runs) for k in ['Qualified','Qualified_and_evaluation_unique','Qualified_and_strict_transfer']}
    summary['Uncertainty_scope']='Monte Carlo uncertainty across sampled allocations of this fixed panel; no independent population or field-cycle inference'
    (a.output/'event_counts.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
    pairs=[]
    for name in ['Reference','Evaluation','Cross','Mixed']:
        cs=[e[name+'_collisions'] for e in evaluations];ds=[e[name+'_pair_denominator'] for e in evaluations]
        rates=np.array(cs)/np.array(ds) if ds else np.array([])
        pairs.append(dict(Comparison=name,Qualified_runs=len(evaluations),Collision_occurrences=sum(cs),Pair_comparisons=sum(ds),Conditional_rate=sum(cs)/sum(ds) if ds else None,Split_rate_MCSE=float(rates.std(ddof=1)/np.sqrt(len(rates))) if len(rates)>1 else None,Empirical_p025=float(np.quantile(rates,.025)) if len(rates) else None,Empirical_p975=float(np.quantile(rates,.975)) if len(rates) else None,Scope='Conditional on qualified selection; repeated pairs are not independent biological observations'))
    (a.output/'pair_counts.json').write_text(json.dumps(pairs,indent=2),encoding='utf-8')
    traits=json.loads((Path(__file__).resolve().parents[1]/'data_schema/trait_metadata.json').read_text(encoding='utf-8'))
    frequency=[dict(Trait=t['id'],Eligible_runs=sum(t['id'] in r['Pool'] for r in all_runs),Selected_runs=sum(t['id'] in (r.get('Selected') or {}).get('Traits_list',[]) for r in all_runs),All_run_denominator=a.runs,Qualified_run_denominator=len(evaluations)) for t in traits]
    (a.output/'trait_selection_counts.json').write_text(json.dumps(frequency,indent=2),encoding='utf-8')
    matched=sum(round(e['Same_variety_full_code_agreement']*e['Same_variety_denominator']) for e in evaluations)
    code_n=sum(e['Same_variety_denominator'] for e in evaluations)
    (a.output/'complete_code_agreement.json').write_text(json.dumps(dict(Matched_variety_occurrences=matched,Variety_occurrences=code_n,Rate=matched/code_n if code_n else None,Scope='Complete reference-image code equality, conditional on qualified selection; not uniqueness'),indent=2),encoding='utf-8')
if __name__=='__main__':main()
