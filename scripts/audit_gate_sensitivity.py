"""Stricter gates project the same saved confirmation simulations; no new audit is claimed."""
from pathlib import Path
import sys,argparse,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
import fingerprint_engine as fe
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--results',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
data=fe.load_data(args.inputs); idx=np.arange(len(data['ids']))
params=fe.fit_codebook(data['reference'],data['traits'],data['kinds'])
pool,table,rc,ic=fe.screen(data,idx,params)
search=pd.read_csv(args.results/'full150/all_subsets.csv')
table['Primary_metric']=table.Kind.map({'continuous':'CCC','count':'CCC','ordered':'linear kappa','nominal':'unweighted kappa'})
table['Primary_threshold']=table.Kind.map({'continuous':.8,'count':.8,'ordered':.6,'nominal':.6})
table['MAPE_rule']=np.where(table.Kind.isin(['continuous','count']),'<10%','not applicable')
for j in range(len(table)):
 good=np.isfinite(data['reference'][:,j])&np.isfinite(data['image'][:,j])
 table.loc[j,'Paired_code_agreement']=np.mean(rc[good,j]==ic[good,j])
table['Reliability_pass_only']=((table.Kind.isin(['continuous','count']) & (table.Primary_agreement>=.8) & (table.MAPE_percent<10)) | (~table.Kind.isin(['continuous','count']) & (table.Primary_agreement>=.6)))
specs=[dict(Name='Primary',CCC=.8,Kappa=.6,MAPE=10,Cat_exact=0,All_code_exact=0),
 dict(Name='CCC085',CCC=.85,Kappa=.6,MAPE=10,Cat_exact=0,All_code_exact=0),
 dict(Name='CCC090',CCC=.9,Kappa=.6,MAPE=10,Cat_exact=0,All_code_exact=0),
 dict(Name='Kappa070',CCC=.8,Kappa=.7,MAPE=10,Cat_exact=0,All_code_exact=0),
 dict(Name='Kappa080',CCC=.8,Kappa=.8,MAPE=10,Cat_exact=0,All_code_exact=0),
 dict(Name='MAPE_lt5',CCC=.8,Kappa=.6,MAPE=5,Cat_exact=0,All_code_exact=0),
 dict(Name='Joint_stricter',CCC=.9,Kappa=.7,MAPE=5,Cat_exact=0,All_code_exact=0)]
for v in [.6,.8,.9]:specs.append(dict(Name=f'Categorical_exact_{v:.0%}',CCC=.8,Kappa=.6,MAPE=10,Cat_exact=v,All_code_exact=0))
for v in [.8,.9]:specs.append(dict(Name=f'All_trait_code_exact_{v:.0%}',CCC=.8,Kappa=.6,MAPE=10,Cat_exact=0,All_code_exact=v))
summaries=[];decisions=[];all_rows=[]
for spec in specs:
 allowed=[]
 for _,r in table.iterrows():
  reasons=[] if r.Eligible else [str(r.Reasons)]
  numeric=r.Kind in ['continuous','count']
  if numeric:
   if r.Primary_agreement<spec['CCC']:reasons.append('sensitivity_CCC')
   if not np.isfinite(r.MAPE_percent) or r.MAPE_percent>=spec['MAPE']:reasons.append('sensitivity_MAPE')
  else:
   if r.Primary_agreement<spec['Kappa']:reasons.append('sensitivity_kappa')
   if not np.isfinite(r.Exact_raw_state_fraction) or r.Exact_raw_state_fraction<spec['Cat_exact']:reasons.append('sensitivity_categorical_exact')
  if r.Paired_code_agreement<spec['All_code_exact']:reasons.append('sensitivity_per_trait_code_agreement')
  ok=not reasons
  if ok:allowed.append(r.Trait)
  decisions.append(dict(Scenario=spec['Name'],Trait=r.Trait,Eligible=ok,Reasons=';'.join(reasons)))
 allowed_set=set(allowed)
 sub=search[search.Traits.map(lambda s:set(s.split(';')).issubset(allowed_set))].copy()
 assert len(sub)==2**len(allowed)-1
 qualified=sub[sub.Development_feasible].sort_values(['Size','Full_code_agreement','Unique_probability'],ascending=[True,False,False])
 best=qualified.iloc[0] if len(qualified) else None
 summaries.append(dict(**spec,Eligible_n=len(allowed),Eligible_traits=';'.join(allowed),Enumerated_subsets=len(sub),Dual_unique_subsets=int(sub.Baseline_dual_unique.sum()),Robustness_qualified_subsets=len(qualified),Minimum_size=int(best.Size) if best is not None else None,Representative=best.Traits if best is not None else '',Complete_code_agreement=best.Full_code_agreement if best is not None else None,Complete_uniqueness=best.Unique_probability if best is not None else None,Scope='Full150 stricter-gate sensitivity; same saved 10000-run confirmation; no new independent audit or nested validation'))
 sub.insert(0,'Scenario',spec['Name']);all_rows.append(sub)
summary=pd.DataFrame(summaries)

summary.to_csv(args.output/'gate_sensitivity.csv',index=False,encoding='utf-8')
pd.DataFrame(decisions).to_csv(args.output/'gate_trait_decisions.csv',index=False,encoding='utf-8')
pd.concat(all_rows).to_csv(args.output/'projected_subsets.csv',index=False,encoding='utf-8')
table.to_csv(args.output/'trait_reliability.csv',index=False,encoding='utf-8')
print(summary[['Name','Eligible_n','Enumerated_subsets','Robustness_qualified_subsets','Minimum_size']].to_string(index=False))
