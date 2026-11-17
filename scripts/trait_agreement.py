"""Recover trait agreement and nested object/rater evidence from frozen records."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
from openpyxl import load_workbook
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
import fingerprint_engine as fe

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--raw',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
p=args.output
target=p/'trait_agreement_and_variability.xlsx'
assert not target.exists()
data=fe.load_data(args.inputs);rng=np.random.default_rng(20261113)
params=fe.fit_codebook(data['reference'],data['traits'],data['kinds'])
_,table,rc,ic=fe.screen(data,np.arange(150),params)
rows=[]
for j,(t,kind) in enumerate(zip(data['traits'],data['kinds'])):
 a=data['reference'][:,j];b=data['image'][:,j];good=np.isfinite(a)&np.isfinite(b);a=a[good];b=b[good];n=len(a)
 row=table.iloc[j].to_dict();row['Primary_metric']='CCC' if kind in ['continuous','count'] else 'unweighted kappa' if kind=='nominal' else 'linear kappa'
 bootstrap=[]
 for run in range(500):
  z=rng.integers(0,n,n);bootstrap.append(fe.agreement(a[z],b[z],kind,t)[0])
 row['Primary_CI025'],row['Primary_CI975']=np.nanquantile(bootstrap,[.025,.975]);row['CI_resampling_unit']='Variety,500 paired bootstrap replicates'
 if kind in ['continuous','count']:
  d=b-a;row.update(Bias=float(d.mean()),LoA_lower=float(d.mean()-1.96*d.std(ddof=1)),LoA_upper=float(d.mean()+1.96*d.std(ddof=1)))
  reg=stats.linregress(a,b);q=stats.t.ppf(.975,n-2)
  row.update(Slope=reg.slope,Slope_CI025=reg.slope-q*reg.stderr,Slope_CI975=reg.slope+q*reg.stderr,Intercept=reg.intercept,Intercept_CI025=reg.intercept-q*reg.intercept_stderr,Intercept_CI975=reg.intercept+q*reg.intercept_stderr)
  bs=[]
  for run in range(500):
   x=d[rng.integers(0,n,n)];mu=x.mean();sd=x.std(ddof=1);bs.append([mu,mu-1.96*sd,mu+1.96*sd])
  ci=np.quantile(bs,[.025,.975],axis=0)
  for k,label in enumerate(['Bias','LoA_lower','LoA_upper']):row[label+'_CI025']=ci[0,k];row[label+'_CI975']=ci[1,k]
 else:
  _,_,_,exact=fe.agreement(a,b,kind,t);z=stats.norm.ppf(.975);den=1+z*z/n;centre=(exact+z*z/(2*n))/den;half=z*np.sqrt(exact*(1-exact)/n+z*z/(4*n*n))/den
  unweighted=fe.agreement(a,b,'nominal',t)[0];linear=fe.agreement(a,b,'ordered',t)[0]
  row.update(Exact_agreement=exact,Exact_Wilson025=centre-half,Exact_Wilson975=centre+half,Unweighted_kappa=unweighted,Linear_kappa=linear,Linear_kappa_role='Primary' if kind=='ordered' else 'Not an interpretation for nominal labels')
 rows.append(row)

w=load_workbook(args.raw,read_only=True,data_only=True)
objects=[]
for r in w['Sheet1'].iter_rows(min_row=2,values_only=True):
 a=np.array(r[3:23],float);b=np.array(r[23:43],float);d=b-a
 objects.append(dict(Variety=r[0],Trait=r[1],Objects=20,Reference_mean=a.mean(),Reference_SD=a.std(ddof=1),Reference_CV=100*a.std(ddof=1)/a.mean(),Image_mean=b.mean(),Image_SD=b.std(ddof=1),Image_CV=100*b.std(ddof=1)/b.mean(),Paired_difference_mean=d.mean(),Paired_difference_SD=d.std(ddof=1),Interpretation='Pooled objects within variety; not separately identifiable field replicate or technical-repeat effects',T28_note='Object ratios descriptive; constrained-selection derived value is ratio of component means' if r[1]=='T28' else ''))

cat={}
for r in w['Sheet2'].iter_rows(min_row=2,values_only=True):
 if not isinstance(r[3],(int,float)):continue
 a=np.array(r[3:12],float).reshape(3,3);b=np.array(r[12:15],float)
 cat.setdefault(r[1],[]).append((r[0],a,b))
pairs=[];summaries=[];image_detail=[]
for t,records in cat.items():
 j=data['traits'].index(t);kind=data['kinds'][j];a=np.concatenate([r[1] for r in records]);b=np.concatenate([r[2] for r in records])
 for u,v in [(0,1),(0,2),(1,2)]:
  primary=fe.agreement(a[:,u],a[:,v],kind,t)[0];unweighted=fe.agreement(a[:,u],a[:,v],'nominal',t)[0]
  pairs.append(dict(Trait=t,Expert_A=u+1,Expert_B=v+1,Varieties=len(records),Images=len(a),Exact=float(np.mean(a[:,u]==a[:,v])),Primary_kappa=primary,Unweighted_kappa=unweighted,Primary_metric='unweighted' if kind=='nominal' else 'linear',Scope='Same-image expert agreement; images clustered in variety; no independent technical repeats'))
 for e in range(3):
  pairs.append(dict(Trait=t,Expert_A=e+1,Expert_B='Program',Varieties=len(records),Images=len(a),Exact=float(np.mean(a[:,e]==b)),Primary_kappa=fe.agreement(a[:,e],b,kind,t)[0],Unweighted_kappa=fe.agreement(a[:,e],b,'nominal',t)[0],Primary_metric='unweighted' if kind=='nominal' else 'linear',Scope='Same-image expert-program agreement'))
 count=0;ties=0
 for variety,x,y in records:
  for k in range(3):
   states,ns=np.unique(x[k],return_counts=True);win=states[ns==ns.max()];mode=float(win[0]) if len(win)==1 else np.nan
   ties+=len(win)>1;count+=bool(np.isfinite(mode) and mode==y[k])
   image_detail.append(dict(Variety=variety,Trait=t,Image=k+1,Expert1=x[k,0],Expert2=x[k,1],Expert3=x[k,2],Unique_expert_mode=mode,Expert_mode_tied=len(win)>1,Program=y[k]))
 summaries.append(dict(Trait=t,Varieties=len(records),Images=len(a),Expert_ratings=len(a)*3,Program_outputs=len(a),Three_experts_unanimous_fraction=float(np.mean((a[:,0]==a[:,1])&(a[:,1]==a[:,2]))),Image_unique_mode_ties=ties,Program_equals_image_expert_mode_count=count,Image_total_denominator=len(a),Interpretation='Image-mode diagnostic; does not replace9-rating variety reference'))
notes=[['Coverage','23/44 means extracted output coverage;10 full-panel candidates pass all constrained-selection rules, not23 validated outputs.'],['Bootstrap','Paired variety bootstrap500 per trait; interval scope is this panel, not field cycles.'],['Expert evidence','3 fixed experts,3 images/variety; expert agreement is a reference benchmark, not a mathematical ceiling on program performance.'],['No fabricated repeats','No split into unknown field replicate groups; no claim of independent reimaging.'],['Nominal labels','Matching distance/unweighted kappa are primary. Linear kappa for nominal numeric labels has no inferential interpretation.']]
mapping={r['id']:r for r in __import__('json').loads((fe.ROOT/'data_schema/trait_metadata.json').read_text(encoding='utf-8'))}
for row in rows:
 for key in ['Unit','Official_type','GB_characteristic','GB_observation','UPOV_correspondence','UPOV_observation','Applicability','Note']:row[key]=mapping[row['Trait']].get(key)
mapping_columns=['Unit','Official_type','GB_characteristic','GB_observation','UPOV_correspondence','UPOV_observation','Applicability','Note']
main_table=pd.DataFrame(rows)
main_table=main_table[[c for c in main_table if c not in mapping_columns]+mapping_columns]
with pd.ExcelWriter(target,engine='openpyxl') as writer:
 for sheet,frame in [('Main_trait23',main_table),('Within_variety1500',pd.DataFrame(objects)),('Expert_pairs_and_program',pd.DataFrame(pairs)),('Image_agreement13',pd.DataFrame(summaries)),('Image_ratings5400',pd.DataFrame(image_detail)),('Notes',pd.DataFrame(notes,columns=['Item','Explanation']))]:frame.to_excel(writer,sheet_name=sheet,index=False)
fe.write_json(p/'trait_evidence_summary.json',dict(Numerical_variety_traits=len(objects),Complete_categorical_variety_traits=sum(len(v) for v in cat.values()),Same_image_records=len(image_detail),Traits=len(rows),Expert_pair_rows=len(pairs)))
print('Saved23-trait evidence,1500 within-variety summaries and',len(image_detail),'matched image-rating records.')
