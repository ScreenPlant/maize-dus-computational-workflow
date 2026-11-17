"""Aggregate private 20-object/3x3-rating records; never invent adjudication or missing values."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from openpyxl import load_workbook
ROOT=Path(__file__).resolve().parents[1]
def mode(values):
    states,counts=np.unique(values,return_counts=True);w=states[counts==counts.max()]
    if len(w)!=1:raise ValueError('Observed tie requires a recorded, type-appropriate adjudication; none is fabricated')
    return int(w[0])
def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--materials',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    w=load_workbook(a.raw,read_only=True,data_only=True);numeric={};categorical={}
    for r in w['Sheet1'].iter_rows(min_row=2,values_only=True):
        if not isinstance(r[0],str):continue
        x=np.array(r[3:23],float);y=np.array(r[23:43],float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Incomplete numeric object record')
        numeric[(r[0],r[1])]=(x,y)
    for r in w['Sheet2'].iter_rows(min_row=2,values_only=True):
        if not isinstance(r[0],str):continue
        x=np.array([z if isinstance(z,(int,float)) else np.nan for z in r[3:12]],float)
        y=np.array([z if isinstance(z,(int,float)) else np.nan for z in r[12:15]],float)
        if np.isfinite(x).all() and np.isfinite(y).all():categorical[(r[0],r[1])]=(mode(x),mode(y))
        elif np.isnan(x).all() and np.isnan(y).all():categorical[(r[0],r[1])]=(np.nan,np.nan)
        else:raise ValueError('Partial rating/output record; define handling before running')
    traits=[t['id'] for t in json.loads((ROOT/'data_schema/trait_metadata.json').read_text(encoding='utf-8'))]
    materials=pd.read_csv(a.materials,dtype=str);rows=[]
    for variety in materials.Variety:
        for t in traits:
            if t=='T28':
                x=numeric[(variety,'T26')];y=numeric[(variety,'T27')]
                ref=float(x[0].mean()/y[0].mean());img=float(x[1].mean()/y[1].mean())
            elif (variety,t) in numeric:
                x,y=numeric[(variety,t)];ref=float(x.mean());img=float(y.mean())
            else:ref,img=categorical[(variety,t)]
            rows.append(dict(Variety=variety,Trait=t,Reference=ref,Image=img))
    # Match the precision serialized in the frozen spreadsheet, rather than changing scientific inputs.
    for r in rows:
        for key in ['Reference','Image']:
            if np.isfinite(r[key]):r[key]=float(format(r[key],'.16g'))
    pd.DataFrame(rows).to_csv(a.output/'variety_summary.csv',index=False,float_format='%.17g',encoding='utf-8')
    materials.to_csv(a.output/'materials.csv',index=False,encoding='utf-8')
    print('Wrote',len(rows),'variety-trait rows; protected inputs must not be committed.')
if __name__=='__main__':main()
