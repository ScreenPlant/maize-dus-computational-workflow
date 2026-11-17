"""Run the revised constrained procedure; protected inputs remain outside the repository."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.setdefault(key,'1')
from pathlib import Path
import argparse,json,sys,hashlib
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
import numpy as np
import pandas as pd
import fingerprint_engine as engine
import selection_runner as runner

DATA=None;PARTS=None
def bind_output(inputs,output,membership):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    paths=[Path(inputs)/'variety_summary.csv',Path(inputs)/'materials.csv',ROOT/'configs/analysis_protocol.json',ROOT/'data_schema/trait_metadata.json',ROOT/'analysis/fingerprint_engine.py',ROOT/'analysis/selection_runner.py',Path(__file__).resolve()]
    if membership:paths.append(Path(membership))
    record={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    lock=output/'input_lock.json'
    if lock.exists():
        if json.loads(lock.read_text(encoding='utf-8'))!=record:raise ValueError('Inputs/protocol changed; use a new output directory')
    else:
        if list(output.glob('*/result.json')):raise ValueError('Unbound cached results; choose a new output directory')
        lock.write_text(json.dumps(record,indent=2),encoding='utf-8')
def initialize(inputs,output,membership):
    global DATA,PARTS
    DATA=engine.load_data(inputs);runner.RESULTS=Path(output).resolve();runner.RESULTS.mkdir(parents=True,exist_ok=True)
    PARTS={}
    if membership:
        m=pd.read_csv(membership);index={v:k for k,v in enumerate(DATA['ids'])}
        for rep,g in m.groupby('Replicate',sort=True):
            if not set(g.Role)<= {'Reference','Evaluation'}:raise ValueError('Unknown membership role')
            a=[index[v] for v in g.loc[g.Role=='Reference','ID']];b=[index[v] for v in g.loc[g.Role=='Evaluation','ID']]
            if len(a)!=120 or len(b)!=30 or len(set(a+b))!=150:raise ValueError('Expected a disjoint 120/30 allocation')
            PARTS[int(rep)]=(np.array(a),np.array(b))

def worker(rep):
    a,b=PARTS[rep]
    r=runner.fit_run(DATA,a,b,f'nested_{rep:03d}',engine.CONFIG['seed']+rep*100)
    return rep,r['Primary_status']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--membership',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=['full','nested','all'],default='all')
    p.add_argument('--first',type=int,default=1);p.add_argument('--last',type=int,default=1000)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    if a.scope in ['nested','all'] and a.membership is None:p.error('--membership is required for complete nested reselection')
    bind_output(a.inputs,a.output,a.membership)
    initialize(a.inputs,a.output,a.membership)
    if a.scope in ['full','all']:
        runner.fit_run(DATA,np.arange(len(DATA['ids'])),np.array([],int),'full150',engine.CONFIG['seed'])
    if a.scope in ['nested','all']:
        reps=list(range(a.first,a.last+1))
        if not set(reps)<=set(PARTS):raise ValueError('Requested runs missing from membership file')
        if a.workers==1:
            for rep in reps:worker(rep)
        else:
            with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a.inputs,a.output,a.membership)) as pool:
                for rep,status in pool.map(worker,reps):print(rep,status,flush=True)
    engine.write_json(a.output/'execution_manifest.json',dict(Method=engine.CONFIG['method_id'],Protocol=engine.CONFIG,Input_sha256=DATA['sha256'],Requested_scope=a.scope,First=a.first,Last=a.last,Upstream_model_refitted=False))
if __name__=='__main__':main()
