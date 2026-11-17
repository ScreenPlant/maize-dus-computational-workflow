"""Predefined alternative discrepancy models; do not retune the primary protocol."""
from pathlib import Path
import sys,argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
import fingerprint_engine as fe
import selection_runner as runner
from run_revised_analysis import bind_output
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();bind_output(a.inputs,a.output,None);runner.RESULTS=a.output
    d=fe.load_data(a.inputs);idx=np.arange(len(d['ids']))
    runner.fit_run(d,idx,np.array([],int),'sensitivity_no_T32',fe.CONFIG['seed']+20_000,exclude_extra=['T32'])
    params=fe.fit_codebook(d['reference'],d['traits'],d['kinds'])
    pool,screen,_,_=fe.screen(d,idx,params);model=fe.fit_model(d,idx,pool,params)
    for label,mode,shared in [('absolute_discrepancies','absolute',False),('shared_categorical_uniform','relative',True)]:
        folder=a.output/label;folder.mkdir(exist_ok=False)
        sim=fe.simulate_all_subsets(model,fe.CONFIG['simulation_confirmation_runs'],fe.CONFIG['seed']+30_000,mode,shared)
        table,choice=fe.build_search(model,sim,screen)
        table.to_csv(folder/'all_subsets.csv',index=False,encoding='utf-8')
        fe.write_json(folder/'summary.json',dict(Label=label,Mode=mode,Shared_categorical=shared,Runs=fe.CONFIG['simulation_confirmation_runs'],Choice=choice,Scope='Alternative-model all-subset confirmation; no independent selected-subset audit or nested validation'))
if __name__=='__main__':main()
