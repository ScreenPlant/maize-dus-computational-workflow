"""Evaluate a named artifact against an explicitly supplied annotation dictionary and split."""
from pathlib import Path
import argparse,json
from ultralytics import YOLO
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--imgsz',type=int,required=True);p.add_argument('--split',choices=['val','test'],default='test');a=p.parse_args()
    model=YOLO(str(a.model),task='segment')
    result=model.val(data=str(a.data.resolve()),split=a.split,imgsz=a.imgsz,batch=1,project=str(a.output.resolve()),name='evaluation',exist_ok=False)
    # These AP values use mask-IoU thresholds; they are not pixel mean IoU.
    dest=Path(result.save_dir)/'artifact_metrics.json'
    dest.write_text(json.dumps(dict(model=str(a.model),data=str(a.data),split=a.split,imgsz=a.imgsz,metrics=dict(result.results_dict)),indent=2),encoding='utf-8')
if __name__=='__main__':main()
