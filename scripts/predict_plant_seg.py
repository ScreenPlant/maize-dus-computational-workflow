"""Segmentation example; it does not implement the unpublished C++ trait-measurement layer."""
from pathlib import Path
import argparse
from ultralytics import YOLO
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--imgsz',type=int,default=1920);p.add_argument('--conf',type=float,default=0.1);p.add_argument('--device',default='cpu');a=p.parse_args()
    model=YOLO(str(a.model),task='segment')
    model.predict(source=str(a.input.resolve()),imgsz=a.imgsz,conf=a.conf,iou=0.7,device=a.device,save=True,save_txt=True,save_conf=True,project=str(a.output.resolve()),name='plant',exist_ok=False)
if __name__=='__main__':main()
