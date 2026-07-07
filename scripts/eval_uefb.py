#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import torch, yaml
from torch.utils.data import DataLoader
from scripts.train import build_model, evaluate
from rlef.datasets.uefb_dataset import UEFBPairedDataset


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--data_root',required=True); ap.add_argument('--output_dir',required=True); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args(); cfg=yaml.safe_load(open(args.config,encoding='utf-8'))
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    model=build_model(cfg).to(device); ckpt=torch.load(args.checkpoint,map_location=device); model.load_state_dict(ckpt['model'])
    ds=UEFBPairedDataset(args.data_root); loader=DataLoader(ds,batch_size=1,shuffle=False)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    metrics, rows=evaluate(model,loader,device,out,'uefb')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics, indent=2))
if __name__=='__main__': main()
