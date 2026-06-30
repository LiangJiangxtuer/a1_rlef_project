#!/usr/bin/env python3
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
paths={
 'LOL-v1 train low': ROOT/'data/LOL-v1/train/low',
 'LOL-v1 test low': ROOT/'data/LOL-v1/test/low',
 'LOL-v2 real train low': ROOT/'data/LOL-v2/Real_captured/Train/Low',
 'LOL-v2 real train normal': ROOT/'data/LOL-v2/Real_captured/Train/Normal',
 'LOL-v2 synthetic train low': ROOT/'data/LOL-v2/Synthetic/Train/Low',
 'unpaired_real': ROOT/'data/unpaired_real',
}
rows=[]
for k,p in paths.items():
    rows.append({'name':k,'path':str(p),'exists':p.exists(),'images':len(list(p.rglob('*.png'))) if p.exists() else 0})
print(json.dumps(rows,indent=2,ensure_ascii=False))
