#!/usr/bin/env python3
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path
PY='/home/user/miniconda3/envs/cutler_dinov3/bin/python'
ROOT=Path(__file__).resolve().parents[1]

def run(cmd):
    print('+',' '.join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)), check=True, cwd=ROOT)

def main():
    data=ROOT/'data/UEFB-v2-smoke'
    source=ROOT/'data/LOL-v2/Real_captured/Train/Normal'
    run([PY,'scripts/make_uefb_v2.py','--source',source,'--output',data,'--num_train','20','--num_test','20','--image_size','96','--seed','3407'])
    rows=[]
    for cfg in ['rlef_m0_restorer_smoke.yml','rlef_m3_adaptive_gauge_smoke.yml','rlef_m4_gate_smoke.yml','rlef_m5_recoverability_smoke.yml']:
        tag=cfg.replace('.yml','')
        out=ROOT/'experiments'/f'p0_{tag}_seed3407'
        run([PY,'scripts/train.py','--config',ROOT/'configs'/cfg,'--output_dir',out,'--device','cuda','--max_steps','30'])
        metrics=json.loads((out/'metrics/eval_metrics.json').read_text())
        rows.append({'run':tag, **metrics, 'run_dir':str(out)})
    table=ROOT/'results/tables/p0_smoke_summary.csv'; table.parent.mkdir(parents=True,exist_ok=True)
    with table.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (ROOT/'results/tables/p0_smoke_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps({'summary':str(table),'rows':rows},indent=2))
if __name__=='__main__': main()
