#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from torch.utils.data import DataLoader, ConcatDataset
import yaml

from rlef.datasets.uefb_dataset import UEFBPairedDataset
from rlef.datasets.paired_rgb_dataset import PairedRGBDataset
from rlef.losses.total_loss import compute_total_loss
from rlef.metrics.full_reference import psnr_torch, ssim_torch
from rlef.metrics.exposure_field import exposure_field_metrics, local_exposure_error, noise_amplification_index, saturation_rate, identity_drop, normalized_abs_error, q_ece
from rlef.models.rlef_former import RLEFFormer
from rlef.utils.image_io import make_contact_sheet, save_tensor_image
from rlef.utils.seed import set_seed, snapshot_env


def load_cfg(path):
    with open(path, encoding='utf-8') as f: return yaml.safe_load(f)


def build_dataset(spec, split):
    typ = spec.get('type', 'uefb')
    if typ == 'uefb':
        return UEFBPairedDataset(spec['root'], crop_size=spec.get('crop_size'), training=(split=='train'), augment=spec.get('augment', split=='train'), max_images=spec.get('max_images'))
    if typ == 'paired_rgb':
        return PairedRGBDataset(spec['low_dir'], spec['high_dir'], crop_size=spec.get('crop_size'), training=(split=='train'), augment=spec.get('augment', split=='train'), name=spec.get('name'), max_images=spec.get('max_images'))
    raise ValueError(f'Unknown dataset type {typ}')


def build_split(cfg, split):
    specs = cfg['data'][split]
    if isinstance(specs, dict): specs = [specs]
    datasets = [build_dataset(s, split) for s in specs]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_model(cfg):
    m = cfg.get('model', {})
    return RLEFFormer(base_channels=m.get('base_channels', 32), e_max=m.get('e_max', 3.5), exposure_branch=m.get('exposure_branch', True), adaptive_gauge=m.get('adaptive_gauge', True), fixed_gauge=m.get('fixed_gauge'), physics_branch=m.get('physics_branch', True), gate_branch=m.get('gate_branch', False), q_branch=m.get('q_branch', False))


def evaluate(model, loader, device, output_dir=None, prefix='eval'):
    model.eval(); rows=[]
    with torch.no_grad():
        for bi,batch in enumerate(loader):
            batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            out = model(batch['low'])
            row = {
                'psnr': psnr_torch(out['I_hat'], batch['high']).item(),
                'ssim': ssim_torch(out['I_hat'], batch['high']).item(),
                'lee': local_exposure_error(out['I_hat'], batch['high']).item(),
                'nai': noise_amplification_index(out['I_hat'], batch['low']).item(),
                'input_psnr': psnr_torch(batch['low'], batch['high']).item(),
                'identity_drop': identity_drop(out['I_hat'], batch['low'], batch['high']).item(),
                'q_ece': q_ece(out['Q'], normalized_abs_error(out['I_hat'], batch['high'])).item(),
            }
            sr = saturation_rate(out['I_hat']); row.update(over=sr['over'].item(), under=sr['under'].item())
            if 'E_gt' in batch:
                row.update(exposure_field_metrics(out['E'], batch['E_gt']))
            rows.append(row)
            if output_dir is not None and bi < 8:
                make_contact_sheet({'low': batch['low'][0:1], 'pred': out['I_hat'][0:1], 'gt': batch['high'][0:1], 'E': out['E'][0:1], 'A': out['A'][0:1], 'Q': out['Q'][0:1]}, Path(output_dir)/'visuals'/f'{prefix}_{bi:03d}_sheet.png')
                save_tensor_image(out['I_hat'][0], Path(output_dir)/'visuals'/f'{prefix}_{bi:03d}_pred.png')
    agg = {k: sum(r[k] for r in rows)/len(rows) for k in rows[0]} if rows else {}
    return agg, rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--output_dir', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max_steps', type=int, default=None)
    args=ap.parse_args()
    cfg=load_cfg(args.config); set_seed(int(cfg.get('seed',3407)))
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    outdir=Path(args.output_dir); (outdir/'checkpoints').mkdir(parents=True, exist_ok=True); (outdir/'metrics').mkdir(exist_ok=True); (outdir/'visuals').mkdir(exist_ok=True)
    (outdir/'config.yml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    (outdir/'env.json').write_text(json.dumps(snapshot_env(), indent=2, ensure_ascii=False), encoding='utf-8')
    train_ds, val_ds = build_split(cfg, 'train'), build_split(cfg, 'val')
    tcfg=cfg.get('training', {})
    train_loader=DataLoader(train_ds, batch_size=tcfg.get('batch_size',4), shuffle=True, num_workers=tcfg.get('num_workers',0), pin_memory=(device.type=='cuda'))
    val_loader=DataLoader(val_ds, batch_size=cfg.get('eval',{}).get('batch_size',tcfg.get('batch_size',4)), shuffle=False, num_workers=0)
    model=build_model(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=tcfg.get('lr',2e-4), weight_decay=tcfg.get('weight_decay',1e-4))
    weights=cfg.get('loss', {'rec':1.0})
    max_steps=args.max_steps or tcfg.get('max_steps', 100)
    log_rows=[]; step=0; start=time.time(); model.train()
    while step < max_steps:
        for batch in train_loader:
            step += 1
            batch={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            out=model(batch['low'])
            loss, scalars=compute_total_loss(out,batch,weights)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get('grad_clip',1.0)); opt.step()
            if step == 1 or step % tcfg.get('log_every',10)==0:
                row={'step':step, 'loss':float(loss.detach().cpu()), **{k:float(v.detach().cpu()) for k,v in scalars.items()}}
                log_rows.append(row); print(json.dumps(row), flush=True)
            if step >= max_steps: break
    val, val_rows=evaluate(model,val_loader,device,outdir,'val')
    torch.save({'model':model.state_dict(),'cfg':cfg,'val_metrics':val,'step':step}, outdir/'checkpoints'/'last.pth')
    (outdir/'metrics'/'eval_metrics.json').write_text(json.dumps(val, indent=2), encoding='utf-8')
    with open(outdir/'metrics'/'train_log.csv','w',newline='',encoding='utf-8') as f:
        if log_rows:
            w=csv.DictWriter(f, fieldnames=list(log_rows[0].keys())); w.writeheader(); w.writerows(log_rows)
    with open(outdir/'metrics'/'val_per_batch.csv','w',newline='',encoding='utf-8') as f:
        if val_rows:
            w=csv.DictWriter(f, fieldnames=list(val_rows[0].keys())); w.writeheader(); w.writerows(val_rows)
    meta={'run_id':outdir.name,'command':' '.join(sys.argv),'steps':step,'elapsed_sec':time.time()-start,'config':cfg,'metrics':val,'checkpoint':str(outdir/'checkpoints'/'last.pth')}
    (outdir/'run_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'DONE': True, **meta}, ensure_ascii=False, indent=2), flush=True)

if __name__=='__main__': main()
