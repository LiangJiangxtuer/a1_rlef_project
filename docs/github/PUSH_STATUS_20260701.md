# GitHub push status — 2026-07-01

## Repository

- Local path: `/home/user/a1_rlef_project`
- Branch: `main`
- Intended remote: `git@github.com:LiangJiangxtuer/a1_rlef_project.git`
- Handoff tag: `rlef-v2-smoke-20260701`

## Completed locally

A1-RLEF v2 has been scaffolded and executed through P0/P1-light:

- full Chinese execution plan: `docs/EXECUTION_PLAN_RLEF_V2_ZH.md`
- executed process report: `docs/EXPERIMENT_PROCESS_REPORT.md`
- claim ledger: `docs/CLAIM_LEDGER.md`
- source guidance copy: `docs/source/A1_RLEF_v2_experiment_guidance_after_process_audit_20260701.md`
- RLEF MVP code: `src/rlef/`
- scripts: `scripts/make_uefb_v2.py`, `scripts/train.py`, `scripts/run_p0_smoke.py`, `scripts/run_p1_light_replication.py`
- real result tables: `results/tables/p0_smoke_summary.*`, `results/tables/p1_light_replication_summary.*`

Large/local-only assets are ignored: `data`, `experiments/`, checkpoints, generated visuals, archives.

## Verification

```text
pytest tests -q
........ [100%]
8 passed
```

`compileall` and JSON validation passed before this handoff note.

## Push attempt result

The push was attempted with SSH:

```bash
git remote add origin git@github.com:LiangJiangxtuer/a1_rlef_project.git
git push -u origin main
git push origin rlef-v2-smoke-20260701
```

It failed because the current machine/session does not have GitHub SSH authentication:

```text
git@github.com: Permission denied (publickey).
fatal: 无法读取远程仓库。
```

## Push after authentication

Option A — SSH key added to GitHub:

```bash
cd /home/user/a1_rlef_project
ssh -T git@github.com
git push -u origin main
git push origin rlef-v2-smoke-20260701
```

Option B — HTTPS + Personal Access Token entered interactively:

```bash
cd /home/user/a1_rlef_project
git remote set-url origin https://github.com/LiangJiangxtuer/a1_rlef_project.git
git push -u origin main
git push origin rlef-v2-smoke-20260701
```

When prompted, use:

```text
Username: LiangJiangxtuer
Password/key: GitHub Personal Access Token with repo Contents read/write
```

Do not store the PAT in project files.
