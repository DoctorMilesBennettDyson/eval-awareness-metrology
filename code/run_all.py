"""Sequential runner (prereg §12): G0 model first, then smallest to largest.

Each model: extract (subprocess, logged) -> (G0 for llama3.1-8b) -> delete that model's HF cache if large.
Skips models whose results/activations/<key>/DONE exists. Never computes aggregate estimators.

    python run_all.py                 # everything
    python run_all.py --only llama3.1-8b
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from data import ROOT
from extract import MODELS

ORDER = ["llama3.1-8b",  # G0 first
         "smollm2-135m", "gemma3-270m", "smollm2-360m", "qwen2.5-0.5b", "gemma3-1b", "llama3.2-1b",
         "qwen2.5-1.5b", "smollm2-1.7b", "qwen2.5-3b", "llama3.2-3b", "gemma3-4b", "qwen2.5-7b",
         "gemma3-12b", "qwen2.5-14b"]
KEEP_CACHE_BELOW_GB = 4.0
PY = sys.executable
LOGS = ROOT / "results" / "logs"


def cache_gb(hf_id):
    from huggingface_hub import scan_cache_dir
    for repo in scan_cache_dir().repos:
        if repo.repo_id == hf_id:
            return repo.size_on_disk / 1e9, repo
    return 0.0, None


def drop_cache(hf_id):
    size, repo = cache_gb(hf_id)
    if repo is None or size < KEEP_CACHE_BELOW_GB:
        return 0.0
    from huggingface_hub import scan_cache_dir
    strategy = scan_cache_dir().delete_revisions(*[r.commit_hash for r in repo.revisions])
    strategy.execute()
    return size


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd)}\n")
        f.flush()
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--extract-args", default="")
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    keys = [args.only] if args.only else ORDER
    for key in keys:
        hf_id = MODELS[key][0]
        done = ROOT / "results" / "activations" / key / "DONE"
        log = LOGS / f"{key}.log"
        if not done.exists():
            cmd = [PY, "extract_lw.py", "--model", key] + args.extract_args.split()
            ref = ROOT / "results" / "reference_standard_path" / key
            if ref.exists():
                cmd += ["--validate", str(ref)]
            rc = run(cmd, log)
            if rc != 0:
                print(f"{key}: extraction FAILED (rc={rc}), see {log}", flush=True)
                continue
        if key == "llama3.1-8b" and not (ROOT / "results" / f"G0_{key}.json").exists():
            run([PY, "g0.py", "--model", key], log)
        # prereg §12: per-model estimators are NOT computed here; analysis starts after the last model is DONE
        freed = drop_cache(hf_id)
        print(f"{key}: ok (freed {freed:.1f} GB cache)", flush=True)


if __name__ == "__main__":
    main()
