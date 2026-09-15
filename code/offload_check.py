"""Threat 5 (prereg §13): numeric consistency of CPU-offloaded bf16 execution vs full-GPU execution.

64 D3 prompts (first 16 of each cell's test split), last-token resid_pre, same model loaded twice:
full GPU vs max 1 GiB on GPU (rest on CPU). Reports relative L2 difference per layer.

    python offload_check.py --model smollm2-1.7b
"""
import argparse
import gc
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import ROOT, load_d3
from extract import MODELS, Extractor, encode_rendered, find_decoder_layers, render_chat


def acts(hf_id, tok, ids_list, max_memory):
    kw = {"device_map": "cuda"} if max_memory is None else {"device_map": "auto", "max_memory": max_memory}
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16, **kw).eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tcfg = getattr(model.config, "text_config", model.config)
    _, layers = find_decoder_layers(model, int(tcfg.num_hidden_layers))
    ex = Extractor(model, list(layers))
    out = np.stack([ex.run([ids], ("pos", [len(ids) - 1]))[0] for ids in ids_list])
    placement = sorted({str(v) for v in getattr(model, "hf_device_map", {"": "cuda"}).values()})
    del ex, model
    gc.collect()
    torch.cuda.empty_cache()
    return out, placement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-1.7b")
    args = ap.parse_args()
    hf_id = MODELS[args.model][0]
    tok = AutoTokenizer.from_pretrained(hf_id)
    d3 = load_d3()
    pick = []
    for c in ("BE", "CD", "CE", "BD"):
        pick += [r for r in d3 if r["cell"] == c and r["split"] == "test"][:16]
    ids_list = [encode_rendered(tok, render_chat(tok, None, r["prompt"])[0]) for r in pick]
    gpu, pg = acts(hf_id, tok, ids_list, None)
    off, po = acts(hf_id, tok, ids_list, {0: "1GiB", "cpu": "30GiB"})
    rel = np.linalg.norm(gpu - off, axis=-1) / np.linalg.norm(gpu, axis=-1).clip(1e-12)  # (64, L)
    res = {"model": args.model, "placement_gpu": pg, "placement_offload": po,
           "rel_l2_max_per_layer": rel.max(0).tolist(), "rel_l2_median_per_layer": np.median(rel, 0).tolist(),
           "rel_l2_max_overall": float(rel.max())}
    (ROOT / "results" / f"offload_check_{args.model}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list) or k.startswith("placement")}, indent=1))


if __name__ == "__main__":
    main()
