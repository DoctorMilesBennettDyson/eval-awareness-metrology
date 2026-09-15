"""Layer-wise extraction: same prompts, reductions and outputs as extract.py, ~2 orders of magnitude faster
for models larger than GPU memory.

Standard offloaded execution moves every offloaded layer's weights to the GPU for every prompt. Here the
weights of layer l are materialised on the GPU once, and every prompt (batch size 1, as in prereg §6) is run
through the model with decoder layers < l replaced by proxies that return the cached input of layer l and
layer l+1 replaced by a proxy that stops the forward pass. The model's own embedding, attention-mask and
rotary code produce the inputs to layer l, so family-specific logic (e.g. Gemma-3 sliding windows) is
reused unchanged. resid_pre of layer l is reduced on the GPU by the same code as extract.Extractor.

Equivalence with extract.py is checked with `--validate` against stored standard-path activations.

    python extract_lw.py --model llama3.1-8b
    python extract_lw.py --model qwen2.5-0.5b-base --validate results/reference_standard_path/qwen2.5-0.5b-base
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from data import ROOT
from extract import MODELS, OFFLOAD_DIR, build_sets, find_decoder_layers, text_param_count, weights_meta, \
    write_meta_and_done


class Stop(Exception):
    pass


class FromCache(torch.nn.Module):
    """Replaces decoder layer 0: returns the cached input of the real layer."""
    def __init__(self):
        super().__init__()
        self.value = None

    def forward(self, hidden_states=None, *args, **kwargs):
        return self.value


class Passthrough(torch.nn.Module):
    def forward(self, hidden_states=None, *args, **kwargs):
        return hidden_states


class Halt(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise Stop()


def device_map_for(hf_id, local):
    cfg = AutoConfig.from_pretrained(hf_id, local_files_only=local)
    tcfg = getattr(cfg, "text_config", cfg)
    L = int(tcfg.num_hidden_layers)
    with init_empty_weights():
        skel = AutoModelForCausalLM.from_config(cfg)
    layers_name, _ = find_decoder_layers(skel, L)
    parent = layers_name.rsplit(".", 1)[0]
    tied = bool(getattr(cfg, "tie_word_embeddings", False) or getattr(tcfg, "tie_word_embeddings", False))
    dm = {}
    for name, _ in list(skel.named_parameters()) + list(skel.named_buffers()):
        if name.startswith(layers_name + "."):
            key = ".".join(name.split(".")[: len(layers_name.split(".")) + 1])
            dm[key] = "disk"
        elif name.startswith(parent + ".embed_tokens") or name.startswith(parent + ".rotary_emb"):
            dm[name.rsplit(".", 1)[0] if name.count(".") > parent.count(".") + 1 else name] = 0
        elif name.startswith("lm_head") and tied:
            dm[name] = 0
        else:
            dm[name] = "cpu"
    del skel
    return dm, L, layers_name


def materialize(layer, device):
    """Load an accelerate-offloaded layer's weights onto `device` once and drop its hooks."""
    for _, sm in layer.named_modules():
        hook = getattr(sm, "_hf_hook", None)
        wm = getattr(hook, "weights_map", None) if hook is not None else None
        if hook is not None and getattr(hook, "offload", False) and wm is not None:
            for pname, _ in list(sm.named_parameters(recurse=False)) + list(sm.named_buffers(recurse=False)):
                if pname in wm:
                    set_module_tensor_to_device(sm, pname, device, value=wm[pname])
    remove_hook_from_module(layer, recurse=True)
    for pname, p in list(layer.named_parameters()) + list(layer.named_buffers()):
        if p.device.type != "cuda":
            raise RuntimeError(f"parameter {pname} still on {p.device}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--sets", default="d1,d2,d3")
    ap.add_argument("--validate", default=None, help="directory with standard-path .npy files to compare against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hf_id, family, nominal = MODELS[args.model]
    out_dir = Path(args.out) if args.out else ROOT / "results" / "activations" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260915)

    t_load = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
        local = True
    except OSError:
        tok = AutoTokenizer.from_pretrained(hf_id)
        local = False
    if not local:  # make sure weights are in the cache before building the device map
        from huggingface_hub import snapshot_download
        snapshot_download(hf_id, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"])
    dm, L, layers_name = device_map_for(hf_id, True)
    model = AutoModelForCausalLM.from_pretrained(hf_id, local_files_only=True, dtype=torch.bfloat16, device_map=dm,
                                                 offload_folder=str(OFFLOAD_DIR))
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tcfg = getattr(model.config, "text_config", model.config)
    d = int(tcfg.hidden_size)
    _, layer_list = find_decoder_layers(model, L)
    real = list(layer_list)
    t_load = time.time() - t_load
    print(f"{args.model}: L={L} d={d} layers={layers_name} (layer-wise) load={t_load:.0f}s", flush=True)

    meta = {"key": args.model, "hf_id": hf_id, "family": family, "nominal_B": nominal, "n_layers": L, "hidden": d,
            "text_params": text_param_count(model), "dtype": "bfloat16", "layers_module": layers_name,
            "placement": {"engine": "layer-wise", "layers": "disk->cuda one at a time", "embed": "cuda"},
            "attn_impl": getattr(model.config, "_attn_implementation", None),
            "transformers": __import__("transformers").__version__, "torch": torch.__version__,
            "load_seconds": t_load, "timings": {}}
    weights_meta(hf_id, meta)

    sets = build_sets(tok, set(args.sets.split(",")), out_dir, meta)
    prompts = []  # (set_idx, row, ids, spec)
    arrays = []
    for si, s in enumerate(sets):
        n = len(s["ids"])
        main_arr = np.lib.format.open_memmap(s["out"], mode="w+", dtype=np.float32, shape=(n, L, d))
        last_arr = (np.lib.format.open_memmap(s["last"], mode="w+", dtype=np.float32, shape=(n, L, d))
                    if s["last"] is not None else None)
        arrays.append((main_arr, last_arr))
        for r in range(n):
            prompts.append((si, r, s["ids"][r], s["spec"][r]))
    N = len(prompts)
    device = torch.device("cuda")
    ml = layer_list  # the ModuleList inside the model
    from_cache, passthrough, halt = FromCache(), Passthrough(), Halt()
    cache = [None] * N
    state = {}

    def pre_hook(module, args_, kwargs):
        h = args_[0] if args_ else kwargs["hidden_states"]
        si, r, ids, (kind, payload) = state["prompt"]
        main_arr, last_arr = arrays[si]
        if kind == "pos":
            main_arr[r, state["l"]] = h[0, payload].float().cpu().numpy()
        else:
            w = torch.tensor(payload, dtype=torch.float32, device=h.device)
            mean = (h[0].float() * w[:, None]).sum(0) / w.sum().clamp_min(1.0)
            main_arr[r, state["l"]] = mean.cpu().numpy()
            if kind == "wmean+last":
                last_arr[r, state["l"]] = h[0, len(ids) - 1].float().cpu().numpy()
        if state["l"] == L - 1:
            raise Stop()

    def post_hook(module, args_, output):
        state["out"] = output[0] if isinstance(output, tuple) else output

    t0 = time.time()
    for l in range(L):
        layer = real[l]
        materialize(layer, device)
        for j in range(L):
            ml[j] = real[j] if j == l else (from_cache if j == 0 else passthrough) if j < l else halt
        hpre = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        hpost = layer.register_forward_hook(post_hook)
        with torch.no_grad():
            for i, (si, r, ids, spec) in enumerate(prompts):
                state.update(prompt=(si, r, ids, spec), l=l, out=None)
                if l > 0:
                    from_cache.value = cache[i].to(device, non_blocking=True)
                try:
                    model(input_ids=torch.tensor([ids], device=device), use_cache=False, logits_to_keep=1)
                except Stop:
                    pass
                if l < L - 1:
                    cache[i] = state["out"].to("cpu")
        hpre.remove()
        hpost.remove()
        # free the layer's GPU weights
        for pname, _ in list(layer.named_parameters()) + list(layer.named_buffers()):
            owner = layer.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else layer
            set_module_tensor_to_device(owner, pname.rsplit(".", 1)[-1], "meta")
        torch.cuda.empty_cache()
        print(f"  layer {l + 1}/{L}  {time.time() - t0:.0f}s", flush=True)
    for j in range(L):
        ml[j] = real[j]
    for main_arr, last_arr in arrays:
        main_arr.flush()
        if last_arr is not None:
            last_arr.flush()
    meta["timings"]["layerwise_total"] = time.time() - t0
    del arrays

    if args.validate:
        ref = Path(args.validate)
        rep = {}
        for s in sets:
            for path in [s["out"]] + ([s["last"]] if s["last"] is not None else []):
                rp = ref / path.name
                if not rp.exists():
                    continue
                a = np.load(path, mmap_mode="r")
                b = np.load(rp, mmap_mode="r")
                num = np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), axis=-1)
                den = np.linalg.norm(np.asarray(b, dtype=np.float64), axis=-1).clip(1e-12)
                rel = num / den
                rep[path.name] = {"max_rel_l2": float(rel.max()), "median_rel_l2": float(np.median(rel)),
                                  "exact_fraction": float((num == 0).mean())}
        print("VALIDATION", rep, flush=True)
        meta["validation_vs_standard"] = rep

    write_meta_and_done(out_dir, meta)
    print(f"{args.model}: done  {meta['timings']}", flush=True)


if __name__ == "__main__":
    main()
