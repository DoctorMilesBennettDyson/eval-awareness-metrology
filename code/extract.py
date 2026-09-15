"""Activation extraction (prereg §6). One model per invocation.

    python extract.py --model qwen2.5-0.5b-base          # pilot (excluded from confirmatory set)
    python extract.py --model smollm2-135m

Outputs to results/activations/<key>/ :
    d1_pos.npy, d1_neg.npy   (204, L, d)  resid_pre at the answer-letter token (P-SAD vector)
    d2_mean.npy              (400, L, d)  token-mean resid_pre over non-BOS, non-whitespace tokens (P-SAD scores)
    d2_last.npy              (400, L, d)  resid_pre at the last prompt token (exploratory transfer test, prereg §15)
    d3_last.npy              (4063, L, d) resid_pre at the last prompt token (2x2 probes)
    meta.json                labels/cells/splits/lengths, model metadata, weight SHA-256s, timings

resid_pre of block l == input of decoder layer l (l = 0..L-1); captured with forward pre-hooks.
SAD texts are never written: only activations, labels and lengths are saved.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import ROOT, load_d1, load_d2, load_d3

MODELS = {
    # pilot only (prereg §11)
    "qwen2.5-0.5b-base": ("Qwen/Qwen2.5-0.5B", "pilot", 0.5),
    # confirmatory set (prereg §4)
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2", 0.135),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2", 0.36),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2", 1.7),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5", 0.5),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5", 1.5),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5", 3.0),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5", 7.0),
    "qwen2.5-14b": ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5", 14.0),
    "gemma3-270m": ("unsloth/gemma-3-270m-it", "Gemma-3", 0.27),
    "gemma3-1b": ("unsloth/gemma-3-1b-it", "Gemma-3", 1.0),
    "gemma3-4b": ("unsloth/gemma-3-4b-it", "Gemma-3", 4.0),
    "gemma3-12b": ("unsloth/gemma-3-12b-it", "Gemma-3", 12.0),
    "llama3.2-1b": ("unsloth/Llama-3.2-1B-Instruct", "Llama-3.x", 1.0),
    "llama3.2-3b": ("unsloth/Llama-3.2-3B-Instruct", "Llama-3.x", 3.0),
    "llama3.1-8b": ("unsloth/Llama-3.1-8B-Instruct", "Llama-3.x", 8.0),
}

MAX_TOKENS = 2048
OFFLOAD_DIR = Path(os.environ.get("EVALAWARE_OFFLOAD", r"C:\venvs\offload"))


def find_decoder_layers(model, n_layers):
    found = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers and name.endswith("layers"):
            if "vision" in name:
                continue
            found = (name, mod)
    if found is None:
        raise RuntimeError("decoder layer list not found")
    return found


def text_param_count(model):
    return int(sum(p.numel() for n, p in model.named_parameters()
                   if "vision" not in n and "multi_modal_projector" not in n))


def render_chat(tok, system, user):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), False
    except Exception:
        merged = [{"role": "user", "content": (system + "\n\n" + user) if system else user}]
        return tok.apply_chat_template(merged, add_generation_prompt=True, tokenize=False), True


def encode_raw(tok, text):
    return tok(text, add_special_tokens=True)["input_ids"]


def encode_rendered(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def answer_letter_position(tok, ids, letter):
    """Last token whose decoded text is exactly the answer letter (reference: token 'A'/'B')."""
    pieces = [tok.decode([t]) for t in ids]
    for i in range(len(ids) - 1, -1, -1):
        if pieces[i].strip() == letter:
            return i, False
    for i in range(len(ids) - 1, -1, -1):
        if letter in pieces[i]:
            return i, True
    raise RuntimeError("answer letter not found")


def token_mean_weights(tok, ids):
    """Reference filter: drop BOS-like tokens and whitespace-only tokens."""
    bos_ids = {t for t in (tok.bos_token_id,) if t is not None}
    w = []
    for t in ids:
        s = tok.decode([t])
        drop = (t in bos_ids or "<|begin_of_text|>" in s or s == "<s>" or s == "<bos>"
                or "<BOS>" in s or s.strip() == "")
        w.append(0.0 if drop else 1.0)
    return w


class Extractor:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.reduce = None
        self.out = [None] * len(layers)
        self.handles = [layer.register_forward_pre_hook(self._make_hook(i), with_kwargs=True)
                        for i, layer in enumerate(layers)]

    def _make_hook(self, i):
        def hook(module, args, kwargs):
            if self.reduce is None:
                return
            h = args[0] if args else kwargs["hidden_states"]
            self.out[i] = self.reduce(h).to("cpu", torch.float32)
        return hook

    @torch.no_grad()
    def run(self, batch_ids, spec):
        """batch_ids: list of token lists (right padded here).
        spec: ('pos', [idx]) -> (B, L, d); ('wmean', [weights]) -> (B, L, d);
              ('wmean+last', [weights]) -> (B, L, 2, d) with [..., 0, :] token-mean and [..., 1, :] last token."""
        device = self.model.get_input_embeddings().weight.device
        T = max(len(x) for x in batch_ids)
        pad = self.model.config.pad_token_id if getattr(self.model.config, "pad_token_id", None) is not None else 0
        ids = torch.full((len(batch_ids), T), pad, dtype=torch.long)
        mask = torch.zeros((len(batch_ids), T), dtype=torch.long)
        for b, x in enumerate(batch_ids):
            ids[b, :len(x)] = torch.tensor(x)
            mask[b, :len(x)] = 1
        kind, payload = spec
        if kind == "pos":
            pos = torch.tensor(payload, dtype=torch.long)

            def reduce(h):
                return h[torch.arange(h.shape[0], device=h.device), pos.to(h.device)].float()
        else:
            w = torch.zeros((len(batch_ids), T), dtype=torch.float32)
            for b, ws in enumerate(payload):
                w[b, :len(ws)] = torch.tensor(ws)

            last = torch.tensor([len(x) - 1 for x in batch_ids], dtype=torch.long)

            def reduce(h):
                ww = w.to(h.device)
                mean = (h.float() * ww[..., None]).sum(1) / ww.sum(1, keepdim=True).clamp_min(1.0)
                if kind == "wmean":
                    return mean
                lt = h[torch.arange(h.shape[0], device=h.device), last.to(h.device)].float()
                return torch.stack([mean, lt], dim=1)
        self.reduce = reduce
        try:
            self.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False, logits_to_keep=1)
        finally:
            self.reduce = None
        return torch.stack(self.out, dim=1).numpy()  # (B, L, d)


def batches(lengths, token_budget, max_batch):
    order = np.argsort(lengths)
    cur, cur_max = [], 0
    for i in order:
        n = lengths[i]
        if cur and (max(cur_max, n) * (len(cur) + 1) > token_budget or len(cur) >= max_batch):
            yield cur
            cur, cur_max = [], 0
        cur.append(int(i))
        cur_max = max(cur_max, n)
    if cur:
        yield cur


def extract_set(ex, name, ids_list, spec_list, out_path, L, d, token_budget, max_batch, last_path=None):
    N = len(ids_list)
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(N, L, d))
    arr_last = (np.lib.format.open_memmap(last_path, mode="w+", dtype=np.float32, shape=(N, L, d))
                if last_path is not None else None)
    lengths = np.array([len(x) for x in ids_list])
    t0 = time.time()
    done = 0
    for bidx in batches(lengths, token_budget, max_batch):
        kind = spec_list[0][0]
        spec = (kind, [spec_list[i][1] for i in bidx])
        res = ex.run([ids_list[i] for i in bidx], spec)
        if kind == "wmean+last":
            arr[bidx] = res[:, :, 0, :]
            arr_last[bidx] = res[:, :, 1, :]
        else:
            arr[bidx] = res
        done += len(bidx)
        if done % 256 < len(bidx):
            print(f"  {name}: {done}/{N}  {time.time() - t0:.0f}s", flush=True)
    arr.flush()
    del arr
    if arr_last is not None:
        arr_last.flush()
        del arr_last
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--gpu-gib", type=float, default=5.5)
    ap.add_argument("--cpu-gib", type=float, default=10.0)
    ap.add_argument("--token-budget", type=int, default=4096)
    # prereg §6 fixes batch size 1: in bf16, right-padded batching changed resid_pre by up to 4% relative L2
    # in the pilot (fp32: 1e-5), and prompt length correlates with class in D2/D3.
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--sets", default="d1,d2,d3")
    args = ap.parse_args()

    hf_id, family, nominal = MODELS[args.model]
    out_dir = ROOT / "results" / "activations" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260915)

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: f"{args.gpu_gib}GiB", "cpu": f"{args.cpu_gib}GiB"},
        offload_folder=str(OFFLOAD_DIR))
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tcfg = getattr(model.config, "text_config", model.config)
    L, d = int(tcfg.num_hidden_layers), int(tcfg.hidden_size)
    layers_name, layers = find_decoder_layers(model, L)
    ex = Extractor(model, list(layers))
    device_map = getattr(model, "hf_device_map", {})
    placement = ({str(v): sum(1 for x in device_map.values() if x == v) for v in set(device_map.values())}
                 or {str(model.device): "all"})
    t_load = time.time() - t_load
    print(f"{args.model}: L={L} d={d} layers={layers_name} placement={placement} load={t_load:.0f}s", flush=True)

    meta = {"key": args.model, "hf_id": hf_id, "family": family, "nominal_B": nominal,
            "n_layers": L, "hidden": d, "text_params": text_param_count(model),
            "dtype": "bfloat16", "layers_module": layers_name, "placement": placement,
            "attn_impl": getattr(model.config, "_attn_implementation", None),
            "transformers": __import__("transformers").__version__, "torch": torch.__version__,
            "load_seconds": t_load, "timings": {}}
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(hf_id, files_metadata=True)
        meta["revision"] = info.sha
        meta["weights_sha256"] = {s.rfilename: s.lfs.sha256 for s in info.siblings
                                  if s.lfs is not None and s.rfilename.endswith(".safetensors")}
    except Exception as e:  # metadata only; never blocks extraction
        meta["weights_sha256_error"] = repr(e)

    sets = set(args.sets.split(","))

    if "d1" in sets:
        d1 = load_d1()
        pos_ids, pos_spec, neg_ids, neg_spec, fallbacks = [], [], [], [], 0
        for it in d1:
            for ans, ids_l, spec_l in ((it["positive"], pos_ids, pos_spec), (it["negative"], neg_ids, neg_spec)):
                ids = encode_raw(tok, f"{it['question']}\n\n {ans}")
                p, fb = answer_letter_position(tok, ids, ans[1])
                fallbacks += fb
                ids_l.append(ids)
                spec_l.append(("pos", p))
        meta["d1"] = {"n": len(d1), "letter_fallbacks": fallbacks}
        meta["timings"]["d1_pos"] = extract_set(ex, "d1_pos", pos_ids, pos_spec, out_dir / "d1_pos.npy", L, d,
                                                args.token_budget, args.max_batch)
        meta["timings"]["d1_neg"] = extract_set(ex, "d1_neg", neg_ids, neg_spec, out_dir / "d1_neg.npy", L, d,
                                                args.token_budget, args.max_batch)

    if "d2" in sets:
        d2 = load_d2()
        ids_list, spec, ntok, nw, truncated, merged = [], [], [], [], 0, 0
        for it in d2:
            text, m = render_chat(tok, it["system"], it["user"])
            merged += m
            ids = encode_rendered(tok, text)
            if len(ids) > MAX_TOKENS:
                ids = ids[:MAX_TOKENS]
                truncated += 1
            w = token_mean_weights(tok, ids)
            ids_list.append(ids)
            spec.append(("wmean+last", w))
            ntok.append(len(ids))
            nw.append(int(sum(w)))
        meta["d2"] = {"n": len(d2), "labels": [it["label"] for it in d2], "nchar": [it["nchar"] for it in d2],
                      "ntok": ntok, "n_weighted_tok": nw, "truncated": truncated, "system_merged": merged}
        # d2_last.npy (last prompt token) is for the exploratory transfer test registered in prereg §15
        meta["timings"]["d2"] = extract_set(ex, "d2", ids_list, spec, out_dir / "d2_mean.npy", L, d,
                                            args.token_budget, args.max_batch, last_path=out_dir / "d2_last.npy")

    if "d3" in sets:
        d3 = load_d3()
        ids_list, spec, ntok, truncated = [], [], [], 0
        for it in d3:
            text, _ = render_chat(tok, None, it["prompt"])
            ids = encode_rendered(tok, text)
            if len(ids) > MAX_TOKENS:
                ids = ids[:MAX_TOKENS]
                truncated += 1
            ids_list.append(ids)
            spec.append(("pos", len(ids) - 1))
            ntok.append(len(ids))
        meta["d3"] = {"n": len(d3), "cell": [r["cell"] for r in d3], "split": [r["split"] for r in d3],
                      "is_eval": [r["is_eval"] for r in d3], "is_bench": [r["is_bench"] for r in d3],
                      "nchar": [r["nchar"] for r in d3], "ntok": ntok, "truncated": truncated}
        meta["timings"]["d3"] = extract_set(ex, "d3", ids_list, spec, out_dir / "d3_last.npy", L, d,
                                            args.token_budget, args.max_batch)

    meta_path = out_dir / "meta.json"
    if meta_path.exists():  # partial re-runs (--sets) must not drop metadata of sets not re-extracted
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old_t = old.get("timings", {})
        old.update({k: v for k, v in meta.items() if k != "timings"})
        old["timings"] = {**old_t, **meta["timings"]}
        meta = old
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    print(f"{args.model}: done  {meta['timings']}", flush=True)


if __name__ == "__main__":
    main()
