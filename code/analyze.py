"""Estimators, nulls, baselines and bootstrap replicates (prereg §7-§9).

Stage 1 (per model):   python analyze.py per-model --model <key>
Stage 2 (all models):  python analyze.py aggregate          (only after every confirmatory model is DONE, prereg §12)

AUROC convention: evaluation class positive. Rank-based AUROC with average ranks for ties.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from data import ROOT, load_d2, load_d3

ACT = ROOT / "results" / "activations"
OUT = ROOT / "results" / "analysis"
SEED = 20260915
B_PERM = 1000        # N1
B_SIGN = 200         # N2
N_HALV = 200         # E_split
B_BOOT = 1000        # item bootstrap
N_HALV_BOOT = 20     # halvings inside each bootstrap replicate (computational approximation, see report)
CHUNK = 256


# ----------------------------------------------------------------------------- AUROC utilities
def auroc_from_scores(scores, y):
    """scores (N, L) or (N,), y (N,) in {0,1} -> AUROC per column."""
    s = scores if scores.ndim == 2 else scores[:, None]
    r = rankdata(s, axis=0)
    n1 = y.sum()
    n0 = len(y) - n1
    a = (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return a if scores.ndim == 2 else a[0]


def auroc_perm(ranks, Y):
    """ranks (N, L) precomputed, Y (B, N) label matrices with fixed n1 -> (B, L)."""
    n1 = Y[0].sum()
    n0 = Y.shape[1] - n1
    return (Y @ ranks - n1 * (n1 + 1) / 2) / (n1 * n0)


def e_naive(auc, skip0=False):
    dev = np.abs(auc - 0.5)
    if skip0:
        dev = dev[..., 1:]
    return dev.max(-1)


def stratified_perms(y, B, rng):
    return np.stack([rng.permutation(y) for _ in range(B)]).astype(np.float64)


def e_split(scores, y, n_halv, rng):
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = np.empty(n_halv)
    for k in range(n_halv):
        p1, p0 = rng.permutation(idx1), rng.permutation(idx0)
        A = np.concatenate([p1[: len(p1) // 2], p0[: len(p0) // 2]])
        Bh = np.concatenate([p1[len(p1) // 2:], p0[len(p0) // 2:]])
        aA = auroc_from_scores(scores[A], y[A])
        l = int(np.argmax(np.abs(aA - 0.5)))
        sgn = np.sign(aA[l] - 0.5) or 1.0
        aB = auroc_from_scores(scores[Bh, l], y[Bh])
        vals[k] = sgn * (aB - 0.5)
    return float(vals.mean())


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def project(acts, V, idx=None):
    """acts memmap (N, L, d); V (L, d) unit vectors -> (N, L) scores, chunked."""
    rows = np.arange(acts.shape[0]) if idx is None else np.asarray(idx)
    out = np.empty((len(rows), V.shape[0]), dtype=np.float64)
    for s in range(0, len(rows), CHUNK):
        a = np.asarray(acts[rows[s:s + CHUNK]], dtype=np.float64)
        out[s:s + CHUNK] = np.einsum("nld,ld->nl", a, V)
    return out


def cell_mean(acts, idx):
    tot = None
    for s in range(0, len(idx), CHUNK):
        a = np.asarray(acts[idx[s:s + CHUNK]], dtype=np.float64).sum(0)
        tot = a if tot is None else tot + a
    return tot / len(idx)


# ----------------------------------------------------------------------------- baselines
def tfidf_cv_auroc(texts, y, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=2000))
    p = cross_val_predict(pipe, texts, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(auroc_from_scores(p, y))


def tfidf_train_test(tr_texts, tr_y, te_texts, te_y):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=2000))
    pipe.fit(tr_texts, tr_y)
    return float(auroc_from_scores(pipe.predict_proba(te_texts)[:, 1], te_y))


# ----------------------------------------------------------------------------- per model
def per_model(key):
    rng = np.random.default_rng(SEED)
    d = ACT / key
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    L = meta["n_layers"]
    res = {"key": key, "family": meta["family"], "nominal_B": meta["nominal_B"], "n_layers": L,
           "hidden": meta["hidden"], "text_params": meta["text_params"]}
    rep = {}

    # ---------------- P-SAD
    pos = np.load(d / "d1_pos.npy", mmap_mode="r")
    neg = np.load(d / "d1_neg.npy", mmap_mode="r")
    diff = np.asarray(pos, dtype=np.float64) - np.asarray(neg, dtype=np.float64)       # (204, L, d)
    V = unit(diff.mean(0))
    d2 = np.load(d / "d2_mean.npy", mmap_mode="r")
    y2 = np.array(meta["d2"]["labels"])
    S2 = project(d2, V)                                                                   # (400, L)
    auc2 = auroc_from_scores(S2, y2)
    res["sad_auc_layers"] = auc2.tolist()
    res["sad_E_naive"] = float(e_naive(auc2))
    res["sad_E_naive_no_l0"] = float(e_naive(auc2, skip0=True))
    res["sad_best_layer"] = int(np.argmax(np.abs(auc2 - 0.5)))
    res["sad_E_fixed"] = float(abs(auc2[int(round(0.5 * (L - 1)))] - 0.5))

    ranks2 = rankdata(S2, axis=0)
    null1 = e_naive(auroc_perm(ranks2, stratified_perms(y2, B_PERM, rng)))
    res["sad_N1_mean"] = float(null1.mean())
    res["sad_N1_p95"] = float(np.quantile(null1, 0.95))
    res["sad_E_corr"] = res["sad_E_naive"] - res["sad_N1_mean"]
    res["sad_N1_p_value"] = float((1 + (null1 >= res["sad_E_naive"]).sum()) / (1 + B_PERM))

    null2 = np.empty(B_SIGN)
    for b in range(B_SIGN):
        s = rng.choice([-1.0, 1.0], size=diff.shape[0])
        Vb = unit(np.einsum("n,nld->ld", s, diff) / diff.shape[0])
        null2[b] = e_naive(auroc_from_scores(project(d2, Vb), y2))
    res["sad_N2_mean"] = float(null2.mean())
    res["sad_N2_p95"] = float(np.quantile(null2, 0.95))

    res["sad_E_split"] = e_split(S2, y2, N_HALV, rng)

    ntok2 = np.array(meta["d2"]["ntok"], dtype=float)
    nchar2 = np.array(meta["d2"]["nchar"], dtype=float)
    res["sad_B1_ntok_auc"] = float(auroc_from_scores(ntok2, y2))
    res["sad_B1_nchar_auc"] = float(auroc_from_scores(nchar2, y2))
    res["sad_B3_l0_auc"] = float(auc2[0])

    # bootstrap replicates (stratified item resampling)
    i1, i0 = np.where(y2 == 1)[0], np.where(y2 == 0)[0]
    bn, bs = np.empty(B_BOOT), np.empty(B_BOOT)
    for b in range(B_BOOT):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        bn[b] = e_naive(auroc_from_scores(S2[idx], y2[idx]))
        bs[b] = e_split(S2[idx], y2[idx], N_HALV_BOOT, rng)
    rep["sad_E_naive"] = bn
    rep["sad_E_split"] = bs

    # ---------------- 2x2
    d3 = np.load(d / "d3_last.npy", mmap_mode="r")
    cell = np.array(meta["d3"]["cell"])
    split = np.array(meta["d3"]["split"])
    idx = {(c, sp): np.where((cell == c) & (split == sp))[0] for c in ("BE", "CD", "CE", "BD")
           for sp in ("train", "val", "test")}
    M = {c: cell_mean(d3, idx[(c, "train")]) for c in ("BE", "CD", "CE", "BD")}
    probes = {
        "STD": unit(M["BE"] - M["CD"]),
        "PAIRED": unit((M["BE"] + M["CE"]) / 2 - (M["CD"] + M["BD"]) / 2),
        "FORMAT": unit((M["BE"] + M["BD"]) / 2 - (M["CE"] + M["CD"]) / 2),
    }
    res["cos_STD_FORMAT_layers"] = np.einsum("ld,ld->l", probes["STD"], probes["FORMAT"]).tolist()
    res["cos_PAIRED_FORMAT_layers"] = np.einsum("ld,ld->l", probes["PAIRED"], probes["FORMAT"]).tolist()

    ev = {sp: np.concatenate([idx[(c, sp)] for c in ("BE", "CD", "CE", "BD")]) for sp in ("val", "test")}
    S = {p: {sp: project(d3, probes[p], ev[sp]) for sp in ("val", "test")} for p in probes}
    y_eval = np.array(meta["d3"]["is_eval"])

    def sub(sp, cells):
        rows = ev[sp]
        m = np.isin(cell[rows], cells)
        return m, y_eval[rows][m]

    def auc_cells(p, sp, cells, rows_mask=None):
        m, yy = sub(sp, cells)
        return auroc_from_scores(S[p][sp][m], yy)

    def compute_2x2(pick):
        """pick(sp) -> boolean mask over ev[sp] rows used (for bootstrap). Returns estimators."""
        out = {}
        # H3: STD, layer selected on val by AUROC(BE vs CD) (signed), crossed AUROC on test
        mv = np.isin(cell[ev["val"]], ["BE", "CD"]) & pick("val")
        a_val_std = auroc_from_scores(S["STD"]["val"][mv], y_eval[ev["val"]][mv])
        l_std = int(np.argmax(a_val_std))
        mt = np.isin(cell[ev["test"]], ["CE", "BD"]) & pick("test")
        out["std_crossed_auc_valsel"] = float(auroc_from_scores(S["STD"]["test"][mt, l_std], y_eval[ev["test"]][mt]))
        out["std_valsel_layer"] = l_std
        # E_naive2x2: STD, max over layers |AUROC(all eval vs all deploy on test) - .5|
        ma = pick("test")
        a_all = auroc_from_scores(S["STD"]["test"][ma], y_eval[ev["test"]][ma])
        out["E_naive2x2"] = float(e_naive(a_all))
        # E_corr2x2: PAIRED, val-selected layer by all-cells AUROC (signed), crossed AUROC - .5 on test
        mvp = pick("val")
        a_val_p = auroc_from_scores(S["PAIRED"]["val"][mvp], y_eval[ev["val"]][mvp])
        l_p = int(np.argmax(a_val_p))
        out["E_corr2x2"] = float(auroc_from_scores(S["PAIRED"]["test"][mt, l_p], y_eval[ev["test"]][mt]) - 0.5)
        out["paired_valsel_layer"] = l_p
        return out

    all_rows = lambda sp: np.ones(len(ev[sp]), dtype=bool)
    res.update(compute_2x2(all_rows))
    # layerwise curves on test (descriptive)
    mt_cross = np.isin(cell[ev["test"]], ["CE", "BD"])
    for p in ("STD", "PAIRED", "FORMAT"):
        res[f"{p}_crossed_auc_layers"] = auroc_from_scores(S[p]["test"][mt_cross], y_eval[ev["test"]][mt_cross]).tolist()
        res[f"{p}_all_auc_layers"] = auroc_from_scores(S[p]["test"], y_eval[ev["test"]]).tolist()

    # EXPLORATORY (prereg §15, 2026-09-15): 2x2-trained probes transferred to SAD at the last prompt token.
    # Layer chosen on 2x2 validation (same layers as H3 / E_corr2x2), so no selection on SAD.
    if (d / "d2_last.npy").exists():
        d2l = np.load(d / "d2_last.npy", mmap_mode="r")
        for p, l_sel in (("STD", res["std_valsel_layer"]), ("PAIRED", res["paired_valsel_layer"]), ("FORMAT", None)):
            a = auroc_from_scores(project(d2l, probes[p]), y2)
            res[f"expl_sad_transfer_{p}_auc_layers"] = a.tolist()
            if l_sel is not None:
                res[f"expl_sad_transfer_{p}_auc_valsel"] = float(a[l_sel])

    ntok3 = np.array(meta["d3"]["ntok"], dtype=float)
    tr = ntok3[ev["test"]]
    res["2x2_B1_ntok_crossed_auc"] = float(auroc_from_scores(tr[mt_cross], y_eval[ev["test"]][mt_cross]))
    res["2x2_B3_l0_std_crossed_auc"] = res["STD_crossed_auc_layers"][0]

    # bootstrap replicates: resample val and test rows within cell
    boots = {k: np.empty(B_BOOT) for k in ("std_crossed_auc_valsel", "E_naive2x2", "E_corr2x2")}
    for b in range(B_BOOT):
        weights = {}
        for sp in ("val", "test"):
            rows = ev[sp]
            counts = np.zeros(len(rows), dtype=int)
            for c in ("BE", "CD", "CE", "BD"):
                pos_c = np.where(cell[rows] == c)[0]
                counts[pos_c] += np.bincount(rng.choice(len(pos_c), len(pos_c)), minlength=len(pos_c))
            weights[sp] = counts
        # expand resampled rows by repetition
        Sb = {p: {sp: np.repeat(S[p][sp], weights[sp], axis=0) for sp in ("val", "test")} for p in S}
        cb = {sp: np.repeat(cell[ev[sp]], weights[sp]) for sp in ("val", "test")}
        yb = {sp: np.repeat(y_eval[ev[sp]], weights[sp]) for sp in ("val", "test")}
        mv = np.isin(cb["val"], ["BE", "CD"])
        l_std = int(np.argmax(auroc_from_scores(Sb["STD"]["val"][mv], yb["val"][mv])))
        mt = np.isin(cb["test"], ["CE", "BD"])
        boots["std_crossed_auc_valsel"][b] = auroc_from_scores(Sb["STD"]["test"][mt, l_std], yb["test"][mt])
        boots["E_naive2x2"][b] = e_naive(auroc_from_scores(Sb["STD"]["test"], yb["test"]))
        l_p = int(np.argmax(auroc_from_scores(Sb["PAIRED"]["val"], yb["val"])))
        boots["E_corr2x2"][b] = auroc_from_scores(Sb["PAIRED"]["test"][mt, l_p], yb["test"][mt]) - 0.5
    rep.update(boots)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{key}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / f"{key}_boot.npz", **rep)
    return res


# ----------------------------------------------------------------------------- text baselines (need texts in memory)
def text_baselines(seed=SEED):
    d2 = load_d2()
    y2 = np.array([it["label"] for it in d2])
    out = {"sad_B2_tfidf_cv_auc": tfidf_cv_auroc([it["user"] for it in d2], y2, seed)}
    d3 = load_d3()
    tr = [r for r in d3 if r["split"] == "train" and r["cell"] in ("BE", "CD")]
    te = [r for r in d3 if r["split"] == "test" and r["cell"] in ("CE", "BD")]
    out["2x2_B2_tfidf_std_crossed_auc"] = tfidf_train_test([r["prompt"] for r in tr], np.array([r["is_eval"] for r in tr]),
                                                           [r["prompt"] for r in te], np.array([r["is_eval"] for r in te]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "text_baselines.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- aggregate (prereg §9)
CONFIRMATORY = ["smollm2-135m", "smollm2-360m", "smollm2-1.7b", "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b",
                "qwen2.5-7b", "qwen2.5-14b", "gemma3-270m", "gemma3-1b", "gemma3-4b", "gemma3-12b",
                "llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]


def fe_design(logn, fam, extra=None):
    fams = sorted(set(fam))
    cols = [logn] + ([extra] if extra is not None else []) + [(np.array(fam) == f).astype(float) for f in fams]
    return np.column_stack(cols)


def fe_slope(y, logn, fam, extra=None):
    X = fe_design(logn, fam, extra)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def summarize_slope(name, y, boot, logn, fam):
    """Point slope, percentile CIs (95%, 98.33%), leave-one-model-out jackknife."""
    s = float(fe_slope(y, logn, fam)[0])
    out = {"estimator": name, "slope_per_decade": s}
    if boot is not None:
        bs = np.array([fe_slope(boot[:, b], logn, fam)[0] for b in range(boot.shape[1])])
        out["ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        out["ci9833"] = [float(np.quantile(bs, 0.00835)), float(np.quantile(bs, 0.99165))]
        out["boot_slopes"] = bs
    jk = np.array([fe_slope(y[m], logn[m], list(np.array(fam)[m]))[0]
                   for m in (np.arange(len(y)) != i for i in range(len(y)))])
    out["jackknife_min_max"] = [float(jk.min()), float(jk.max())]
    out["jackknife_se"] = float(np.sqrt((len(jk) - 1) / len(jk) * ((jk - jk.mean()) ** 2).sum()))
    return out


def aggregate():
    from scipy.stats import spearmanr
    rows, boots = [], []
    missing = [k for k in CONFIRMATORY if not (OUT / f"{k}.json").exists()]
    for k in CONFIRMATORY:
        if (OUT / f"{k}.json").exists():
            rows.append(json.loads((OUT / f"{k}.json").read_text(encoding="utf-8")))
            boots.append(dict(np.load(OUT / f"{k}_boot.npz")))
    fam = [r["family"] for r in rows]
    logn = np.log10([r["nominal_B"] * 1e9 for r in rows])
    L = np.array([r["n_layers"] for r in rows], dtype=float)
    col = lambda key: np.array([r[key] for r in rows], dtype=float)
    bcol = lambda key: np.stack([b[key] for b in boots])  # (models, B)

    rep = {"models_analysed": [r["key"] for r in rows], "models_missing": missing, "n_models": len(rows)}

    # H1a / H1b
    n1 = col("sad_N1_mean")
    rep["H1a"] = {"min_N1_mean": float(n1.min()), "argmin": rows[int(n1.argmin())]["key"],
                  "confirmed": bool(n1.min() >= 0.02)}
    rho = spearmanr(n1, L).statistic
    rng = np.random.default_rng(SEED)
    perm = np.array([spearmanr(n1, rng.permutation(L)).statistic for _ in range(10000)])
    p_h1b = float((1 + (perm >= rho).sum()) / (1 + len(perm)))
    rep["H1b"] = {"spearman_rho": float(rho), "p_one_sided": p_h1b,
                  "confirmed_bonferroni": bool(rho > 0 and p_h1b < 0.05 / 3), "confirmed_uncorrected": bool(rho > 0 and p_h1b < 0.05)}

    # slopes
    S = {}
    for name, key, has_boot in (("sad_E_naive", "sad_E_naive", True), ("sad_E_split", "sad_E_split", True),
                                ("sad_E_fixed", "sad_E_fixed", False), ("sad_E_naive_no_l0", "sad_E_naive_no_l0", False),
                                ("E_naive2x2", "E_naive2x2", True), ("E_corr2x2", "E_corr2x2", True),
                                ("std_crossed_auc_valsel", "std_crossed_auc_valsel", True)):
        S[name] = summarize_slope(name, col(key), bcol(key) if has_boot else None, logn, fam)
    # E_corr on SAD: E_naive bootstrap minus fixed N1 mean
    S["sad_E_corr"] = summarize_slope("sad_E_corr", col("sad_E_corr"), bcol("sad_E_naive") - n1[:, None], logn, fam)
    for p in ("STD", "PAIRED"):
        k = f"expl_sad_transfer_{p}_auc_valsel"
        if all(k in r for r in rows):
            S[f"EXPLORATORY_{k}"] = summarize_slope(k, col(k), None, logn, fam)

    def ci_excl0(d, lvl):
        lo, hi = d[lvl]
        return bool(lo > 0 or hi < 0)

    rep["H2"] = {"slope": S["sad_E_naive"]["slope_per_decade"], "ci9833": S["sad_E_naive"]["ci9833"],
                 "ci95": S["sad_E_naive"]["ci95"],
                 "confirmed_bonferroni": bool(S["sad_E_naive"]["slope_per_decade"] > 0 and S["sad_E_naive"]["ci9833"][0] > 0),
                 "confirmed_uncorrected": bool(S["sad_E_naive"]["slope_per_decade"] > 0 and S["sad_E_naive"]["ci95"][0] > 0)}
    frac = float((col("std_crossed_auc_valsel") < 0.5).mean())
    rep["H3"] = {"fraction_models_crossed_below_0.5": frac, "confirmed": bool(frac >= 0.8)}
    diff = S["E_naive2x2"]["boot_slopes"] - S["E_corr2x2"]["boot_slopes"]
    d0 = S["E_naive2x2"]["slope_per_decade"] - S["E_corr2x2"]["slope_per_decade"]
    rep["H4"] = {"slope_diff": float(d0), "ci9833": [float(np.quantile(diff, 0.00835)), float(np.quantile(diff, 0.99165))],
                 "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
                 "E_corr2x2_values": col("E_corr2x2").tolist(),
                 "ceiling_models": int((col("E_corr2x2") >= 0.49).sum()),
                 "caveat": "PAIRED is trained on CE and BD; E_corr2x2 may sit at ceiling (prereg §15, 2026-09-15 00:20)"}
    rep["H4"]["confirmed_bonferroni"] = bool(d0 > 0 and rep["H4"]["ci9833"][0] > 0)
    rep["H4"]["confirmed_uncorrected"] = bool(d0 > 0 and rep["H4"]["ci95"][0] > 0)
    sn = S["E_naive2x2"]["slope_per_decade"]
    rep["H5"] = {"slope_E_corr2x2": S["E_corr2x2"]["slope_per_decade"], "ci95": S["E_corr2x2"]["ci95"],
                 "ratio_corr_over_naive": float(S["E_corr2x2"]["slope_per_decade"] / sn) if sn != 0 else None}
    b6 = fe_slope(col("sad_E_naive"), logn, fam, extra=np.log10(L))
    rep["H6"] = {"coef_log10N": float(b6[0]), "coef_log10L": float(b6[1]),
                 "smollm2": [{"key": r["key"], "L": r["n_layers"], "N_B": r["nominal_B"], "E_naive": r["sad_E_naive"],
                              "N1_mean": r["sad_N1_mean"]} for r in rows if r["family"] == "SmolLM2"]}
    rep["slopes"] = {k: {kk: vv for kk, vv in v.items() if kk != "boot_slopes"} for k, v in S.items()}
    rep["per_model"] = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
    tb = OUT / "text_baselines.json"
    if tb.exists():
        rep["text_baselines"] = json.loads(tb.read_text(encoding="utf-8"))
    (OUT / "AGGREGATE.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p1 = sp.add_parser("per-model")
    p1.add_argument("--model", required=True)
    sp.add_parser("text-baselines")
    sp.add_parser("aggregate")
    args = ap.parse_args()
    if args.cmd == "per-model":
        r = per_model(args.model)
        print(json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}, indent=1))
    elif args.cmd == "text-baselines":
        print(json.dumps(text_baselines(), indent=1))
    elif args.cmd == "aggregate":
        r = aggregate()
        print(json.dumps({k: v for k, v in r.items() if k not in ("per_model", "slopes")}, indent=1))


if __name__ == "__main__":
    main()
