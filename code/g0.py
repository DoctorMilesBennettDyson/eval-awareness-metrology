"""Validation gate G0 (prereg §10): replicate Devbunova (2026) on Llama-3.1-8B-Instruct.

PASS iff  STD logistic probe (BE vs CD) classifies >= 80% of BD test items as evaluation
     and  PAIRED logistic probe (BE, CE vs CD, BD) classifies <= 20% of BD test items as evaluation.
Layer 24 (resid_pre of block 24), last prompt token, StandardScaler + L2 logistic regression (C = 1.0).

    python g0.py [--model llama3.1-8b] [--layer 24]
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import ROOT

ACT = ROOT / "results" / "activations"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--layer", type=int, default=24)
    args = ap.parse_args()
    d = ACT / args.model
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    X = np.load(d / "d3_last.npy", mmap_mode="r")
    cell = np.array(meta["d3"]["cell"])
    split = np.array(meta["d3"]["split"])

    def rows(cells, sp):
        return np.where(np.isin(cell, cells) & (split == sp))[0]

    def fit(pos_cells, neg_cells):
        tr = np.concatenate([rows(pos_cells, "train"), rows(neg_cells, "train")])
        y = np.isin(cell[tr], pos_cells).astype(int)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(np.asarray(X[tr, args.layer], dtype=np.float64), y)
        return clf

    out = {"model": args.model, "layer": args.layer}
    for name, pos_c, neg_c in (("STD", ["BE"], ["CD"]), ("PAIRED", ["BE", "CE"], ["CD", "BD"])):
        clf = fit(pos_c, neg_c)
        for c in ("BE", "CD", "CE", "BD"):
            te = rows([c], "test")
            out[f"{name}_pct_eval_{c}"] = float(100 * clf.predict(np.asarray(X[te, args.layer], dtype=np.float64)).mean())
    out["published"] = {"STD_pct_eval_BD": 94.5, "PAIRED_pct_eval_BD": 7.2}
    out["PASS"] = bool(out["STD_pct_eval_BD"] >= 80.0 and out["PAIRED_pct_eval_BD"] <= 20.0)
    (ROOT / "results" / f"G0_{args.model}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
