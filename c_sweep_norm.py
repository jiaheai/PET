"""
Seed sweep for Option C (concatenated dual independent autoencoders),
WITH z-score correction added -- matches b_sweep_norm.py's pattern,
so C is on equal footing with B for the raw-vs-corrected comparison.

Same 30-seed, torch-seed-only convention as c_sweep.py. Only addition:
after fitting the classifier on AUGSBURG's concatenated latents, also
compute z-score-corrected PRE-RAPID probabilities (aligned to AUGSBURG's
probability distribution) and report both raw and corrected results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from c import Autoencoder3D, train_autoencoder, encode_concat
from nifti_loader import load_all_cohorts
from cnn import zscore_shift_correct   # reuse the same correction used for the CNN baseline

# Defaults -- overridable via --data-path / --results-path for automation
# (e.g. running the same sweep against raw / -ZSCORE / -HISTMATCH data
# without editing this file each time).
DEFAULT_DATA_PATH    = "CUBES-Labelled-COHORTS-HISTMATCH"
DEFAULT_RESULTS_PATH = "c_sweep_norm_results.jsonl"

TRAIN_COHORT   = "AUGSBURG"
HOLDOUT_COHORT = "PRE-RAPID"

TORCH_SEEDS    = list(range(30))   # 0..29 -- matches A/B/CNN sweep convention
VAL_SPLIT_SEED = 40                # fixed -- same per-cohort train/val split every run
N_EPOCHS       = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Option C sweep (30 seeds, with z-score correction)")
    parser.add_argument(
        "--data-path", default=DEFAULT_DATA_PATH,
        help=f"Directory containing AUGSBURG/PRE-RAPID cohort data (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--results-path", default=DEFAULT_RESULTS_PATH,
        help=f"Where to write/resume sweep results (default: {DEFAULT_RESULTS_PATH}). "
             "Use a distinct path per data variant so results from different "
             "preprocessing (raw/z-score/histogram-match) don't get mixed together.",
    )
    return parser.parse_args()


def load_existing_results(results_path: Path) -> dict[int, dict]:
    """Load already-completed seed results, keyed by torch seed, so a rerun can skip them.

    NOTE: this version adds "holdout_cohort_corrected" to each result.
    If results_path was written by c_sweep.py (no correction) instead of
    this script, delete it before rerunning -- summarize() will KeyError
    on the missing field otherwise.
    """
    if not results_path.exists():
        return {}
    results = {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            results[r["torch_seed"]] = r
    return results


def append_result(r: dict, results_path: Path) -> None:
    with open(results_path, "a") as f:
        f.write(json.dumps(r) + "\n")


def eval_on(y: np.ndarray, y_prob: np.ndarray) -> dict:
    """Same metrics as before, but takes precomputed (y, y_prob) directly
    so it can score either raw or z-score corrected probabilities
    through the same code path."""
    y_pred = (y_prob >= 0.5).astype(int)
    acc  = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    auc  = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "acc": float(acc), "balanced_acc": float(bacc), "auc": float(auc),
        "recall_pos": float(recall_pos), "recall_neg": float(recall_neg),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_one_seed(torch_seed: int, aug_patients: list, pr_patients: list) -> dict:
    # stratified so val doesn't accidentally end up class-skewed at n~10
    train_aug, val_aug = train_test_split(
        aug_patients, test_size=0.2, random_state=VAL_SPLIT_SEED,
        stratify=[p.label for p in aug_patients],
    )
    train_pr,  val_pr  = train_test_split(
        pr_patients, test_size=0.2, random_state=VAL_SPLIT_SEED,
        stratify=[p.label for p in pr_patients],
    )

    torch.manual_seed(torch_seed)
    model_aug = Autoencoder3D(latent_dim=64)
    model_aug = train_autoencoder(
        model_aug, train_aug, val_aug, n_epochs=N_EPOCHS,
        checkpoint_path=None, label=f"AUGSBURG seed={torch_seed}",
    )

    torch.manual_seed(torch_seed + 1000)   # offset so the two encoders don't get identical init
    model_pr = Autoencoder3D(latent_dim=64)
    model_pr = train_autoencoder(
        model_pr, train_pr, val_pr, n_epochs=N_EPOCHS,
        checkpoint_path=None, label=f"PRE-RAPID seed={torch_seed}",
    )

    Z_aug, y_aug = encode_concat(model_aug.encoder, model_pr.encoder, aug_patients)
    Z_pr,  y_pr  = encode_concat(model_aug.encoder, model_pr.encoder, pr_patients)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(Z_aug, y_aug)

    prob_aug = clf.predict_proba(Z_aug)[:, 1]
    prob_pr  = clf.predict_proba(Z_pr)[:, 1]

    # -- z-score correction: align PRE-RAPID's predicted-probability -----
    # distribution to AUGSBURG's (unlabeled statistics only, no PRE-RAPID
    # labels used in computing the correction itself -- same pattern as
    # b_sweep_norm.py and cnn.py).
    prob_pr_corrected = zscore_shift_correct(prob_source=prob_aug, prob_target=prob_pr)

    return {
        "torch_seed": torch_seed,
        "train_cohort": eval_on(y_aug, prob_aug),                      # in-sample AUGSBURG, reference only
        "holdout_cohort": eval_on(y_pr, prob_pr),                      # raw -- original headline result
        "holdout_cohort_corrected": eval_on(y_pr, prob_pr_corrected),  # z-score corrected
    }


def summarize(results: list, key: str, label: str) -> None:
    accs   = [r[key]["acc"] for r in results]
    baccs  = [r[key]["balanced_acc"] for r in results]
    aucs   = [r[key]["auc"] for r in results]
    rec_p  = [r[key]["recall_pos"] for r in results]
    rec_n  = [r[key]["recall_neg"] for r in results]

    print(f"\n=== {label} across {len(results)} seeds ===")
    print(f"acc         : {np.nanmean(accs):.3f} +/- {np.nanstd(accs):.3f}")
    print(f"bal acc     : {np.nanmean(baccs):.3f} +/- {np.nanstd(baccs):.3f}")
    print(f"auc         : {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")
    print(f"recall(pos) : {np.nanmean(rec_p):.3f} +/- {np.nanstd(rec_p):.3f}")
    print(f"recall(neg) : {np.nanmean(rec_n):.3f} +/- {np.nanstd(rec_n):.3f}")
    print(f"per-seed bal acc: {[round(b, 3) for b in baccs]}")
    print(f"per-seed auc    : {[round(a, 3) for a in aucs]}")
    print(f"per-seed rec+   : {[round(r, 3) for r in rec_p]}")
    print(f"per-seed rec-   : {[round(r, 3) for r in rec_n]}")


if __name__ == "__main__":
    args = parse_args()
    data_path = Path(args.data_path)
    results_path = Path(args.results_path)

    print(f"data path    : {data_path}")
    print(f"results path : {results_path}")

    all_cohorts = load_all_cohorts(data_path)
    aug_patients = all_cohorts[TRAIN_COHORT]
    pr_patients  = all_cohorts[HOLDOUT_COHORT]

    existing = load_existing_results(results_path)
    if existing:
        print(f"found {len(existing)} completed seed(s), will skip those: {sorted(existing.keys())}")

    results = []
    for torch_seed in TORCH_SEEDS:
        if torch_seed in existing:
            results.append(existing[torch_seed])
            continue
        print(f"\n{'='*60}\nTORCH SEED {torch_seed}\n{'='*60}")
        r = run_one_seed(torch_seed, aug_patients, pr_patients)
        append_result(r, results_path)
        results.append(r)
        print(f"  {TRAIN_COHORT} (in-sample): {r['train_cohort']}")
        print(f"  {HOLDOUT_COHORT} (test, raw)      : {r['holdout_cohort']}")
        print(f"  {HOLDOUT_COHORT} (test, corrected): {r['holdout_cohort_corrected']}")

    summarize(results, "train_cohort", f"{TRAIN_COHORT} (in-sample, reference only)")
    summarize(results, "holdout_cohort", f"{HOLDOUT_COHORT} (test -- raw, original headline result)")
    summarize(results, "holdout_cohort_corrected", f"{HOLDOUT_COHORT} (test -- z-score corrected)")