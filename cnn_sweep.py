"""
Compares CNN baseline capacity (latent_dim) with z-score shift
correction, across multiple torch seeds, to see whether more capacity
(64-dim) helps or hurts once corrected, and how stable each is.

Reuses cnn.py's model/training/correction code directly rather than
duplicating it.

Result schema matches a_sweep_norm.py / b_sweep_norm.py / c_sweep_norm.py:
each seed's result has "train_cohort" (in-sample AUGSBURG, reference
only), "holdout_cohort" (raw PRE-RAPID), "holdout_cohort_corrected"
(z-score corrected PRE-RAPID) -- each a dict with acc/balanced_acc/auc/
recall_pos/recall_neg/tn/fp/fn/tp. Keeping this consistent across all
four sweep scripts means one shared consolidation script (see
run_all_and_export_csv.py) can parse all of them the same way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from cnn import CNNClassifier3D, train_cnn_baseline, zscore_shift_correct
from nifti_loader import load_all_cohorts

# Defaults -- overridable via --data-path / --results-path for automation
# (e.g. running the same sweep against raw / -ZSCORE / -HISTMATCH data
# without editing this file each time).
DEFAULT_DATA_PATH    = "CUBES-Labelled-COHORTS"
DEFAULT_RESULTS_PATH = "cnn_sweep_results.jsonl"

TRAIN_COHORT   = "AUGSBURG"
HOLDOUT_COHORT = "PRE-RAPID"

LATENT_DIM     = 16               # fixed -- 16-dim beat 64-dim in the earlier 5-seed comparison
TORCH_SEEDS    = list(range(30))  # 0..29 -- matches the 30-seed convention used for harmonization sweeps
VAL_SPLIT_SEED = 40                # fixed -- same AUGSBURG train/val split across all runs
N_EPOCHS       = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN baseline sweep (30 seeds, with z-score correction)")
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


def get_probs(model, plist, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for p in plist:
            vol = torch.from_numpy(p.pet_masked.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
            probs.append(torch.sigmoid(model(vol)).item())
            ys.append(p.label)
    return np.array(probs), np.array(ys)


def load_existing_results(results_path: Path) -> dict[int, dict]:
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


def eval_on(y: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Same metric shape as a/b/c_sweep_norm.py's eval_set/eval_on --
    acc, auc, recall_pos, recall_neg, balanced_acc, confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "acc": float(acc), "auc": float(auc),
        "recall_pos": float(recall_pos), "recall_neg": float(recall_neg),
        "balanced_acc": float(bacc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_one(torch_seed: int, aug_patients: list, pr_patients: list) -> dict:
    # stratified so val doesn't accidentally end up class-skewed at n~10
    train_aug, val_aug = train_test_split(
        aug_patients, test_size=0.2, random_state=VAL_SPLIT_SEED,
        stratify=[p.label for p in aug_patients],
    )

    torch.manual_seed(torch_seed)
    model = CNNClassifier3D(latent_dim=LATENT_DIM)
    model = train_cnn_baseline(
        model,
        train_patients=train_aug, val_patients=val_aug,
        n_epochs=N_EPOCHS,
        checkpoint_path=None,
    )

    prob_aug, y_aug = get_probs(model, aug_patients)
    prob_pr,  y_pr  = get_probs(model, pr_patients)

    # -- z-score correction: align PRE-RAPID's predicted-probability -----
    # distribution to AUGSBURG's (unlabeled statistics only, no PRE-RAPID
    # labels used in computing the correction itself).
    prob_pr_corrected = zscore_shift_correct(prob_source=prob_aug, prob_target=prob_pr)

    return {
        "torch_seed": torch_seed,
        "train_cohort": eval_on(y_aug, prob_aug),                      # in-sample AUGSBURG, reference only
        "holdout_cohort": eval_on(y_pr, prob_pr),                      # raw -- original headline result
        "holdout_cohort_corrected": eval_on(y_pr, prob_pr_corrected),  # z-score corrected
    }


def summarize(results: list, key: str, label: str) -> None:
    accs          = [r[key]["acc"] for r in results if r[key] is not None]
    aucs          = [r[key]["auc"] for r in results if r[key] is not None]
    recalls_pos   = [r[key]["recall_pos"] for r in results if r[key] is not None]
    recalls_neg   = [r[key]["recall_neg"] for r in results if r[key] is not None]
    balanced_accs = [r[key]["balanced_acc"] for r in results if r[key] is not None]

    print(f"\n=== {label} across {len(accs)} torch seeds ===")
    print(f"acc            : {np.nanmean(accs):.3f} +/- {np.nanstd(accs):.3f}")
    print(f"auc            : {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")
    print(f"recall(pos)    : {np.nanmean(recalls_pos):.3f} +/- {np.nanstd(recalls_pos):.3f}")
    print(f"recall(neg)    : {np.nanmean(recalls_neg):.3f} +/- {np.nanstd(recalls_neg):.3f}")
    print(f"balanced acc   : {np.nanmean(balanced_accs):.3f} +/- {np.nanstd(balanced_accs):.3f}")
    print(f"per-seed acc   : {[round(a, 3) for a in accs]}")
    print(f"per-seed auc   : {[round(a, 3) for a in aucs]}")
    print(f"per-seed rec+  : {[round(r, 3) for r in recalls_pos]}")
    print(f"per-seed rec-  : {[round(r, 3) for r in recalls_neg]}")
    print(f"per-seed bacc  : {[round(b, 3) for b in balanced_accs]}")


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
        r = run_one(torch_seed, aug_patients, pr_patients)
        append_result(r, results_path)
        results.append(r)
        print(f"  {TRAIN_COHORT} (in-sample): {r['train_cohort']}")
        print(f"  {HOLDOUT_COHORT} (test, raw)      : {r['holdout_cohort']}")
        print(f"  {HOLDOUT_COHORT} (test, corrected): {r['holdout_cohort_corrected']}")

    summarize(results, "train_cohort", f"{TRAIN_COHORT} (in-sample, reference only)")
    summarize(results, "holdout_cohort", f"{HOLDOUT_COHORT} (test -- raw, original headline result)")
    summarize(results, "holdout_cohort_corrected", f"{HOLDOUT_COHORT} (test -- z-score corrected)")