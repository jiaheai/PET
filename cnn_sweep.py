"""
Compares CNN baseline capacity (latent_dim) with z-score shift
correction, across multiple torch seeds, to see whether more capacity
(64-dim) helps or hurts once corrected, and how stable each is.

Reuses cnn_baseline_clean.py's model/training/correction code directly
rather than duplicating it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from cnn import (
    CNNClassifier3D,
    train_cnn_baseline,
    zscore_shift_correct,
    eval_on_probs,
)
from nifti_loader import load_all_cohorts

DATA_PATH      = "CUBES-Labelled-COHORTS-HISTMATCH"
TRAIN_COHORT   = "AUGSBURG"
HOLDOUT_COHORT = "PRE-RAPID"

LATENT_DIM    = 16               # fixed -- 16-dim beat 64-dim in the earlier 5-seed comparison
TORCH_SEEDS   = list(range(30))  # 0..29 -- matches the 30-seed convention used for harmonization sweeps
VAL_SPLIT_SEED = 40          # fixed -- same AUGSBURG train/val split across all runs
N_EPOCHS      = 200
RESULTS_PATH  = Path("cnn_sweep_results.jsonl")


def get_probs(model, plist, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for p in plist:
            vol = torch.from_numpy(p.pet_masked.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
            probs.append(torch.sigmoid(model(vol)).item())
            ys.append(p.label)
    return np.array(probs), np.array(ys)


def load_existing_results() -> dict:
    if not RESULTS_PATH.exists():
        return {}
    results = {}
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            results[(r["latent_dim"], r["torch_seed"])] = r
    return results


def append_result(r: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(r) + "\n")


def run_one(torch_seed: int, aug_patients: list, pr_patients: list) -> dict:
    train_aug, val_aug = train_test_split(aug_patients, test_size=0.2, random_state=VAL_SPLIT_SEED)

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

    prob_pr_corrected = zscore_shift_correct(prob_source=prob_aug, prob_target=prob_pr)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_pr, prob_pr) if len(np.unique(y_pr)) > 1 else float("nan")

    print(f"\n--- latent_dim={LATENT_DIM}  torch_seed={torch_seed} ---")
    result_raw = eval_on_probs(y_pr, prob_pr, f"PRE-RAPID raw (seed={torch_seed})", threshold=0.5)
    result_corrected = eval_on_probs(y_pr, prob_pr_corrected,
                                      f"PRE-RAPID z-score corrected (seed={torch_seed})",
                                      threshold=0.5)

    return {
        "latent_dim": LATENT_DIM,
        "torch_seed": torch_seed,
        "auc": float(auc),
        "raw": result_raw,
        "corrected": result_corrected,
    }


def summarize(results: list) -> None:
    if not results:
        return
    aucs           = [r["auc"] for r in results]
    raw_baccs      = [r["raw"]["bacc"] for r in results]
    raw_recall_p   = [r["raw"]["recall_pos"] for r in results]
    raw_recall_n   = [r["raw"]["recall_neg"] for r in results]
    corr_baccs     = [r["corrected"]["bacc"] for r in results]
    corr_recall_p  = [r["corrected"]["recall_pos"] for r in results]
    corr_recall_n  = [r["corrected"]["recall_neg"] for r in results]

    print(f"\n=== latent_dim={LATENT_DIM} across {len(results)} seeds ===")
    print(f"raw auc             : {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}  "
          f"(unchanged by correction -- threshold-independent)")
    print(f"raw bal acc         : {np.nanmean(raw_baccs):.3f} +/- {np.nanstd(raw_baccs):.3f}")
    print(f"raw recall(+)       : {np.nanmean(raw_recall_p):.3f} +/- {np.nanstd(raw_recall_p):.3f}")
    print(f"raw recall(-)       : {np.nanmean(raw_recall_n):.3f} +/- {np.nanstd(raw_recall_n):.3f}")
    print(f"corrected bal acc   : {np.nanmean(corr_baccs):.3f} +/- {np.nanstd(corr_baccs):.3f}")
    print(f"corrected recall(+) : {np.nanmean(corr_recall_p):.3f} +/- {np.nanstd(corr_recall_p):.3f}")
    print(f"corrected recall(-) : {np.nanmean(corr_recall_n):.3f} +/- {np.nanstd(corr_recall_n):.3f}")
    print(f"per-seed corrected bal acc: {[round(b, 3) for b in corr_baccs]}")
    print(f"per-seed raw bal acc      : {[round(b, 3) for b in raw_baccs]}")
    print(f"per-seed raw recall(+)    : {[round(r, 3) for r in raw_recall_p]}")
    print(f"per-seed raw recall(-)    : {[round(r, 3) for r in raw_recall_n]}")
    print(f"per-seed raw auc          : {[round(a, 3) for a in aucs]}")


if __name__ == "__main__":
    all_cohorts = load_all_cohorts(Path(DATA_PATH))
    aug_patients = all_cohorts[TRAIN_COHORT]
    pr_patients  = all_cohorts[HOLDOUT_COHORT]

    existing = load_existing_results()
    if existing:
        print(f"found {len(existing)} completed seed(s), will skip those: "
              f"{sorted(seed for (_, seed) in existing.keys())}")

    results = []
    for torch_seed in TORCH_SEEDS:
        key = (LATENT_DIM, torch_seed)
        if key in existing:
            results.append(existing[key])
            continue
        r = run_one(torch_seed, aug_patients, pr_patients)
        append_result(r)
        results.append(r)

    summarize(results)