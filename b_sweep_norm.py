from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from b import HarmonizationModel, train_harmonization
from classifier import encode_patients
from nifti_loader import load_all_cohorts
from cnn import zscore_shift_correct   # reuse the same correction used for the CNN baseline

DATA_PATH      = "CUBES-Labelled-COHORTS"
TRAIN_COHORT   = "AUGSBURG"    # entire cohort -- always the classifier's training set
HOLDOUT_COHORT = "PRE-RAPID"   # entire cohort -- always the classifier's test set

TORCH_SEEDS  = list(range(30))   # 0..9 -- init/training-stochasticity sweep
VAL_FRACTION = 0.2               # per-cohort val split, used only for early stopping
VAL_SPLIT_SEED = 40              # fixed -- keeps the same val patients across all torch seeds
N_EPOCHS     = 1000
RESULTS_PATH = Path("b_sweep_norm_results.jsonl")


def load_existing_results() -> dict[int, dict]:
    """Load already-completed seed results, keyed by torch seed, so a rerun can skip them.

    NOTE: this version adds "holdout_cohort_corrected" to each result.
    If RESULTS_PATH was written by an earlier version of this script
    (missing that field, or the older "threshold" field), delete it
    before rerunning -- summarize() will KeyError on the missing field
    otherwise.
    """
    if not RESULTS_PATH.exists():
        return {}
    results = {}
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            results[r["torch_seed"]] = r
    return results


def append_result(r: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(r) + "\n")


def run_one_seed(torch_seed: int, all_cohorts: dict) -> dict:
    aug_patients = all_cohorts[TRAIN_COHORT]     # all 50
    pr_patients  = all_cohorts[HOLDOUT_COHORT]   # all 28

    train_aug, val_aug = train_test_split(aug_patients, test_size=VAL_FRACTION, random_state=VAL_SPLIT_SEED)
    train_pr,  val_pr  = train_test_split(pr_patients,  test_size=VAL_FRACTION, random_state=VAL_SPLIT_SEED)

    torch.manual_seed(torch_seed)
    model = HarmonizationModel(latent_dim=64)
    model = train_harmonization(
        model,
        train_aug=train_aug, val_aug=val_aug,
        train_pr=train_pr,   val_pr=val_pr,
        n_epochs=N_EPOCHS,
        lambda_mmd=1,
        checkpoint_path=None,
    )

    # classifier: fit on ALL of AUGSBURG, test on ALL of PRE-RAPID --
    Z_train, y_train = encode_patients(model, aug_patients)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(Z_train, y_train)

    def eval_set(plist, prob_override=None):
        if not plist:
            return None
        Z, y = encode_patients(model, plist)
        y_prob = clf.predict_proba(Z)[:, 1] if prob_override is None else prob_override
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        recall_pos = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        recall_neg = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        balanced_acc = float(np.nanmean([recall_pos, recall_neg]))
        return {
            "acc": float(acc), "auc": float(auc),
            "recall_pos": float(recall_pos), "recall_neg": float(recall_neg),
            "balanced_acc": balanced_acc,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    # -- z-score correction: align PRE-RAPID's predicted-probability -----
    # distribution to AUGSBURG's (unlabeled statistics only, no PRE-RAPID
    # labels used in computing the correction itself -- same principle
    # as cnn.py's zscore_shift_correct, applied here to clf's probabilities
    # instead of the CNN's sigmoid outputs).
    Z_aug_full, _ = encode_patients(model, aug_patients)
    Z_pr_full,  _ = encode_patients(model, pr_patients)
    prob_aug = clf.predict_proba(Z_aug_full)[:, 1]
    prob_pr  = clf.predict_proba(Z_pr_full)[:, 1]
    prob_pr_corrected = zscore_shift_correct(prob_source=prob_aug, prob_target=prob_pr)

    return {
        "torch_seed": torch_seed,
        "train_cohort": eval_set(aug_patients),                              # in-sample AUGSBURG, reference only
        "holdout_cohort": eval_set(pr_patients),                              # raw -- the original headline result
        "holdout_cohort_corrected": eval_set(pr_patients, prob_override=prob_pr_corrected),  # z-score corrected
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
    all_cohorts = load_all_cohorts(Path(DATA_PATH))

    existing = load_existing_results()
    if existing:
        print(f"found {len(existing)} completed torch seed(s) in {RESULTS_PATH}, will skip those: "
              f"{sorted(existing.keys())}")

    results = []
    for torch_seed in TORCH_SEEDS:
        if torch_seed in existing:
            results.append(existing[torch_seed])
            continue

        print(f"\n{'='*60}\nTORCH SEED {torch_seed}\n{'='*60}")
        r = run_one_seed(torch_seed, all_cohorts)
        append_result(r)   # persist immediately, so a crash doesn't lose this seed
        results.append(r)
        print(f"  {TRAIN_COHORT} (in-sample): {r['train_cohort']}")
        print(f"  {HOLDOUT_COHORT} (test)    : {r['holdout_cohort']}")

    summarize(results, "train_cohort", f"{TRAIN_COHORT} (in-sample, reference only)")
    summarize(results, "holdout_cohort", f"{HOLDOUT_COHORT} (test -- raw, original headline result)")
    summarize(results, "holdout_cohort_corrected", f"{HOLDOUT_COHORT} (test -- z-score corrected)")