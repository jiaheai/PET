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

from harmonization import HarmonizationModel, train_harmonization
from classifier import encode_patients
from nifti_loader import load_all_cohorts

DATA_PATH      = "CUBES-Labelled-COHORTS"
TRAIN_COHORT   = "AUGSBURG"    # entire cohort -- always the classifier's training set
HOLDOUT_COHORT = "PRE-RAPID"   # entire cohort -- always the classifier's test set

# No data splitting anymore: per the supervisor's design, harmonization
# sees ALL patients from both cohorts (no held-out carve-out), and the
# classifier always trains on the full TRAIN_COHORT / tests on the full
# HOLDOUT_COHORT. The only thing left to vary across a sweep is the
# harmonization model's random initialization / training stochasticity.
TORCH_SEEDS  = list(range(30))   # 0..9 -- init/training-stochasticity sweep
VAL_FRACTION = 0.2               # per-cohort val split, used only for early stopping
VAL_SPLIT_SEED = 40              # fixed -- keeps the same val patients across all torch seeds
N_EPOCHS     = 1000
RESULTS_PATH = Path("sweep_results.jsonl")


def load_existing_results() -> dict[int, dict]:
    """Load already-completed seed results, keyed by torch seed, so a rerun can skip them.

    NOTE: if RESULTS_PATH was written by an older (data-split-based) version
    of this script, delete it before rerunning -- the schema here has no
    "threshold" field and is keyed differently, so old entries won't match.
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

    # val split is ONLY for harmonizer early stopping -- fixed across all
    # torch seeds so every run's encoder is trained/selected against the
    # same val patients, isolating init/training stochasticity as the one
    # variable. Both train and val patients still count as "seen" by the
    # harmonizer -- nothing is held out from it, per the supervisor's note
    # that harmonization may see all data.
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
    # no split, per the supervisor's cohort-as-train/cohort-as-test design.
    Z_train, y_train = encode_patients(model, aug_patients)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(Z_train, y_train)

    def eval_set(plist):
        if not plist:
            return None
        Z, y = encode_patients(model, plist)
        y_pred = clf.predict(Z)          # default 0.5 threshold, no tuning
        y_prob = clf.predict_proba(Z)[:, 1]
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

    return {
        "torch_seed": torch_seed,
        "train_cohort": eval_set(aug_patients),      # in-sample AUGSBURG, reference only
        "holdout_cohort": eval_set(pr_patients),      # the real result -- all 28 PRE-RAPID
    }


def summarize(results: list, key: str, label: str) -> None:
    aucs          = [r[key]["auc"] for r in results if r[key] is not None]
    recalls_pos   = [r[key]["recall_pos"] for r in results if r[key] is not None]
    recalls_neg   = [r[key]["recall_neg"] for r in results if r[key] is not None]
    balanced_accs = [r[key]["balanced_acc"] for r in results if r[key] is not None]

    print(f"\n=== {label} across {len(aucs)} torch seeds ===")
    print(f"balanced acc   : {np.nanmean(balanced_accs):.3f} +/- {np.nanstd(balanced_accs):.3f}")
    print(f"auc            : {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")
    print(f"recall(pos)    : {np.nanmean(recalls_pos):.3f} +/- {np.nanstd(recalls_pos):.3f}")
    print(f"recall(neg)    : {np.nanmean(recalls_neg):.3f} +/- {np.nanstd(recalls_neg):.3f}")
    print(f"per-seed bacc  : {[round(b, 3) for b in balanced_accs]}")
    print(f"per-seed auc   : {[round(a, 3) for a in aucs]}")
    print(f"per-seed rec+  : {[round(r, 3) for r in recalls_pos]}")
    print(f"per-seed rec-  : {[round(r, 3) for r in recalls_neg]}")


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
    summarize(results, "holdout_cohort", f"{HOLDOUT_COHORT} (test)")