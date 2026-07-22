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
TRAIN_COHORT   = "AUGSBURG"
HOLDOUT_COHORT = "PRE-RAPID"
SEEDS          = list(range(20))   # 0..19
N_EPOCHS       = 1000
RESULTS_PATH   = Path("sweep_results.jsonl")


def load_existing_results() -> dict[int, dict]:
    """Load already-completed seed results, keyed by seed, so a rerun can skip them.

    NOTE: if RESULTS_PATH was written by an older version of this script
    (missing recall_neg / balanced_acc), delete it before rerunning, or
    summarize() will KeyError on those old entries.
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
            results[r["seed"]] = r
    return results


def append_result(r: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(r) + "\n")


def run_one_seed(seed: int, all_cohorts: dict) -> dict:
    aug_patients = all_cohorts[TRAIN_COHORT]
    pr_patients  = all_cohorts[HOLDOUT_COHORT]

    trainval_aug, test_aug = train_test_split(aug_patients, test_size=0.2, random_state=seed)
    train_aug,    val_aug  = train_test_split(trainval_aug, test_size=0.25, random_state=seed)

    trainval_pr, test_pr = train_test_split(pr_patients, test_size=0.2, random_state=seed)
    train_pr,    val_pr  = train_test_split(trainval_pr, test_size=0.25, random_state=seed)

    torch.manual_seed(seed)
    model = HarmonizationModel(latent_dim=64)
    model = train_harmonization(
        model,
        train_aug=train_aug, val_aug=val_aug,
        train_pr=train_pr,   val_pr=val_pr,
        n_epochs=N_EPOCHS,
        lambda_mmd=1,
        checkpoint_path=None,
    )

    Z_train, y_train = encode_patients(model, train_aug + val_aug)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(Z_train, y_train)

    def eval_set(plist):
        if not plist:
            return None
        Z, y = encode_patients(model, plist)
        y_pred = clf.predict(Z)
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
        "seed": seed,
        "train_cohort_test": eval_set(test_aug),
        "holdout_cohort_test": eval_set(test_pr),
    }


def summarize(results: list, key: str, label: str) -> None:
    accs          = [r[key]["acc"] for r in results if r[key] is not None]
    aucs          = [r[key]["auc"] for r in results if r[key] is not None]
    recalls_pos   = [r[key]["recall_pos"] for r in results if r[key] is not None]
    recalls_neg   = [r[key]["recall_neg"] for r in results if r[key] is not None]
    balanced_accs = [r[key]["balanced_acc"] for r in results if r[key] is not None]

    print(f"\n=== {label} across {len(accs)} seeds ===")
    print(f"acc            : {np.nanmean(accs):.3f} \u00b1 {np.nanstd(accs):.3f}")
    print(f"auc            : {np.nanmean(aucs):.3f} \u00b1 {np.nanstd(aucs):.3f}")
    print(f"recall(pos)    : {np.nanmean(recalls_pos):.3f} \u00b1 {np.nanstd(recalls_pos):.3f}")
    print(f"recall(neg)    : {np.nanmean(recalls_neg):.3f} \u00b1 {np.nanstd(recalls_neg):.3f}")
    print(f"balanced acc   : {np.nanmean(balanced_accs):.3f} \u00b1 {np.nanstd(balanced_accs):.3f}")
    print(f"per-seed acc   : {[round(a, 3) for a in accs]}")
    print(f"per-seed auc   : {[round(a, 3) for a in aucs]}")
    print(f"per-seed rec+  : {[round(r, 3) for r in recalls_pos]}")
    print(f"per-seed rec-  : {[round(r, 3) for r in recalls_neg]}")
    print(f"per-seed bacc  : {[round(b, 3) for b in balanced_accs]}")


if __name__ == "__main__":
    all_cohorts = load_all_cohorts(Path(DATA_PATH))

    existing = load_existing_results()
    if existing:
        print(f"found {len(existing)} completed seed(s) in {RESULTS_PATH}, will skip those: "
              f"{sorted(existing.keys())}")

    results = []
    for seed in SEEDS:
        if seed in existing:
            results.append(existing[seed])
            continue

        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        r = run_one_seed(seed, all_cohorts)
        append_result(r)   # persist immediately, so a crash doesn't lose this seed
        results.append(r)
        print(f"  {TRAIN_COHORT} test : {r['train_cohort_test']}")
        print(f"  {HOLDOUT_COHORT} test: {r['holdout_cohort_test']}")

    summarize(results, "train_cohort_test", f"{TRAIN_COHORT} TEST")
    summarize(results, "holdout_cohort_test", f"{HOLDOUT_COHORT} TEST")