"""
Approach A, leave-one-cohort-out sweep across 2 cohorts.

Uses only AUGSBURG and SWISS.

For each target cohort:
- 50% of the target is used as the unlabeled harmonization half.
- 50% is fully held out.
- The source cohort uses all of its patients.
- The target harmonization half is split 80/20 without label stratification.
- The source cohort is split 80/20 with label stratification.
- A2's shared encoder/decoder is trained on exactly two groups, so there is
  exactly one MMD pair: source <-> target harmonization half.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from a_2 import HarmonizationModel, train_harmonization
from nifti_loader import load_all_cohorts


DEFAULT_DATA_PATH = "CUBES-Labelled-COHORTS_2"
DEFAULT_RESULTS_PATH = "a_sweep_2cohort_results.jsonl"

COHORT_NAMES = ["AUGSBURG", "SWISS"]
TORCH_SEEDS = list(range(5))
VAL_SPLIT_SEED = 40
TARGET_HOLDOUT_SEED = 123
LAMBDA_MMD = 0.7
DECODER_FREEZE_EPOCH = 50
N_EPOCHS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approach A, leave-one-cohort-out sweep across AUGSBURG and SWISS"
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=f"Directory containing AUGSBURG and SWISS (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--results-path",
        default=DEFAULT_RESULTS_PATH,
        help=f"Where to write sweep results (default: {DEFAULT_RESULTS_PATH})",
    )
    return parser.parse_args()


def append_result(r: dict, results_path: Path) -> None:
    r = {"record_type": "run", **r}
    with open(results_path, "a") as f:
        f.write(json.dumps(r) + "\n")


def append_summary(s: dict, results_path: Path) -> None:
    s = {"record_type": "summary", **s}
    with open(results_path, "a") as f:
        f.write(json.dumps(s) + "\n")


def split_target_cohort(patients: list) -> tuple[list, list]:
    harmonization_half, heldout_half = train_test_split(
        patients,
        test_size=0.5,
        random_state=TARGET_HOLDOUT_SEED,
        stratify=[p.label for p in patients],
    )
    return harmonization_half, heldout_half


def encode_patients(
    model: HarmonizationModel,
    patients: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()

    zs, ys = [], []

    with torch.no_grad():
        for patient in patients:
            vol = (
                torch.from_numpy(patient.pet_masked.astype("float32"))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            z = model.encode(vol).squeeze(0).cpu().numpy()
            zs.append(z)
            ys.append(patient.label)

    return np.stack(zs), np.array(ys)


def eval_set(model, clf, plist) -> dict | None:
    if not plist:
        return None

    Z, y = encode_patients(model, plist)
    y_prob = clf.predict_proba(Z)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    balanced_acc = float(np.nanmean([recall_pos, recall_neg]))

    return {
        "acc": float(acc),
        "auc": float(auc),
        "recall_pos": float(recall_pos),
        "recall_neg": float(recall_neg),
        "balanced_acc": balanced_acc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_model_once(
    torch_seed: int,
    all_cohorts: dict,
    target_cohort: str,
) -> tuple[HarmonizationModel, dict, list, list]:
    target_harmonization, target_heldout = split_target_cohort(
        all_cohorts[target_cohort]
    )

    source_cohort = next(
        cohort for cohort in COHORT_NAMES
        if cohort != target_cohort
    )

    source_patients = all_cohorts[source_cohort]

    train_source, val_source = train_test_split(
        source_patients,
        test_size=0.2,
        random_state=VAL_SPLIT_SEED,
        stratify=[p.label for p in source_patients],
    )

    train_target_harm, val_target_harm = train_test_split(
        target_harmonization,
        test_size=0.2,
        random_state=VAL_SPLIT_SEED,
        stratify=None,
    )

    torch.manual_seed(torch_seed)

    model = HarmonizationModel(latent_dim=64)

    model = train_harmonization(
        model,
        train_cohorts={
            source_cohort: train_source,
            target_cohort: train_target_harm,
        },
        val_cohorts={
            source_cohort: val_source,
            target_cohort: val_target_harm,
        },
        n_epochs=N_EPOCHS,
        lambda_mmd=LAMBDA_MMD,
        decoder_freeze_epoch=DECODER_FREEZE_EPOCH,
        checkpoint_path=None,
    )

    return model, all_cohorts, target_harmonization, target_heldout


def evaluate_target(
    torch_seed: int,
    model: HarmonizationModel,
    cohort_all: dict,
    target_cohort: str,
    target_harmonization: list,
    target_heldout: list,
) -> dict:
    source_cohort = next(
        cohort for cohort in COHORT_NAMES
        if cohort != target_cohort
    )

    source_patients = cohort_all[source_cohort]

    Z_source, y_source = encode_patients(model, source_patients)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    )
    clf.fit(Z_source, y_source)

    return {
        "model": "A",
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "source_cohort": source_cohort,
        "known_cohort_insample": eval_set(
            model,
            clf,
            source_patients,
        ),
        "target_cohort_harmonization_half": eval_set(
            model,
            clf,
            target_harmonization,
        ),
        "target_cohort_raw": eval_set(
            model,
            clf,
            target_heldout,
        ),
    }


def summarize(results: list, key: str, label: str) -> dict:
    accs = [r[key]["acc"] for r in results if r[key] is not None]
    aucs = [r[key]["auc"] for r in results if r[key] is not None]
    recalls_pos = [r[key]["recall_pos"] for r in results if r[key] is not None]
    recalls_neg = [r[key]["recall_neg"] for r in results if r[key] is not None]
    balanced_accs = [
        r[key]["balanced_acc"]
        for r in results
        if r[key] is not None
    ]

    print(f"\n=== {label} across {len(accs)} runs ===")
    print(f"acc            : {np.nanmean(accs):.3f} +/- {np.nanstd(accs):.3f}")
    print(f"auc            : {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")
    print(f"recall(pos)    : {np.nanmean(recalls_pos):.3f} +/- {np.nanstd(recalls_pos):.3f}")
    print(f"recall(neg)    : {np.nanmean(recalls_neg):.3f} +/- {np.nanstd(recalls_neg):.3f}")
    print(f"balanced acc   : {np.nanmean(balanced_accs):.3f} +/- {np.nanstd(balanced_accs):.3f}")
    print(f"per-run acc    : {[round(a, 3) for a in accs]}")
    print(f"per-run auc    : {[round(a, 3) for a in aucs]}")
    print(f"per-run rec+   : {[round(r, 3) for r in recalls_pos]}")
    print(f"per-run rec-   : {[round(r, 3) for r in recalls_neg]}")
    print(f"per-run bacc   : {[round(b, 3) for b in balanced_accs]}")

    return {
        "model": "A",
        "label": label,
        "key": key,
        "n_runs": len(accs),
        "acc_mean": float(np.nanmean(accs)),
        "acc_std": float(np.nanstd(accs)),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "recall_pos_mean": float(np.nanmean(recalls_pos)),
        "recall_pos_std": float(np.nanstd(recalls_pos)),
        "recall_neg_mean": float(np.nanmean(recalls_neg)),
        "recall_neg_std": float(np.nanstd(recalls_neg)),
        "balanced_acc_mean": float(np.nanmean(balanced_accs)),
        "balanced_acc_std": float(np.nanstd(balanced_accs)),
        "per_run_acc": [round(a, 3) for a in accs],
        "per_run_auc": [round(a, 3) for a in aucs],
        "per_run_recall_pos": [round(r, 3) for r in recalls_pos],
        "per_run_recall_neg": [round(r, 3) for r in recalls_neg],
        "per_run_balanced_acc": [round(b, 3) for b in balanced_accs],
    }


if __name__ == "__main__":
    args = parse_args()

    data_path = Path(args.data_path)
    results_path = Path(args.results_path)

    print(f"data path    : {data_path}")
    print(f"results path : {results_path}")
    print(f"cohorts      : {COHORT_NAMES}")

    loaded_cohorts = load_all_cohorts(data_path)

    for name in COHORT_NAMES:
        if name not in loaded_cohorts:
            raise ValueError(
                f"Cohort '{name}' not found. "
                f"Available cohorts: {list(loaded_cohorts.keys())}"
            )

    all_cohorts = {
        name: loaded_cohorts[name]
        for name in COHORT_NAMES
    }

    results_path.write_text("")

    results = []

    for torch_seed in TORCH_SEEDS:
        for target_cohort in COHORT_NAMES:
            print(
                f"\n{'=' * 60}\n"
                f"TORCH SEED {torch_seed}  target={target_cohort}  "
                f"(50% harmonization, 50% fully held out)\n"
                f"{'=' * 60}"
            )

            (
                model,
                cohort_all,
                target_harmonization,
                target_heldout,
            ) = train_model_once(
                torch_seed,
                all_cohorts,
                target_cohort,
            )

            print(
                f"\n--- evaluating target={target_cohort}, "
                f"seed={torch_seed} "
                f"(held out: {len(target_heldout)}, "
                f"in-training: {len(target_harmonization)}) ---"
            )

            r = evaluate_target(
                torch_seed,
                model,
                cohort_all,
                target_cohort,
                target_harmonization,
                target_heldout,
            )

            append_result(r, results_path)
            results.append(r)

            print(
                f"  source cohort (in-sample)      : "
                f"{r['known_cohort_insample']}"
            )
            print(
                f"  {target_cohort} (harmonization half) : "
                f"{r['target_cohort_harmonization_half']}"
            )
            print(
                f"  {target_cohort} (held out)            : "
                f"{r['target_cohort_raw']}"
            )

    summary_stats = []

    for target_cohort in COHORT_NAMES:
        subset = [
            r for r in results
            if r["target_cohort"] == target_cohort
        ]

        summary_stats.append(
            summarize(
                subset,
                "target_cohort_raw",
                f"target={target_cohort} (held-out)",
            )
        )

    summary_stats.append(
        summarize(
            results,
            "target_cohort_raw",
            "ALL target cohorts pooled (held-out)",
        )
    )

    for s in summary_stats:
        append_summary(s, results_path)

    print(f"\nsummary appended to: {results_path}")
