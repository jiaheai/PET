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


DEFAULT_DATA_PATH = "CUBES-Labelled-COHORTS"
DEFAULT_RESULTS_PATH = "cnn_sweep_multi_harmonization_half_results.jsonl"

COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "SWISS"]
LATENT_DIM = 16
TORCH_SEEDS = list(range(25))
VAL_SPLIT_SEED = 40
TARGET_HOLDOUT_SEED = 123
N_EPOCHS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original CNN baseline evaluated on the target harmonization half"
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def append_result(r: dict, results_path: Path) -> None:
    with open(results_path, "a") as f:
        f.write(json.dumps({"record_type": "run", **r}) + "\n")


def append_summary(s: dict, results_path: Path) -> None:
    with open(results_path, "a") as f:
        f.write(json.dumps({"record_type": "summary", **s}) + "\n")


def load_existing_results(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []

    runs = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.pop("record_type", None) == "run":
                runs.append(rec)
    return runs


def complete_seeds(runs: list[dict], cohort_names: list[str]) -> set[int]:
    by_seed: dict[int, set[str]] = {}
    for r in runs:
        by_seed.setdefault(r["torch_seed"], set()).add(r["target_cohort"])

    return {
        seed
        for seed, cohorts in by_seed.items()
        if cohorts == set(cohort_names)
    }


def split_target_cohort(patients: list) -> tuple[list, list]:
    harmonization_half, heldout_half = train_test_split(
        patients,
        test_size=0.5,
        random_state=TARGET_HOLDOUT_SEED,
        stratify=[p.label for p in patients],
    )
    return harmonization_half, heldout_half


def get_probs(
    model,
    plist,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()
    ys, probs = [], []

    with torch.no_grad():
        for p in plist:
            vol = (
                torch.from_numpy(p.pet_masked.astype("float32"))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            probs.append(torch.sigmoid(model(vol)).item())
            ys.append(p.label)

    return np.array(probs), np.array(ys)


def eval_on(
    y: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict | None:
    if len(y) == 0:
        return None

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    auc = (
        roc_auc_score(y, y_prob)
        if len(np.unique(y)) > 1
        else float("nan")
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        y_pred,
        labels=[0, 1],
    ).ravel()

    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "acc": float(acc),
        "auc": float(auc),
        "recall_pos": float(recall_pos),
        "recall_neg": float(recall_neg),
        "balanced_acc": float(bacc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_model_once(
    torch_seed: int,
    all_cohorts: dict,
    target_cohort: str,
) -> tuple[CNNClassifier3D, dict, list]:
    target_harmonization, _ = split_target_cohort(
        all_cohorts[target_cohort]
    )

    known_cohorts = [
        c for c in COHORT_NAMES
        if c != target_cohort
    ]
    known_patients = [
        p
        for name in known_cohorts
        for p in all_cohorts[name]
    ]

    train_known, val_known = train_test_split(
        known_patients,
        test_size=0.2,
        random_state=VAL_SPLIT_SEED,
        stratify=[p.label for p in known_patients],
    )

    torch.manual_seed(torch_seed)

    model = CNNClassifier3D(latent_dim=LATENT_DIM)
    model = train_cnn_baseline(
        model,
        train_patients=train_known,
        val_patients=val_known,
        n_epochs=N_EPOCHS,
        checkpoint_path=None,
    )

    return model, all_cohorts, target_harmonization


def evaluate_target(
    torch_seed: int,
    model: CNNClassifier3D,
    cohort_all: dict,
    target_cohort: str,
    target_harmonization: list,
) -> dict:
    known_cohorts = [
        c for c in COHORT_NAMES
        if c != target_cohort
    ]
    known_patients = [
        p
        for name in known_cohorts
        for p in cohort_all[name]
    ]

    prob_known, y_known = get_probs(
        model,
        known_patients,
    )
    prob_target, y_target = get_probs(
        model,
        target_harmonization,
    )

    prob_target_corrected = zscore_shift_correct(
        prob_source=prob_known,
        prob_target=prob_target,
    )

    return {
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "known_cohorts": known_cohorts,
        "known_cohort_insample": eval_on(
            y_known,
            prob_known,
        ),
        "target_cohort_raw": eval_on(
            y_target,
            prob_target,
        ),
        "target_cohort_corrected": eval_on(
            y_target,
            prob_target_corrected,
        ),
    }


def summarize(
    results: list,
    key: str,
    label: str,
) -> dict:
    accs = [r[key]["acc"] for r in results if r[key] is not None]
    aucs = [r[key]["auc"] for r in results if r[key] is not None]
    recalls_pos = [
        r[key]["recall_pos"]
        for r in results
        if r[key] is not None
    ]
    recalls_neg = [
        r[key]["recall_neg"]
        for r in results
        if r[key] is not None
    ]
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

    all_cohorts = load_all_cohorts(data_path)

    for name in COHORT_NAMES:
        if name not in all_cohorts:
            raise ValueError(
                f"Cohort '{name}' not found. "
                f"Available cohorts: {list(all_cohorts.keys())}"
            )

    existing_runs = (
        []
        if args.fresh
        else load_existing_results(results_path)
    )

    done_seeds = (
        complete_seeds(existing_runs, COHORT_NAMES)
        & set(TORCH_SEEDS)
    )

    results = [
        r
        for r in existing_runs
        if r["torch_seed"] in done_seeds
    ]

    discarded_seeds = {
        r["torch_seed"]
        for r in existing_runs
    } - done_seeds

    if discarded_seeds:
        print(
            f"discarding incomplete/outdated seed(s) found in "
            f"{results_path}: {sorted(discarded_seeds)} "
            f"(redoing from scratch)"
        )

    results_path.write_text("")

    for r in results:
        append_result(r, results_path)

    seeds_to_run = [
        s
        for s in TORCH_SEEDS
        if s not in done_seeds
    ]

    for torch_seed in seeds_to_run:
        for target_cohort in COHORT_NAMES:
            print(
                f"\n{'=' * 60}\n"
                f"TORCH SEED {torch_seed}  target={target_cohort}\n"
                f"{'=' * 60}"
            )

            (
                model,
                cohort_all,
                target_harmonization,
            ) = train_model_once(
                torch_seed,
                all_cohorts,
                target_cohort,
            )

            r = evaluate_target(
                torch_seed,
                model,
                cohort_all,
                target_cohort,
                target_harmonization,
            )

            append_result(r, results_path)
            results.append(r)

            print(
                f"  known cohorts (in-sample) : "
                f"{r['known_cohort_insample']}"
            )
            print(
                f"  {target_cohort} "
                f"(harmonization half, raw) : "
                f"{r['target_cohort_raw']}"
            )
            print(
                f"  {target_cohort} "
                f"(harmonization half, corrected) : "
                f"{r['target_cohort_corrected']}"
            )

    summary_stats = []

    for target_cohort in COHORT_NAMES:
        subset = [
            r
            for r in results
            if r["target_cohort"] == target_cohort
        ]

        if not subset:
            continue

        summary_stats.append(
            summarize(
                subset,
                "target_cohort_raw",
                f"target={target_cohort} "
                f"(harmonization half, raw)",
            )
        )
        summary_stats.append(
            summarize(
                subset,
                "target_cohort_corrected",
                f"target={target_cohort} "
                f"(harmonization half, z-score corrected)",
            )
        )

    summary_stats.append(
        summarize(
            results,
            "target_cohort_raw",
            "ALL target cohorts pooled "
            "(harmonization half, raw)",
        )
    )
    summary_stats.append(
        summarize(
            results,
            "target_cohort_corrected",
            "ALL target cohorts pooled "
            "(harmonization half, z-score corrected)",
        )
    )

    for s in summary_stats:
        append_summary(s, results_path)

    print(f"\nsummary appended to: {results_path}")
