"""CNN2 leave-one-cohort-out sweep across AUGSBURG and SWISS.

For each target cohort, one CNN is trained on labeled patients from the two
source cohorts. The target cohort is split 50/50 with TARGET_HOLDOUT_SEED=123
to preserve the same held-out membership used by A2/B2/C2. CNN2 model weights
never use target labels or images; when correction is enabled,
the harmonization half fits only the unsupervised normalization statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from experiment_utils import (
    experiment_metadata,
    normalize_split,
    prepare_results_file,
    seed_everything,
)
from cnn_2 import (
    CNNClassifier3D,
    eval_on,
    predict_probs,
    train_cnn_baseline,
)
from nifti_loader import load_all_cohorts


DEFAULT_DATA_PATH = "CUBES-Labelled-COHORTS_2"
DEFAULT_RESULTS_PATH = "cnn_2.jsonl"

COHORT_NAMES = ["AUGSBURG", "SWISS"]
TORCH_SEEDS = list(range(10))
VAL_SPLIT_SEED = 40
TARGET_HOLDOUT_SEED = 123

LATENT_DIM = 16
DROPOUT = 0.5
N_EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNN2 leave-one-cohort-out sweep"
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=f"Directory containing cohort data (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--results-path",
        default=DEFAULT_RESULTS_PATH,
        help=f"Where to write/resume results (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Wipe --results-path and rerun every seed from scratch.",
    )
    parser.add_argument(
        "--zscore-correction",
        action="store_true",
        help="Fit cohort normalization within each target fold.",
    )
    return parser.parse_args()


def append_result(
    result: dict,
    results_path: Path,
) -> None:
    with open(results_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "record_type": "run",
                    **result,
                }
            )
            + "\n"
        )


def append_summary(
    summary: dict,
    results_path: Path,
) -> None:
    with open(results_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "record_type": "summary",
                    **summary,
                }
            )
            + "\n"
        )


def split_target_cohort(
    patients: list,
) -> tuple[list, list]:
    return train_test_split(
        patients,
        test_size=0.5,
        random_state=TARGET_HOLDOUT_SEED,
        stratify=[
            p.label
            for p in patients
        ],
    )


def split_source_cohort(
    patients: list,
) -> tuple[list, list]:
    return train_test_split(
        patients,
        test_size=0.2,
        random_state=VAL_SPLIT_SEED,
        stratify=[
            p.label
            for p in patients
        ],
    )


def train_model_once(
    torch_seed: int,
    all_cohorts: dict,
    target_cohort: str,
    zscore_correction: bool = False,
) -> tuple[
    CNNClassifier3D,
    dict,
    list[str],
    list,
    list,
]:
    target_harmonization, target_heldout = (
        split_target_cohort(
            all_cohorts[target_cohort]
        )
    )

    source_cohorts = [
        name
        for name in COHORT_NAMES
        if name != target_cohort
    ]

    cohort_train = {}
    cohort_val = {}

    for name in source_cohorts:
        train_patients, val_patients = (
            split_source_cohort(
                all_cohorts[name]
            )
        )

        cohort_train[name] = train_patients
        cohort_val[name] = val_patients

    # The target harmonization half is allowed to fit unsupervised correction;
    # the held-out half is transformed only after all statistics are frozen.
    cohort_train[target_cohort] = []
    cohort_val[target_cohort] = []
    prepared_cohorts = all_cohorts
    if zscore_correction:
        (
            prepared_cohorts,
            cohort_train,
            cohort_val,
            target_harmonization,
            target_heldout,
        ) = normalize_split(
            all_cohorts=all_cohorts,
            cohort_train=cohort_train,
            cohort_val=cohort_val,
            target_cohort=target_cohort,
            target_harmonization=target_harmonization,
            target_heldout=target_heldout,
        )

    source_train = [
        patient for name in source_cohorts for patient in cohort_train[name]
    ]
    source_val = [
        patient for name in source_cohorts for patient in cohort_val[name]
    ]

    seed_everything(
        torch_seed
    )

    model = CNNClassifier3D(
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    )

    model = train_cnn_baseline(
        model,
        train_patients=source_train,
        val_patients=source_val,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        checkpoint_path=None,
    )

    return (
        model,
        prepared_cohorts,
        source_cohorts,
        target_harmonization,
        target_heldout,
    )


def evaluate_target(
    torch_seed: int,
    model: CNNClassifier3D,
    all_cohorts: dict,
    target_cohort: str,
    source_cohorts: list[str],
    target_harmonization: list,
    target_heldout: list,
) -> dict:
    source_patients = [
        patient
        for name in source_cohorts
        for patient in all_cohorts[name]
    ]

    prob_source, y_source = (
        predict_probs(
            model,
            source_patients,
        )
    )
    prob_harm, y_harm = (
        predict_probs(
            model,
            target_harmonization,
        )
    )
    prob_heldout, y_heldout = (
        predict_probs(
            model,
            target_heldout,
        )
    )

    return {
        "model": "CNN2",
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "source_cohorts": source_cohorts,
        "source_cohorts_insample": eval_on(
            y_source,
            prob_source,
        ),
        "target_harmonization": eval_on(
            y_harm,
            prob_harm,
        ),
        "target_heldout": eval_on(
            y_heldout,
            prob_heldout,
        ),
    }


def summarize(
    results: list,
    key: str,
    label: str,
) -> dict:
    rows = [
        result[key]
        for result in results
        if result.get(key) is not None
    ]

    metrics = {
        "acc": [
            row["acc"]
            for row in rows
        ],
        "auc": [
            row["auc"]
            for row in rows
        ],
        "recall_pos": [
            row["recall_pos"]
            for row in rows
        ],
        "recall_neg": [
            row["recall_neg"]
            for row in rows
        ],
        "balanced_acc": [
            row["balanced_acc"]
            for row in rows
        ],
    }

    print(
        f"\n=== {label} across "
        f"{len(rows)} runs ==="
    )

    for name, values in metrics.items():
        print(
            f"{name:15s}: "
            f"{np.nanmean(values):.3f} +/- "
            f"{np.nanstd(values):.3f}"
        )

    return {
        "model": "CNN2",
        "label": label,
        "key": key,
        "n_runs": len(rows),
        **{
            f"{name}_mean": float(
                np.nanmean(values)
            )
            for name, values in metrics.items()
        },
        **{
            f"{name}_std": float(
                np.nanstd(values)
            )
            for name, values in metrics.items()
        },
    }


def prepare_results(
    results_path: Path,
    fresh: bool,
    experiment: dict,
) -> tuple[
    list[dict],
    list[int],
]:
    return prepare_results_file(
        path=results_path,
        fresh=fresh,
        experiment_id=experiment["experiment_id"],
        cohort_names=COHORT_NAMES,
        torch_seeds=TORCH_SEEDS,
        experiment=experiment,
    )


def run_sweep(
    args: argparse.Namespace,
) -> None:
    data_path = Path(
        args.data_path
    )
    results_path = Path(
        args.results_path
    )

    print("[CNN2 SWEEP]")
    print(
        f"data path    : {data_path}"
    )
    print(
        f"results path : {results_path}"
    )
    print(
        f"cohorts      : {COHORT_NAMES}"
    )
    print(
        f"torch seeds  : {TORCH_SEEDS}"
    )
    print(
        "target harmonization half is used only to fit unsupervised correction"
        if args.zscore_correction
        else "target cohort is never used for CNN training"
    )

    all_cohorts = load_all_cohorts(
        data_path
    )

    for name in COHORT_NAMES:
        if name not in all_cohorts:
            raise ValueError(
                f"Cohort '{name}' not found. "
                f"Available cohorts: "
                f"{list(all_cohorts.keys())}"
            )

    all_cohorts = {name: all_cohorts[name] for name in COHORT_NAMES}

    metadata = experiment_metadata(
        model="CNN2",
        data_path=data_path,
        cohort_names=COHORT_NAMES,
        zscore_correction=args.zscore_correction,
        parameters={
            "latent_dim": LATENT_DIM,
            "dropout": DROPOUT,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "val_split_seed": VAL_SPLIT_SEED,
            "target_holdout_seed": TARGET_HOLDOUT_SEED,
        },
        code_paths=[Path(__file__), Path(__file__).with_name("cnn_2.py")],
    )
    results, seeds_to_run = (
        prepare_results(
            results_path,
            args.fresh,
            metadata,
        )
    )

    for torch_seed in seeds_to_run:
        for target_cohort in COHORT_NAMES:
            print(
                f"\n{'=' * 60}\n"
                f"TORCH SEED {torch_seed}  "
                f"target={target_cohort}  "
                f"(target held-out half entirely excluded)\n"
                f"{'=' * 60}"
            )

            (
                model,
                prepared_cohorts,
                source_cohorts,
                target_harmonization,
                target_heldout,
            ) = train_model_once(
                torch_seed,
                all_cohorts,
                target_cohort,
                args.zscore_correction,
            )

            result = evaluate_target(
                torch_seed,
                model,
                prepared_cohorts,
                target_cohort,
                source_cohorts,
                target_harmonization,
                target_heldout,
            )
            result["experiment_id"] = metadata["experiment_id"]
            result["experiment"] = metadata

            append_result(
                result,
                results_path,
            )
            results.append(
                result
            )

            print(
                f"  sources in-sample : "
                f"{result['source_cohorts_insample']}"
            )
            print(
                f"  target harm       : "
                f"{result['target_harmonization']}"
            )
            print(
                f"  target held-out   : "
                f"{result['target_heldout']}"
            )

    summary_stats = []

    for target_cohort in COHORT_NAMES:
        subset = [
            result
            for result in results
            if result["target_cohort"] == target_cohort
        ]

        if subset:
            summary_stats.append(
                summarize(
                    subset,
                    "target_heldout",
                    (
                        f"target={target_cohort} "
                        f"(held-out)"
                    ),
                )
            )

    if results:
        summary_stats.append(
            summarize(
                results,
                "target_heldout",
                (
                    "ALL target cohorts pooled "
                    "(held-out)"
                ),
            )
        )

    for summary in summary_stats:
        append_summary(
            summary,
            results_path,
        )

    print(
        f"\nsummary appended to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    run_sweep(
        parse_args()
    )
