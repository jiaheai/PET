"""Approach C3 leave-one-cohort-out sweep across AUGSBURG, PRE-RAPID, and SWISS.

One independent autoencoder is trained per cohort. Source-cohort
autoencoders use their source-training splits; the target-cohort autoencoder uses
only the target harmonization half. Every patient is encoded by all three
encoders and the three latent vectors are concatenated before fitting one
source-training-only StandardScaler + LogisticRegression classifier.

The target cohort is split 50/50 with TARGET_HOLDOUT_SEED=123. The target
held-out half never touches autoencoder training or classifier fitting.
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

from experiment_utils import (
    experiment_metadata,
    normalize_split,
    prepare_results_file,
    seed_everything,
)
from c_3 import Autoencoder3D, Encoder3D, encode_concat, train_autoencoder
from nifti_loader import load_all_cohorts


DEFAULT_DATA_PATH = "CUBES-Labelled-COHORTS_3"
DEFAULT_RESULTS_PATH = "c_3.jsonl"

COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "SWISS"]
TORCH_SEEDS = list(range(10))
VAL_SPLIT_SEED = 40
TARGET_HOLDOUT_SEED = 123

N_EPOCHS = 200
LATENT_DIM = 32
SEED_OFFSET = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approach C3 leave-one-cohort-out sweep"
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


def append_result(r: dict, results_path: Path) -> None:
    with open(results_path, "a") as f:
        f.write(json.dumps({"record_type": "run", **r}) + "\n")


def append_summary(s: dict, results_path: Path) -> None:
    with open(results_path, "a") as f:
        f.write(json.dumps({"record_type": "summary", **s}) + "\n")


def split_target_cohort(patients: list) -> tuple[list, list]:
    return train_test_split(
        patients,
        test_size=0.5,
        random_state=TARGET_HOLDOUT_SEED,
        stratify=[p.label for p in patients],
    )


def eval_on(
    y: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict | None:
    if len(y) == 0:
        return None

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
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

    recall_pos = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else float("nan")
    )
    recall_neg = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    return {
        "acc": float(acc),
        "auc": float(auc),
        "recall_pos": float(recall_pos),
        "recall_neg": float(recall_neg),
        "balanced_acc": float(
            np.nanmean([recall_pos, recall_neg])
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_model_once(
    torch_seed: int,
    all_cohorts: dict,
    target_cohort: str,
    zscore_correction: bool = False,
) -> tuple[dict[str, Encoder3D], dict, list[str], list, list, list]:
    target_harmonization, target_heldout = split_target_cohort(
        all_cohorts[target_cohort]
    )

    source_cohorts = [
        name
        for name in COHORT_NAMES
        if name != target_cohort
    ]

    training_pools = {
        name: all_cohorts[name]
        for name in source_cohorts
    }
    training_pools[target_cohort] = target_harmonization

    cohort_train, cohort_val = {}, {}
    for name in COHORT_NAMES:
        pool = training_pools[name]
        is_target = name == target_cohort
        train_p, val_p = train_test_split(
            pool,
            test_size=0.2,
            random_state=VAL_SPLIT_SEED,
            stratify=(
                None
                if is_target
                else [p.label for p in pool]
            ),
        )
        cohort_train[name], cohort_val[name] = train_p, val_p

    prepared_cohorts = all_cohorts
    if zscore_correction:
        prepared_cohorts, cohort_train, cohort_val, target_harmonization, target_heldout = normalize_split(
            all_cohorts=all_cohorts, cohort_train=cohort_train,
            cohort_val=cohort_val, target_cohort=target_cohort,
            target_harmonization=target_harmonization,
            target_heldout=target_heldout,
        )

    encoders: dict[str, Encoder3D] = {}
    for i, name in enumerate(COHORT_NAMES):

        seed_everything(
            torch_seed + i * SEED_OFFSET
        )

        model = Autoencoder3D(
            latent_dim=LATENT_DIM,
        )
        model = train_autoencoder(
            model,
            cohort_train[name],
            cohort_val[name],
            n_epochs=N_EPOCHS,
            checkpoint_path=None,
            label=(
                f"{name} "
                f"(target={target_cohort}) "
                f"seed={torch_seed}"
            ),
        )

        encoders[name] = model.encoder

    return (
        encoders,
        prepared_cohorts,
        source_cohorts,
        [patient for name in source_cohorts for patient in cohort_train[name]],
        target_harmonization,
        target_heldout,
    )


def evaluate_target(
    torch_seed: int,
    encoders: dict[str, Encoder3D],
    all_cohorts: dict,
    target_cohort: str,
    source_cohorts: list[str],
    classifier_train: list,
    target_harmonization: list,
    target_heldout: list,
) -> dict:
    source_patients = [
        p
        for name in source_cohorts
        for p in all_cohorts[name]
    ]

    Z_source, y_source = encode_concat(
        encoders,
        COHORT_NAMES,
        source_patients,
    )
    Z_classifier_train, y_classifier_train = encode_concat(
        encoders, COHORT_NAMES, classifier_train,
    )
    Z_target_harm, y_target_harm = encode_concat(
        encoders,
        COHORT_NAMES,
        target_harmonization,
    )
    Z_target_heldout, y_target_heldout = encode_concat(
        encoders,
        COHORT_NAMES,
        target_heldout,
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    )
    clf.fit(
        Z_classifier_train,
        y_classifier_train,
    )

    prob_source = clf.predict_proba(
        Z_source
    )[:, 1]
    prob_target_harm = clf.predict_proba(
        Z_target_harm
    )[:, 1]
    prob_target_heldout = clf.predict_proba(
        Z_target_heldout
    )[:, 1]

    return {
        "model": "C3",
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "source_cohorts": source_cohorts,
        "source_cohorts_insample": eval_on(
            y_source,
            prob_source,
        ),
        "target_harmonization": eval_on(
            y_target_harm,
            prob_target_harm,
        ),
        "target_heldout": eval_on(
            y_target_heldout,
            prob_target_heldout,
        ),
    }


def summarize(
    results: list,
    key: str,
    label: str,
) -> dict:
    rows = [
        r[key]
        for r in results
        if r.get(key) is not None
    ]

    metrics = {
        "acc": [r["acc"] for r in rows],
        "auc": [r["auc"] for r in rows],
        "recall_pos": [
            r["recall_pos"]
            for r in rows
        ],
        "recall_neg": [
            r["recall_neg"]
            for r in rows
        ],
        "balanced_acc": [
            r["balanced_acc"]
            for r in rows
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
        "model": "C3",
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
) -> tuple[list[dict], list[int]]:
    return prepare_results_file(
        path=results_path, fresh=fresh,
        experiment_id=experiment["experiment_id"],
        cohort_names=COHORT_NAMES, torch_seeds=TORCH_SEEDS,
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

    print("[C3 SWEEP]")
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
        model="C3", data_path=data_path, cohort_names=COHORT_NAMES,
        zscore_correction=args.zscore_correction,
        parameters={
            "n_epochs": N_EPOCHS, "latent_dim": LATENT_DIM,
            "seed_offset": SEED_OFFSET, "val_split_seed": VAL_SPLIT_SEED,
            "target_holdout_seed": TARGET_HOLDOUT_SEED,
        },
        code_paths=[Path(__file__), Path(__file__).with_name("c_3.py")],
    )
    results, seeds_to_run = prepare_results(
        results_path,
        args.fresh,
        metadata,
    )

    for torch_seed in seeds_to_run:
        for target_cohort in COHORT_NAMES:
            print(
                f"\n{'=' * 60}\n"
                f"TORCH SEED {torch_seed}  "
                f"target={target_cohort}  "
                f"(50% harmonization, "
                f"50% held out)\n"
                f"{'=' * 60}"
            )

            (
                encoders,
                prepared_cohorts,
                source_cohorts,
                classifier_train,
                target_harmonization,
                target_heldout,
            ) = train_model_once(
                torch_seed,
                all_cohorts,
                target_cohort,
                args.zscore_correction,
            )

            r = evaluate_target(
                torch_seed,
                encoders,
                prepared_cohorts,
                target_cohort,
                source_cohorts,
                classifier_train,
                target_harmonization,
                target_heldout,
            )
            r["experiment_id"] = metadata["experiment_id"]
            r["experiment"] = metadata

            append_result(
                r,
                results_path,
            )
            results.append(r)

            print(
                f"  sources in-sample : "
                f"{r['source_cohorts_insample']}"
            )
            print(
                f"  target harm       : "
                f"{r['target_harmonization']}"
            )
            print(
                f"  target held-out   : "
                f"{r['target_heldout']}"
            )

    summary_stats = []

    for target_cohort in COHORT_NAMES:
        subset = [
            r
            for r in results
            if r["target_cohort"] == target_cohort
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

    for s in summary_stats:
        append_summary(
            s,
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
