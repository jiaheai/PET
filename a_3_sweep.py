"""
A3 leave-one-cohort-out sweep across AUGSBURG, PRE-RAPID, and SWISS.

Matches the D3 training/evaluation setup while using A3's one shared
encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from experiment_utils import (
    experiment_metadata,
    normalize_split,
    prepare_results_file,
)
from a_3 import (
    HarmonizationModel,
    train_harmonization,
    alternating_classifier_finetune,
)
from nifti_loader import load_all_cohorts


DEFAULT_DATA_PATH = "CUBES-Labelled-COHORTS_3"
DEFAULT_RESULTS_PATH = "a_3_sweep.jsonl"

COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "SWISS"]
TORCH_SEEDS = list(range(10))
VAL_SPLIT_SEED = 40
TARGET_HOLDOUT_SEED = 123

LATENT_DIM = 64
BATCH_SIZE = 16

N_EPOCHS = 1000
LAMBDA_MMD = 100
LATENT_GAMMA_MODE = "adaptive"
DECODER_FREEZE_EPOCH = 50
PATIENCE = 50

ALT_N_ROUNDS = 5
ALT_EPOCHS_PER_ROUND = 10
ALT_LAMBDA_MMD = 100
ALT_LR = 1e-4

PAIR_WEIGHTING = "adaptive"
TARGET_PAIR_WEIGHT = 1.0
WEIGHTING_EMA_BETA = 0.9
WEIGHTING_TEMPERATURE = 1.0
WEIGHTING_FLOOR = 1e-3
WEIGHTING_CEIL = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A3 leave-one-cohort-out sweep with D3-matched training"
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--fresh", action="store_true")
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


def predict_probs(
    model: HarmonizationModel,
    patients: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()

    probs, ys = [], []

    with torch.no_grad():
        for patient in patients:
            vol = (
                torch.from_numpy(patient.pet_masked.astype("float32"))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            z = model.encode(vol)
            logit = model.classify(z)
            probs.append(torch.sigmoid(logit).item())
            ys.append(patient.label)

    return np.asarray(probs), np.asarray(ys)


def eval_set(
    model: HarmonizationModel,
    patients: list,
) -> dict | None:
    if not patients:
        return None

    y_prob, y = predict_probs(model, patients)
    y_pred = (y_prob >= 0.5).astype(int)

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
) -> tuple[HarmonizationModel, dict, list, list]:
    target_harmonization, target_heldout = split_target_cohort(
        all_cohorts[target_cohort]
    )

    cohort_train = {}
    cohort_val = {}
    cohort_all = {
        name: all_cohorts[name]
        for name in COHORT_NAMES
    }

    for name in COHORT_NAMES:
        is_target = name == target_cohort

        patients = (
            target_harmonization
            if is_target
            else all_cohorts[name]
        )

        train_p, val_p = train_test_split(
            patients,
            test_size=0.2,
            random_state=VAL_SPLIT_SEED,
            stratify=(
                None
                if is_target
                else [p.label for p in patients]
            ),
        )

        cohort_train[name] = train_p
        cohort_val[name] = val_p

    if zscore_correction:
        (
            cohort_all,
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

    torch.manual_seed(torch_seed)

    model = HarmonizationModel(
        latent_dim=LATENT_DIM,
    )

    model = train_harmonization(
        model,
        cohort_train=cohort_train,
        cohort_val=cohort_val,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lambda_mmd=LAMBDA_MMD,
        decoder_freeze_epoch=DECODER_FREEZE_EPOCH,
        latent_gamma_mode=LATENT_GAMMA_MODE,
        target_cohort=target_cohort,
        target_pair_weight=TARGET_PAIR_WEIGHT,
        pair_weighting=PAIR_WEIGHTING,
        weighting_ema_beta=WEIGHTING_EMA_BETA,
        weighting_temperature=WEIGHTING_TEMPERATURE,
        weighting_floor=WEIGHTING_FLOOR,
        weighting_ceil=WEIGHTING_CEIL,
        patience=PATIENCE,
        checkpoint_path=None,
    )

    model = alternating_classifier_finetune(
        model,
        cohort_train=cohort_train,
        cohort_val=cohort_val,
        target_cohort=target_cohort,
        n_rounds=ALT_N_ROUNDS,
        epochs_per_round=ALT_EPOCHS_PER_ROUND,
        lambda_mmd=ALT_LAMBDA_MMD,
        batch_size=BATCH_SIZE,
        lr=ALT_LR,
        target_pair_weight=TARGET_PAIR_WEIGHT,
        pair_weighting=PAIR_WEIGHTING,
        weighting_ema_beta=WEIGHTING_EMA_BETA,
        weighting_temperature=WEIGHTING_TEMPERATURE,
        weighting_floor=WEIGHTING_FLOOR,
        weighting_ceil=WEIGHTING_CEIL,
        checkpoint_path=None,
    )

    return (
        model,
        cohort_all,
        target_harmonization,
        target_heldout,
    )


def evaluate_target(
    torch_seed: int,
    model: HarmonizationModel,
    cohort_all: dict,
    target_cohort: str,
    target_harmonization: list,
    target_heldout: list,
) -> dict:
    source_cohorts = [
        name
        for name in COHORT_NAMES
        if name != target_cohort
    ]

    source_patients = [
        patient
        for name in source_cohorts
        for patient in cohort_all[name]
    ]

    return {
        "model": "A3",
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "source_cohorts": source_cohorts,
        "source_cohorts_insample": eval_set(
            model,
            source_patients,
        ),
        "target_harmonization": eval_set(
            model,
            target_harmonization,
        ),
        "target_heldout": eval_set(
            model,
            target_heldout,
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
        if r[key] is not None
    ]

    metrics = {
        "acc": [r["acc"] for r in rows],
        "auc": [r["auc"] for r in rows],
        "recall_pos": [r["recall_pos"] for r in rows],
        "recall_neg": [r["recall_neg"] for r in rows],
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
        "model": "A3",
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


if __name__ == "__main__":
    args = parse_args()

    data_path = Path(args.data_path)
    results_path = Path(args.results_path)

    print(f"data path    : {data_path}")
    print(f"results path : {results_path}")
    print(f"cohorts      : {COHORT_NAMES}")

    loaded = load_all_cohorts(data_path)

    for name in COHORT_NAMES:
        if name not in loaded:
            raise ValueError(
                f"Cohort '{name}' not found. "
                f"Available cohorts: "
                f"{list(loaded.keys())}"
            )

    all_cohorts = {
        name: loaded[name]
        for name in COHORT_NAMES
    }

    metadata = experiment_metadata(
        model="A3",
        data_path=data_path,
        cohort_names=COHORT_NAMES,
        zscore_correction=args.zscore_correction,
        parameters={
            "latent_dim": LATENT_DIM,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "lambda_mmd": LAMBDA_MMD,
            "latent_gamma_mode": LATENT_GAMMA_MODE,
            "decoder_freeze_epoch": DECODER_FREEZE_EPOCH,
            "patience": PATIENCE,
            "alt_n_rounds": ALT_N_ROUNDS,
            "alt_epochs_per_round": ALT_EPOCHS_PER_ROUND,
            "alt_lambda_mmd": ALT_LAMBDA_MMD,
            "alt_lr": ALT_LR,
            "pair_weighting": PAIR_WEIGHTING,
            "target_pair_weight": TARGET_PAIR_WEIGHT,
            "weighting_ema_beta": WEIGHTING_EMA_BETA,
            "weighting_temperature": WEIGHTING_TEMPERATURE,
            "weighting_floor": WEIGHTING_FLOOR,
            "weighting_ceil": WEIGHTING_CEIL,
            "val_split_seed": VAL_SPLIT_SEED,
            "target_holdout_seed": TARGET_HOLDOUT_SEED,
        },
        code_paths=[Path(__file__), Path(__file__).with_name("a_3.py")],
    )
    results, seeds_to_run = prepare_results_file(
        path=results_path,
        fresh=args.fresh,
        experiment_id=metadata["experiment_id"],
        cohort_names=COHORT_NAMES,
        torch_seeds=TORCH_SEEDS,
    )

    for torch_seed in seeds_to_run:
        for target_cohort in COHORT_NAMES:
            print(
                f"\n{'=' * 60}\n"
                f"TORCH SEED {torch_seed}  "
                f"target={target_cohort}\n"
                f"{'=' * 60}"
            )

            (
                model,
                cohort_all,
                target_harm,
                target_heldout,
            ) = train_model_once(
                torch_seed,
                all_cohorts,
                target_cohort,
                args.zscore_correction,
            )
            r = evaluate_target(
                torch_seed,
                model,
                cohort_all,
                target_cohort,
                target_harm,
                target_heldout,
            )
            r["experiment_id"] = metadata["experiment_id"]
            r["experiment"] = metadata

            append_result(r, results_path)
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

    summaries = []

    for target_cohort in COHORT_NAMES:
        subset = [
            r
            for r in results
            if r["target_cohort"] == target_cohort
        ]

        summaries.append(
            summarize(
                subset,
                "target_heldout",
                f"target={target_cohort} "
                f"(held-out)",
            )
        )

    summaries.append(
        summarize(
            results,
            "target_heldout",
            "ALL target cohorts pooled "
            "(held-out)",
        )
    )

    for summary in summaries:
        append_summary(
            summary,
            results_path,
        )

    print(
        f"\nsummary appended to: "
        f"{results_path}"
    )
