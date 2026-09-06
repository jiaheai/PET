"""Approach C2 leave-one-cohort-out sweep across AUGSBURG and SWISS.

One independent autoencoder is trained per cohort. The source-cohort
autoencoder uses the full source cohort; the target-cohort autoencoder uses
only the target harmonization half. Every patient is encoded by both
encoders and the two latent vectors are concatenated before fitting one
source-only StandardScaler + LogisticRegression classifier.

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

from c_2 import Autoencoder3D, Encoder3D, encode_concat, train_autoencoder
from nifti_loader import load_all_cohorts


DEFAULT_DATA_PATH = "CUBES-Labelled-COHORTS_2"
DEFAULT_RESULTS_PATH = "c_2.jsonl"

COHORT_NAMES = ["AUGSBURG", "SWISS"]
TORCH_SEEDS = list(range(10))
VAL_SPLIT_SEED = 40
TARGET_HOLDOUT_SEED = 123

N_EPOCHS = 200
LATENT_DIM = 32
SEED_OFFSET = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approach C2 leave-one-cohort-out sweep"
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


def complete_seeds(
    runs: list[dict],
    cohort_names: list[str],
) -> set[int]:
    by_seed: dict[int, set[str]] = {}

    for r in runs:
        if r.get("target_heldout") is None:
            continue

        by_seed.setdefault(
            r["torch_seed"],
            set(),
        ).add(r["target_cohort"])

    return {
        seed
        for seed, completed_targets in by_seed.items()
        if completed_targets == set(cohort_names)
    }


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
) -> tuple[dict[str, Encoder3D], list[str], list, list]:
    target_harmonization, target_heldout = split_target_cohort(
        all_cohorts[target_cohort]
    )

    source_cohorts = [
        name
        for name in COHORT_NAMES
        if name != target_cohort
    ]

    source_cohort = source_cohorts[0]

    training_pools = {
        source_cohort: all_cohorts[source_cohort],
        target_cohort: target_harmonization,
    }

    encoders: dict[str, Encoder3D] = {}

    for i, name in enumerate(COHORT_NAMES):
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

        torch.manual_seed(
            torch_seed + i * SEED_OFFSET
        )

        model = Autoencoder3D(
            latent_dim=LATENT_DIM,
        )
        model = train_autoencoder(
            model,
            train_p,
            val_p,
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
        source_cohorts,
        target_harmonization,
        target_heldout,
    )


def evaluate_target(
    torch_seed: int,
    encoders: dict[str, Encoder3D],
    all_cohorts: dict,
    target_cohort: str,
    source_cohorts: list[str],
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
        Z_source,
        y_source,
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
        "model": "C2",
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
        "model": "C2",
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
) -> tuple[list[dict], list[int]]:
    existing_runs = (
        []
        if fresh
        else load_existing_results(
            results_path
        )
    )

    done_seeds = (
        complete_seeds(
            existing_runs,
            COHORT_NAMES,
        )
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
            f"discarding incomplete/outdated "
            f"seed(s) found in {results_path}: "
            f"{sorted(discarded_seeds)} "
            f"(redoing from scratch)"
        )

    results_path.write_text("")

    for r in results:
        append_result(
            r,
            results_path,
        )

    seeds_to_run = [
        seed
        for seed in TORCH_SEEDS
        if seed not in done_seeds
    ]

    if fresh:
        print(
            f"--fresh: running all "
            f"{len(seeds_to_run)} seeds: "
            f"{seeds_to_run}"
        )
    elif done_seeds:
        print(
            f"resuming: {len(done_seeds)} "
            f"seed(s) already complete "
            f"{sorted(done_seeds)}, running "
            f"{len(seeds_to_run)} more: "
            f"{seeds_to_run}"
        )
    else:
        print(
            f"no usable prior results -- "
            f"running all "
            f"{len(seeds_to_run)} seeds: "
            f"{seeds_to_run}"
        )

    return (
        results,
        seeds_to_run,
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

    print("[C2 SWEEP]")
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

    results, seeds_to_run = prepare_results(
        results_path,
        args.fresh,
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
                source_cohorts,
                target_harmonization,
                target_heldout,
            ) = train_model_once(
                torch_seed,
                all_cohorts,
                target_cohort,
            )

            r = evaluate_target(
                torch_seed,
                encoders,
                all_cohorts,
                target_cohort,
                source_cohorts,
                target_harmonization,
                target_heldout,
            )

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
