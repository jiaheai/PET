"""Combined B2 sweep: train/load once, evaluate both target halves, summarize held-out only."""

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
from b_2 import (
    HarmonizationModel,
    train_harmonization_multi,
    alternating_classifier_finetune,
)
from nifti_loader import load_all_cohorts


DEFAULT_DATA_PATH = "CUBES-Labelled-COHORTS_2"
DEFAULT_RESULTS_PATH = "b_2.jsonl"

COHORT_NAMES = ['AUGSBURG', 'SWISS']
TORCH_SEEDS = list(range(10))
VAL_SPLIT_SEED = 40
TARGET_HOLDOUT_SEED = 123

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
WEIGHTING_TEMPERATURE = 1
WEIGHTING_FLOOR = 1e-3
WEIGHTING_CEIL = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B2 sweep -- harmonization + held-out metrics in one run"
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument(
        "--load-dir",
        default=None,
        help="Optional directory to load existing checkpoints from.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to save newly trained checkpoints to.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Reset results and rerun every seed. Checkpoint loading is controlled by --load-dir.",
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


def checkpoint_path_for(
    directory: Path,
    torch_seed: int,
    target_cohort: str,
) -> Path:
    return directory / f"seed{torch_seed}_target{target_cohort}.pt"


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
            z = model.encode(patient.cohort, vol)
            logit = model.classify(z)
            probs.append(torch.sigmoid(logit).item())
            ys.append(patient.label)

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
        "balanced_acc": float(np.nanmean([recall_pos, recall_neg])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize(results: list, key: str, label: str) -> dict:
    rows = [r[key] for r in results if r.get(key) is not None]

    metrics = {
        "acc": [r["acc"] for r in rows],
        "auc": [r["auc"] for r in rows],
        "recall_pos": [r["recall_pos"] for r in rows],
        "recall_neg": [r["recall_neg"] for r in rows],
        "balanced_acc": [r["balanced_acc"] for r in rows],
    }

    print(f"\n=== {label} across {len(rows)} runs ===")
    for name, values in metrics.items():
        print(
            f"{name:15s}: "
            f"{np.nanmean(values):.3f} +/- {np.nanstd(values):.3f}"
        )

    return {
        "model": "B2",
        "label": label,
        "key": key,
        "n_runs": len(rows),
        **{
            f"{name}_mean": float(np.nanmean(values))
            for name, values in metrics.items()
        },
        **{
            f"{name}_std": float(np.nanstd(values))
            for name, values in metrics.items()
        },
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

    cohort_train, cohort_val, cohort_all = {}, {}, {}

    for name in COHORT_NAMES:
        cohort_all[name] = all_cohorts[name]
        is_target = name == target_cohort
        patients_for_training = (
            target_harmonization
            if is_target
            else all_cohorts[name]
        )

        train_p, val_p = train_test_split(
            patients_for_training,
            test_size=0.2,
            random_state=VAL_SPLIT_SEED,
            stratify=(
                None
                if is_target
                else [p.label for p in patients_for_training]
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
        cohort_names=COHORT_NAMES,
        latent_dim=64,
    )

    model = train_harmonization_multi(
        model,
        cohort_train=cohort_train,
        cohort_val=cohort_val,
        n_epochs=N_EPOCHS,
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
        checkpoint_path=None,
        patience=PATIENCE,
    )

    model = alternating_classifier_finetune(
        model,
        cohort_train=cohort_train,
        cohort_val=cohort_val,
        target_cohort=target_cohort,
        n_rounds=ALT_N_ROUNDS,
        epochs_per_round=ALT_EPOCHS_PER_ROUND,
        lambda_mmd=ALT_LAMBDA_MMD,
        lr=ALT_LR,
        target_pair_weight=TARGET_PAIR_WEIGHT,
        pair_weighting=PAIR_WEIGHTING,
        weighting_ema_beta=WEIGHTING_EMA_BETA,
        weighting_temperature=WEIGHTING_TEMPERATURE,
        weighting_floor=WEIGHTING_FLOOR,
        weighting_ceil=WEIGHTING_CEIL,
        checkpoint_path=None,
    )

    return model, cohort_all, target_harmonization, target_heldout


def get_or_train_model(
    torch_seed: int,
    all_cohorts: dict,
    target_cohort: str,
    load_dir: Path | None,
    experiment: dict,
    zscore_correction: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[HarmonizationModel, dict, list, list, bool]:
    target_harmonization, target_heldout = split_target_cohort(
        all_cohorts[target_cohort]
    )
    cohort_all = {
        name: all_cohorts[name]
        for name in COHORT_NAMES
    }

    if load_dir is not None:
        ckpt_path = checkpoint_path_for(
            load_dir,
            torch_seed,
            target_cohort,
        )

        if ckpt_path.exists():
            print(f"  loading checkpoint from {ckpt_path}")
            model = HarmonizationModel(
                cohort_names=COHORT_NAMES,
                latent_dim=64,
            )
            payload = torch.load(ckpt_path, map_location="cpu")
            if (
                isinstance(payload, dict)
                and payload.get("experiment_id") == experiment["experiment_id"]
                and "state_dict" in payload
            ):
                if zscore_correction:
                    source_train, source_val = {}, {}
                    for name in COHORT_NAMES:
                        patients = (
                            target_harmonization
                            if name == target_cohort
                            else all_cohorts[name]
                        )
                        train_p, val_p = train_test_split(
                            patients,
                            test_size=0.2,
                            random_state=VAL_SPLIT_SEED,
                            stratify=(
                                None
                                if name == target_cohort
                                else [p.label for p in patients]
                            ),
                        )
                        source_train[name] = train_p
                        source_val[name] = val_p
                    (
                        cohort_all,
                        _,
                        _,
                        target_harmonization,
                        target_heldout,
                    ) = normalize_split(
                        all_cohorts=all_cohorts,
                        cohort_train=source_train,
                        cohort_val=source_val,
                        target_cohort=target_cohort,
                        target_harmonization=target_harmonization,
                        target_heldout=target_heldout,
                    )
                model.load_state_dict(payload["state_dict"])
                return (
                    model.to(device),
                    cohort_all,
                    target_harmonization,
                    target_heldout,
                    True,
                )
            print("  checkpoint provenance mismatch -- training")

        print(f"  no checkpoint at {ckpt_path} -- training")

    model, cohort_all, target_harmonization, target_heldout = train_model_once(
        torch_seed,
        all_cohorts,
        target_cohort,
        zscore_correction,
    )

    return (
        model,
        cohort_all,
        target_harmonization,
        target_heldout,
        False,
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
        c
        for c in COHORT_NAMES
        if c != target_cohort
    ]
    source_patients = [
        p
        for name in source_cohorts
        for p in cohort_all[name]
    ]

    prob_source, y_source = predict_probs(model, source_patients)
    prob_harm, y_harm = predict_probs(model, target_harmonization)
    prob_heldout, y_heldout = predict_probs(model, target_heldout)

    return {
        "model": "B2",
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "source_cohorts": source_cohorts,
        "source_cohorts_insample": eval_on(y_source, prob_source),
        "target_harmonization": eval_on(y_harm, prob_harm),
        "target_heldout": eval_on(y_heldout, prob_heldout),
    }


def prepare_results(
    results_path: Path,
    fresh: bool,
    experiment_id: str,
) -> tuple[list[dict], list[int]]:
    return prepare_results_file(
        path=results_path,
        fresh=fresh,
        experiment_id=experiment_id,
        cohort_names=COHORT_NAMES,
        torch_seeds=TORCH_SEEDS,
    )


def run_combined(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    results_path = Path(args.results_path)
    load_dir = Path(args.load_dir) if args.load_dir is not None else None
    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("[B2 SWEEP -- harmonization + held-out in one pass]")
    print(f"data path    : {data_path}")
    print(f"results path : {results_path}")
    print(f"load dir     : {load_dir if load_dir is not None else '(none)'}")
    print(f"output dir   : {output_dir if output_dir is not None else '(none)'}")
    print(f"cohorts      : {COHORT_NAMES}")
    print(
        f"alt fine-tune: {ALT_N_ROUNDS} rounds x "
        f"{ALT_EPOCHS_PER_ROUND} epochs  "
        f"lambda_mmd={ALT_LAMBDA_MMD}  lr={ALT_LR}"
    )

    all_cohorts = load_all_cohorts(data_path)

    for name in COHORT_NAMES:
        if name not in all_cohorts:
            raise ValueError(
                f"Cohort '{name}' not found. "
                f"Available cohorts: {list(all_cohorts.keys())}"
            )

    all_cohorts = {name: all_cohorts[name] for name in COHORT_NAMES}

    metadata = experiment_metadata(
        model="B2",
        data_path=data_path,
        cohort_names=COHORT_NAMES,
        zscore_correction=args.zscore_correction,
        parameters={
            "latent_dim": 64, "batch_size": 16,
            "n_epochs": N_EPOCHS, "lambda_mmd": LAMBDA_MMD,
            "latent_gamma_mode": LATENT_GAMMA_MODE,
            "decoder_freeze_epoch": DECODER_FREEZE_EPOCH, "patience": PATIENCE,
            "alt_n_rounds": ALT_N_ROUNDS,
            "alt_epochs_per_round": ALT_EPOCHS_PER_ROUND,
            "alt_lambda_mmd": ALT_LAMBDA_MMD, "alt_lr": ALT_LR,
            "pair_weighting": PAIR_WEIGHTING,
            "target_pair_weight": TARGET_PAIR_WEIGHT,
            "weighting_ema_beta": WEIGHTING_EMA_BETA,
            "weighting_temperature": WEIGHTING_TEMPERATURE,
            "weighting_floor": WEIGHTING_FLOOR,
            "weighting_ceil": WEIGHTING_CEIL,
            "val_split_seed": VAL_SPLIT_SEED,
            "target_holdout_seed": TARGET_HOLDOUT_SEED,
        },
        code_paths=[Path(__file__), Path(__file__).with_name("b_2.py")],
    )
    results, seeds_to_run = prepare_results(
        results_path,
        args.fresh,
        metadata["experiment_id"],
    )

    for torch_seed in seeds_to_run:
        for target_cohort in COHORT_NAMES:
            print(
                f"\n{'=' * 60}\n"
                f"TORCH SEED {torch_seed}  target={target_cohort}  "
                f"(50% harmonization, 50% held out)\n"
                f"{'=' * 60}"
            )

            (
                model,
                cohort_all,
                target_harmonization,
                target_heldout,
                was_loaded,
            ) = get_or_train_model(
                torch_seed,
                all_cohorts,
                target_cohort,
                load_dir,
                metadata,
                args.zscore_correction,
            )

            if was_loaded:
                print("  using loaded checkpoint")
            elif output_dir is not None:
                ckpt_path = checkpoint_path_for(
                    output_dir,
                    torch_seed,
                    target_cohort,
                )
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "experiment_id": metadata["experiment_id"],
                        "experiment": metadata,
                    },
                    ckpt_path,
                )
                print(f"  saved checkpoint -> {ckpt_path}")

            r = evaluate_target(
                torch_seed,
                model,
                cohort_all,
                target_cohort,
                target_harmonization,
                target_heldout,
            )
            r["experiment_id"] = metadata["experiment_id"]
            r["experiment"] = metadata

            append_result(r, results_path)
            results.append(r)

            print(f"  sources in-sample : {r['source_cohorts_insample']}")
            print(f"  target harm       : {r['target_harmonization']}")
            print(f"  target held-out   : {r['target_heldout']}")

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
                    f"target={target_cohort} (held-out)",
                )
            )

    if results:
        summary_stats.append(
            summarize(
                results,
                "target_heldout",
                "ALL target cohorts pooled (held-out)",
            )
        )

    for s in summary_stats:
        append_summary(s, results_path)

    print(f"\nsummary appended to: {results_path}")


if __name__ == "__main__":
    run_combined(parse_args())
