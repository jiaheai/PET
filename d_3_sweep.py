"""Combined multi-cohort sweep: train/load once and score both target halves in the same run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from d_3 import (
    HarmonizationModel,
    train_harmonization_multi,
    alternating_classifier_finetune,
)
from nifti_loader import load_all_cohorts


DEFAULT_DATA_PATH = "CUBES-Labelled-COHORTS"
DEFAULT_TUNE_RESULTS_PATH = "tune_sweep_multi_results.jsonl"
DEFAULT_HELDOUT_RESULTS_PATH = "heldout_sweep_multi_results.jsonl"
DEFAULT_MODEL_DIR = "models_tuned"

COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "SWISS"]
TORCH_SEEDS = list(range(25))
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
        description="Combined sweep -- harmonization + held-out metrics in one run"
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-path", default="sweep_multi_combined_results.jsonl")
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
        if (
            r.get("target_cohort_harmonization_half") is None
            or r.get("target_cohort_heldout") is None
        ):
            continue
        by_seed.setdefault(r["torch_seed"], set()).add(r["target_cohort"])
    return {
        seed
        for seed, cohorts in by_seed.items()
        if cohorts == set(cohort_names)
    }


def checkpoint_path_for(
    directory: Path,
    torch_seed: int,
    target_cohort: str,
) -> Path:
    return directory / f"seed{torch_seed}_target{target_cohort}.pt"


def split_target_cohort(patients: list) -> tuple[list, list]:
    harmonization_half, heldout_half = train_test_split(
        patients,
        test_size=0.5,
        random_state=TARGET_HOLDOUT_SEED,
        stratify=[p.label for p in patients],
    )
    return harmonization_half, heldout_half


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
        "balanced_acc": float(np.nanmean([recall_pos, recall_neg])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
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
    print(
        f"recall(pos)    : "
        f"{np.nanmean(recalls_pos):.3f} +/- {np.nanstd(recalls_pos):.3f}"
    )
    print(
        f"recall(neg)    : "
        f"{np.nanmean(recalls_neg):.3f} +/- {np.nanstd(recalls_neg):.3f}"
    )
    print(
        f"balanced acc   : "
        f"{np.nanmean(balanced_accs):.3f} +/- {np.nanstd(balanced_accs):.3f}"
    )
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
        "per_run_balanced_acc": [
            round(b, 3)
            for b in balanced_accs
        ],
    }


def train_model_once(
    torch_seed: int,
    all_cohorts: dict,
    target_cohort: str,
) -> tuple[HarmonizationModel, dict, list]:
    target_harmonization, _ = split_target_cohort(
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

    return model, cohort_all, target_harmonization


def get_or_train_model(
    torch_seed: int,
    all_cohorts: dict,
    target_cohort: str,
    load_dir: Path | None,
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
            model.load_state_dict(
                torch.load(ckpt_path, map_location="cpu")
            )
            return (
                model.to(device),
                cohort_all,
                target_harmonization,
                target_heldout,
                True,
            )

        print(f"  no checkpoint at {ckpt_path} -- training")

    model, cohort_all, target_harmonization = train_model_once(
        torch_seed,
        all_cohorts,
        target_cohort,
    )
    _, target_heldout = split_target_cohort(
        all_cohorts[target_cohort]
    )

    return (
        model,
        cohort_all,
        target_harmonization,
        target_heldout,
        False,
    )


def load_heldout_model(
    torch_seed: int,
    all_cohorts: dict,
    target_cohort: str,
    model_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[HarmonizationModel, dict, list]:
    _, target_heldout = split_target_cohort(
        all_cohorts[target_cohort]
    )

    ckpt_path = checkpoint_path_for(
        model_dir,
        torch_seed,
        target_cohort,
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {ckpt_path} -- run this script in tune mode "
            f"with --output-dir {model_dir} first "
            f"(seed={torch_seed}, target_cohort={target_cohort!r})"
        )

    model = HarmonizationModel(
        cohort_names=COHORT_NAMES,
        latent_dim=64,
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu")
    )

    return (
        model.to(device),
        all_cohorts,
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
    known_cohorts = [
        c
        for c in COHORT_NAMES
        if c != target_cohort
    ]
    known_patients = [
        p
        for name in known_cohorts
        for p in cohort_all[name]
    ]

    prob_known, y_known = predict_probs(
        model,
        known_patients,
    )
    prob_harm, y_harm = predict_probs(
        model,
        target_harmonization,
    )
    prob_heldout, y_heldout = predict_probs(
        model,
        target_heldout,
    )

    return {
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "known_cohorts": known_cohorts,
        "known_cohort_insample": eval_on(
            y_known,
            prob_known,
        ),
        "target_cohort_harmonization_half": eval_on(
            y_harm,
            prob_harm,
        ),
        "target_cohort_heldout": eval_on(
            y_heldout,
            prob_heldout,
        ),
    }


def prepare_results(
    results_path: Path,
    fresh: bool,
) -> tuple[list[dict], list[int]]:
    existing_runs = (
        []
        if fresh
        else load_existing_results(results_path)
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
            f"discarding incomplete/outdated seed(s) found in "
            f"{results_path}: {sorted(discarded_seeds)} "
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
            f"--fresh: running all {len(seeds_to_run)} seeds: "
            f"{seeds_to_run}"
        )
    elif done_seeds:
        print(
            f"resuming: {len(done_seeds)} seed(s) already complete "
            f"{sorted(done_seeds)}, running {len(seeds_to_run)} more: "
            f"{seeds_to_run}"
        )
    else:
        print(
            f"no usable prior results -- running all "
            f"{len(seeds_to_run)} seeds: {seeds_to_run}"
        )

    return results, seeds_to_run


def run_combined(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    results_path = Path(args.results_path)
    load_dir = (
        Path(args.load_dir)
        if args.load_dir is not None
        else None
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else None
    )

    if output_dir is not None:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "[COMBINED RUN -- harmonization + held-out in one pass]"
    )
    print(f"data path    : {data_path}")
    print(f"results path : {results_path}")
    print(
        f"load dir     : "
        f"{load_dir if load_dir is not None else '(none)'}"
    )
    print(
        f"output dir   : "
        f"{output_dir if output_dir is not None else '(none)'}"
    )
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

    results, seeds_to_run = prepare_results(
        results_path,
        args.fresh,
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
                    model.state_dict(),
                    ckpt_path,
                )
                print(
                    f"  saved checkpoint -> {ckpt_path}"
                )

            r = evaluate_target(
                torch_seed,
                model,
                cohort_all,
                target_cohort,
                target_harmonization,
                target_heldout,
            )

            append_result(
                r,
                results_path,
            )
            results.append(r)

            print(
                f"  known cohorts (in-sample)      : "
                f"{r['known_cohort_insample']}"
            )
            print(
                f"  {target_cohort} (harmonization half) : "
                f"{r['target_cohort_harmonization_half']}"
            )
            print(
                f"  {target_cohort} (held-out half)      : "
                f"{r['target_cohort_heldout']}"
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
                "target_cohort_harmonization_half",
                f"target={target_cohort} (harmonization half)",
            )
        )
        summary_stats.append(
            summarize(
                subset,
                "target_cohort_heldout",
                f"target={target_cohort} (held-out)",
            )
        )

    summary_stats.append(
        summarize(
            results,
            "target_cohort_harmonization_half",
            "ALL target cohorts pooled (harmonization half)",
        )
    )
    summary_stats.append(
        summarize(
            results,
            "target_cohort_heldout",
            "ALL target cohorts pooled (held-out)",
        )
    )

    for s in summary_stats:
        append_summary(
            s,
            results_path,
        )

    print(f"\nsummary appended to: {results_path}")


if __name__ == "__main__":
    run_combined(parse_args())
