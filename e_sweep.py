from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from e import CNNClassifier3D, train_cnn_baseline
from nifti_loader import load_all_cohorts


DEFAULT_DATA_PATH = "CUBES-Labelled-COHORTS"
DEFAULT_RESULTS_PATH = "tune_sweep_cnn_d_matched_results.jsonl"

COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "SWISS"]
TORCH_SEEDS = list(range(25))
VAL_SPLIT_SEED = 40
TARGET_HOLDOUT_SEED = 123

LATENT_DIM = 16
ENCODER_DROPOUT = 0.2
CLASSIFIER_DROPOUT = 0.2
N_EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNN baseline matched to D where applicable"
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--load-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def append_result(record: dict, path: Path) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"record_type": "run", **record}) + "\n")


def append_summary(record: dict, path: Path) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"record_type": "summary", **record}) + "\n")


def load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []

    runs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.pop("record_type", None) == "run":
                runs.append(record)
    return runs


def complete_seeds(runs: list[dict]) -> set[int]:
    by_seed: dict[int, set[str]] = {}
    for record in runs:
        by_seed.setdefault(record["torch_seed"], set()).add(
            record["target_cohort"]
        )

    return {
        seed
        for seed, cohorts in by_seed.items()
        if cohorts == set(COHORT_NAMES)
    }


def checkpoint_path_for(
    directory: Path,
    seed: int,
    target: str,
) -> Path:
    return directory / f"seed{seed}_target{target}.pt"


def split_target_cohort(patients: list) -> tuple[list, list]:
    return train_test_split(
        patients,
        test_size=0.5,
        random_state=TARGET_HOLDOUT_SEED,
        stratify=[p.label for p in patients],
    )


def split_source_data(
    all_cohorts: dict,
    target_cohort: str,
) -> tuple[list, list, list]:
    target_harmonization, _ = split_target_cohort(
        all_cohorts[target_cohort]
    )

    source_train = []
    source_val = []

    for name in COHORT_NAMES:
        if name == target_cohort:
            continue

        train_p, val_p = train_test_split(
            all_cohorts[name],
            test_size=0.2,
            random_state=VAL_SPLIT_SEED,
            stratify=[p.label for p in all_cohorts[name]],
        )
        source_train.extend(train_p)
        source_val.extend(val_p)

    return source_train, source_val, target_harmonization


def predict_probs(
    model: CNNClassifier3D,
    patients: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()

    probs = []
    ys = []

    with torch.no_grad():
        for patient in patients:
            vol = (
                torch.from_numpy(patient.pet_masked.astype("float32"))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            probs.append(torch.sigmoid(model(vol)).item())
            ys.append(patient.label)

    return np.array(probs), np.array(ys)


def eval_on(
    y: np.ndarray,
    y_prob: np.ndarray,
) -> dict | None:
    if len(y) == 0:
        return None

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
        if tp + fn > 0
        else float("nan")
    )
    recall_neg = (
        tn / (tn + fp)
        if tn + fp > 0
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
    seed: int,
    all_cohorts: dict,
    target_cohort: str,
) -> tuple[CNNClassifier3D, list]:
    source_train, source_val, target_harmonization = split_source_data(
        all_cohorts,
        target_cohort,
    )

    torch.manual_seed(seed)

    model = CNNClassifier3D(
        latent_dim=LATENT_DIM,
        encoder_dropout=ENCODER_DROPOUT,
        classifier_dropout=CLASSIFIER_DROPOUT,
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

    return model, target_harmonization


def get_or_train_model(
    seed: int,
    all_cohorts: dict,
    target_cohort: str,
    load_dir: Path | None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[CNNClassifier3D, list, bool]:
    _, _, target_harmonization = split_source_data(
        all_cohorts,
        target_cohort,
    )

    if load_dir is not None:
        checkpoint = checkpoint_path_for(
            load_dir,
            seed,
            target_cohort,
        )

        if checkpoint.exists():
            print(f"  loading checkpoint from {checkpoint}")
            model = CNNClassifier3D(
                latent_dim=LATENT_DIM,
                encoder_dropout=ENCODER_DROPOUT,
                classifier_dropout=CLASSIFIER_DROPOUT,
            )
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu")
            )
            return model.to(device), target_harmonization, True

        print(f"  no checkpoint at {checkpoint} -- training")

    model, target_harmonization = train_model_once(
        seed,
        all_cohorts,
        target_cohort,
    )
    return model, target_harmonization, False


def evaluate_target(
    seed: int,
    model: CNNClassifier3D,
    all_cohorts: dict,
    target_cohort: str,
    target_harmonization: list,
) -> dict:
    source_cohorts = [
        name
        for name in COHORT_NAMES
        if name != target_cohort
    ]
    source_patients = [
        p
        for name in source_cohorts
        for p in all_cohorts[name]
    ]

    prob_source, y_source = predict_probs(
        model,
        source_patients,
    )
    prob_target, y_target = predict_probs(
        model,
        target_harmonization,
    )

    return {
        "torch_seed": seed,
        "target_cohort": target_cohort,
        "known_cohorts": source_cohorts,
        "known_cohort_insample": eval_on(
            y_source,
            prob_source,
        ),
        "target_cohort_harmonization_half": eval_on(
            y_target,
            prob_target,
        ),
    }


def summarize(
    results: list[dict],
    key: str,
    label: str,
) -> dict:
    metrics = {
        "acc": [r[key]["acc"] for r in results],
        "auc": [r[key]["auc"] for r in results],
        "recall_pos": [r[key]["recall_pos"] for r in results],
        "recall_neg": [r[key]["recall_neg"] for r in results],
        "balanced_acc": [
            r[key]["balanced_acc"]
            for r in results
        ],
    }

    print(f"\n=== {label} across {len(results)} runs ===")

    for name, values in metrics.items():
        print(
            f"{name:15s}: "
            f"{np.nanmean(values):.3f} +/- "
            f"{np.nanstd(values):.3f}"
        )

    return {
        "label": label,
        "key": key,
        "n_runs": len(results),
        **{
            f"{name}_mean": float(np.nanmean(values))
            for name, values in metrics.items()
        },
        **{
            f"{name}_std": float(np.nanstd(values))
            for name, values in metrics.items()
        },
        **{
            f"per_run_{name}": [
                round(v, 3)
                for v in values
            ]
            for name, values in metrics.items()
        },
    }


if __name__ == "__main__":
    args = parse_args()

    data_path = Path(args.data_path)
    results_path = Path(args.results_path)
    load_dir = (
        Path(args.load_dir)
        if args.load_dir
        else None
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else None
    )

    if output_dir is not None:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("[CNN baseline matched to D where applicable]")
    print(f"channels           : 16 -> 32 -> 64")
    print(f"latent             : {LATENT_DIM}")
    print(f"encoder dropout    : {ENCODER_DROPOUT}")
    print(f"classifier dropout : {CLASSIFIER_DROPOUT}")
    print(f"batch              : {BATCH_SIZE}")
    print(f"optimizer          : AdamW")
    print(f"lr                 : {LR}")
    print(f"weight decay       : {WEIGHT_DECAY}")

    all_cohorts = load_all_cohorts(data_path)

    for name in COHORT_NAMES:
        if name not in all_cohorts:
            raise ValueError(
                f"Cohort '{name}' not found. "
                f"Available: {list(all_cohorts.keys())}"
            )

    existing_runs = (
        []
        if args.fresh
        else load_existing_results(results_path)
    )

    done_seeds = (
        complete_seeds(existing_runs)
        & set(TORCH_SEEDS)
    )

    results = [
        r
        for r in existing_runs
        if r["torch_seed"] in done_seeds
    ]

    results_path.write_text("")
    for record in results:
        append_result(record, results_path)

    for seed in [
        s
        for s in TORCH_SEEDS
        if s not in done_seeds
    ]:
        for target_cohort in COHORT_NAMES:
            print(
                f"\n{'=' * 60}\n"
                f"TORCH SEED {seed}  "
                f"target={target_cohort}\n"
                f"{'=' * 60}"
            )

            (
                model,
                target_harmonization,
                was_loaded,
            ) = get_or_train_model(
                seed,
                all_cohorts,
                target_cohort,
                load_dir,
            )

            if (
                not was_loaded
                and output_dir is not None
            ):
                checkpoint = checkpoint_path_for(
                    output_dir,
                    seed,
                    target_cohort,
                )
                torch.save(
                    model.state_dict(),
                    checkpoint,
                )
                print(
                    f"  saved checkpoint -> {checkpoint}"
                )

            record = evaluate_target(
                seed,
                model,
                all_cohorts,
                target_cohort,
                target_harmonization,
            )

            append_result(
                record,
                results_path,
            )
            results.append(record)

            print(
                f"  {target_cohort} "
                f"(harmonization half): "
                f"{record['target_cohort_harmonization_half']}"
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
                "target_cohort_harmonization_half",
                f"target={target_cohort} "
                f"(harmonization half)",
            )
        )

    summaries.append(
        summarize(
            results,
            "target_cohort_harmonization_half",
            "ALL target cohorts pooled "
            "(harmonization half)",
        )
    )

    for summary in summaries:
        append_summary(
            summary,
            results_path,
        )

    print(
        f"\nsummary appended to: {results_path}"
    )
