"""
Leave-one-cohort-out sweep using the alternating block-coordinate
fine-tune (harmonization_stratified.alternating_classifier_finetune)
instead of aux_clf trained jointly by SGD the whole time.

Per (seed, target_cohort): train stages 1-2 (train_harmonization_multi,
lambda_clf=0 -- pure recon+MMD, no classification signal yet), then hand
the harmonized model to alternating_classifier_finetune, which alternates
between fitting a full sklearn LogisticRegression on the current z and
training the encoders against that frozen fit (MMD still active every
round). After that returns, model.aux_clf already IS the classifier to
use -- evaluate_target reads it directly, no separate downstream refit.
"""

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
from cnn import zscore_shift_correct

DEFAULT_DATA_PATH    = "CUBES-Labelled-COHORTS"
DEFAULT_RESULTS_PATH = "alt_sweep_multi_results.jsonl"

COHORT_NAMES   = ["AUGSBURG", "PRE-RAPID", "SWISS"]
TORCH_SEEDS    = list(range(25))
VAL_SPLIT_SEED = 40
TARGET_HOLDOUT_SEED = 123

# -- stage 1-2 (harmonization) settings -------------------------------------
N_EPOCHS       = 1000
LAMBDA_MMD     = 100
LATENT_GAMMA_MODE = "adaptive"
DECODER_FREEZE_EPOCH = 50
PATIENCE = 50   # only counts post-freeze, see train_harmonization_multi's patience_active gate

# -- alternating fine-tune settings ------------------------------------------
ALT_N_ROUNDS        = 5
ALT_EPOCHS_PER_ROUND = 10
ALT_LAMBDA_MMD       = 100   # MMD weight during alternating rounds -- kept equal to stage-2's
                              # LAMBDA_MMD by default; change independently if you want to test
                              # a different alignment strength for the fine-tune phase specifically
ALT_LR               = 1e-4

# -- adaptive pair weighting (applied in BOTH stage 1-2 and every alternating
# round -- see train_harmonization_multi / alternating_classifier_finetune
# docstrings for the full mechanics). "static" + target_pair_weight=1.0 is a
# no-op, identical to having no weighting at all -- change PAIR_WEIGHTING to
# "adaptive" to actually turn this on.
PAIR_WEIGHTING       = "adaptive"
TARGET_PAIR_WEIGHT   = 1.0
WEIGHTING_EMA_BETA    = 0.9
WEIGHTING_TEMPERATURE = 1.0
WEIGHTING_FLOOR       = 1e-3
WEIGHTING_CEIL        = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-cohort-out sweep, alternating classifier fine-tune"
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument(
        "--fresh", action="store_true",
        help="Wipe --results-path and rerun every seed from scratch. Default: resume.",
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
    return {seed for seed, cohorts in by_seed.items() if cohorts == set(cohort_names)}


def split_target_cohort(patients: list) -> tuple[list, list]:
    harmonization_half, heldout_half = train_test_split(
        patients, test_size=0.5, random_state=TARGET_HOLDOUT_SEED,
        stratify=[p.label for p in patients],
    )
    return harmonization_half, heldout_half


def predict_probs(
    model: HarmonizationModel,
    patients: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Same as b_sweep_multi.py's predict_probs -- reads straight from
    model.aux_clf. After alternating_classifier_finetune, aux_clf holds a
    real, fully-converged sklearn fit (folded into linear-layer form), not
    a jointly-SGD-trained proxy -- so this is exactly the classifier that
    was used throughout the fine-tune, not a stand-in for it.
    """
    model = model.to(device)
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for patient in patients:
            vol = (
                torch.from_numpy(patient.pet_masked.astype("float32"))
                .unsqueeze(0).unsqueeze(0)
                .to(device)
            )
            z = model.encode(patient.cohort, vol)
            logit = model.classify(z)
            probs.append(torch.sigmoid(logit).item())
            ys.append(patient.label)
    return np.array(probs), np.array(ys)


def eval_on(y: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict | None:
    if len(y) == 0:
        return None
    y_pred = (y_prob >= threshold).astype(int)
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


def train_model_once(
    torch_seed: int, all_cohorts: dict, target_cohort: str
) -> tuple[HarmonizationModel, dict, list, list]:
    target_harmonization, target_heldout = split_target_cohort(all_cohorts[target_cohort])

    cohort_train, cohort_val, cohort_all = {}, {}, {}
    for name in COHORT_NAMES:
        cohort_all[name] = all_cohorts[name]
        is_target = name == target_cohort
        patients_for_training = target_harmonization if is_target else all_cohorts[name]

        # target's split is NOT label-stratified -- a real deployment-time
        # unlabeled target cohort has no labels to stratify by, so
        # stratifying here would be testing under friendlier conditions
        # than the model will actually see. Known (source) cohorts keep
        # stratification since their labels genuinely drive lambda_clf
        # (during the alternating fine-tune) and the classifier fit.
        train_p, val_p = train_test_split(
            patients_for_training, test_size=0.2, random_state=VAL_SPLIT_SEED,
            stratify=None if is_target else [p.label for p in patients_for_training],
        )
        cohort_train[name] = train_p
        cohort_val[name] = val_p

    torch.manual_seed(torch_seed)
    model = HarmonizationModel(cohort_names=COHORT_NAMES, latent_dim=64)

    # -- stage 1-2: pure recon+MMD, no classification signal yet ------------
    model = train_harmonization_multi(
        model,
        cohort_train=cohort_train, cohort_val=cohort_val,
        n_epochs=N_EPOCHS,
        lambda_mmd=LAMBDA_MMD,
        lambda_clf=0.0,
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

    # -- alternating fine-tune: fit-classifier / train-encoders, repeat -----
    model = alternating_classifier_finetune(
        model,
        cohort_train=cohort_train, cohort_val=cohort_val,
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


def evaluate_target(
    torch_seed: int,
    model: HarmonizationModel,
    cohort_all: dict,
    target_cohort: str,
    target_harmonization: list,
    target_heldout: list,
) -> dict:
    known_cohorts = [c for c in COHORT_NAMES if c != target_cohort]
    known_patients = [p for name in known_cohorts for p in cohort_all[name]]

    prob_known, y_known = predict_probs(model, known_patients)
    prob_harm, y_harm = predict_probs(model, target_harmonization)
    prob_target, y_target = predict_probs(model, target_heldout)

    prob_target_corrected = zscore_shift_correct(prob_source=prob_known, prob_target=prob_target)

    return {
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "known_cohorts": known_cohorts,
        "known_cohort_insample": eval_on(y_known, prob_known),
        "target_cohort_harmonization_half": eval_on(y_harm, prob_harm),
        "target_cohort_raw": eval_on(y_target, prob_target),
        "target_cohort_corrected": eval_on(y_target, prob_target_corrected),
    }


def summarize(results: list, key: str, label: str) -> dict:
    accs          = [r[key]["acc"] for r in results if r[key] is not None]
    aucs          = [r[key]["auc"] for r in results if r[key] is not None]
    recalls_pos   = [r[key]["recall_pos"] for r in results if r[key] is not None]
    recalls_neg   = [r[key]["recall_neg"] for r in results if r[key] is not None]
    balanced_accs = [r[key]["balanced_acc"] for r in results if r[key] is not None]

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
        "acc_mean": float(np.nanmean(accs)), "acc_std": float(np.nanstd(accs)),
        "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs)),
        "recall_pos_mean": float(np.nanmean(recalls_pos)), "recall_pos_std": float(np.nanstd(recalls_pos)),
        "recall_neg_mean": float(np.nanmean(recalls_neg)), "recall_neg_std": float(np.nanstd(recalls_neg)),
        "balanced_acc_mean": float(np.nanmean(balanced_accs)), "balanced_acc_std": float(np.nanstd(balanced_accs)),
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
    print(f"alt fine-tune: {ALT_N_ROUNDS} rounds x {ALT_EPOCHS_PER_ROUND} epochs  "
          f"lambda_mmd={ALT_LAMBDA_MMD}  lr={ALT_LR}")

    all_cohorts = load_all_cohorts(data_path)
    for name in COHORT_NAMES:
        if name not in all_cohorts:
            raise ValueError(f"Cohort '{name}' not found. Available cohorts: {list(all_cohorts.keys())}")

    existing_runs = [] if args.fresh else load_existing_results(results_path)
    done_seeds = complete_seeds(existing_runs, COHORT_NAMES) & set(TORCH_SEEDS)
    results = [r for r in existing_runs if r["torch_seed"] in done_seeds]

    discarded_seeds = {r["torch_seed"] for r in existing_runs} - done_seeds
    if discarded_seeds:
        print(f"discarding incomplete/outdated seed(s) found in {results_path}: "
              f"{sorted(discarded_seeds)} (redoing from scratch)")

    results_path.write_text("")
    for r in results:
        append_result(r, results_path)

    seeds_to_run = [s for s in TORCH_SEEDS if s not in done_seeds]
    if args.fresh:
        print(f"--fresh: running all {len(seeds_to_run)} seeds: {seeds_to_run}")
    elif done_seeds:
        print(f"resuming: {len(done_seeds)} seed(s) already complete {sorted(done_seeds)}, "
              f"running {len(seeds_to_run)} more: {seeds_to_run}")
    else:
        print(f"no usable prior results -- running all {len(seeds_to_run)} seeds: {seeds_to_run}")

    for torch_seed in seeds_to_run:
        for target_cohort in COHORT_NAMES:
            print(f"\n{'='*60}\nTORCH SEED {torch_seed}  target={target_cohort}  "
                  f"(50% of {target_cohort} in training, 50% fully held out)\n{'='*60}")
            model, cohort_all, target_harmonization, target_heldout = train_model_once(
                torch_seed, all_cohorts, target_cohort
            )

            print(f"\n--- evaluating target={target_cohort}, seed={torch_seed} "
                  f"(held out: {len(target_heldout)}, in-training: {len(target_harmonization)}) ---")
            r = evaluate_target(
                torch_seed, model, cohort_all, target_cohort, target_harmonization, target_heldout
            )
            append_result(r, results_path)
            results.append(r)
            print(f"  known cohorts (in-sample)      : {r['known_cohort_insample']}")
            print(f"  {target_cohort} (harmonization half) : {r['target_cohort_harmonization_half']}")
            print(f"  {target_cohort} (held out, raw)       : {r['target_cohort_raw']}")
            print(f"  {target_cohort} (held out, corrected) : {r['target_cohort_corrected']}")

    summary_stats = []
    for target_cohort in COHORT_NAMES:
        subset = [r for r in results if r["target_cohort"] == target_cohort]
        if not subset:
            continue
        summary_stats.append(summarize(subset, "target_cohort_raw", f"target={target_cohort} (held-out, raw)"))
        summary_stats.append(summarize(subset, "target_cohort_corrected", f"target={target_cohort} (held-out, z-score corrected)"))

    summary_stats.append(summarize(results, "target_cohort_raw", "ALL target cohorts pooled (held-out, raw)"))
    summary_stats.append(summarize(results, "target_cohort_corrected", "ALL target cohorts pooled (held-out, z-score corrected)"))

    for s in summary_stats:
        append_summary(s, results_path)
    print(f"\nsummary appended to: {results_path}")