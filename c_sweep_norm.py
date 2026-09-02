"""
Approach C, leave-one-cohort-out sweep across 3 cohorts, with z-score
correction (matches a_sweep_multi.py's pattern, so A and C are on equal
footing for the raw-vs-corrected, cross-approach comparison).

Direct 3-cohort generalization of c.py's "build 2 autoencoders, concat
their latents" philosophy: train ONE independent autoencoder per cohort
(3 total, each seeing only its own cohort's volumes -- the target
cohort's autoencoder trains only on its harmonization-half, never its
held-out half). Every patient is then encoded by pushing it through all
3 encoders and concatenating the results (3 * latent_dim = 192-dim
features) -- same mechanism as the original encode_concat, generalized
from 2 encoders to N.

The classifier is fit on the concatenated latents of the two KNOWN
(non-target) cohorts' full patient sets (their true labels), and
evaluated on the target cohort's held-out half. 50% of the target
cohort is stratified-split off and never touches any training step (not
any autoencoder, not the classifier) -- same TARGET_HOLDOUT_SEED as
a_sweep_multi.py / b_sweep_multi.py / cnn_sweep_multi.py, so all
approaches hold out identical patients for a valid cross-approach
comparison.
"""

from __future__ import annotations

import copy
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

from c import Autoencoder3D, Encoder3D, train_autoencoder
from nifti_loader import load_all_cohorts
from cnn import zscore_shift_correct   # reuse the same correction used for the CNN baseline

# Defaults -- overridable via --data-path / --results-path for automation.
DEFAULT_DATA_PATH    = "CUBES-Labelled-COHORTS"
DEFAULT_RESULTS_PATH = "c_sweep_multi_results.jsonl"

COHORT_NAMES   = ["AUGSBURG", "PRE-RAPID", "SWISS"]
TORCH_SEEDS    = list(range(5, 10))     # matches a_sweep_multi.py's convention (3x more expensive than the 2-cohort sweeps)
VAL_SPLIT_SEED = 40                 # fixed -- keeps the same val patients across all torch seeds
TARGET_HOLDOUT_SEED = 123           # fixed -- SAME value as a_sweep_multi.py / b_sweep_multi.py / cnn_sweep_multi.py
N_EPOCHS       = 200                # matches c.py / c_sweep_norm.py's per-autoencoder epoch budget (no MMD term here)
SEED_OFFSET    = 1000               # per-encoder seed offset so the 3 encoders don't get identical init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approach C, leave-one-cohort-out sweep across all cohort combinations and torch seeds"
    )
    parser.add_argument(
        "--data-path", default=DEFAULT_DATA_PATH,
        help=f"Directory containing all cohorts' data (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--results-path", default=DEFAULT_RESULTS_PATH,
        help=f"Where to write/resume sweep results (default: {DEFAULT_RESULTS_PATH}).",
    )
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
    # a seed only counts as complete if every cohort has a run record for it
    by_seed: dict[int, set[str]] = {}
    for r in runs:
        by_seed.setdefault(r["torch_seed"], set()).add(r["target_cohort"])
    return {seed for seed, cohorts in by_seed.items() if cohorts == set(cohort_names)}


def split_target_cohort(patients: list) -> tuple[list, list]:
    """Stratified 50/50 split of the target cohort's patients into a
    harmonization half (its own autoencoder trains on this half only)
    and a held-out half (never touches training in any way -- not any
    autoencoder, not the classifier). Fixed random_state, same value as
    the other approaches' sweeps, so all approaches hold out identical
    patients.
    """
    harmonization_half, heldout_half = train_test_split(
        patients, test_size=0.5, random_state=TARGET_HOLDOUT_SEED,
        stratify=[p.label for p in patients],
    )
    return harmonization_half, heldout_half


def encode_concat_multi(
    encoders: dict[str, Encoder3D],
    cohort_order: list[str],
    patients: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Push every patient through ALL encoders (regardless of the
    patient's own cohort) in a fixed cohort_order, concatenate the
    resulting latent vectors. Generalizes c.py's encode_concat (which
    is hardcoded to exactly 2 encoders) to any number of cohorts.

    cohort_order is fixed (== COHORT_NAMES) so the concat vector's layout
    is identical across every (seed, target_cohort) run -- position i in
    the feature vector is always that cohort's encoder output, regardless
    of which cohort happens to be the target that run.

    Returns (Z, y): Z is (N, len(cohort_order) * latent_dim), y is (N,) labels.
    """
    for name in cohort_order:
        encoders[name] = encoders[name].to(device).eval()

    zs, ys = [], []
    with torch.no_grad():
        for p in patients:
            vol = torch.from_numpy(p.pet_masked.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
            parts = [encoders[name](vol).squeeze(0).cpu().numpy() for name in cohort_order]
            zs.append(np.concatenate(parts))
            ys.append(p.label)
    return np.stack(zs), np.array(ys)


def eval_on(y: np.ndarray, y_prob: np.ndarray) -> dict | None:
    if len(y) == 0:
        return None
    y_pred = (y_prob >= 0.5).astype(int)
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
) -> tuple[dict[str, Encoder3D], list, list, list]:
    """Train Approach C's 3 independent autoencoders for this
    (torch_seed, target_cohort) combination -- one per cohort in
    COHORT_NAMES. The known (non-target) cohorts' autoencoders each
    train on their own full patient set; the target cohort's autoencoder
    trains only on its harmonization-half (never its held-out half).
    Retrained per (seed, target_cohort) pair since which cohort is
    "target" changes what data its own autoencoder sees.
    """
    target_harmonization, target_heldout = split_target_cohort(all_cohorts[target_cohort])
    known_cohorts = [c for c in COHORT_NAMES if c != target_cohort]

    # -- per-cohort training pool: known cohorts use their full patient --
    # set; the target cohort uses only its harmonization-half.
    training_pools = {name: all_cohorts[name] for name in known_cohorts}
    training_pools[target_cohort] = target_harmonization

    encoders: dict[str, Encoder3D] = {}
    for i, name in enumerate(COHORT_NAMES):
        pool = training_pools[name]
        is_target = name == target_cohort
        # target's split is NOT label-stratified -- a real deployment-time
        # unlabeled target cohort has no labels to stratify by, so
        # stratifying here would be testing its own autoencoder's
        # early-stopping split under friendlier conditions than the model
        # will actually see. Known (source) cohorts keep stratification
        # since their labels genuinely drive the classifier fit later.
        train_p, val_p = train_test_split(
            pool, test_size=0.2, random_state=VAL_SPLIT_SEED,
            stratify=None if is_target else [p.label for p in pool],
        )
        torch.manual_seed(torch_seed + i * SEED_OFFSET)
        model = Autoencoder3D(latent_dim=64)
        model = train_autoencoder(
            model, train_p, val_p, n_epochs=N_EPOCHS,
            checkpoint_path=None, label=f"{name} (target={target_cohort}) seed={torch_seed}",
        )
        encoders[name] = model.encoder

    return encoders, known_cohorts, target_harmonization, target_heldout


def evaluate_target(
    torch_seed: int,
    encoders: dict[str, Encoder3D],
    all_cohorts: dict,
    target_cohort: str,
    known_cohorts: list,
    target_harmonization: list,
    target_heldout: list,
) -> dict:
    """Fit the classifier on the known cohorts' concatenated (3-encoder)
    latents + their true labels, evaluate on target's held-out half --
    never seen by target's own autoencoder (which only saw
    target_harmonization) or by the classifier (only ever sees known
    cohorts' patients).
    """
    known_patients = [p for name in known_cohorts for p in all_cohorts[name]]

    Z_known, y_known = encode_concat_multi(encoders, COHORT_NAMES, known_patients)
    Z_target_harm, y_target_harm = encode_concat_multi(encoders, COHORT_NAMES, target_harmonization)
    Z_target_heldout, y_target_heldout = encode_concat_multi(encoders, COHORT_NAMES, target_heldout)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(Z_known, y_known)

    prob_known = clf.predict_proba(Z_known)[:, 1]
    prob_target_harm = clf.predict_proba(Z_target_harm)[:, 1]
    prob_target_heldout = clf.predict_proba(Z_target_heldout)[:, 1]

    # -- z-score correction: align target's predicted-probability -------
    # distribution to the known cohorts' -- exactly the data clf was fit
    # on. Never uses target's own labels, same principle as the
    # 2-cohort script's use of zscore_shift_correct.
    prob_target_corrected = zscore_shift_correct(prob_source=prob_known, prob_target=prob_target_heldout)

    return {
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "known_cohorts": known_cohorts,
        "known_cohort_insample": eval_on(y_known, prob_known),
        "target_cohort_harmonization_half": eval_on(y_target_harm, prob_target_harm),
        "target_cohort_raw": eval_on(y_target_heldout, prob_target_heldout),
        "target_cohort_corrected": eval_on(y_target_heldout, prob_target_corrected),
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
        "label": label, "key": key, "n_runs": len(accs),
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
                  f"(50% of {target_cohort} in training [unsupervised], 50% fully held out)\n{'='*60}")
            encoders, known_cohorts, target_harmonization, target_heldout = train_model_once(
                torch_seed, all_cohorts, target_cohort
            )

            print(f"\n--- evaluating target={target_cohort}, seed={torch_seed} "
                  f"(held out: {len(target_heldout)}, in-training: {len(target_harmonization)}) ---")
            r = evaluate_target(
                torch_seed, encoders, all_cohorts,
                target_cohort, known_cohorts, target_harmonization, target_heldout,
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