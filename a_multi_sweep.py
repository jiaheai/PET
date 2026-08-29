"""
Approach A, leave-one-cohort-out sweep across 3 cohorts.

Approach A's shared single encoder means MMD only ever needs two groups
of volumes to compare -- it never dispatches per-cohort the way the
multi-encoder approach (b_3_multi/b_3_anchor) does. So the natural 3+
cohort adaptation needs no logic changes in a.py: train_harmonization's
"known" cohort is the POOLED known (non-target) cohorts, and its "target"
cohort is target's harmonization-half -- the model directly aligns
"everything we have labels for" against "the unlabeled target," which is
arguably a cleaner match for the actual goal than pairwise/anchor
mechanics need to approximate.

NOTE: a.py's train_harmonization now takes train_cohorts/val_cohorts as
{name: [patients]} dicts (generalized for N-cohort support) instead of
the old train_aug/val_aug/train_pr/val_pr positional args. This sweep
still only ever passes 2 entries ("known" and "target") -- with N=2,
compute_mmd_multi reduces to exactly one pairwise MMD term, identical to
the old behavior -- so nothing about the sweep's actual logic changes,
only the call site's kwargs below.

Same 50/50 held-out split as b_sweep_multi.py / cnn_sweep_multi.py, using
the SAME TARGET_HOLDOUT_SEED -- so target_heldout is identical across all
three approaches' sweeps, required for a valid cross-approach comparison.
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

from a import HarmonizationModel, train_harmonization
from nifti_loader import load_all_cohorts
from cnn import zscore_shift_correct   # reuse the same correction used for the CNN baseline

# Defaults -- overridable via --data-path / --results-path for automation.
DEFAULT_DATA_PATH    = "CUBES-Labelled-COHORTS"
DEFAULT_RESULTS_PATH = "a_sweep_multi_results.jsonl"

COHORT_NAMES   = ["AUGSBURG", "PRE-RAPID", "SWISS"]
TORCH_SEEDS    = list(range(5))
VAL_SPLIT_SEED = 40             # fixed -- keeps the same val patients across all torch seeds
TARGET_HOLDOUT_SEED = 123       # fixed -- SAME value as b_sweep_multi.py / cnn_sweep_multi.py,
                                 # so target_heldout is identical across every approach's sweep
LAMBDA_MMD     = 1
DECODER_FREEZE_EPOCH = 100
N_EPOCHS       = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approach A, leave-one-cohort-out sweep across all cohort combinations and torch seeds"
    )
    parser.add_argument(
        "--data-path", default=DEFAULT_DATA_PATH,
        help=f"Directory containing all cohorts' data (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--results-path", default=DEFAULT_RESULTS_PATH,
        help=f"Where to write/resume sweep results (default: {DEFAULT_RESULTS_PATH}).",
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


def split_target_cohort(patients: list) -> tuple[list, list]:
    """Stratified 50/50 split of the target cohort's patients into a
    harmonization half (enters train_harmonization's "target" side,
    unsupervised only) and a held-out half (never touches training in any
    way -- not the harmonization model, not the classifier). Fixed
    random_state, same value as the other approaches' sweeps, so all
    three hold out identical patients.
    """
    harmonization_half, heldout_half = train_test_split(
        patients, test_size=0.5, random_state=TARGET_HOLDOUT_SEED,
        stratify=[p.label for p in patients],
    )
    return harmonization_half, heldout_half


def encode_patients(
    model: HarmonizationModel,
    patients: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Encode patients into latent vectors. Approach A has a single shared
    encoder with no per-cohort dispatch (model.encode(x) takes no cohort
    argument at all) -- simpler than the multi-encoder approach's version.
    """
    model = model.to(device)
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for patient in patients:
            vol = (
                torch.from_numpy(patient.pet_masked.astype("float32"))
                .unsqueeze(0).unsqueeze(0)
                .to(device)
            )
            z = model.encode(vol).squeeze(0).cpu().numpy()
            zs.append(z)
            ys.append(patient.label)
    return np.stack(zs), np.array(ys)


def eval_set(model, clf, plist, prob_override=None) -> dict | None:
    if not plist:
        return None
    Z, y = encode_patients(model, plist)
    y_prob = clf.predict_proba(Z)[:, 1] if prob_override is None else prob_override
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
) -> tuple[HarmonizationModel, dict, list, list]:
    """Train Approach A's shared-encoder model for this (torch_seed,
    target_cohort) combination. train_harmonization's "known" cohort is
    the POOLED known (non-target) cohorts; its "target" cohort is
    target's harmonization-half -- the model has no concept of individual
    cohort identity beyond this train-time split, since the encoder is
    fully shared. Retrained per (seed, target_cohort) pair, same reason
    as the other sweeps: the training data composition changes with
    target_cohort.
    """
    target_harmonization, target_heldout = split_target_cohort(all_cohorts[target_cohort])

    known_cohorts = [c for c in COHORT_NAMES if c != target_cohort]
    known_patients = [p for name in known_cohorts for p in all_cohorts[name]]

    train_known, val_known = train_test_split(
        known_patients, test_size=0.2, random_state=VAL_SPLIT_SEED,
        stratify=[p.label for p in known_patients],
    )
    train_target_harm, val_target_harm = train_test_split(
        target_harmonization, test_size=0.2, random_state=VAL_SPLIT_SEED,
        stratify=[p.label for p in target_harmonization],
    )

    torch.manual_seed(torch_seed)
    model = HarmonizationModel(latent_dim=64)
    model = train_harmonization(
        model,
        train_cohorts={"known": train_known, "target": train_target_harm},
        val_cohorts={"known": val_known, "target": val_target_harm},
        n_epochs=N_EPOCHS,
        lambda_mmd=LAMBDA_MMD,
        decoder_freeze_epoch=DECODER_FREEZE_EPOCH,
        checkpoint_path=None,
    )
    return model, all_cohorts, target_harmonization, target_heldout


def evaluate_target(
    torch_seed: int,
    model: HarmonizationModel,
    cohort_all: dict,
    target_cohort: str,
    target_harmonization: list,
    target_heldout: list,
) -> dict:
    """Fit the classifier on the known (non-target) cohorts' LABELS and
    evaluate on target's held-out half -- never seen by the harmonization
    model (train_harmonization only saw target_harmonization) or the
    classifier (only ever sees known_patients).
    """
    known_cohorts = [c for c in COHORT_NAMES if c != target_cohort]
    known_patients = [p for name in known_cohorts for p in cohort_all[name]]

    Z_known, y_known = encode_patients(model, known_patients)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(Z_known, y_known)

    target_patients = target_heldout

    # -- z-score correction: align target's predicted-probability ---
    # distribution to the POOLED known cohorts' -- exactly the data clf
    # was fit on. Never uses target's own labels, same principle as the
    # 2-cohort script's use of zscore_shift_correct.
    Z_target, _ = encode_patients(model, target_patients)
    prob_known = clf.predict_proba(Z_known)[:, 1]
    prob_target = clf.predict_proba(Z_target)[:, 1]
    prob_target_corrected = zscore_shift_correct(prob_source=prob_known, prob_target=prob_target)

    return {
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "known_cohorts": known_cohorts,
        "known_cohort_insample": eval_set(model, clf, known_patients),
        "target_cohort_harmonization_half": eval_set(model, clf, target_harmonization),
        "target_cohort_raw": eval_set(model, clf, target_patients),
        "target_cohort_corrected": eval_set(model, clf, target_patients, prob_override=prob_target_corrected),
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

    # Always run fresh -- no resume/skip logic. Truncate results_path up
    # front so this run's results aren't mixed with any prior run's.
    results_path.write_text("")

    results = []
    for torch_seed in TORCH_SEEDS:
        for target_cohort in COHORT_NAMES:
            print(f"\n{'='*60}\nTORCH SEED {torch_seed}  target={target_cohort}  "
                  f"(50% of {target_cohort} in training [unsupervised], 50% fully held out)\n{'='*60}")
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