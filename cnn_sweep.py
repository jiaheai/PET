"""
CNN baseline, leave-one-cohort-out sweep across 3 cohorts.

Adapts cnn_sweep_norm.py's 2-cohort (fixed AUGSBURG-train / PRE-RAPID-test)
design to N cohorts, matching b_sweep_multi.py's structure: every cohort
takes a turn as target, the CNN trains on the pooled labels of the other
two ("known") cohorts, and is evaluated on a genuinely held-out half of
target -- never seen in any form, since this CNN (unlike the harmonization
approaches) has no unsupervised stage that could see target's images
either. The 50/50 split is kept anyway so target's evaluation set here is
IDENTICAL to the harmonization sweeps' held-out half -- this is the
no-harmonization baseline those approaches need to beat, and that
comparison is only valid if all approaches are scored on the same patients.

Reuses cnn.py's model/training/correction code directly rather than
duplicating it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from cnn import CNNClassifier3D, train_cnn_baseline, zscore_shift_correct
from nifti_loader import load_all_cohorts

# Defaults -- overridable via --data-path / --results-path for automation.
DEFAULT_DATA_PATH    = "CUBES-Labelled-COHORTS"
DEFAULT_RESULTS_PATH = "cnn_sweep_multi_results.jsonl"

COHORT_NAMES   = ["AUGSBURG", "PRE-RAPID", "SWISS"]
LATENT_DIM     = 16               # fixed -- 16-dim beat 64-dim in the earlier 5-seed 2-cohort comparison
TORCH_SEEDS    = list(range(50))   # matches the seed count used by the harmonization sweeps this session
VAL_SPLIT_SEED = 40                # fixed -- same known-cohort train/val split across all runs
TARGET_HOLDOUT_SEED = 123          # fixed -- same held-out half of target across all torch seeds,
                                    # and the SAME patients the harmonization sweeps hold out --
                                    # required for a fair cross-approach comparison
N_EPOCHS       = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN baseline, leave-one-cohort-out sweep across all cohort combinations and torch seeds")
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
        help="Wipe --results-path and rerun every seed in TORCH_SEEDS from scratch, "
             "ignoring anything already there. Default: resume -- reuse any torch_seed "
             "that already has a complete run (all cohorts as target) in --results-path, "
             "and only run the seeds still missing.",
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
    """Load previously written 'run' records from results_path, if any.
    'summary' records are dropped unconditionally -- the summary is always
    regenerated fresh from the full (resumed + new) result set at the end
    of this run, never carried over from a prior run's aggregation.
    """
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
                runs.append(rec)   # record_type popped -- shape now matches evaluate_target()'s return
    return runs


def complete_seeds(runs: list[dict], cohort_names: list[str]) -> set[int]:
    """A torch_seed counts as complete only if it has a run record for
    EVERY cohort as target. A seed interrupted partway through its cohort
    rotation is NOT complete -- its partial rows get discarded and the
    whole seed is redone from scratch below, rather than silently
    averaging in an incomplete seed. Also means a COHORT_NAMES change
    (e.g. a cohort added) correctly invalidates old seeds that only
    covered the previous, smaller cohort set.
    """
    by_seed: dict[int, set[str]] = {}
    for r in runs:
        by_seed.setdefault(r["torch_seed"], set()).add(r["target_cohort"])
    return {seed for seed, cohorts in by_seed.items() if cohorts == set(cohort_names)}


def split_target_cohort(patients: list) -> tuple[list, list]:
    """Stratified 50/50 split of the target cohort's patients. The CNN
    never trains on either half (no unsupervised stage exists to use the
    "harmonization half" the way the harmonization approaches do) -- both
    halves are equally untouched by training. Kept anyway so
    target_heldout is IDENTICAL to the harmonization sweeps' held-out set,
    which is required for a valid apples-to-apples baseline comparison.
    Fixed random_state so the same held-out half is used regardless of
    torch_seed, matching the harmonization sweeps exactly (same seed value
    too -- TARGET_HOLDOUT_SEED=123 in both).
    """
    harmonization_half, heldout_half = train_test_split(
        patients, test_size=0.5, random_state=TARGET_HOLDOUT_SEED,
        stratify=[p.label for p in patients],
    )
    return harmonization_half, heldout_half


def get_probs(model, plist, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for p in plist:
            vol = torch.from_numpy(p.pet_masked.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
            probs.append(torch.sigmoid(model(vol)).item())
            ys.append(p.label)
    return np.array(probs), np.array(ys)


def eval_on(y: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict | None:
    """Same metric shape as b_sweep_multi.py's eval_set -- acc, auc,
    recall_pos, recall_neg, balanced_acc, confusion matrix."""
    if len(y) == 0:
        return None
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "acc": float(acc), "auc": float(auc),
        "recall_pos": float(recall_pos), "recall_neg": float(recall_neg),
        "balanced_acc": float(bacc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_model_once(
    torch_seed: int, all_cohorts: dict, target_cohort: str
) -> tuple[CNNClassifier3D, dict, list, list]:
    """Train the CNN for this (torch_seed, target_cohort) combination on
    the POOLED labels of the two known (non-target) cohorts -- the
    multi-cohort analogue of "train on all of AUGSBURG" in the 2-cohort
    script. Because which cohorts are "known" changes with target_cohort,
    the model is retrained per (seed, target_cohort) pair, same as the
    harmonization sweeps.

    Returns the trained model, cohort_all (all patients per cohort,
    unsplit), the target cohort's harmonization half (untouched by
    training, kept as a diagnostic to compare against the true held-out
    half below), and the target cohort's held-out half (the real test,
    identical to the harmonization sweeps' held-out set).
    """
    target_harmonization, target_heldout = split_target_cohort(all_cohorts[target_cohort])

    known_cohorts = [c for c in COHORT_NAMES if c != target_cohort]
    known_patients = [p for name in known_cohorts for p in all_cohorts[name]]

    train_known, val_known = train_test_split(
        known_patients, test_size=0.2, random_state=VAL_SPLIT_SEED,
        stratify=[p.label for p in known_patients],
    )

    torch.manual_seed(torch_seed)
    model = CNNClassifier3D(latent_dim=LATENT_DIM)
    model = train_cnn_baseline(
        model,
        train_patients=train_known, val_patients=val_known,
        n_epochs=N_EPOCHS,
        checkpoint_path=None,
    )
    return model, all_cohorts, target_harmonization, target_heldout


def evaluate_target(
    torch_seed: int,
    model: CNNClassifier3D,
    cohort_all: dict,
    target_cohort: str,
    target_harmonization: list,
    target_heldout: list,
) -> dict:
    """Evaluate on the target cohort's held-out half -- the CNN never saw
    ANY of target's data in any form during training (unlike the
    harmonization approaches' unsupervised harmonization-half exposure),
    so target_cohort_harmonization_half here is purely a diagnostic
    comparison point, not evidence of a train/test gap the way it is for
    the harmonization sweeps.
    """
    known_cohorts = [c for c in COHORT_NAMES if c != target_cohort]
    known_patients = [p for name in known_cohorts for p in cohort_all[name]]

    prob_known, y_known = get_probs(model, known_patients)
    prob_harm, y_harm = get_probs(model, target_harmonization)
    prob_target, y_target = get_probs(model, target_heldout)

    # -- z-score correction: align target's predicted-probability ---
    # distribution to the POOLED known cohorts' -- no target labels used.
    prob_target_corrected = zscore_shift_correct(prob_source=prob_known, prob_target=prob_target)

    return {
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "known_cohorts": known_cohorts,
        "known_cohort_insample": eval_on(y_known, prob_known),               # in-sample, reference only
        "target_cohort_harmonization_half": eval_on(y_harm, prob_harm),      # diagnostic only -- see docstring
        "target_cohort_raw": eval_on(y_target, prob_target),                 # held-out half, raw
        "target_cohort_corrected": eval_on(y_target, prob_target_corrected), # held-out half, z-score corrected
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

    # -- resume/fresh setup ---------------------------------------------
    # See b_sweep_multi.py for the identical scheme: --fresh (or no prior
    # file) starts empty; otherwise keep only seeds that are BOTH complete
    # (every cohort covered) AND still in the current TORCH_SEEDS, discard
    # everything else (incomplete seeds, seeds outside the current list,
    # all prior "summary" lines), and rewrite results_path to contain
    # exactly the kept run records before appending anything new.
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
                  f"(CNN never trains on ANY of {target_cohort}; 50% is a diagnostic-only "
                  f"'harmonization half', 50% is the real held-out test)\n{'='*60}")
            model, cohort_all, target_harmonization, target_heldout = train_model_once(
                torch_seed, all_cohorts, target_cohort
            )

            print(f"\n--- evaluating target={target_cohort}, seed={torch_seed} "
                  f"(held out: {len(target_heldout)}, diagnostic-only: {len(target_harmonization)}) ---")
            r = evaluate_target(
                torch_seed, model, cohort_all, target_cohort, target_harmonization, target_heldout
            )
            append_result(r, results_path)   # persist immediately, so a crash doesn't lose this run
            results.append(r)
            print(f"  known cohorts (in-sample)      : {r['known_cohort_insample']}")
            print(f"  {target_cohort} (diagnostic-only half) : {r['target_cohort_harmonization_half']}")
            print(f"  {target_cohort} (held out, raw)       : {r['target_cohort_raw']}")
            print(f"  {target_cohort} (held out, corrected) : {r['target_cohort_corrected']}")

    # -- per-target-cohort summaries -----------------------------------------
    # results spans EVERY seed in TORCH_SEEDS -- resumed seeds loaded above
    # plus whatever just ran -- so this averages across all requested
    # seeds, not just the ones run this invocation.
    summary_stats = []
    for target_cohort in COHORT_NAMES:
        subset = [r for r in results if r["target_cohort"] == target_cohort]
        if not subset:
            continue
        summary_stats.append(summarize(subset, "target_cohort_raw", f"target={target_cohort} (held-out, raw)"))
        summary_stats.append(summarize(subset, "target_cohort_corrected", f"target={target_cohort} (held-out, z-score corrected)"))

    # -- overall summary across every target-cohort choice -------------------
    summary_stats.append(summarize(results, "target_cohort_raw", "ALL target cohorts pooled (held-out, raw)"))
    summary_stats.append(summarize(results, "target_cohort_corrected", "ALL target cohorts pooled (held-out, z-score corrected)"))

    for s in summary_stats:
        append_summary(s, results_path)
    print(f"\nsummary appended to: {results_path}")