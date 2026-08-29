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

from b_3_multi import HarmonizationModel, train_harmonization_multi
from nifti_loader import load_all_cohorts
from cnn import zscore_shift_correct   # reuse the same correction used for the CNN baseline

# Defaults -- overridable via --data-path / --results-path for automation.
DEFAULT_DATA_PATH    = "CUBES-Labelled-COHORTS"
DEFAULT_RESULTS_PATH = "b_sweep_multi_results.jsonl"   # contains both record_type="run" (per-seed,
                                                         # per-target-cohort) and record_type="summary"
                                                         # (aggregate, appended at the end) lines.

COHORT_NAMES   = ["AUGSBURG", "PRE-RAPID", "SWISS"]
TORCH_SEEDS    = list(range(25))
VAL_SPLIT_SEED = 40             # fixed -- keeps the same val patients across all torch seeds
TARGET_HOLDOUT_SEED = 123       # fixed -- keeps the same held-out half of the target cohort across
                                 # all torch seeds, so only the harmonization training itself varies
                                 # with torch_seed, not which patients were held out
N_EPOCHS       = 1000
LAMBDA_MMD     = 100
LATENT_GAMMA_MODE = "adaptive"   # "adaptive" (recompute every batch) or "fixed" (frozen
                                  # once at decoder_freeze_epoch, then held constant) --
                                  # see train_harmonization_multi's docstring for what this
                                  # isolates. Change this and rerun to compare.
DECODER_FREEZE_EPOCH = 50
PATIENCE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-cohort-out sweep across all cohort combinations and torch seeds"
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
    """Stratified 50/50 split of the target cohort's patients into a
    harmonization half (the only part of this cohort that enters the
    harmonization model's training/val data) and a held-out half (never
    touches training in any way -- not the harmonization model, not the
    classifier -- and is only scored once at the end on the finished
    model). Fixed random_state so the same held-out half is used
    regardless of torch_seed.
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
    """Encode patients into latent vectors using each patient's OWN cohort's
    encoder (model.encoders[patient.cohort]) -- mirrors classifier.py's
    _ENCODE_FN_BY_COHORT dispatch for the 2-encoder version, generalized to
    N cohorts via a ModuleDict instead of a fixed aug/pr attribute lookup.
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
            z = model.encode(patient.cohort, vol).squeeze(0).cpu().numpy()
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
    """Train the harmonization model for this (torch_seed, target_cohort)
    combination.

    Only a stratified 50% of the target cohort's patients (the
    "harmonization half", from split_target_cohort) enters the
    harmonization model's training/val data at all. The other 50% (the
    "held-out half") never touches training -- not the harmonization
    model, not the classifier -- and is reserved for the final
    evaluation only. Known (non-target) cohorts are unaffected and still
    contribute all of their patients. Because the training data
    composition now changes with target_cohort, the model can no longer
    be trained once per seed and reused across all three target_cohort
    rotations -- it's retrained for each (seed, target_cohort) pair.

    Returns the trained model, cohort_all (all patients per cohort,
    unsplit -- used for known-cohort evaluation and classifier fitting),
    the target cohort's harmonization half (patients the model DID see,
    useful as a diagnostic), and the target cohort's held-out half
    (patients the model never saw, the real test).
    """
    target_harmonization, target_heldout = split_target_cohort(all_cohorts[target_cohort])

    cohort_train, cohort_val, cohort_all = {}, {}, {}
    for name in COHORT_NAMES:
        cohort_all[name] = all_cohorts[name]  # full cohort, unsplit -- for eval/classifier fitting
        patients_for_training = target_harmonization if name == target_cohort else all_cohorts[name]

        train_p, val_p = train_test_split(
            patients_for_training, test_size=0.2, random_state=VAL_SPLIT_SEED,
            stratify=[p.label for p in patients_for_training],
        )
        cohort_train[name] = train_p
        cohort_val[name] = val_p

    torch.manual_seed(torch_seed)
    model = HarmonizationModel(cohort_names=COHORT_NAMES, latent_dim=64)
    model = train_harmonization_multi(
        model,
        cohort_train=cohort_train, cohort_val=cohort_val,
        n_epochs=N_EPOCHS,
        lambda_mmd=LAMBDA_MMD,
        decoder_freeze_epoch=DECODER_FREEZE_EPOCH,
        latent_gamma_mode=LATENT_GAMMA_MODE,
        checkpoint_path=None,
        patience=PATIENCE,
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
    """Fit the classifier on the known (non-target) cohorts' LABELS and
    evaluate on the target cohort's held-out half -- the classifier never
    sees target-cohort labels either way, but now the headline result
    (target_cohort_raw / target_cohort_corrected) is scored only on
    patients the harmonization model never trained on at all, not just
    patients the classifier didn't fit on.
    """
    known_cohorts = [c for c in COHORT_NAMES if c != target_cohort]

    # classifier: fit on ALL patients from the known (non-target) cohorts,
    # pooled together -- this is the multi-cohort analogue of "fit on all
    # of AUGSBURG" in the 2-cohort script, generalized to N-1 known cohorts.
    known_patients = [p for name in known_cohorts for p in cohort_all[name]]
    Z_known, y_known = encode_patients(model, known_patients)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(Z_known, y_known)

    # The headline result: the held-out half of the target cohort. This
    # data never entered the harmonization model's training/val split
    # (see split_target_cohort / train_model_once) and never entered the
    # classifier fit either -- it's untouched until this line.
    target_patients = target_heldout

    # -- z-score correction: align target cohort's predicted-probability ---
    # distribution to the POOLED known cohorts' -- i.e. exactly the same
    # data clf was just fit on (known_patients, above). The classifier's
    # 0.5 threshold is calibrated against that pooled distribution by
    # construction (that's what .fit() saw), so that's the correct
    # reference to measure target's shift against -- not a single cohort.
    # Never uses target's own labels either way, same principle as the
    # 2-cohort script's use of zscore_shift_correct.
    prob_known = clf.predict_proba(Z_known)[:, 1]
    Z_target, _ = encode_patients(model, target_patients)
    prob_target = clf.predict_proba(Z_target)[:, 1]
    prob_target_corrected = zscore_shift_correct(prob_source=prob_known, prob_target=prob_target)

    return {
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "known_cohorts": known_cohorts,
        "known_cohort_insample": eval_set(model, clf, known_patients),   # in-sample, reference only
        # diagnostic: patients from the target cohort that the harmonization
        # model DID train on (unsupervised) but the classifier never saw --
        # useful to compare against the truly-unseen result below
        "target_cohort_harmonization_half": eval_set(model, clf, target_harmonization),
        "target_cohort_raw": eval_set(model, clf, target_patients),      # held-out half -- never seen by anything, raw
        "target_cohort_corrected": eval_set(model, clf, target_patients, prob_override=prob_target_corrected),  # held-out half, z-score corrected
    }


def summarize(results: list, key: str, label: str) -> dict:
    """Print the aggregate stats for `key` across `results` (unchanged
    behavior) AND return them as a dict so callers can persist them --
    previously this function only printed, so these numbers existed
    nowhere but stdout.
    """
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

    all_cohorts = load_all_cohorts(data_path)
    for name in COHORT_NAMES:
        if name not in all_cohorts:
            raise ValueError(f"Cohort '{name}' not found. Available cohorts: {list(all_cohorts.keys())}")

    # -- resume/fresh setup ---------------------------------------------
    # --fresh (or no prior file) -> nothing to resume, empty start.
    # Otherwise: load prior runs, keep only seeds that are BOTH complete
    # (every cohort covered) AND still requested in the current TORCH_SEEDS
    # (so shrinking/changing the seed list doesn't silently keep averaging
    # in seeds you no longer asked for). Anything else -- incomplete seeds,
    # seeds outside the current list, all prior "summary" lines -- is
    # dropped; results_path is then rewritten to contain exactly the kept
    # run records, so it never carries stale rows or a stale summary
    # forward into this run.
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
            append_result(r, results_path)   # persist immediately, so a crash doesn't lose this run
            results.append(r)
            print(f"  known cohorts (in-sample)      : {r['known_cohort_insample']}")
            print(f"  {target_cohort} (harmonization half) : {r['target_cohort_harmonization_half']}")
            print(f"  {target_cohort} (held out, raw)       : {r['target_cohort_raw']}")
            print(f"  {target_cohort} (held out, corrected) : {r['target_cohort_corrected']}")

    # -- per-target-cohort summaries -----------------------------------------
    # results here spans EVERY seed in TORCH_SEEDS -- resumed seeds loaded
    # above plus whatever just ran -- so this averages across all 20 (or
    # however many), not just the ones run this invocation.
    summary_stats = []
    for target_cohort in COHORT_NAMES:
        subset = [r for r in results if r["target_cohort"] == target_cohort]
        if not subset:
            continue
        summary_stats.append(summarize(subset, "target_cohort_raw", f"target={target_cohort} (held-out, raw)"))
        summary_stats.append(summarize(subset, "target_cohort_corrected", f"target={target_cohort} (held-out, z-score corrected)"))

    # -- overall summary across every target-cohort choice -------------------
    # This is the number that actually answers the feasibility question:
    # does harmonization generalize regardless of *which* cohort gets left
    # out, not just for one arbitrarily chosen train/test split.
    summary_stats.append(summarize(results, "target_cohort_raw", "ALL target cohorts pooled (held-out, raw)"))
    summary_stats.append(summarize(results, "target_cohort_corrected", "ALL target cohorts pooled (held-out, z-score corrected)"))

    # Appended to the SAME results_path as the per-run records, at the end
    # of the file, each tagged record_type="summary" so a reader can tell
    # them apart from record_type="run" lines without guessing from shape.
    for s in summary_stats:
        append_summary(s, results_path)
    print(f"\nsummary appended to: {results_path}")