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

from b_3_anchor import HarmonizationModel, train_harmonization_anchor
from nifti_loader import load_all_cohorts
from cnn import zscore_shift_correct   # reuse the same correction used for the CNN baseline

# Defaults -- overridable via --data-path / --results-path for automation.
DEFAULT_DATA_PATH    = "CUBES-Labelled-COHORTS"
DEFAULT_RESULTS_PATH = "b_sweep_anchor_results.jsonl"   # separate from the pairwise sweep's results --
                                                          # same JSON schema, different training method,
                                                          # don't want them silently pooled in summarize()

COHORT_NAMES   = ["AUGSBURG", "PRE-RAPID", "SWISS"]
ANCHOR_COHORT  = "SWISS"   # fixed reference cohort every run -- your largest/most-trusted cohort;
                                # stays the same across all target_cohort rotations below. If a given
                                # combination has target_cohort == ANCHOR_COHORT, anchor_mmd degrades to
                                # fully pooled for that run (verified safe, but loses the anchor benefit
                                # for that one combination -- see the anchor-vs-pairwise discussion).
TORCH_SEEDS    = list(range(5))
VAL_SPLIT_SEED = 40             # fixed -- keeps the same val patients across all torch seeds
N_EPOCHS       = 1000
LAMBDA_MMD     = 1000
DECODER_FREEZE_EPOCH = 100


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
    return parser.parse_args()


def append_result(r: dict, results_path: Path) -> None:
    with open(results_path, "a") as f:
        f.write(json.dumps(r) + "\n")


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


def train_model_once(torch_seed: int, all_cohorts: dict) -> tuple[HarmonizationModel, dict]:
    """Train the harmonization model ONCE for this seed. Harmonization no
    longer depends on target_cohort at all -- MMD is fully pooled and
    label-blind regardless of which cohort you're about to evaluate
    against (anchor_cohort is a separate, fixed choice, unrelated to
    target_cohort) -- so training a fresh model per target_cohort
    combination would just be identical work repeated N times for
    nothing. Returns the trained model plus cohort_all (all patients per
    cohort, unsplit) for the evaluation step.
    """
    cohort_train, cohort_val, cohort_all = {}, {}, {}
    for name in COHORT_NAMES:
        patients = all_cohorts[name]
        cohort_all[name] = patients
        train_p, val_p = train_test_split(
            patients, test_size=0.2, random_state=VAL_SPLIT_SEED,
            stratify=[p.label for p in patients],
        )
        cohort_train[name] = train_p
        cohort_val[name] = val_p

    torch.manual_seed(torch_seed)
    model = HarmonizationModel(cohort_names=COHORT_NAMES, latent_dim=64)
    model = train_harmonization_anchor(
        model,
        cohort_train=cohort_train, cohort_val=cohort_val,
        n_epochs=N_EPOCHS,
        lambda_mmd=LAMBDA_MMD,
        decoder_freeze_epoch=DECODER_FREEZE_EPOCH,
        anchor_cohort=ANCHOR_COHORT,
        checkpoint_path=None,
    )
    return model, cohort_all


def evaluate_target(torch_seed: int, model: HarmonizationModel, cohort_all: dict, target_cohort: str) -> dict:
    """Fit the classifier on the known (non-target) cohorts' LABELS and
    evaluate on target_cohort -- reuses an already-trained model, doesn't
    retrain anything. This is the part that actually varies per
    (target_cohort, torch_seed) combination now.
    """
    known_cohorts = [c for c in COHORT_NAMES if c != target_cohort]

    # classifier: fit on ALL patients from the known (non-target) cohorts,
    # pooled together -- this is the multi-cohort analogue of "fit on all
    # of AUGSBURG" in the 2-cohort script, generalized to N-1 known cohorts.
    known_patients = [p for name in known_cohorts for p in cohort_all[name]]
    Z_train, y_train = encode_patients(model, known_patients)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(Z_train, y_train)

    target_patients = cohort_all[target_cohort]

    # -- z-score correction: align target cohort's predicted-probability ---
    # distribution to a SINGLE reference cohort's, not the pooled known
    # cohorts. Pooling known_patients together only makes sense as a
    # calibration reference if the known cohorts share one distribution --
    # exactly the thing we can't assume, since domain shift between them is
    # possible too. Use whichever known cohort has the most patients this
    # rotation as the reference (larger cohort -> more stable mean/std
    # estimate for the correction) -- never the target's own labels, same
    # principle as the 2-cohort script's use of zscore_shift_correct.
    reference_cohort = max(known_cohorts, key=lambda c: len(cohort_all[c]))
    reference_patients = cohort_all[reference_cohort]

    Z_reference_full, _ = encode_patients(model, reference_patients)
    Z_target_full, _ = encode_patients(model, target_patients)
    prob_reference = clf.predict_proba(Z_reference_full)[:, 1]
    prob_target = clf.predict_proba(Z_target_full)[:, 1]
    prob_target_corrected = zscore_shift_correct(prob_source=prob_reference, prob_target=prob_target)

    return {
        "torch_seed": torch_seed,
        "target_cohort": target_cohort,
        "known_cohorts": known_cohorts,
        "calibration_reference_cohort": reference_cohort,
        "known_cohort_insample": eval_set(model, clf, known_patients),                                  # in-sample, reference only
        "target_cohort_raw": eval_set(model, clf, target_patients),                                      # raw -- the headline cross-cohort result
        "target_cohort_corrected": eval_set(model, clf, target_patients, prob_override=prob_target_corrected),  # z-score corrected
    }


def summarize(results: list, key: str, label: str) -> None:
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
        print(f"\n{'='*60}\nTORCH SEED {torch_seed}  (training once, evaluating against: {COHORT_NAMES})\n{'='*60}")
        model, cohort_all = train_model_once(torch_seed, all_cohorts)

        for target_cohort in COHORT_NAMES:
            print(f"\n--- evaluating target={target_cohort}, seed={torch_seed} ---")
            r = evaluate_target(torch_seed, model, cohort_all, target_cohort)
            append_result(r, results_path)   # persist immediately, so a crash doesn't lose this run
            results.append(r)
            print(f"  known cohorts (in-sample) : {r['known_cohort_insample']}")
            print(f"  {target_cohort} (target)  : {r['target_cohort_raw']}")

    # -- per-target-cohort summaries -----------------------------------------
    for target_cohort in COHORT_NAMES:
        subset = [r for r in results if r["target_cohort"] == target_cohort]
        if not subset:
            continue
        summarize(subset, "target_cohort_raw", f"target={target_cohort} (raw, held-out)")
        summarize(subset, "target_cohort_corrected", f"target={target_cohort} (z-score corrected)")

    # -- overall summary across every target-cohort choice -------------------
    # This is the number that actually answers the feasibility question:
    # does harmonization generalize regardless of *which* cohort gets left
    # out, not just for one arbitrarily chosen train/test split.
    summarize(results, "target_cohort_raw", "ALL target cohorts pooled (raw, held-out)")
    summarize(results, "target_cohort_corrected", "ALL target cohorts pooled (z-score corrected)")