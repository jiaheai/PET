from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harmonization import HarmonizationModel

DATA_PATH = "CUBES-Labelled-COHORTS"
_ENCODE_FN_BY_COHORT = {
    "AUGSBURG": "encode_aug",
    "PRE-RAPID": "encode_pr",
}


# ── latent encoding ───────────────────────────────────────────────────────────

def encode_patients(
    model: HarmonizationModel,
    patients: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Encode patients into latent vectors using their cohort's encoder.

    Returns (Z, y): Z is (N, latent_dim), y is (N,) with 0 = negative
    (low risk), 1 = positive (high risk).
    """
    model = model.to(device)
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for patient in patients:
            encode_fn = getattr(model, _ENCODE_FN_BY_COHORT[patient.cohort])
            vol = (
                torch.from_numpy(patient.pet_masked.astype("float32"))
                .unsqueeze(0).unsqueeze(0)
                .to(device)
            )
            z = encode_fn(vol).squeeze(0).cpu().numpy()
            zs.append(z)
            ys.append(patient.label)
    return np.stack(zs), np.array(ys)


def load_val_patients(
    aug_patients: list, pr_patients: list, val_ids_path: str | Path
) -> list:
    """Return only the patients present in val_patient_ids.json (never
    seen by gradient descent during encoder training)."""
    with open(val_ids_path) as f:
        val_ids = json.load(f)

    val_aug_ids = set(val_ids["AUGSBURG"])
    val_pr_ids = set(val_ids["PRE-RAPID"])

    val_aug = [p for p in aug_patients if p.patient_id in val_aug_ids]
    val_pr = [p for p in pr_patients if p.patient_id in val_pr_ids]

    # sanity check — if these don't match, the JSON is stale or patient_id
    # values don't line up between runs (e.g. reloaded/renamed data).
    if len(val_aug) != len(val_aug_ids):
        missing = val_aug_ids - {p.patient_id for p in val_aug}
        print(f"WARNING: {len(missing)} AUGSBURG val IDs not found in loaded patients: {missing}")
    if len(val_pr) != len(val_pr_ids):
        missing = val_pr_ids - {p.patient_id for p in val_pr}
        print(f"WARNING: {len(missing)} PRE-RAPID val IDs not found in loaded patients: {missing}")

    return val_aug + val_pr


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_cv(
    Z: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    label: str = "",
) -> dict:
    """Stratified k-fold CV of a StandardScaler + LogisticRegression pipeline.

    Scaler and classifier are refit per fold to avoid leakage from the
    held-out fold into the training statistics.

    Returns a dict of summary stats (mean/std acc & auc, pooled confusion
    matrix) so callers can compare runs programmatically, not just by eye.
    """
    n_splits = min(n_splits, np.bincount(y).min())  # can't have more folds than the smaller class
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, aucs = [], []
    tn = fp = fn = tp = 0

    if label:
        print(f"\n=== {label} ===")
    for fold, (train_idx, test_idx) in enumerate(skf.split(Z, y)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(Z[train_idx], y[train_idx])
        y_pred = clf.predict(Z[test_idx])
        y_prob = clf.predict_proba(Z[test_idx])[:, 1]

        acc = accuracy_score(y[test_idx], y_pred)
        if len(np.unique(y[test_idx])) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y[test_idx], y_prob)

        accs.append(acc)
        aucs.append(auc)
        tn_, fp_, fn_, tp_ = confusion_matrix(y[test_idx], y_pred, labels=[0, 1]).ravel()
        tn += tn_; fp += fp_; fn += fn_; tp += tp_
        print(f"fold {fold}  n_test {len(test_idx)}  acc {acc:.3f}  auc {auc:.3f}")

    mean_acc, std_acc = np.nanmean(accs), np.nanstd(accs)
    mean_auc, std_auc = np.nanmean(aucs), np.nanstd(aucs)
    print(f"\nmean acc {mean_acc:.3f} \u00b1 {std_acc:.3f}")
    print(f"mean auc {mean_auc:.3f} \u00b1 {std_auc:.3f}")
    print(f"pooled confusion matrix  tn {tn}  fp {fp}  fn {fn}  tp {tp}")

    return {
        "mean_acc": mean_acc, "std_acc": std_acc,
        "mean_auc": mean_auc, "std_auc": std_auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def run_evaluation(
    model: HarmonizationModel,
    patients: list,
    label: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Encode a patient list, print latent-matrix summary, and run CV.

    Single entry point so full-pool and val-only evaluations share one
    code path instead of duplicating encode+print+evaluate at each call
    site.
    """
    Z, y = encode_patients(model, patients)
    print(f"\n{label} latent matrix {Z.shape}  positives {int(y.sum())}  negatives {int((1 - y).sum())}")
    stats = evaluate_cv(Z, y, n_splits=n_splits, random_state=random_state, label=label.upper())
    return Z, y, stats


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Latent-space risk classifier CV")
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--checkpoint", default="models/best_harmonization.pt")
    parser.add_argument("--val-ids-path", default="models/val_patient_ids.json")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--full-n-splits", type=int, default=5)
    parser.add_argument("--val-n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-classifier", default="models/latent_logreg.joblib")
    args = parser.parse_args()

    from nifti_loader import load_all_cohorts

    models_dir = Path(args.checkpoint).parent
    models_dir.mkdir(exist_ok=True)

    model = HarmonizationModel(latent_dim=args.latent_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    cohorts = load_all_cohorts(Path(args.data_path))
    aug_patients = cohorts["AUGSBURG"]
    pr_patients = cohorts["PRE-RAPID"]
    patients = aug_patients + pr_patients
    print(f"loaded {len(patients)} patients from {list(cohorts.keys())}")

    # ── full pool (includes encoder-train patients) ────────────────────────
    Z_all, y_all, full_stats = run_evaluation(
        model, patients, "full pool (includes encoder-train patients)",
        n_splits=args.full_n_splits, random_state=args.random_state,
    )

    # ── val-only pool (never touched by gradient descent) ──────────────────
    eval_patients = load_val_patients(aug_patients, pr_patients, args.val_ids_path)
    Z_val, y_val, val_stats = run_evaluation(
        model, eval_patients, "val-only pool (encoder-untrained-on patients)",
        n_splits=args.val_n_splits, random_state=args.random_state,
    )

    print(
        f"\nsummary: full acc {full_stats['mean_acc']:.3f} vs "
        f"val-only acc {val_stats['mean_acc']:.3f}"
    )

    # final classifier fit on all pooled latents, for downstream use
    final_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    final_clf.fit(Z_all, y_all)
    joblib.dump(final_clf, args.out_classifier)
    print(f"saved final classifier to {args.out_classifier}")


if __name__ == "__main__":
    main()