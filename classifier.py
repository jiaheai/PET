from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from b import HarmonizationModel

DATA_PATH        = "CUBES-Labelled-COHORTS"
CHECKPOINT_PATH  = "models/best_harmonization.pt"
LATENT_DIM       = 64
FULL_N_SPLITS    = 5
RANDOM_STATE     = 42
OUT_CLASSIFIER   = "models/latent_logreg.joblib"

TRAIN_COHORT     = "AUGSBURG"    # cohort the classifier is fit on (entire cohort)
HOLDOUT_COHORT   = "PRE-RAPID"   # cohort the classifier is evaluated on (entire cohort)

_ENCODE_FN_BY_COHORT = {
    "AUGSBURG": "encode_aug",
    "PRE-RAPID": "encode_pr",
}


# -- latent encoding ------------------------------------------------------------

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


# -- evaluation ------------------------------------------------------------------

def evaluate_cv(
    Z: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    label: str = "",
) -> dict:
    """Stratified k-fold CV of a StandardScaler + LogisticRegression pipeline.

    Scaler and classifier are refit per fold to avoid leakage from the
    held-out fold into the training statistics. This is a sanity check
    ("is there any learnable signal on the training cohort at all"), not
    the headline cross-cohort result -- that's eval_on() below.

    Returns a dict of summary stats (mean/std acc, balanced acc & auc,
    pooled confusion matrix) so callers can compare runs programmatically,
    not just by eye.
    """
    n_splits = min(n_splits, np.bincount(y).min())  # can't have more folds than the smaller class
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, baccs, aucs = [], [], []
    tn = fp = fn = tp = 0

    if label:
        print(f"\n=== {label} ===")
    for fold, (train_idx, test_idx) in enumerate(skf.split(Z, y)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(Z[train_idx], y[train_idx])
        y_pred = clf.predict(Z[test_idx])
        y_prob = clf.predict_proba(Z[test_idx])[:, 1]

        acc = accuracy_score(y[test_idx], y_pred)
        bacc = balanced_accuracy_score(y[test_idx], y_pred)
        if len(np.unique(y[test_idx])) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y[test_idx], y_prob)

        tn_, fp_, fn_, tp_ = confusion_matrix(y[test_idx], y_pred, labels=[0, 1]).ravel()
        recall_pos = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else float("nan")
        recall_neg = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else float("nan")

        accs.append(acc)
        baccs.append(bacc)
        aucs.append(auc)
        tn += tn_; fp += fp_; fn += fn_; tp += tp_
        print(f"fold {fold}  n_test {len(test_idx)}  acc {acc:.3f}  bacc {bacc:.3f}  auc {auc:.3f}  "
              f"recall+ {recall_pos:.3f}  recall- {recall_neg:.3f}")

    mean_acc, std_acc = np.nanmean(accs), np.nanstd(accs)
    mean_bacc, std_bacc = np.nanmean(baccs), np.nanstd(baccs)
    mean_auc, std_auc = np.nanmean(aucs), np.nanstd(aucs)
    pooled_recall_pos = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    pooled_recall_neg = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    print(f"\nmean acc       {mean_acc:.3f} \u00b1 {std_acc:.3f}")
    print(f"mean bal. acc  {mean_bacc:.3f} \u00b1 {std_bacc:.3f}")
    print(f"mean auc       {mean_auc:.3f} \u00b1 {std_auc:.3f}")
    print(f"pooled confusion matrix  tn {tn}  fp {fp}  fn {fn}  tp {tp}")
    print(f"pooled recall(pos) {pooled_recall_pos:.3f}  recall(neg) {pooled_recall_neg:.3f}")

    return {
        "mean_acc": mean_acc, "std_acc": std_acc,
        "mean_bacc": mean_bacc, "std_bacc": std_bacc,
        "mean_auc": mean_auc, "std_auc": std_auc,
        "recall_pos": pooled_recall_pos, "recall_neg": pooled_recall_neg,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def eval_on(model, final_clf, label: str, plist: list) -> None:
    """Apply an already-fit classifier to a fixed patient list (no refitting)."""
    if not plist:
        print(f"\n=== {label} ===\n(no patients)")
        return
    Z, y = encode_patients(model, plist)
    y_pred = final_clf.predict(Z)
    y_prob = final_clf.predict_proba(Z)[:, 1]
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    print(f"\n=== {label} ===")
    print(f"n {len(plist)}  acc {acc:.3f}  bal. acc {bacc:.3f}  auc {auc:.3f}")
    print(f"recall(pos) {recall_pos:.3f}  recall(neg) {recall_neg:.3f}")
    print(f"confusion matrix  tn {tn}  fp {fp}  fn {fn}  tp {tp}")


# -- entry point -------------------------------------------------------------

if __name__ == "__main__":
    from nifti_loader import load_all_cohorts

    models_dir = Path(CHECKPOINT_PATH).parent
    models_dir.mkdir(exist_ok=True)

    model = HarmonizationModel(latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    cohorts = load_all_cohorts(Path(DATA_PATH))

    for name in (TRAIN_COHORT, HOLDOUT_COHORT):
        if name not in cohorts:
            raise ValueError(
                f"Cohort '{name}' not found. Available cohorts: {list(cohorts.keys())}"
            )

    # -- use the ENTIRE cohort on each side ---------------------------------
    # No train/test split within either cohort: the classifier is fit on
    # every AUGSBURG patient's labels and evaluated on every PRE-RAPID
    # patient's labels. This is valid (no leakage into the reported
    # number) because PRE-RAPID's labels never touch classifier training
    # -- only its latents (produced by the harmonization encoder, which
    # was trained without labels on both cohorts) are used at test time.
    train_patients   = cohorts[TRAIN_COHORT]     # all AUGSBURG patients
    holdout_patients = cohorts[HOLDOUT_COHORT]   # all PRE-RAPID patients

    print(f"training classifier on {TRAIN_COHORT} (all {len(train_patients)} patients)")
    print(f"evaluating on {HOLDOUT_COHORT} (all {len(holdout_patients)} patients)")

    # -- fit on the full train cohort ----------------------------------------
    Z_train, y_train = encode_patients(model, train_patients)
    print(f"\ntrain latent matrix {Z_train.shape}  positives {int(y_train.sum())}  negatives {int((1 - y_train).sum())}")

    # CV within the training cohort, as a sanity check that the latents
    # are learnable at all before trusting the cross-cohort number below.
    train_cv_stats = evaluate_cv(Z_train, y_train, n_splits=FULL_N_SPLITS, random_state=RANDOM_STATE, label="TRAIN-COHORT CV")

    final_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    final_clf.fit(Z_train, y_train)
    joblib.dump(final_clf, OUT_CLASSIFIER)
    print(f"saved final classifier to {OUT_CLASSIFIER}")

    # -- evaluate on the full holdout cohort ---------------------------------
    eval_on(model, final_clf, f"HELD-OUT COHORT ({HOLDOUT_COHORT}, all patients)", holdout_patients)
    eval_on(model, final_clf, f"TRAIN COHORT, in-sample ({TRAIN_COHORT}, all patients)", train_patients)