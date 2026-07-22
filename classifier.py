from __future__ import annotations

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

DATA_PATH        = "CUBES-Labelled-COHORTS"
CHECKPOINT_PATH  = "models/best_harmonization.pt"
TEST_IDS_PATH    = "models/test_patient_ids.txt"
LATENT_DIM       = 64
FULL_N_SPLITS    = 5
RANDOM_STATE     = 42
OUT_CLASSIFIER   = "models/latent_logreg.joblib"

TRAIN_COHORT     = "AUGSBURG"    # cohort used to fit the classifier
HOLDOUT_COHORT   = "PRE-RAPID"   # cohort evaluated as the generalization test

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


def eval_on(model, final_clf, label: str, plist: list) -> None:
    """Apply an already-fit classifier to a fixed patient list (no refitting)."""
    if not plist:
        print(f"\n=== {label} ===\n(no patients)")
        return
    Z, y = encode_patients(model, plist)
    y_pred = final_clf.predict(Z)
    y_prob = final_clf.predict_proba(Z)[:, 1]
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    print(f"\n=== {label} ===")
    print(f"n {len(plist)}  acc {acc:.3f}  auc {auc:.3f}")
    print(f"confusion matrix  tn {tn}  fp {fp}  fn {fn}  tp {tp}")


# ── entry point ───────────────────────────────────────────────────────────────

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

    # ── load the test-patient keys saved during harmonization training ──
    # NOTE: this file and CHECKPOINT_PATH must come from the same
    # harmonization training run — if you retrain with a different split
    # or dataset version, regenerate this file too, or the "held-out"
    # evaluation below will be silently wrong (encoder may have trained
    # on patients this script treats as unseen).
    test_ids_path = Path(TEST_IDS_PATH)
    if not test_ids_path.exists():
        raise FileNotFoundError(
            f"{test_ids_path} not found — run harmonization.py first to "
            f"generate the train/val/test split and save test patient ids."
        )
    with open(test_ids_path) as f:
        test_keys = set(tuple(line.strip().split("\t")) for line in f if line.strip())

    all_patients = [p for plist in cohorts.values() for p in plist]

    test_patients     = [p for p in all_patients if (p.cohort, p.patient_id) in test_keys]
    trainval_patients = [p for p in all_patients if (p.cohort, p.patient_id) not in test_keys]

    train_patients = [p for p in trainval_patients if p.cohort == TRAIN_COHORT]
    test_train_cohort   = [p for p in test_patients if p.cohort == TRAIN_COHORT]
    test_holdout_cohort = [p for p in test_patients if p.cohort == HOLDOUT_COHORT]

    print(f"loaded {len(all_patients)} patients total")
    print(f"training on {TRAIN_COHORT} trainval ({len(train_patients)} patients)")
    print(f"held-out test — {TRAIN_COHORT}: {len(test_train_cohort)}  "
          f"{HOLDOUT_COHORT}: {len(test_holdout_cohort)}")

    # ── fit on train cohort's trainval patients only ────────────────────
    Z_train, y_train = encode_patients(model, train_patients)
    print(f"\ntrain latent matrix {Z_train.shape}  positives {int(y_train.sum())}  negatives {int((1 - y_train).sum())}")

    # CV within the training cohort's trainval set, sanity-check it's learnable at all
    train_cv_stats = evaluate_cv(Z_train, y_train, n_splits=FULL_N_SPLITS, random_state=RANDOM_STATE, label="TRAIN-COHORT CV")

    final_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    final_clf.fit(Z_train, y_train)
    joblib.dump(final_clf, OUT_CLASSIFIER)
    print(f"saved final classifier to {OUT_CLASSIFIER}")

    # ── evaluate on true held-out test patients ──────────────────────────
    eval_on(model, final_clf, f"HELD-OUT COHORT ({HOLDOUT_COHORT} test)", test_holdout_cohort)
    eval_on(model, final_clf, f"TRAIN COHORT, held-out ({TRAIN_COHORT} test)", test_train_cohort)
    eval_on(model, final_clf, "ALL TEST PATIENTS COMBINED", test_patients)