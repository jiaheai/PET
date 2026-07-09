"""
Positive/negative risk classifier on harmonized PET reconstructions.

Pipeline: encode each patient's harmonized volume into the 64-dim latent
space with the trained HarmonizationModel encoder for its cohort
(encode_aug for AUGSBURG, encode_pr for PRE-RAPID), pool latents from
both cohorts, then fit logistic regression for positive (1 = high risk)
vs negative (0 = low risk) classification.
"""

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

PATH = "CUBES-Labelled-COHORTS"

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

def evaluate_cv(Z: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> None:
    """Stratified k-fold CV of a StandardScaler + LogisticRegression pipeline.

    Scaler and classifier are refit per fold to avoid leakage from the
    held-out fold into the training statistics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accs, aucs = [], []
    tn = fp = fn = tp = 0

    for fold, (train_idx, test_idx) in enumerate(skf.split(Z, y)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(Z[train_idx], y[train_idx])

        y_pred = clf.predict(Z[test_idx])
        y_prob = clf.predict_proba(Z[test_idx])[:, 1]

        acc = accuracy_score(y[test_idx], y_pred)
        auc = roc_auc_score(y[test_idx], y_prob)
        accs.append(acc)
        aucs.append(auc)

        tn_, fp_, fn_, tp_ = confusion_matrix(y[test_idx], y_pred, labels=[0, 1]).ravel()
        tn += tn_; fp += fp_; fn += fn_; tp += tp_

        print(f"fold {fold}  n_test {len(test_idx)}  acc {acc:.3f}  auc {auc:.3f}")

    print(f"\nmean acc {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"mean auc {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"pooled confusion matrix  tn {tn}  fp {fp}  fn {fn}  tp {tp}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from nifti_loader import load_all_cohorts

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model = HarmonizationModel(latent_dim=64)
    model.load_state_dict(torch.load(models_dir / "best_harmonization.pt", map_location="cpu"))
    model.eval()

    cohorts = load_all_cohorts(Path(PATH))
    patients = [p for plist in cohorts.values() for p in plist]
    print(f"loaded {len(patients)} patients from {list(cohorts.keys())}")

    Z, y = encode_patients(model, patients)
    print(f"latent matrix {Z.shape}  positives {int(y.sum())}  negatives {int((1 - y).sum())}")

    evaluate_cv(Z, y)

    # final classifier fit on all pooled latents, for downstream use
    final_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    final_clf.fit(Z, y)
    joblib.dump(final_clf, models_dir / "latent_logreg.joblib")
    print(f"saved final classifier to {models_dir / 'latent_logreg.joblib'}")
