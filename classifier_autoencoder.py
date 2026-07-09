"""
Positive/negative risk classifier on plain-autoencoder latents.

Same architecture as classifier.py, but uses the plain per-cohort
Autoencoder3D encoders (trained independently, without any
harmonization/MMD loss) instead of the shared HarmonizationModel:
models/best_autoencoder_AUGSBURG.pt and models/best_autoencoder_PRE-RAPID.pt.
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

from autoencoder import Autoencoder3D

PATH = "CUBES-Labelled-COHORTS"
LATENT_DIM = 64

_CHECKPOINT_BY_COHORT = {
    "AUGSBURG": "best_autoencoder_AUGSBURG.pt",
    "PRE-RAPID": "best_autoencoder_PRE-RAPID.pt",
}


# ── model loading ─────────────────────────────────────────────────────────────

def load_cohort_encoders(
    models_dir: Path,
    latent_dim: int = LATENT_DIM,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Autoencoder3D]:
    """Load each cohort's independently-trained plain autoencoder."""
    models = {}
    for cohort, filename in _CHECKPOINT_BY_COHORT.items():
        model = Autoencoder3D(latent_dim=latent_dim)
        model.load_state_dict(torch.load(models_dir / filename, map_location="cpu"))
        model = model.to(device)
        model.eval()
        models[cohort] = model
    return models


# ── latent encoding ───────────────────────────────────────────────────────────

def encode_patients(
    models: dict[str, Autoencoder3D],
    patients: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Encode patients into latent vectors using their cohort's own autoencoder.

    Returns (Z, y): Z is (N, latent_dim), y is (N,) with 0 = negative
    (low risk), 1 = positive (high risk).
    """
    zs, ys = [], []
    with torch.no_grad():
        for patient in patients:
            model = models[patient.cohort]
            vol = (
                torch.from_numpy(patient.pet_masked.astype("float32"))
                .unsqueeze(0).unsqueeze(0)
                .to(device)
            )
            z = model.encoder(vol).squeeze(0).cpu().numpy()
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

    models = load_cohort_encoders(models_dir)

    cohorts = load_all_cohorts(Path(PATH))
    patients = [p for plist in cohorts.values() for p in plist]
    print(f"loaded {len(patients)} patients from {list(cohorts.keys())}")

    Z, y = encode_patients(models, patients)
    print(f"latent matrix {Z.shape}  positives {int(y.sum())}  negatives {int((1 - y).sum())}")

    evaluate_cv(Z, y)

    # final classifier fit on all pooled latents, for downstream use
    final_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    final_clf.fit(Z, y)
    joblib.dump(final_clf, models_dir / "latent_logreg_autoencoder.joblib")
    print(f"saved final classifier to {models_dir / 'latent_logreg_autoencoder.joblib'}")
