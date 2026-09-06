"""
build 2 autoencoders, then train one classifier on the concat latent space of all augsburg, and test on concat of pre-rapid's TRULY held-out half

Changed from the original version of this baseline: "held out" now means
the same thing it means in a_sweep_multi.py / b_sweep_multi.py. PRE-RAPID
is split with the SAME TARGET_HOLDOUT_SEED into a harmonization half
(which model_pr's autoencoder trains and early-stops on -- unsupervised,
its labels are never used) and a held-out half that NEVER touches
training in any way. Before this fix, model_pr was trained on 100% of
PRE-RAPID and "held out" only meant "labels weren't used" -- not
comparable to the MMD approaches' held-out numbers, where the held-out
half is untouched entirely. Now it is a fair comparison.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH      = "CUBES-Labelled-COHORTS"
TRAIN_COHORT   = "AUGSBURG"
HOLDOUT_COHORT = "PRE-RAPID"

VAL_SPLIT_SEED = 40
TARGET_HOLDOUT_SEED = 123   # same value as a_sweep_multi.py / b_sweep_multi.py --
                            # keeps this baseline's held-out patients identical
                            # to theirs, required for a valid cross-approach comparison


# -- building blocks (same shapes as elsewhere in this project) -------------------

class Encoder3D(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64 * 4 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.fc(h)


class Decoder3D(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 64, 4, 4, 4)
        return self.deconv(h)


class Autoencoder3D(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder3D(latent_dim)
        self.decoder = Decoder3D(latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# -- dataset ------------------------------------------------------------------

class VolumeDataset(Dataset):
    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> torch.Tensor:
        vol = self.patients[idx].pet_masked.astype("float32")
        return torch.from_numpy(vol).unsqueeze(0)


def split_holdout_cohort(patients: list) -> tuple[list, list]:
    """Stratified 50/50 split into a harmonization half (the autoencoder
    trains and early-stops on this, unsupervised -- labels never used) and
    a held-out half that never touches training in any way. Same
    TARGET_HOLDOUT_SEED as a_sweep_multi.py / b_sweep_multi.py's
    split_target_cohort, so all three approaches hold out identical
    patients.
    """
    harmonization_half, heldout_half = train_test_split(
        patients, test_size=0.5, random_state=TARGET_HOLDOUT_SEED,
        stratify=[p.label for p in patients],
    )
    return harmonization_half, heldout_half


# -- training (one independent autoencoder per cohort) ----------------------------

def train_autoencoder(
    model: Autoencoder3D,
    train_patients: list,
    val_patients: list,
    n_epochs: int = 200,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 50,
    checkpoint_path: str | None = None,
    label: str = "",
) -> Autoencoder3D:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(VolumeDataset(train_patients), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(VolumeDataset(val_patients),   batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = F.mse_loss(x_hat, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
            n_train += batch.size(0)
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_hat, _ = model(batch)
                loss = F.mse_loss(x_hat, batch)
                val_loss += loss.item() * batch.size(0)
                n_val += batch.size(0)
        val_loss /= max(n_val, 1)

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"[{label}] epoch {epoch:3d}  train_loss {train_loss:.5f}  val_loss {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"[{label}] no val_loss improvement for {patience} epochs "
                  f"(best was {best_val_loss:.5f} at epoch {best_epoch}) -- stopping early")
            break

    print(f"[{label}] training done -- best val_loss {best_val_loss:.5f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -- concatenated encoding ------------------------------------------------------

def encode_concat(
    encoder_aug: Encoder3D,
    encoder_pr: Encoder3D,
    patients: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Push every patient through BOTH encoders regardless of their own
    cohort, concatenate the two resulting latent vectors.

    Returns (Z, y): Z is (N, 2*latent_dim), y is (N,) labels.
    """
    encoder_aug = encoder_aug.to(device).eval()
    encoder_pr  = encoder_pr.to(device).eval()
    zs, ys = [], []
    with torch.no_grad():
        for p in patients:
            vol = torch.from_numpy(p.pet_masked.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
            z_aug = encoder_aug(vol).squeeze(0).cpu().numpy()
            z_pr  = encoder_pr(vol).squeeze(0).cpu().numpy()
            zs.append(np.concatenate([z_aug, z_pr]))
            ys.append(p.label)
    return np.stack(zs), np.array(ys)


def eval_on(clf, Z, y, label) -> None:
    y_pred = clf.predict(Z)
    y_prob = clf.predict_proba(Z)[:, 1]
    acc  = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    auc  = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    print(f"\n=== {label} ===")
    print(f"n {len(y)}  acc {acc:.3f}  bal. acc {bacc:.3f}  auc {auc:.3f}")
    print(f"recall(pos) {recall_pos:.3f}  recall(neg) {recall_neg:.3f}")
    print(f"confusion matrix  tn {tn}  fp {fp}  fn {fn}  tp {tp}")


# -- entry point ------------------------------------------------------------
if __name__ == "__main__":
    from nifti_loader import load_all_cohorts

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    all_cohorts = load_all_cohorts(Path(DATA_PATH))
    aug_patients = all_cohorts[TRAIN_COHORT]     # all 50 -- fully available, like a/b's "known" cohorts

    # PRE-RAPID is split into a harmonization half (autoencoder trains and
    # early-stops on this, unsupervised) and a held-out half that never
    # touches training -- matching a_sweep_multi.py / b_sweep_multi.py's
    # target-cohort handling, so this baseline's held-out numbers are
    # actually comparable to theirs.
    pr_harmonization, pr_heldout = split_holdout_cohort(all_cohorts[HOLDOUT_COHORT])
    print(f"{HOLDOUT_COHORT}: {len(pr_harmonization)} harmonization (autoencoder-trained) "
          f"/ {len(pr_heldout)} held out (never touches training)")

    # -- train / val split, per cohort, purely for each autoencoder's ------
    # early stopping. AUGSBURG uses all its patients; PRE-RAPID uses only
    # its harmonization half -- pr_heldout is never touched here or below.
    train_aug, val_aug = train_test_split(
        aug_patients, test_size=0.2, random_state=VAL_SPLIT_SEED,
        stratify=[p.label for p in aug_patients],
    )
    train_pr, val_pr = train_test_split(
        pr_harmonization, test_size=0.2, random_state=VAL_SPLIT_SEED,
        stratify=[p.label for p in pr_harmonization],
    )

    print(f"AUGSBURG : {len(train_aug)} train / {len(val_aug)} val  (all {len(aug_patients)} used)")
    print(f"PRE-RAPID: {len(train_pr)} train / {len(val_pr)} val  "
          f"(of {len(pr_harmonization)} harmonization patients; "
          f"{len(pr_heldout)} more held out entirely)")

    # -- train two INDEPENDENT autoencoders, one per cohort -----------------
    torch.manual_seed(41)
    model_aug = Autoencoder3D(latent_dim=64)
    model_aug = train_autoencoder(
        model_aug, train_aug, val_aug, n_epochs=200,
        checkpoint_path=str(models_dir / "best_autoencoder_aug_c.pt"), label="AUGSBURG",
    )

    torch.manual_seed(42)
    model_pr = Autoencoder3D(latent_dim=64)
    model_pr = train_autoencoder(
        model_pr, train_pr, val_pr, n_epochs=200,
        checkpoint_path=str(models_dir / "best_autoencoder_pr_c.pt"), label="PRE-RAPID",
    )

    # -- concatenated latents: every patient through BOTH encoders ----------
    Z_aug_concat, y_aug         = encode_concat(model_aug.encoder, model_pr.encoder, aug_patients)
    Z_pr_heldout_concat, y_pr_h = encode_concat(model_aug.encoder, model_pr.encoder, pr_heldout)
    print(f"\nconcatenated latent shape: {Z_aug_concat.shape}  (should be N x {2*64})")

    # -- classifier: fit on ALL of AUGSBURG's concat latents, test on -------
    # PRE-RAPID's TRULY held-out half only.
    clf_concat = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf_concat.fit(Z_aug_concat, y_aug)

    eval_on(clf_concat, Z_pr_heldout_concat, y_pr_h,
            f"HELD-OUT COHORT ({HOLDOUT_COHORT}, held-out half, concatenated latents)")
    eval_on(clf_concat, Z_aug_concat, y_aug,
            f"TRAIN COHORT, in-sample ({TRAIN_COHORT}, concatenated latents, all patients)")