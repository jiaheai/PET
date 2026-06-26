"""
3D autoencoder with MMD-based domain harmonization for 32x32x32 PET VOIs.

Step 2 of the harmonization pipeline: two per-site encoders (AUGSBURG,
PRE-RAPID) feeding into a shared decoder, with an MMD alignment loss
pushing the two cohorts' latent distributions together.

Architecture
------------
  encoder_aug  ──┐
                 ├──► shared decoder ──► x_hat_{aug,pr}
  encoder_pr   ──┘
       │
       └──► MMD loss (z_aug vs z_pr)

Loss
----
  total = MSE(x_hat_aug, x_aug) + MSE(x_hat_pr, x_pr) + lambda_mmd * MMD(z_aug, z_pr)

The single shared decoder forces both encoders to map into a compatible
latent space; the MMD term explicitly aligns their distributions.

Run locally (requires torch + your CUBES-Labelled-COHORTS data).
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── building blocks ───────────────────────────────────────────────────────────

class Encoder3D(nn.Module):
    """Shared encoder architecture — instantiated once per cohort."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # 32^3 -> 16^3 -> 8^3 -> 4^3
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),   # 32 -> 16
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64 * 4 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.fc(h)


class Decoder3D(nn.Module):
    """Single shared decoder — reconstructs from the aligned latent space."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),   # 16 -> 32
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 64, 4, 4, 4)
        return self.deconv(h)


# ── harmonization model ───────────────────────────────────────────────────────

class HarmonizationModel(nn.Module):
    """Dual-encoder, shared-decoder harmonization autoencoder.

    Each cohort gets its own encoder (learns site-specific -> shared mapping).
    One decoder reconstructs from the shared latent space for both cohorts.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder_aug = Encoder3D(latent_dim)
        self.encoder_pr  = Encoder3D(latent_dim)
        self.decoder     = Decoder3D(latent_dim)

    def forward(
        self,
        x_aug: torch.Tensor,
        x_pr:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_aug = self.encoder_aug(x_aug)
        z_pr  = self.encoder_pr(x_pr)
        x_hat_aug = self.decoder(z_aug)
        x_hat_pr  = self.decoder(z_pr)
        return x_hat_aug, x_hat_pr, z_aug, z_pr

    def encode_aug(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single AUGSBURG volume (inference helper)."""
        return self.encoder_aug(x)

    def encode_pr(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single PRE-RAPID volume (inference helper)."""
        return self.encoder_pr(x)


# ── MMD loss ──────────────────────────────────────────────────────────────────

def compute_mmd(
    z_aug: torch.Tensor,
    z_pr:  torch.Tensor,
    sigma: float | None = None,
) -> torch.Tensor:
    """RBF-kernel Maximum Mean Discrepancy between two sets of latent vectors.

    Args:
        z_aug: (N, latent_dim) latent vectors from AUGSBURG encoder.
        z_pr:  (M, latent_dim) latent vectors from PRE-RAPID encoder.
        sigma: RBF bandwidth. If None, uses the median heuristic (median
               pairwise distance across both sets combined) — a standard,
               data-adaptive choice.

    Returns:
        Scalar MMD² estimate. Zero when distributions are identical;
        positive otherwise. Differentiable w.r.t. both z_aug and z_pr.
    """
    if sigma is None:
        # Median heuristic: set sigma to median pairwise distance
        # across the combined set of latent vectors.
        combined = torch.cat([z_aug, z_pr], dim=0)
        dists = torch.cdist(combined, combined)
        sigma = dists.median().item()
        sigma = max(sigma, 1e-6)   # guard against degenerate case

    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.cdist(a, b).pow(2) / (2 * sigma ** 2))

    kxx = rbf(z_aug, z_aug).mean()
    kyy = rbf(z_pr,  z_pr ).mean()
    kxy = rbf(z_aug, z_pr ).mean()
    return kxx + kyy - 2 * kxy


# ── dataset ───────────────────────────────────────────────────────────────────

class VolumeDataset(Dataset):
    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> torch.Tensor:
        vol = self.patients[idx].pet_masked.astype("float32")
        return torch.from_numpy(vol).unsqueeze(0)   # (1, 32, 32, 32)


# ── training loop ─────────────────────────────────────────────────────────────

def train_harmonization(
    model:         HarmonizationModel,
    train_aug:     list,
    val_aug:       list,
    train_pr:      list,
    val_pr:        list,
    n_epochs:      int   = 100,
    batch_size:    int   = 8,
    lr:            float = 1e-3,
    lambda_mmd:    float = 1.0,
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu",
    patience:      int   = 35,
    checkpoint_path: str | None = "best_harmonization.pt",
) -> HarmonizationModel:
    """Train the dual-encoder harmonization model.

    Each iteration samples one batch from each cohort, computes:
      - reconstruction loss for both cohorts
      - MMD loss between the two cohorts' latent vectors
      - combined loss = recon + lambda_mmd * MMD

    lambda_mmd controls the reconstruction/alignment tradeoff:
      - too high  → encoders collapse to one point (MMD≈0, reconstruction fails)
      - too low   → alignment pressure too weak, cohorts stay separated
      - start at 1.0 and tune based on printed MMD vs recon components

    zip() over the two loaders stops at the shorter one (PRE-RAPID).
    AUGSBURG's excess batches are skipped each epoch — this is intentional
    to keep paired (aug, pr) batches for the MMD term. An alternative is
    to cycle the shorter loader; see comment below if you want that behaviour.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_aug_loader = DataLoader(VolumeDataset(train_aug), batch_size=batch_size, shuffle=True)
    train_pr_loader  = DataLoader(VolumeDataset(train_pr),  batch_size=batch_size, shuffle=True)
    val_aug_loader   = DataLoader(VolumeDataset(val_aug),   batch_size=batch_size, shuffle=False)
    val_pr_loader    = DataLoader(VolumeDataset(val_pr),    batch_size=batch_size, shuffle=False)

    # To cycle the shorter loader instead of stopping early each epoch,
    # replace zip(...) with zip(train_aug_loader, itertools.cycle(train_pr_loader))
    # (and import itertools at the top). This uses all of AUGSBURG each epoch
    # but sees PRE-RAPID patients more than once per epoch.

    best_val_loss        = float("inf")
    best_epoch           = -1
    best_state           = None
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        # ── training ─────────────────────────────────────────────────
        model.train()
        train_recon  = 0.0
        train_mmd    = 0.0
        n_train      = 0
        n_train_batches = 0

        for batch_aug, batch_pr in zip(train_aug_loader, itertools.cycle(train_pr_loader)):
            batch_aug = batch_aug.to(device)
            batch_pr  = batch_pr.to(device)

            optimizer.zero_grad()
            x_hat_aug, x_hat_pr, z_aug, z_pr = model(batch_aug, batch_pr)

            recon = F.mse_loss(x_hat_aug, batch_aug) + F.mse_loss(x_hat_pr, batch_pr)
            mmd   = compute_mmd(z_aug, z_pr)
            loss  = recon + lambda_mmd * mmd

            loss.backward()
            optimizer.step()

            n = batch_aug.size(0) + batch_pr.size(0)
            train_recon += recon.item() * n
            train_mmd   += mmd.item()
            n_train     += n
            n_train_batches += 1

        train_recon /= max(n_train, 1)
        train_mmd   /= max(n_train_batches, 1)
        train_loss   = train_recon + lambda_mmd * train_mmd

        # ── validation ───────────────────────────────────────────────
        model.eval()
        val_recon  = 0.0
        val_mmd    = 0.0
        n_val      = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch_aug, batch_pr in zip(val_aug_loader, val_pr_loader):
                batch_aug = batch_aug.to(device)
                batch_pr  = batch_pr.to(device)
                x_hat_aug, x_hat_pr, z_aug, z_pr = model(batch_aug, batch_pr)

                recon = F.mse_loss(x_hat_aug, batch_aug) + F.mse_loss(x_hat_pr, batch_pr)
                mmd   = compute_mmd(z_aug, z_pr)

                n = batch_aug.size(0) + batch_pr.size(0)
                val_recon += recon.item() * n
                val_mmd   += mmd.item()
                n_val     += n
                n_val_batches += 1

        val_recon /= max(n_val, 1)
        val_mmd   /= max(n_val_batches, 1)
        val_loss   = val_recon + lambda_mmd * val_mmd

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:3d}  "
                f"train_loss {train_loss:.5f}  (recon {train_recon:.5f}  mmd {train_mmd:.5f})  "
                f"val_loss {val_loss:.5f}  (recon {val_recon:.5f}  mmd {val_mmd:.5f})"
            )

        # ── checkpointing ─────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss            = val_loss
            best_epoch               = epoch
            epochs_since_improvement = 0
            best_state               = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_since_improvement += 1

        # ── early stopping ────────────────────────────────────────────
        if epochs_since_improvement >= patience:
            print(
                f"no val_loss improvement for {patience} epochs "
                f"(best was {best_val_loss:.5f} at epoch {best_epoch}) — stopping early"
            )
            break

    print(f"training done — best val_loss {best_val_loss:.5f} at epoch {best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from nifti_loader import load_all_cohorts
    from sklearn.model_selection import train_test_split

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    all_cohorts = load_all_cohorts(Path("CUBES-Labelled-COHORTS"))

    aug_patients = all_cohorts["AUGSBURG"]
    pr_patients  = all_cohorts["PRE-RAPID"]

    train_aug, val_aug = train_test_split(aug_patients, test_size=0.2, random_state=42)
    train_pr,  val_pr  = train_test_split(pr_patients,  test_size=0.2, random_state=42)

    print(f"AUGSBURG : {len(train_aug)} train / {len(val_aug)} val")
    print(f"PRE-RAPID: {len(train_pr)} train / {len(val_pr)} val")

    torch.manual_seed(42)
    model = HarmonizationModel(latent_dim=64)

    model = train_harmonization(
        model,
        train_aug=train_aug, val_aug=val_aug,
        train_pr=train_pr,   val_pr=val_pr,
        n_epochs=100,
        lambda_mmd=1.0,   # tune this — start at 1.0, watch recon vs mmd components
        checkpoint_path=str(models_dir / "best_harmonization.pt"),
    )