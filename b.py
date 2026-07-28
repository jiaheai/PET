from __future__ import annotations

import copy
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader


# -- experiment config ------------------------------------------------------------
DATA_PATH          = "CUBES-Labelled-COHORTS"
DOMAIN_SHIFT_GAMMA = 5.3919e-06


# -- building blocks ------------------------------------------------------------

class Encoder3D(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
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
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 64, 4, 4, 4)
        return self.deconv(h)


# -- harmonization model ------------------------------------------------------

class HarmonizationModel(nn.Module):
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
        return self.encoder_aug(x)

    def encode_pr(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_pr(x)


# -- MMD loss ------------------------------------------------------------

def compute_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """RBF-kernel MMD in image space.

    Uses DOMAIN_SHIFT_GAMMA so training MMD is on the same scale as
    the domain_shift.py evaluation metric (baseline: 0.096446).

    x, y: (N, D) and (M, D) -- flattened reconstructed volumes.
    """
    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-DOMAIN_SHIFT_GAMMA * torch.cdist(a, b).pow(2))

    kxx = rbf(x, x).mean()
    kyy = rbf(y, y).mean()
    kxy = rbf(x, y).mean()
    return kxx + kyy - 2 * kxy


# -- dataset ------------------------------------------------------------

class VolumeDataset(Dataset):
    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> torch.Tensor:
        vol = self.patients[idx].pet_masked.astype("float32")
        return torch.from_numpy(vol).unsqueeze(0)   # (1, 32, 32, 32)


# -- training loop ------------------------------------------------------------

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
    patience:      int   = 200,
    checkpoint_path: str | None = "best_harmonization.pt",
) -> HarmonizationModel:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"gamma        : {DOMAIN_SHIFT_GAMMA:.4e}")
    print(f"baseline MMD : 0.096446")

    train_aug_loader = DataLoader(VolumeDataset(train_aug), batch_size=batch_size, shuffle=True)
    train_pr_loader  = DataLoader(VolumeDataset(train_pr),  batch_size=batch_size, shuffle=True)
    val_aug_loader   = DataLoader(VolumeDataset(val_aug),   batch_size=batch_size, shuffle=False)
    val_pr_loader    = DataLoader(VolumeDataset(val_pr),    batch_size=batch_size, shuffle=False)

    best_val_loss        = float("inf")
    best_epoch           = -1
    best_state           = None
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        # -- training --------------------------------------------------
        model.train()
        train_recon     = 0.0
        train_mmd       = 0.0
        n_train         = 0
        n_train_batches = 0

        if len(train_aug_loader) >= len(train_pr_loader):
            train_iter = zip(train_aug_loader, itertools.cycle(train_pr_loader))
        else:
            train_iter = zip(itertools.cycle(train_aug_loader), train_pr_loader)

        for batch_aug, batch_pr in train_iter:
            batch_aug = batch_aug.to(device)
            batch_pr  = batch_pr.to(device)

            optimizer.zero_grad()
            x_hat_aug, x_hat_pr, z_aug, z_pr = model(batch_aug, batch_pr)

            recon = F.mse_loss(x_hat_aug, batch_aug) + F.mse_loss(x_hat_pr, batch_pr)
            mmd   = compute_mmd(
                x_hat_aug.flatten(start_dim=1),
                x_hat_pr.flatten(start_dim=1),
            )
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

        # -- validation --------------------------------------------------
        model.eval()
        val_recon     = 0.0
        val_mmd       = 0.0
        n_val         = 0
        n_val_batches = 0

        if len(val_aug_loader) >= len(val_pr_loader):
            val_iter = zip(val_aug_loader, itertools.cycle(val_pr_loader))
        else:
            val_iter = zip(itertools.cycle(val_aug_loader), val_pr_loader)

        with torch.no_grad():
            for batch_aug, batch_pr in val_iter:
                batch_aug = batch_aug.to(device)
                batch_pr  = batch_pr.to(device)
                x_hat_aug, x_hat_pr, z_aug, z_pr = model(batch_aug, batch_pr)

                recon = F.mse_loss(x_hat_aug, batch_aug) + F.mse_loss(x_hat_pr, batch_pr)
                mmd   = compute_mmd(
                    x_hat_aug.flatten(start_dim=1),
                    x_hat_pr.flatten(start_dim=1),
                )

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
                f"train_loss {train_loss:.5f}  (recon {train_recon:.5f}  mmd {train_mmd:.6f})  "
                f"val_loss {val_loss:.5f}  (recon {val_recon:.5f}  mmd {val_mmd:.6f})"
            )

        if val_loss < best_val_loss:
            best_val_loss            = val_loss
            best_epoch               = epoch
            epochs_since_improvement = 0
            best_state               = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(
                f"no val_loss improvement for {patience} epochs "
                f"(best was {best_val_loss:.5f} at epoch {best_epoch}) -- stopping early"
            )
            break

    print(f"training done -- best val_loss {best_val_loss:.5f} at epoch {best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# -- reconstruction saving ------------------------------------------------------

def save_harmonized_reconstructions(
    model:    HarmonizationModel,
    patients: list,
    cohort:   str,
    out_root: str | Path = "harmonized_reconstructions",
    device:   str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    import nibabel as nib
    from pathlib import Path as _Path

    if cohort == "AUGSBURG":
        encode_fn = model.encode_aug
    elif cohort == "PRE-RAPID":
        encode_fn = model.encode_pr
    else:
        raise ValueError(f"Unknown cohort '{cohort}'. Expected 'AUGSBURG' or 'PRE-RAPID'.")

    out_root = _Path(out_root)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for patient in patients:
            vol = (
                torch.from_numpy(patient.pet_masked.astype("float32"))
                .unsqueeze(0).unsqueeze(0)
                .to(device)
            )
            z     = encode_fn(vol)
            x_hat = model.decoder(z)
            recon = x_hat.squeeze().cpu().numpy()

            out_dir = out_root / cohort
            out_dir.mkdir(parents=True, exist_ok=True)

            nib.save(
                nib.Nifti1Image(recon, patient.affine),
                str(out_dir / f"{patient.patient_id}_PET_res_{patient.label}.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(patient.mask.astype(np.float32), patient.affine),
                str(out_dir / f"{patient.patient_id}_prostate_mask_res.nii.gz"),
            )

    print(f"saved {len(patients)} harmonized reconstructions to {out_root / cohort}")


# -- entry point ------------------------------------------------------------
if __name__ == "__main__":
    from nifti_loader import load_all_cohorts
    from sklearn.model_selection import train_test_split

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    all_cohorts = load_all_cohorts(Path(DATA_PATH))

    aug_patients = all_cohorts["AUGSBURG"]
    pr_patients  = all_cohorts["PRE-RAPID"]

    # stratified so val doesn't accidentally end up class-skewed at n~10
    train_aug, val_aug = train_test_split(
        aug_patients, test_size=0.2, random_state=40,
        stratify=[p.label for p in aug_patients],
    )
    train_pr,  val_pr  = train_test_split(
        pr_patients, test_size=0.2, random_state=40,
        stratify=[p.label for p in pr_patients],
    )

    print(f"AUGSBURG : {len(train_aug)} train / {len(val_aug)} val  (all {len(aug_patients)} used)")
    print(f"PRE-RAPID: {len(train_pr)} train / {len(val_pr)} val  (all {len(pr_patients)} used)")

    torch.manual_seed(41)
    model = HarmonizationModel(latent_dim=64)

    model = train_harmonization(
        model,
        train_aug=train_aug, val_aug=val_aug,
        train_pr=train_pr,   val_pr=val_pr,
        n_epochs=1000,
        lambda_mmd=1,
        checkpoint_path=str(models_dir / "best_harmonization.pt"),
    )
    
    save_harmonized_reconstructions(model, aug_patients, "AUGSBURG",
                                     out_root="harmonized_reconstructions")
    save_harmonized_reconstructions(model, pr_patients, "PRE-RAPID",
                                     out_root="harmonized_reconstructions")