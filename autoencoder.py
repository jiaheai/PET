"""
Plain 3D autoencoder for 32x32x32 PET VOIs.

Step 1 of the harmonization pipeline: get a single encoder/decoder
reconstructing volumes well, before adding per-center encoders and
the VAE/alignment loss on top.

Run locally (requires torch + your CUBES-Labelled-COHORTS data).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── model ────────────────────────────────────────────────────────────────────

class Encoder3D(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # 32^3 -> 16^3 -> 8^3 -> 4^3
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),  # 32 -> 16
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
        # x: (B, 1, 32, 32, 32)
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        z = self.fc(h)
        return z


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
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 64, 4, 4, 4)
        x_hat = self.deconv(h)
        return x_hat

class Autoencoder3D(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder3D(latent_dim)
        self.decoder = Decoder3D(latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ── dataset ──────────────────────────────────────────────────────────────────

class VolumeDataset(Dataset):
    """Wraps a list of PatientVolumes for PyTorch training.

    Expects each volume already masked / normalised as desired upstream
    (e.g. via your normalize.zscore_patient).
    """

    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> torch.Tensor:
        vol = self.patients[idx].pet_masked.astype("float32")
        return torch.from_numpy(vol).unsqueeze(0)  # add channel dim -> (1, 32, 32, 32)


# ── training loop ────────────────────────────────────────────────────────────

def train_autoencoder(
    model: Autoencoder3D,
    train_patients: list,
    val_patients: list,
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 35,
    checkpoint_path: str | None = "best_autoencoder.pt",
) -> Autoencoder3D:
    """Train the autoencoder, tracking the best validation loss seen.

    Adds two things the original loop didn't have:
      - checkpointing: whenever val_loss improves, the current model
        weights are saved (both in memory and, if checkpoint_path is
        given, to disk via torch.save).
      - early stopping: if val_loss hasn't improved for `patience`
        epochs in a row, training stops early instead of running the
        full n_epochs regardless.

    Returns the BEST model seen (by val_loss), not necessarily the
    model from the final epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_loader = DataLoader(VolumeDataset(train_patients), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(VolumeDataset(val_patients), batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = F.mse_loss(x_hat, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_patients)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_hat, _ = model(batch)
                loss = F.mse_loss(x_hat, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= max(len(val_patients), 1)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:3d}  train_loss {train_loss:.5f}  val_loss {val_loss:.5f}")

        # ── checkpointing ────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            # Keep a copy of the weights in memory.
            best_state = copy.deepcopy(model.state_dict())
            # Optionally also persist to disk, so the best model survives
            # even if the process crashes or you want it later without
            # re-running training.
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_since_improvement += 1

        # ── early stopping ───────────────────────────────────────────
        if epochs_since_improvement >= patience:
            print(
                f"no val_loss improvement for {patience} epochs "
                f"(best was {best_val_loss:.5f} at epoch {best_epoch}) — stopping early"
            )
            break

    print(f"training done — best val_loss {best_val_loss:.5f} at epoch {best_epoch}")

    # Load the best weights back into the model before returning, so the
    # caller gets the best-validated model, not whatever the final epoch
    # happened to leave behind.
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ── reconstruction ───────────────────────────────────────────────────────────

def save_reconstructions(
    model: Autoencoder3D,
    patients: list,
    out_root: str | Path = "reconstructions",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Run all patients through the trained model and save outputs as NIfTI.

    Output layout mirrors the source data:
        reconstructions/{cohort}/{patient_id}_PET_reconstructed.nii.gz
    """
    import nibabel as nib
    from pathlib import Path as _Path

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
            x_hat, _ = model(vol)
            recon = x_hat.squeeze().cpu().numpy()

            out_dir = out_root / patient.cohort
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{patient.patient_id}_PET_reconstructed.nii.gz"
            nib.save(nib.Nifti1Image(recon, patient.affine), str(out_path))
            # print(f"saved {out_path}")


if __name__ == "__main__":
    from pathlib import Path
    from nifti_loader import load_all_cohorts, PatientVolumes
    from sklearn.model_selection import train_test_split

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    all_cohorts = load_all_cohorts(Path("CUBES-Labelled-COHORTS"))

    for cohort_name, patients in all_cohorts.items():
        print(f"\n=== {cohort_name} ({len(patients)} patients) ===")
        train_p, val_p = train_test_split(patients, test_size=0.2, random_state=42)
        print(f"train: {len(train_p)}  val: {len(val_p)}")
        torch.manual_seed(42)

        model = Autoencoder3D(latent_dim=128)
        model = train_autoencoder(
            model, train_p, val_p, n_epochs=100,
            checkpoint_path=str(models_dir / f"best_autoencoder_{cohort_name}.pt"),
        )

        save_reconstructions(model, patients, out_root=Path("reconstructions"))