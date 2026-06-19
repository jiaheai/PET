"""
Plain 3D autoencoder for 32x32x32 PET VOIs.

Step 1 of the harmonization pipeline: get a single encoder/decoder
reconstructing volumes well, before adding per-center encoders and
the VAE/alignment loss on top.

Run locally (requires torch + your CUBES-Labelled-COHORTS data).
"""

from __future__ import annotations

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
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True), 

            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # 8 -> 4
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
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 4 -> 8
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 8 -> 16
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 16 -> 32
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
) -> Autoencoder3D:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_loader = DataLoader(VolumeDataset(train_patients), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(VolumeDataset(val_patients), batch_size=batch_size, shuffle=False)

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

        print(f"epoch {epoch:3d}  train_loss {train_loss:.5f}  val_loss {val_loss:.5f}")

    return model


if __name__ == "__main__":
    # Example wiring - adapt paths/imports to your actual project layout.
    #
    # from pathlib import Path
    # from nifti_loader import load_all_cohorts
    # from normalize import zscore_all_cohorts
    # from sklearn.model_selection import train_test_split
    #
    # all_cohorts = load_all_cohorts(Path("CUBES-Labelled-COHORTS"))
    # all_cohorts = zscore_all_cohorts(all_cohorts)
    # all_patients = [p for plist in all_cohorts.values() for p in plist]
    #
    # train_p, val_p = train_test_split(all_patients, test_size=0.2, random_state=42)
    #
    # model = Autoencoder3D(latent_dim=64)
    # model = train_autoencoder(model, train_p, val_p, n_epochs=100)
    pass