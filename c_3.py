"""Approach C3 core implementation.

Three independent cohort-specific autoencoders are trained separately.
Every patient is then encoded by all three encoders and the latent vectors
are concatenated for downstream classification.

This module contains the model/training utilities. Use c_3_sweep.py for
the standardized AUGSBURG/SWISS leave-one-cohort-out experiment.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DATA_PATH = "CUBES-Labelled-COHORTS_3"
COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "SWISS"]


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
            nn.Identity(),
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


class VolumeDataset(Dataset):
    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> torch.Tensor:
        vol = self.patients[idx].pet_masked.astype("float32")
        return torch.from_numpy(vol).unsqueeze(0)


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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_loader = DataLoader(
        VolumeDataset(train_patients),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        VolumeDataset(val_patients),
        batch_size=batch_size,
        shuffle=False,
    )

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
            print(
                f"[{label}] epoch {epoch:3d}  "
                f"train_loss {train_loss:.5f}  "
                f"val_loss {val_loss:.5f}"
            )

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
            print(
                f"[{label}] no val_loss improvement for {patience} epochs "
                f"(best was {best_val_loss:.5f} at epoch {best_epoch}) -- stopping early"
            )
            break

    print(
        f"[{label}] training done -- "
        f"best val_loss {best_val_loss:.5f} at epoch {best_epoch}"
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def encode_concat(
    encoders: dict[str, Encoder3D],
    cohort_order: list[str],
    patients: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    for name in cohort_order:
        encoders[name] = encoders[name].to(device).eval()

    zs, ys = [], []

    with torch.no_grad():
        for patient in patients:
            vol = (
                torch.from_numpy(patient.pet_masked.astype("float32"))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            parts = [
                encoders[name](vol).squeeze(0).cpu().numpy()
                for name in cohort_order
            ]

            zs.append(np.concatenate(parts))
            ys.append(patient.label)

    return np.stack(zs), np.array(ys)
