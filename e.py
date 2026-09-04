from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class CNNClassifier3D(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        encoder_dropout: float = 0.2,
        classifier_dropout: float = 0.2,
    ):
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
        self.encoder_dropout = nn.Dropout3d(p=encoder_dropout)
        self.fc = nn.Linear(64 * 4 * 4 * 4, latent_dim)
        self.classifier_dropout = nn.Dropout(p=classifier_dropout)
        self.clf_head = nn.Linear(latent_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.encoder_dropout(h)
        h = h.flatten(start_dim=1)
        return self.fc(h)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        z = self.classifier_dropout(z)
        return self.clf_head(z).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x))


class LabeledVolumeDataset(Dataset):
    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.patients[idx]
        vol = torch.from_numpy(p.pet_masked.astype("float32")).unsqueeze(0)
        label = torch.tensor(p.label, dtype=torch.float32)
        return vol, label


def train_cnn_baseline(
    model: CNNClassifier3D,
    train_patients: list,
    val_patients: list,
    n_epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 50,
    checkpoint_path: str | None = None,
) -> CNNClassifier3D:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    n_pos = sum(p.label for p in train_patients)
    n_neg = len(train_patients) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    train_loader = DataLoader(
        LabeledVolumeDataset(train_patients),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        LabeledVolumeDataset(val_patients),
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

        for vols, labels in train_loader:
            vols = vols.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(vols)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=pos_weight,
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * vols.size(0)
            n_train += vols.size(0)

        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for vols, labels in val_loader:
                vols = vols.to(device)
                labels = labels.to(device)
                logits = model(vols)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    pos_weight=pos_weight,
                )
                val_loss += loss.item() * vols.size(0)
                n_val += vols.size(0)

        val_loss /= max(n_val, 1)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:3d}  "
                f"train_loss {train_loss:.5f}  val_loss {val_loss:.5f}"
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
                f"no val_loss improvement for {patience} epochs "
                f"(best was {best_val_loss:.5f} at epoch {best_epoch}) -- stopping early"
            )
            break

    print(
        f"training done -- best val_loss {best_val_loss:.5f} "
        f"at epoch {best_epoch}"
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
