"""CNN3 baseline: plain supervised 3D CNN, no harmonization.

The model is trained only on labeled source cohorts. Target-cohort data never
contributes to the loss. Use cnn_3_sweep.py for the standardized 3-cohort
leave-one-target-out evaluation.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset


DATA_PATH = "CUBES-Labelled-COHORTS"
COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "SWISS"]


class CNNClassifier3D(nn.Module):
    def __init__(self, latent_dim: int = 16, dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.dropout1 = nn.Dropout3d(p=dropout * 0.4)
        self.fc = nn.Linear(32 * 4 * 4 * 4, latent_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.clf_head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.dropout1(h)
        h = h.flatten(start_dim=1)
        z = F.relu(self.fc(h))
        z = self.dropout2(z)
        return self.clf_head(z).squeeze(-1)


class LabeledVolumeDataset(Dataset):
    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient = self.patients[idx]
        vol = torch.from_numpy(
            patient.pet_masked.astype("float32")
        ).unsqueeze(0)
        label = torch.tensor(
            patient.label,
            dtype=torch.float32,
        )
        return vol, label


def train_cnn_baseline(
    model: CNNClassifier3D,
    train_patients: list,
    val_patients: list,
    n_epochs: int = 200,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 50,
    checkpoint_path: str | None = None,
) -> CNNClassifier3D:
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    n_pos = sum(p.label for p in train_patients)
    n_neg = len(train_patients) - n_pos
    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)],
        device=device,
    )

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
                f"train_loss {train_loss:.5f}  "
                f"val_loss {val_loss:.5f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            best_state = copy.deepcopy(
                model.state_dict()
            )

            if checkpoint_path is not None:
                torch.save(
                    best_state,
                    checkpoint_path,
                )
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(
                f"no val_loss improvement for {patience} epochs "
                f"(best was {best_val_loss:.5f} at epoch {best_epoch}) "
                f"-- stopping early"
            )
            break

    print(
        f"training done -- best val_loss "
        f"{best_val_loss:.5f} at epoch {best_epoch}"
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_probs(
    model: CNNClassifier3D,
    patients: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()

    probs, ys = [], []

    with torch.no_grad():
        for patient in patients:
            vol = (
                torch.from_numpy(
                    patient.pet_masked.astype("float32")
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            logit = model(vol)

            probs.append(
                torch.sigmoid(logit).item()
            )
            ys.append(patient.label)

    return (
        np.asarray(probs),
        np.asarray(ys),
    )


def eval_on(
    y: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict | None:
    if len(y) == 0:
        return None

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    auc = (
        roc_auc_score(y, y_prob)
        if len(np.unique(y)) > 1
        else float("nan")
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        y_pred,
        labels=[0, 1],
    ).ravel()

    recall_pos = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else float("nan")
    )
    recall_neg = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    return {
        "acc": float(acc),
        "auc": float(auc),
        "recall_pos": float(recall_pos),
        "recall_neg": float(recall_neg),
        "balanced_acc": float(
            np.nanmean([recall_pos, recall_neg])
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
