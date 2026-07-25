"""
CNN baseline: plain end-to-end supervised classifier, no harmonization.

Trains a single CNN directly on AUGSBURG's labels (no MMD, no shared/
dual encoders, no cross-cohort alignment mechanism at all), then applies
it directly to PRE-RAPID's raw volumes. This is the no-harmonization
baseline that any harmonization approach (A/B/C) needs to beat to
justify its added complexity.

Same train/test cohort rule as everything else in this project: AUGSBURG
labels only ever touch this model's training; PRE-RAPID is evaluated
once, at the end, never contributing to any loss.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = "CUBES-Labelled-COHORTS"
TRAIN_COHORT   = "AUGSBURG"
HOLDOUT_COHORT = "PRE-RAPID"


# -- model ------------------------------------------------------------------

class CNNClassifier3D(nn.Module):
    def __init__(self, latent_dim: int = 16, dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=4, stride=2, padding=1),    # 32 -> 16
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d(8, 16, kernel_size=4, stride=2, padding=1),   # 16 -> 8
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.dropout1 = nn.Dropout3d(p=dropout * 0.4)
        self.fc       = nn.Linear(32 * 4 * 4 * 4, latent_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.clf_head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.dropout1(h)
        h = h.flatten(start_dim=1)
        z = F.relu(self.fc(h))
        z = self.dropout2(z)
        logit = self.clf_head(z).squeeze(-1)
        return logit


# -- dataset ------------------------------------------------------------------

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


# -- training loop ------------------------------------------------------------

def train_cnn_baseline(
    model:           CNNClassifier3D,
    train_patients:  list,
    val_patients:    list,
    n_epochs:        int   = 200,
    batch_size:      int   = 8,
    lr:              float = 1e-3,
    weight_decay:    float = 1e-3,
    device:          str   = "cuda" if torch.cuda.is_available() else "cpu",
    patience:        int   = 50,
    checkpoint_path: str | None = "best_cnn_baseline.pt",
) -> CNNClassifier3D:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(LabeledVolumeDataset(train_patients), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(LabeledVolumeDataset(val_patients),   batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_epoch    = -1
    best_state    = None
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for vols, labels in train_loader:
            vols, labels = vols.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(vols)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
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
                vols, labels = vols.to(device), labels.to(device)
                logits = model(vols)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                val_loss += loss.item() * vols.size(0)
                n_val += vols.size(0)
        val_loss /= max(n_val, 1)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:3d}  train_loss {train_loss:.5f}  val_loss {val_loss:.5f}")

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
            print(f"no val_loss improvement for {patience} epochs "
                  f"(best was {best_val_loss:.5f} at epoch {best_epoch}) -- stopping early")
            break

    print(f"training done -- best val_loss {best_val_loss:.5f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -- evaluation ------------------------------------------------------------------

def eval_on(model: CNNClassifier3D, label: str, plist: list,
            device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    if not plist:
        print(f"\n=== {label} ===\n(no patients)")
        return {}
    model = model.to(device)
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for p in plist:
            vol = torch.from_numpy(p.pet_masked.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(vol)
            probs.append(torch.sigmoid(logit).item())
            ys.append(p.label)
    y = np.array(ys)
    y_prob = np.array(probs)
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    auc  = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print(f"\n=== {label} ===")
    print(f"n {len(plist)}  acc {acc:.3f}  bal. acc {bacc:.3f}  auc {auc:.3f}")
    print(f"recall(pos) {recall_pos:.3f}  recall(neg) {recall_neg:.3f}")
    print(f"confusion matrix  tn {tn}  fp {fp}  fn {fn}  tp {tp}")

    return {"acc": acc, "bacc": bacc, "auc": auc,
            "recall_pos": recall_pos, "recall_neg": recall_neg,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# -- entry point ------------------------------------------------------------
if __name__ == "__main__":
    from nifti_loader import load_all_cohorts

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    all_cohorts = load_all_cohorts(Path(DATA_PATH))
    aug_patients = all_cohorts[TRAIN_COHORT]
    pr_patients  = all_cohorts[HOLDOUT_COHORT]

    train_aug, val_aug = train_test_split(aug_patients, test_size=0.2, random_state=40)
    print(f"AUGSBURG: {len(train_aug)} train / {len(val_aug)} val  (all {len(aug_patients)} used)")
    print(f"PRE-RAPID: held out entirely until final evaluation ({len(pr_patients)} patients)")

    torch.manual_seed(41)
    model = CNNClassifier3D(latent_dim=16)

    model = train_cnn_baseline(
        model,
        train_patients=train_aug, val_patients=val_aug,
        n_epochs=200,
        checkpoint_path=str(models_dir / "best_cnn_baseline.pt"),
    )

    eval_on(model, f"HELD-OUT COHORT ({HOLDOUT_COHORT}, all patients)", pr_patients)
    eval_on(model, f"TRAIN COHORT, in-sample ({TRAIN_COHORT}, all patients)", aug_patients)