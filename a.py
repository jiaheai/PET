from __future__ import annotations

import copy
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader


DATA_PATH = "CUBES-Labelled-COHORTS-ZSCORE"


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


class HarmonizationModel(nn.Module):
    """Shared encoder/decoder across an arbitrary number of cohorts."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder3D(latent_dim)
        self.decoder = Decoder3D(latent_dim)

    def forward(self, xs: list[torch.Tensor]):
        """xs: list of per-cohort batches, same shared encoder/decoder for each.
        Returns (x_hats, zs), both lists in the same cohort order as xs.
        """
        zs = [self.encoder(x) for x in xs]
        x_hats = [self.decoder(z) for z in zs]
        return x_hats, zs

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def set_decoder_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the shared decoder (used for stage-2 encoder-only training)."""
        for p in self.decoder.parameters():
            p.requires_grad = trainable


# -- MMD loss ------------------------------------------------------------

def compute_mmd(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """RBF-kernel MMD between two cohorts' latents (or flattened voxels)."""
    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-gamma * torch.cdist(a, b).pow(2))

    kxx = rbf(x, x).mean()
    kyy = rbf(y, y).mean()
    kxy = rbf(x, y).mean()
    return kxx + kyy - 2 * kxy


def median_heuristic_gamma(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """Median-heuristic RBF bandwidth: gamma = 1 / (2 * median(pairwise sq dist)).
    Computed under no_grad -- chooses a bandwidth constant, not differentiated through.
    """
    pooled = torch.cat([x, y], dim=0)
    with torch.no_grad():
        d2 = torch.cdist(pooled, pooled).pow(2)
        n = d2.size(0)
        off_diag = d2[~torch.eye(n, dtype=torch.bool, device=d2.device)]
        if off_diag.numel() == 0:
            return 1.0
        med = off_diag.median().clamp_min(eps)
        gamma = 1.0 / (2.0 * med)
    return gamma.item()


def compute_mmd_multi(zs: list[torch.Tensor]) -> torch.Tensor:
    """Pairwise MMD across N cohorts, averaged over all C(N,2) pairs.
    Each pair gets its own median-heuristic gamma (pairs can sit at
    different scales in latent space). Averaging (not summing) keeps
    lambda_mmd's effective scale roughly stable as cohort count changes --
    with N=2 there's exactly one pair, so this matches the old 2-cohort
    behavior exactly.
    """
    pairs = list(itertools.combinations(range(len(zs)), 2))
    if not pairs:
        return torch.zeros((), device=zs[0].device)
    total = 0.0
    for i, j in pairs:
        gamma = median_heuristic_gamma(zs[i], zs[j])
        total = total + compute_mmd(zs[i], zs[j], gamma=gamma)
    return total / len(pairs)


def compute_mmd_stratified(*args, **kwargs):
    raise NotImplementedError(
        "compute_mmd_stratified was removed: it required target-cohort labels, "
        "which aren't available at deployment time for a real unlabeled target "
        "cohort. Use compute_mmd_multi(zs) instead -- pooled, no labels needed."
    )


# -- dataset ------------------------------------------------------------

class VolumeDataset(Dataset):
    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> torch.Tensor:
        vol = self.patients[idx].pet_masked.astype("float32")
        return torch.from_numpy(vol).unsqueeze(0)   # (1, 32, 32, 32)


def _zip_cycled(loaders: list[DataLoader]):
    """zip N loaders together, cycling all but the longest so every batch
    from the longest loader gets paired with something from every other
    cohort (generalizes the old 2-loader `if len(a) >= len(b)` cycling)."""
    lengths = [len(l) for l in loaders]
    max_len = max(lengths)
    cycled = [l if len(l) == max_len else itertools.cycle(l) for l in loaders]
    return zip(*cycled)


# -- training loop ------------------------------------------------------------

def train_harmonization(
    model: HarmonizationModel,
    train_cohorts: dict[str, list], val_cohorts: dict[str, list],
    n_epochs: int = 100, batch_size: int = 8, lr: float = 1e-3,
    lambda_mmd: float = 1.0,
    decoder_freeze_epoch: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 200,
    checkpoint_path: str | None = "best_harmonization_shared.pt",
) -> HarmonizationModel:
    """
    train_cohorts / val_cohorts : {cohort_name: [patients]} -- any number of
                         cohorts (2 or more), all sharing one encoder/decoder.
    decoder_freeze_epoch : if set, splits training into two stages:
                         stage 1 (epoch < decoder_freeze_epoch) trains the
                         full model (encoder + decoder) with the mmd term
                         excluded from the loss entirely (lambda_mmd_t=0),
                         i.e. pure reconstruction. At decoder_freeze_epoch
                         the decoder is frozen and mmd turns on at its full
                         lambda_mmd weight; from then on only the (shared)
                         encoder can respond to it, since the decoder it
                         renders through is locked in as whatever it
                         learned to do well in stage 1.
                         None disables this and trains with mmd on at
                         lambda_mmd the whole run, as before.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    decoder_frozen = False

    cohort_names = list(train_cohorts.keys())
    n_pairs = len(list(itertools.combinations(cohort_names, 2)))
    print(f"cohorts      : {cohort_names}  ({n_pairs} pairs for MMD)")
    if decoder_freeze_epoch is not None:
        print(f"decoder freeze at epoch {decoder_freeze_epoch} (stage-2 encoder-only training)")

    train_loaders = [
        DataLoader(VolumeDataset(train_cohorts[name]), batch_size=batch_size, shuffle=True)
        for name in cohort_names
    ]
    val_loaders = [
        DataLoader(VolumeDataset(val_cohorts[name]), batch_size=batch_size, shuffle=False)
        for name in cohort_names
    ]

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        # -- stage-2 switch: freeze decoder, rebuild optimizer over remaining params --
        if decoder_freeze_epoch is not None and epoch == decoder_freeze_epoch and not decoder_frozen:
            model.set_decoder_trainable(False)
            decoder_frozen = True
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)
            # val_loss's scale jumps here (mmd term switches on): reset both
            # the patience counter AND best_val_loss, not just the counter.
            epochs_since_improvement = 0
            best_val_loss = float("inf")
            print(f"epoch {epoch:3d}  -- decoder frozen, mmd turned on (lambda_mmd={lambda_mmd}), optimizing encoder only from here on")

        if decoder_freeze_epoch is not None and epoch < decoder_freeze_epoch:
            lambda_mmd_t = 0.0
        else:
            lambda_mmd_t = lambda_mmd

        # -- training --------------------------------------------------
        model.train()
        train_recon      = 0.0   # mean per-batch, NOT sample-weighted (mse_loss is already mean-reduced)
        train_mmd_latent = 0.0   # drives the loss -- averaged over all cohort pairs
        n_train_batches = 0

        for batches in _zip_cycled(train_loaders):
            batches = [b.to(device) for b in batches]

            optimizer.zero_grad()
            x_hats, zs = model(batches)

            recon = sum(F.mse_loss(xh, x) for xh, x in zip(x_hats, batches))

            # Latent-space MMD drives the loss: aligns the encoder's codes
            # across all cohorts (averaged over every pair) instead of
            # asking different patients' reconstructed pixels to match.
            # Pooled per pair -- NOT label-stratified, since a real
            # deployment-time target cohort won't have labels.
            mmd_latent = compute_mmd_multi(zs)
            loss = recon + lambda_mmd_t * mmd_latent

            loss.backward()
            optimizer.step()

            train_recon      += recon.item()
            train_mmd_latent += mmd_latent.item()
            n_train_batches  += 1

        train_recon      /= max(n_train_batches, 1)
        train_mmd_latent /= max(n_train_batches, 1)
        train_loss = train_recon + lambda_mmd_t * train_mmd_latent

        # -- validation --------------------------------------------------
        model.eval()
        val_recon      = 0.0
        val_mmd_latent = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batches in _zip_cycled(val_loaders):
                batches = [b.to(device) for b in batches]

                x_hats, zs = model(batches)

                recon = sum(F.mse_loss(xh, x) for xh, x in zip(x_hats, batches))
                mmd_latent = compute_mmd_multi(zs)

                val_recon      += recon.item()
                val_mmd_latent += mmd_latent.item()
                n_val_batches  += 1

        val_recon      /= max(n_val_batches, 1)
        val_mmd_latent /= max(n_val_batches, 1)
        val_loss = val_recon + lambda_mmd_t * val_mmd_latent

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:3d}  lambda_mmd {lambda_mmd_t:.4f}  "
                f"train_loss {train_loss:.5f}  (recon {train_recon:.5f}  "
                f"mmd_z {train_mmd_latent:.6f})  "
                f"val_loss {val_loss:.5f}  (recon {val_recon:.5f}  "
                f"mmd_z {val_mmd_latent:.6f})"
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
            print(f"no val_loss improvement for {patience} epochs "
                  f"(best was {best_val_loss:.5f} at epoch {best_epoch}) -- stopping early")
            break

    print(f"training done -- best val_loss {best_val_loss:.5f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -- reconstruction saving ------------------------------------------------------

def save_harmonized_reconstructions(
    model: HarmonizationModel,
    patients: list,
    cohort: str,
    out_root: str | Path = "harmonized_reconstructions_shared",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    import nibabel as nib
    from pathlib import Path as _Path

    out_root = _Path(out_root)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for patient in patients:
            vol = torch.from_numpy(patient.pet_masked.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
            z = model.encode(vol)
            x_hat = model.decoder(z)
            recon = x_hat.squeeze().cpu().numpy()

            out_dir = out_root / cohort
            out_dir.mkdir(parents=True, exist_ok=True)

            nib.save(nib.Nifti1Image(recon, patient.affine),
                      str(out_dir / f"{patient.patient_id}_PET_res_{patient.label}.nii.gz"))
            nib.save(nib.Nifti1Image(patient.mask.astype(np.float32), patient.affine),
                      str(out_dir / f"{patient.patient_id}_prostate_mask_res.nii.gz"))

    print(f"saved {len(patients)} harmonized reconstructions to {out_root / cohort}")


# -- entry point ------------------------------------------------------------
if __name__ == "__main__":
    from nifti_loader import load_all_cohorts
    from sklearn.model_selection import train_test_split

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    all_cohorts = load_all_cohorts(Path(DATA_PATH))   # {cohort_name: [patients]}, any number of cohorts

    train_cohorts: dict[str, list] = {}
    val_cohorts: dict[str, list] = {}

    for name, patients in all_cohorts.items():
        # stratified so val doesn't accidentally end up class-skewed at small n
        train_p, val_p = train_test_split(
            patients, test_size=0.2, random_state=40,
            stratify=[p.label for p in patients],
        )
        train_cohorts[name] = train_p
        val_cohorts[name] = val_p
        print(f"{name:10s}: {len(train_p)} train / {len(val_p)} val  (all {len(patients)} used)")

    torch.manual_seed(41)
    model = HarmonizationModel(latent_dim=64)

    model = train_harmonization(
        model,
        train_cohorts=train_cohorts, val_cohorts=val_cohorts,
        n_epochs=1000,
        lambda_mmd=1,
        decoder_freeze_epoch=100,
        checkpoint_path=str(models_dir / "best_harmonization_shared.pt"),
    )

    for name, patients in all_cohorts.items():
        save_harmonized_reconstructions(model, patients, name)