from __future__ import annotations

import copy
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader


DATA_PATH          = "CUBES-Labelled-COHORTS-ZSCORE"
DOMAIN_SHIFT_GAMMA = 5.3919e-06


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
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder3D(latent_dim)
        self.decoder = Decoder3D(latent_dim)

    def forward(self, x_aug: torch.Tensor, x_pr: torch.Tensor):
        z_aug = self.encoder(x_aug)
        z_pr  = self.encoder(x_pr)
        x_hat_aug = self.decoder(z_aug)
        x_hat_pr  = self.decoder(z_pr)
        return x_hat_aug, x_hat_pr, z_aug, z_pr

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_aug(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_pr(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def set_decoder_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the shared decoder (used for stage-2 encoder-only training)."""
        for p in self.decoder.parameters():
            p.requires_grad = trainable


# -- MMD loss ------------------------------------------------------------

def compute_mmd(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """RBF-kernel MMD.

    x, y: (N, D) and (M, D) -- flattened tensors (image-space voxels or
    latent codes, caller's choice; gamma must match the scale of whichever
    is passed in).
    """
    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-gamma * torch.cdist(a, b).pow(2))

    kxx = rbf(x, x).mean()
    kyy = rbf(y, y).mean()
    kxy = rbf(x, y).mean()
    return kxx + kyy - 2 * kxy


def median_heuristic_gamma(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """Median-heuristic RBF bandwidth: gamma = 1 / (2 * median(pairwise sq dist)).

    DOMAIN_SHIFT_GAMMA is a fixed constant tuned for image-space voxel scale
    (to match domain_shift.py). Latent codes have no such fixed, known scale
    -- raw nn.Linear output, unbounded and free to drift during training --
    so a fixed gamma there would give a kernel with no signal at either
    extreme. The median heuristic instead reads the bandwidth off the
    actual batch each step, adapting as the latent scale shifts.

    Computed under no_grad -- this only chooses a bandwidth constant, it
    isn't meant to be differentiated through.
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


def compute_mmd_stratified(
    x: torch.Tensor,
    y: torch.Tensor,
    labels_x: torch.Tensor,
    labels_y: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Label-stratified MMD: averages MMD within each shared label instead of
    matching the pooled batch distribution. Prevents the model from closing
    the aug/pr gap by blurring label-specific features when the two cohorts
    have different label mixes.

    gamma is computed once from the full (unstratified) batch by the caller
    and reused across all label subsets here -- estimating it separately per
    label would be noisy with only a handful of samples per label per batch.

    Falls back to 0.0 (not the pooled MMD) if no label has >=2 samples on
    both sides in this batch.
    """
    shared_labels = set(labels_x.tolist()) & set(labels_y.tolist())
    per_label = []
    for lbl in shared_labels:
        xi = x[labels_x == lbl]
        yi = y[labels_y == lbl]
        if xi.size(0) < 2 or yi.size(0) < 2:
            continue
        per_label.append(compute_mmd(xi, yi, gamma))
    if not per_label:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return torch.stack(per_label).mean()


# -- dataset ------------------------------------------------------------

class VolumeDataset(Dataset):
    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient = self.patients[idx]
        vol = patient.pet_masked.astype("float32")
        label = torch.tensor(int(patient.label), dtype=torch.long)
        return torch.from_numpy(vol).unsqueeze(0), label   # (1, 32, 32, 32), scalar


# -- training loop ------------------------------------------------------------

def train_harmonization(
    model: HarmonizationModel,
    train_aug: list, val_aug: list, train_pr: list, val_pr: list,
    n_epochs: int = 100, batch_size: int = 8, lr: float = 1e-3,
    lambda_mmd: float = 1.0,
    decoder_freeze_epoch: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 200,
    checkpoint_path: str | None = "best_harmonization_shared.pt",
) -> HarmonizationModel:
    """
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

    print(f"gamma        : {DOMAIN_SHIFT_GAMMA:.4e}   (image-space, informational only -- not used in the loss)")
    print(f"baseline MMD : 0.096446")
    if decoder_freeze_epoch is not None:
        print(f"decoder freeze at epoch {decoder_freeze_epoch} (stage-2 encoder-only training)")

    train_aug_loader = DataLoader(VolumeDataset(train_aug), batch_size=batch_size, shuffle=True)
    train_pr_loader  = DataLoader(VolumeDataset(train_pr),  batch_size=batch_size, shuffle=True)
    val_aug_loader   = DataLoader(VolumeDataset(val_aug),   batch_size=batch_size, shuffle=False)
    val_pr_loader    = DataLoader(VolumeDataset(val_pr),    batch_size=batch_size, shuffle=False)

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
            # Otherwise every stage-2 val_loss (which now includes a
            # non-negative mmd term) is compared against stage-1's best
            # (pure recon, no mmd added) and can essentially never win --
            # the final checkpoint would silently stay pinned to a
            # pre-freeze, pre-harmonization snapshot no matter how long
            # stage 2 runs.
            epochs_since_improvement = 0
            best_val_loss = float("inf")
            print(f"epoch {epoch:3d}  -- decoder frozen, mmd turned on (lambda_mmd={lambda_mmd}), optimizing encoder only from here on")

        # Stage 1 (epoch < decoder_freeze_epoch): pure reconstruction, mmd
        # term excluded from the loss entirely -- not just small, exactly 0.
        # Stage 2 (epoch >= decoder_freeze_epoch, decoder frozen): mmd turns
        # on at its full target weight and only the encoder can respond to it.
        # If decoder_freeze_epoch is None, mmd is on at lambda_mmd the whole run.
        if decoder_freeze_epoch is not None and epoch < decoder_freeze_epoch:
            lambda_mmd_t = 0.0
        else:
            lambda_mmd_t = lambda_mmd

        # -- training --------------------------------------------------
        model.train()
        train_recon      = 0.0
        train_mmd_latent = 0.0   # drives the loss
        train_mmd_image  = 0.0   # logged only, for domain_shift.py comparability
        n_train = 0
        n_train_batches = 0

        if len(train_aug_loader) >= len(train_pr_loader):
            train_iter = zip(train_aug_loader, itertools.cycle(train_pr_loader))
        else:
            train_iter = zip(itertools.cycle(train_aug_loader), train_pr_loader)

        for (batch_aug, label_aug), (batch_pr, label_pr) in train_iter:
            batch_aug = batch_aug.to(device)
            batch_pr  = batch_pr.to(device)
            label_aug = label_aug.to(device)
            label_pr  = label_pr.to(device)

            optimizer.zero_grad()
            x_hat_aug, x_hat_pr, z_aug, z_pr = model(batch_aug, batch_pr)

            recon = F.mse_loss(x_hat_aug, batch_aug) + F.mse_loss(x_hat_pr, batch_pr)

            # Latent-space MMD drives the loss: aligns the encoder's codes
            # for the two cohorts instead of asking two different patients'
            # reconstructed pixels to match.
            latent_gamma = median_heuristic_gamma(z_aug, z_pr)
            mmd_latent = compute_mmd_stratified(
                z_aug, z_pr, label_aug, label_pr, gamma=latent_gamma,
            )
            loss = recon + lambda_mmd_t * mmd_latent

            loss.backward()
            optimizer.step()

            # Image-space MMD: logged only (not in loss), kept purely so the
            # printed number stays in the same units as domain_shift.py's
            # metric -- it does NOT receive gradients or affect training.
            with torch.no_grad():
                mmd_image = compute_mmd_stratified(
                    x_hat_aug.flatten(start_dim=1),
                    x_hat_pr.flatten(start_dim=1),
                    label_aug, label_pr, gamma=DOMAIN_SHIFT_GAMMA,
                )

            n = batch_aug.size(0) + batch_pr.size(0)
            train_recon      += recon.item() * n
            train_mmd_latent += mmd_latent.item()
            train_mmd_image  += mmd_image.item()
            n_train     += n
            n_train_batches += 1

        train_recon      /= max(n_train, 1)
        train_mmd_latent /= max(n_train_batches, 1)
        train_mmd_image  /= max(n_train_batches, 1)
        train_loss = train_recon + lambda_mmd_t * train_mmd_latent

        # -- validation --------------------------------------------------
        model.eval()
        val_recon      = 0.0
        val_mmd_latent = 0.0
        val_mmd_image  = 0.0
        n_val = 0
        n_val_batches = 0

        if len(val_aug_loader) >= len(val_pr_loader):
            val_iter = zip(val_aug_loader, itertools.cycle(val_pr_loader))
        else:
            val_iter = zip(itertools.cycle(val_aug_loader), val_pr_loader)

        with torch.no_grad():
            for (batch_aug, label_aug), (batch_pr, label_pr) in val_iter:
                batch_aug = batch_aug.to(device)
                batch_pr  = batch_pr.to(device)
                label_aug = label_aug.to(device)
                label_pr  = label_pr.to(device)

                x_hat_aug, x_hat_pr, z_aug, z_pr = model(batch_aug, batch_pr)

                recon = F.mse_loss(x_hat_aug, batch_aug) + F.mse_loss(x_hat_pr, batch_pr)
                latent_gamma = median_heuristic_gamma(z_aug, z_pr)
                mmd_latent = compute_mmd_stratified(
                    z_aug, z_pr, label_aug, label_pr, gamma=latent_gamma,
                )
                mmd_image = compute_mmd_stratified(
                    x_hat_aug.flatten(start_dim=1),
                    x_hat_pr.flatten(start_dim=1),
                    label_aug, label_pr, gamma=DOMAIN_SHIFT_GAMMA,
                )

                n = batch_aug.size(0) + batch_pr.size(0)
                val_recon      += recon.item() * n
                val_mmd_latent += mmd_latent.item()
                val_mmd_image  += mmd_image.item()
                n_val     += n
                n_val_batches += 1

        val_recon      /= max(n_val, 1)
        val_mmd_latent /= max(n_val_batches, 1)
        val_mmd_image  /= max(n_val_batches, 1)
        val_loss = val_recon + lambda_mmd_t * val_mmd_latent

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:3d}  lambda_mmd {lambda_mmd_t:.4f}  "
                f"train_loss {train_loss:.5f}  (recon {train_recon:.5f}  "
                f"mmd_z {train_mmd_latent:.6f}  mmd_img {train_mmd_image:.6f})  "
                f"val_loss {val_loss:.5f}  (recon {val_recon:.5f}  "
                f"mmd_z {val_mmd_latent:.6f}  mmd_img {val_mmd_image:.6f})"
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

    if cohort not in ("AUGSBURG", "PRE-RAPID"):
        raise ValueError(f"Unknown cohort '{cohort}'. Expected 'AUGSBURG' or 'PRE-RAPID'.")

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

    all_cohorts = load_all_cohorts(Path(DATA_PATH))

    aug_patients = all_cohorts["AUGSBURG"]
    pr_patients  = all_cohorts["PRE-RAPID"]

    # stratified so val doesn't accidentally end up class-skewed at n~10
    train_aug, val_aug = train_test_split(
        aug_patients, test_size=0.2, random_state=40,
        stratify=[p.label for p in aug_patients],
    )
    train_pr, val_pr = train_test_split(
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
        train_pr=train_pr, val_pr=val_pr,
        n_epochs=1000,
        lambda_mmd=1,
        decoder_freeze_epoch=100,
        checkpoint_path=str(models_dir / "best_harmonization_shared.pt"),
    )

    save_harmonized_reconstructions(model, aug_patients, "AUGSBURG")
    save_harmonized_reconstructions(model, pr_patients, "PRE-RAPID")