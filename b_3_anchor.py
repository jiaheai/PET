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

# Replace with your actual cohort names -- order doesn't matter for pairwise,
# every cohort gets compared against every other cohort.
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
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 64, 4, 4, 4)
        return self.deconv(h)


class HarmonizationModel(nn.Module):
    """One Encoder3D per cohort, one shared decoder. Each cohort's encoder
    specializes to that cohort's acquisition quirks; the anchor cohort's
    encoder never receives gradient from the MMD term (anchor_mmd detaches
    its z), so only the non-anchor cohorts' encoders get pulled toward it.
    All encoders (including the anchor's) still update from the
    reconstruction loss and, during stage 1, from the shared decoder too."""

    def __init__(self, cohort_names: list[str], latent_dim: int = 64):
        super().__init__()
        self.cohort_names = list(cohort_names)
        self.encoders = nn.ModuleDict({name: Encoder3D(latent_dim) for name in cohort_names})
        self.decoder = Decoder3D(latent_dim)

    def encode(self, cohort_name: str, x: torch.Tensor) -> torch.Tensor:
        return self.encoders[cohort_name](x)

    def reconstruct(self, cohort_name: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoders[cohort_name](x)
        x_hat = self.decoder(z)
        return x_hat, z

    def set_decoder_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the shared decoder (used for stage-2 encoder-only training)."""
        for p in self.decoder.parameters():
            p.requires_grad = trainable


# -- MMD loss ------------------------------------------------------------

def compute_mmd(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """RBF-kernel MMD. x, y: (N, D) and (M, D) -- gamma must match their scale."""
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


def compute_mmd_stratified(*args, **kwargs):
    raise NotImplementedError(
        "compute_mmd_stratified was removed: it required target-cohort labels, "
        "which aren't available at deployment time for a real unlabeled target "
        "cohort -- and since this sweep rotates which cohort plays target, no "
        "single cohort can be assumed safe to stratify. Use compute_mmd(x, y, "
        "gamma) instead -- pooled, no labels needed."
    )


def anchor_mmd(
    zs: dict[str, torch.Tensor],
    anchor_cohort: str,
    gamma_fn=median_heuristic_gamma,
    fixed_gamma: float | None = None,
) -> tuple[torch.Tensor, dict[tuple[str, str], float]]:
    """Averages POOLED MMD over each non-anchor cohort vs the anchor -- N-1
    comparisons instead of pairwise_mmd's C(N,2), and the anchor never
    moves: its tensor is detached before every comparison, so only the
    non-anchor cohorts' encoders receive gradient from this term. No
    labels used anywhere. The anchor cohort should be your largest /
    most-trusted cohort -- see the reasoning from the anchor-vs-pairwise
    discussion: fewer patient representations get perturbed (only
    non-anchor cohorts move), and they're moving toward a target
    estimated from more data.

    fixed_gamma overrides the per-comparison adaptive gamma (used for the
    image-space logging variant, for comparability with domain_shift.py).
    """
    names = list(zs.keys())
    if anchor_cohort not in names:
        raise ValueError(f"anchor_cohort {anchor_cohort!r} not among cohorts {names}")
    non_anchor = [n for n in names if n != anchor_cohort]
    if not non_anchor:
        raise ValueError("anchor_mmd needs at least one non-anchor cohort")

    z_anchor = zs[anchor_cohort].detach()   # anchor never moves from this term
    terms = {}
    for other in non_anchor:
        gamma = fixed_gamma if fixed_gamma is not None else gamma_fn(z_anchor, zs[other])
        terms[(anchor_cohort, other)] = compute_mmd(z_anchor, zs[other], gamma=gamma)
    avg = torch.stack(list(terms.values())).mean()
    terms_logged = {k: v.item() for k, v in terms.items()}
    return avg, terms_logged


# -- dataset ------------------------------------------------------------

class VolumeDataset(Dataset):
    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> torch.Tensor:
        vol = self.patients[idx].pet_masked.astype("float32")
        return torch.from_numpy(vol).unsqueeze(0)   # (1, 32, 32, 32)


def _cycling_batches(loaders: dict[str, DataLoader]):
    """Yields one dict[cohort_name] = vol_batch per step, iterating for as
    many steps as the longest loader has, cycling the shorter ones --
    generalization of the 2-cohort zip/cycle trick to N cohorts of
    different sizes.
    """
    names = list(loaders.keys())
    lengths = {n: len(loaders[n]) for n in names}
    driver = max(names, key=lambda n: lengths[n])
    iters = [loaders[n] if n == driver else itertools.cycle(loaders[n]) for n in names]
    for batches in zip(*iters):
        yield dict(zip(names, batches))


# -- training loop ------------------------------------------------------------

def train_harmonization_anchor(
    model: HarmonizationModel,
    cohort_train: dict[str, list],
    cohort_val: dict[str, list],
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    lambda_mmd: float = 1.0,
    decoder_freeze_epoch: int | None = None,
    anchor_cohort: str | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 200,
    checkpoint_path: str | None = "best_harmonization_anchor.pt",
) -> HarmonizationModel:
    """
    cohort_train / cohort_val : dict mapping cohort name -> list of patients.
                         Any number of cohorts (2+). MMD is POOLED -- no
                         labels used anywhere in the loss (labels are used
                         only downstream, for classifier fitting/eval in
                         the sweep script, never for harmonization).
    anchor_cohort : the fixed reference cohort every other cohort's latent
                         codes align TO -- N-1 comparisons instead of
                         pairwise_mmd's C(N,2), and the anchor's tensor is
                         detached every step so its encoder gets zero
                         gradient from the MMD term (only its own
                         reconstruction loss still updates it). Use your
                         largest / most-trusted cohort here: fewer patient
                         representations get perturbed (only non-anchor
                         cohorts move), and they move toward a target
                         estimated from more data. Required -- there's no
                         sensible default.
    decoder_freeze_epoch : same two-stage pattern as the 2-cohort version --
                         stage 1 (epoch < decoder_freeze_epoch) trains
                         encoder+decoder with mmd excluded from the loss
                         entirely (lambda_mmd_t=0), pure reconstruction.
                         At decoder_freeze_epoch the decoder freezes and mmd
                         turns on at lambda_mmd; only the encoder adapts
                         from then on. None disables this (mmd on the whole
                         run).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    decoder_frozen = False
    cohort_names = list(cohort_train.keys())

    if anchor_cohort is None:
        raise ValueError("anchor_cohort is required -- pick your largest/most-trusted cohort")
    if anchor_cohort not in cohort_names:
        raise ValueError(f"anchor_cohort {anchor_cohort!r} not among cohorts {cohort_names}")
    non_anchor = [c for c in cohort_names if c != anchor_cohort]

    print(f"cohorts      : {cohort_names}")
    print(f"anchor cohort: {anchor_cohort!r} (fixed -- detached, never moves; {len(non_anchor)} comparisons, "
          f"all pooled -- no labels used: {[(anchor_cohort, c) for c in non_anchor]})")
    print(f"gamma        : {DOMAIN_SHIFT_GAMMA:.4e}   (image-space, informational only -- not used in the loss)")
    print(f"baseline MMD : 0.096446")
    if decoder_freeze_epoch is not None:
        print(f"decoder freeze at epoch {decoder_freeze_epoch} (stage-2 encoder-only training)")

    train_loaders = {
        name: DataLoader(VolumeDataset(cohort_train[name]), batch_size=batch_size, shuffle=True)
        for name in cohort_names
    }
    val_loaders = {
        name: DataLoader(VolumeDataset(cohort_val[name]), batch_size=batch_size, shuffle=False)
        for name in cohort_names
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        if decoder_freeze_epoch is not None and epoch == decoder_freeze_epoch and not decoder_frozen:
            model.set_decoder_trainable(False)
            decoder_frozen = True
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)
            # val_loss's scale jumps here (mmd term switches on): reset both
            # the patience counter AND best_val_loss -- see 2-cohort version
            # for why keeping only the counter reset silently pins the
            # returned model to a pre-freeze, pre-harmonization checkpoint.
            epochs_since_improvement = 0
            best_val_loss = float("inf")
            print(f"epoch {epoch:3d}  -- decoder frozen, mmd turned on (lambda_mmd={lambda_mmd}), optimizing encoder only from here on")

        if decoder_freeze_epoch is not None and epoch < decoder_freeze_epoch:
            lambda_mmd_t = 0.0
        else:
            lambda_mmd_t = lambda_mmd

        # -- training --------------------------------------------------
        model.train()
        train_recon      = 0.0
        train_mmd_latent = 0.0
        train_mmd_image  = 0.0
        n_train = 0
        n_train_batches = 0

        for batch_dict in _cycling_batches(train_loaders):
            optimizer.zero_grad()

            zs, x_hats, xs = {}, {}, {}
            recon = torch.zeros((), device=device)
            for name, vol in batch_dict.items():
                vol = vol.to(device)
                x_hat, z = model.reconstruct(name, vol)
                recon = recon + F.mse_loss(x_hat, vol)
                zs[name] = z
                x_hats[name] = x_hat.flatten(start_dim=1)   # flatten for MMD's cdist
                xs[name] = vol

            mmd_latent, _ = anchor_mmd(zs, anchor_cohort=anchor_cohort)  # adaptive gamma, drives the loss
            loss = recon + lambda_mmd_t * mmd_latent

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mmd_image, _ = anchor_mmd(x_hats, anchor_cohort=anchor_cohort, fixed_gamma=DOMAIN_SHIFT_GAMMA)

            n = sum(v.size(0) for v in batch_dict.values())
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

        with torch.no_grad():
            for batch_dict in _cycling_batches(val_loaders):
                zs, x_hats = {}, {}
                recon = torch.zeros((), device=device)
                for name, vol in batch_dict.items():
                    vol = vol.to(device)
                    x_hat, z = model.reconstruct(name, vol)
                    recon = recon + F.mse_loss(x_hat, vol)
                    zs[name] = z
                    x_hats[name] = x_hat.flatten(start_dim=1)

                mmd_latent, _ = anchor_mmd(zs, anchor_cohort=anchor_cohort)
                mmd_image, _ = anchor_mmd(x_hats, anchor_cohort=anchor_cohort, fixed_gamma=DOMAIN_SHIFT_GAMMA)

                n = sum(v.size(0) for v in batch_dict.values())
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
    out_root: str | Path = "harmonized_reconstructions_multi",
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
            x_hat, _ = model.reconstruct(cohort, vol)
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

    cohort_train, cohort_val, cohort_all = {}, {}, {}
    for name in COHORT_NAMES:
        patients = all_cohorts[name]
        cohort_all[name] = patients
        train_p, val_p = train_test_split(
            patients, test_size=0.2, random_state=40,
            stratify=[p.label for p in patients],
        )
        cohort_train[name] = train_p
        cohort_val[name] = val_p
        print(f"{name:12s}: {len(train_p)} train / {len(val_p)} val  (all {len(patients)} used)")

    torch.manual_seed(41)
    model = HarmonizationModel(cohort_names=COHORT_NAMES, latent_dim=64)

    model = train_harmonization_anchor(
        model,
        cohort_train=cohort_train,
        cohort_val=cohort_val,
        n_epochs=1000,
        lambda_mmd=1,
        decoder_freeze_epoch=100,
        anchor_cohort="AUGSBURG",    # <-- your largest/most-trusted cohort; stays fixed
        checkpoint_path=str(models_dir / "best_harmonization_anchor.pt"),
    )

    for name in COHORT_NAMES:
        save_harmonized_reconstructions(model, cohort_all[name], name)