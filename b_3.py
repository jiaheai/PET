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
COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "COHORT_C"]


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
    """Single shared encoder across all cohorts -- scales to N cohorts without
    adding parameters per cohort, unlike a per-cohort-encoder design."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder3D(latent_dim)
        self.decoder = Decoder3D(latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
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


def compute_mmd_stratified(
    x: torch.Tensor,
    y: torch.Tensor,
    labels_x: torch.Tensor,
    labels_y: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Label-stratified MMD between two cohorts: averages MMD within each
    shared label instead of matching the pooled distribution, so the model
    can't close the gap by blurring label-specific features when the two
    cohorts have different label mixes.

    Falls back to 0.0 if no label has >=2 samples on both sides in this batch.
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


def pairwise_mmd(
    zs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    target_cohort: str | None = None,
    gamma_fn=median_heuristic_gamma,
    fixed_gamma: float | None = None,
) -> tuple[torch.Tensor, dict[tuple[str, str], float]]:
    """Averages MMD over every cohort pair. Returns the averaged scalar (for
    the loss) and a dict of each pair's individual value (for logging).

    target_cohort : the cohort standing in for "the real unlabeled
    deployment target" in this run. Any pair touching it uses POOLED MMD
    (no labels) -- using its labels here is the exact leakage that inflated
    PRE-RAPID's numbers in the 2-cohort version before the revert. Pairs
    between the other cohorts (which really would have labels available in
    production) use label-stratified MMD, which is legitimate for them.
    If target_cohort is None, every pair is pooled -- the fully-conservative
    default, equivalent to b.py's current behavior extended to N cohorts.

    fixed_gamma overrides the per-pair adaptive gamma (used for the
    image-space logging variant, which needs DOMAIN_SHIFT_GAMMA for
    comparability with domain_shift.py rather than an adaptive one).
    """
    names = list(zs.keys())
    terms = {}
    for a, b in itertools.combinations(names, 2):
        gamma = fixed_gamma if fixed_gamma is not None else gamma_fn(zs[a], zs[b])
        touches_target = target_cohort is not None and target_cohort in (a, b)
        if target_cohort is None or touches_target:
            terms[(a, b)] = compute_mmd(zs[a], zs[b], gamma=gamma)
        else:
            terms[(a, b)] = compute_mmd_stratified(zs[a], zs[b], labels[a], labels[b], gamma=gamma)
    avg = torch.stack(list(terms.values())).mean()
    terms_logged = {k: v.item() for k, v in terms.items()}
    return avg, terms_logged


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
        return torch.from_numpy(vol).unsqueeze(0), label


def _cycling_batches(loaders: dict[str, DataLoader]):
    """Yields one dict[cohort_name] = (vol_batch, label_batch) per step,
    iterating for as many steps as the longest loader has, cycling the
    shorter ones -- generalization of the 2-cohort zip/cycle trick to N
    cohorts of different sizes.
    """
    names = list(loaders.keys())
    lengths = {n: len(loaders[n]) for n in names}
    driver = max(names, key=lambda n: lengths[n])
    iters = [loaders[n] if n == driver else itertools.cycle(loaders[n]) for n in names]
    for batches in zip(*iters):
        yield dict(zip(names, batches))


# -- training loop ------------------------------------------------------------

def train_harmonization_multi(
    model: HarmonizationModel,
    cohort_train: dict[str, list],
    cohort_val: dict[str, list],
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    lambda_mmd: float = 1.0,
    decoder_freeze_epoch: int | None = None,
    target_cohort: str | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 200,
    checkpoint_path: str | None = "best_harmonization_multi.pt",
) -> HarmonizationModel:
    """
    cohort_train / cohort_val : dict mapping cohort name -> list of patients.
                         Any number of cohorts (2+). Pairwise MMD is computed
                         over every cohort pair and averaged -- O(N^2) pairs,
                         so this gets expensive with many cohorts (10 cohorts
                         = 45 pairs/step); fine for a handful.
    target_cohort : the cohort playing "unlabeled deployment target" this
                         run. Any pair involving it uses pooled (label-blind)
                         MMD; pairs among the other cohorts use label-
                         stratified MMD, which is legitimate since those
                         cohorts really do have labels in production. If
                         you're rotating which cohort is held out across
                         combinations (train/test over all cohort pairings),
                         pass a different target_cohort each run -- never
                         hardcode one. None pools every pair (safest default,
                         matches b.py's fully-reverted behavior).
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
    n_pairs = len(list(itertools.combinations(cohort_names, 2)))

    print(f"cohorts      : {cohort_names}  ({n_pairs} pairs)")
    if target_cohort is not None:
        stratified_pairs = [(a, b) for a, b in itertools.combinations(cohort_names, 2) if target_cohort not in (a, b)]
        pooled_pairs = [(a, b) for a, b in itertools.combinations(cohort_names, 2) if target_cohort in (a, b)]
        print(f"target cohort: {target_cohort!r} (treated as unlabeled -- pooled MMD)")
        print(f"  stratified pairs (labels used, legitimate): {stratified_pairs}")
        print(f"  pooled pairs (no labels, avoids leakage):   {pooled_pairs}")
    else:
        print(f"target cohort: none set -- pooling every pair (no stratification anywhere)")
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

            zs, x_hats, xs, labels = {}, {}, {}, {}
            recon = torch.zeros((), device=device)
            for name, (vol, lbl) in batch_dict.items():
                vol = vol.to(device)
                lbl = lbl.to(device)
                x_hat, z = model.reconstruct(vol)
                recon = recon + F.mse_loss(x_hat, vol)
                zs[name] = z
                x_hats[name] = x_hat.flatten(start_dim=1)   # flatten for MMD's cdist
                xs[name] = vol
                labels[name] = lbl

            mmd_latent, _ = pairwise_mmd(zs, labels, target_cohort=target_cohort)  # adaptive per-pair gamma, drives the loss
            loss = recon + lambda_mmd_t * mmd_latent

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mmd_image, _ = pairwise_mmd(x_hats, labels, target_cohort=target_cohort, fixed_gamma=DOMAIN_SHIFT_GAMMA)

            n = sum(v.size(0) for v, _ in batch_dict.values())
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
                zs, x_hats, labels = {}, {}, {}
                recon = torch.zeros((), device=device)
                for name, (vol, lbl) in batch_dict.items():
                    vol = vol.to(device)
                    lbl = lbl.to(device)
                    x_hat, z = model.reconstruct(vol)
                    recon = recon + F.mse_loss(x_hat, vol)
                    zs[name] = z
                    x_hats[name] = x_hat.flatten(start_dim=1)
                    labels[name] = lbl

                mmd_latent, _ = pairwise_mmd(zs, labels, target_cohort=target_cohort)
                mmd_image, _ = pairwise_mmd(x_hats, labels, target_cohort=target_cohort, fixed_gamma=DOMAIN_SHIFT_GAMMA)

                n = sum(v.size(0) for v, _ in batch_dict.values())
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
            x_hat, _ = model.reconstruct(vol)
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
    model = HarmonizationModel(latent_dim=64)

    model = train_harmonization_multi(
        model,
        cohort_train=cohort_train,
        cohort_val=cohort_val,
        n_epochs=1000,
        lambda_mmd=1,
        decoder_freeze_epoch=100,
        target_cohort="COHORT_C",   # <-- set to whichever cohort is playing "test" this run;
                                     #     rotate this across combinations, never hardcode one
        checkpoint_path=str(models_dir / "best_harmonization_multi.pt"),
    )

    for name in COHORT_NAMES:
        save_harmonized_reconstructions(model, cohort_all[name], name)