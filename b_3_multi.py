from __future__ import annotations

import copy
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


DATA_PATH          = "CUBES-Labelled-COHORTS-ZSCORE"
DOMAIN_SHIFT_GAMMA = 5.3919e-06

COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "SWISS"]

# Cohort playing unlabeled held-out target this run. Its pairs stay pooled;
# the other two get class-stratified MMD. None disables stratification.
TARGET_COHORT = "SWISS"


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
        for p in self.decoder.parameters():
            p.requires_grad = trainable


def compute_mmd(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-gamma * torch.cdist(a, b).pow(2))

    kxx = rbf(x, x).mean()
    kyy = rbf(y, y).mean()
    kxy = rbf(x, y).mean()
    return kxx + kyy - 2 * kxy


def compute_mmd_stratified(
    x: torch.Tensor,
    y: torch.Tensor,
    x_labels: torch.Tensor,
    y_labels: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    # source-source only -- never call this on a pair touching an unlabeled target
    common_classes = sorted(set(x_labels.tolist()) & set(y_labels.tolist()))
    if not common_classes:
        return compute_mmd(x, y, gamma)  # no shared class this batch -- fall back to pooled

    terms = []
    for c in common_classes:
        xc = x[x_labels == c]
        yc = y[y_labels == c]
        terms.append(compute_mmd(xc, yc, gamma))
    return torch.stack(terms).mean()


def median_heuristic_gamma(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
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


def pairwise_mmd(
    zs: dict[str, torch.Tensor],
    gamma_fn=median_heuristic_gamma,
    fixed_gamma: float | None = None,
    fixed_gammas: dict[tuple[str, str], float] | None = None,
    target_cohort: str | None = None,
    labels: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[tuple[str, str], float]]:
    # pairs touching target_cohort stay pooled; source-source pairs stratify
    # when labels are present for both sides. Caller must omit target's
    # labels -- this function only enforces the target_cohort side of it.
    names = list(zs.keys())
    terms = {}
    for a, b in itertools.combinations(names, 2):
        if fixed_gammas is not None:
            gamma = fixed_gammas[(a, b)]
        elif fixed_gamma is not None:
            gamma = fixed_gamma
        else:
            gamma = gamma_fn(zs[a], zs[b])

        touches_target = target_cohort is not None and target_cohort in (a, b)
        has_labels = labels is not None and a in labels and b in labels

        if not touches_target and has_labels:
            terms[(a, b)] = compute_mmd_stratified(zs[a], zs[b], labels[a], labels[b], gamma=gamma)
        else:
            terms[(a, b)] = compute_mmd(zs[a], zs[b], gamma=gamma)

    avg = torch.stack(list(terms.values())).mean()
    terms_logged = {k: v.item() for k, v in terms.items()}
    return avg, terms_logged


class VolumeDataset(Dataset):
    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        patient = self.patients[idx]
        vol = patient.pet_masked.astype("float32")
        return torch.from_numpy(vol).unsqueeze(0), int(patient.label)


def _balanced_sampler(patients: list) -> WeightedRandomSampler:
    # inverse-class-frequency so stratified batches usually have both classes
    labels = np.array([p.label for p in patients])
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(patients), replacement=True)


def _cycling_batches(loaders: dict[str, DataLoader]):
    names = list(loaders.keys())
    lengths = {n: len(loaders[n]) for n in names}
    driver = max(names, key=lambda n: lengths[n])
    iters = [loaders[n] if n == driver else itertools.cycle(loaders[n]) for n in names]
    for batches in zip(*iters):
        yield dict(zip(names, batches))


def train_harmonization_multi(
    model: HarmonizationModel,
    cohort_train: dict[str, list],
    cohort_val: dict[str, list],
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    lambda_mmd: float = 1.0,
    decoder_freeze_epoch: int | None = None,
    latent_gamma_mode: str = "adaptive",
    target_cohort: str | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 100,
    checkpoint_path: str | None = "best_harmonization_multi.pt",
) -> HarmonizationModel:
    if latent_gamma_mode not in ("adaptive", "fixed"):
        raise ValueError(f"latent_gamma_mode must be 'adaptive' or 'fixed', got {latent_gamma_mode!r}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    decoder_frozen = False
    fixed_latent_gammas: dict | None = None
    cohort_names = list(cohort_train.keys())
    n_pairs = len(list(itertools.combinations(cohort_names, 2)))
    # mirrors the actual gate in the loops below, not just target_cohort alone
    stratified_pairs = [
        (a, b) for a, b in itertools.combinations(cohort_names, 2)
        if target_cohort is not None and target_cohort not in (a, b)
    ]

    print(f"cohorts      : {cohort_names}  ({n_pairs} pairs)")
    print(f"target       : {target_cohort!r}  (pairs touching it stay pooled)")
    print(f"stratified   : {stratified_pairs}")
    print(f"latent gamma : {latent_gamma_mode}")
    print(f"gamma        : {DOMAIN_SHIFT_GAMMA:.4e}   (image-space, informational only -- not used in the loss)")
    print(f"baseline MMD : 0.096446")
    if decoder_freeze_epoch is not None:
        print(f"decoder freeze at epoch {decoder_freeze_epoch} (stage-2 encoder-only training)")

    train_loaders = {}
    for name in cohort_names:
        dataset = VolumeDataset(cohort_train[name])
        if target_cohort is not None and name != target_cohort:
            train_loaders[name] = DataLoader(
                dataset, batch_size=batch_size, sampler=_balanced_sampler(cohort_train[name])
            )
        else:
            train_loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

            # carry over Adam state for encoder params instead of restarting momentum
            old_state = {p: optimizer.state[p] for p in trainable_params if p in optimizer.state}
            optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)
            optimizer.state.update(old_state)

            # val_loss scale jumps here (mmd turns on) -- reset best + patience together
            epochs_since_improvement = 0
            best_val_loss = float("inf")
            print(f"epoch {epoch:3d}  -- decoder frozen, mmd turned on (lambda_mmd={lambda_mmd}), optimizing encoder only from here on")

        if decoder_freeze_epoch is not None and epoch < decoder_freeze_epoch:
            lambda_mmd_t = 0.0
        else:
            lambda_mmd_t = lambda_mmd

        model.train()
        if decoder_frozen:
            model.decoder.eval()  # freeze BN running stats too, not just weights

        train_recon      = 0.0
        train_mmd_latent = 0.0
        train_mmd_image  = 0.0
        n_train_batches = 0

        for batch_dict in _cycling_batches(train_loaders):
            optimizer.zero_grad()

            zs, x_hats, batch_labels = {}, {}, {}
            recon = torch.zeros((), device=device)
            for name, (vol, label) in batch_dict.items():
                vol = vol.to(device)
                x_hat, z = model.reconstruct(name, vol)
                recon = recon + F.mse_loss(x_hat, vol)
                zs[name] = z
                x_hats[name] = x_hat.flatten(start_dim=1)
                # guard against target_cohort=None: "name != None" is always
                # True for a string, so without the None-check every cohort
                # would land in batch_labels and get stratified regardless
                if target_cohort is not None and name != target_cohort:
                    batch_labels[name] = label.to(device)

            if latent_gamma_mode == "fixed" and decoder_frozen and fixed_latent_gammas is None:
                fixed_latent_gammas = {
                    pair: median_heuristic_gamma(zs[pair[0]], zs[pair[1]])
                    for pair in itertools.combinations(cohort_names, 2)
                }
                print(f"epoch {epoch:3d}  -- latent gamma frozen at stage-2 start: {fixed_latent_gammas}")

            mmd_latent, _ = pairwise_mmd(
                zs, fixed_gammas=fixed_latent_gammas,
                target_cohort=target_cohort, labels=batch_labels,
            )
            loss = recon + lambda_mmd_t * mmd_latent

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # diagnostic only, always pooled
                mmd_image, _ = pairwise_mmd(x_hats, fixed_gamma=DOMAIN_SHIFT_GAMMA)

            train_recon      += recon.item()
            train_mmd_latent += mmd_latent.item()
            train_mmd_image  += mmd_image.item()
            n_train_batches += 1

        train_recon      /= max(n_train_batches, 1)
        train_mmd_latent /= max(n_train_batches, 1)
        train_mmd_image  /= max(n_train_batches, 1)
        train_loss = train_recon + lambda_mmd_t * train_mmd_latent

        model.eval()
        val_recon      = 0.0
        val_mmd_latent = 0.0
        val_mmd_image  = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch_dict in _cycling_batches(val_loaders):
                zs, x_hats, batch_labels = {}, {}, {}
                recon = torch.zeros((), device=device)
                for name, (vol, label) in batch_dict.items():
                    vol = vol.to(device)
                    x_hat, z = model.reconstruct(name, vol)
                    recon = recon + F.mse_loss(x_hat, vol)
                    zs[name] = z
                    x_hats[name] = x_hat.flatten(start_dim=1)
                    if target_cohort is not None and name != target_cohort:
                        batch_labels[name] = label.to(device)

                mmd_latent, _ = pairwise_mmd(
                    zs, fixed_gammas=fixed_latent_gammas,
                    target_cohort=target_cohort, labels=batch_labels,
                )
                mmd_image, _ = pairwise_mmd(x_hats, fixed_gamma=DOMAIN_SHIFT_GAMMA)

                val_recon      += recon.item()
                val_mmd_latent += mmd_latent.item()
                val_mmd_image  += mmd_image.item()
                n_val_batches += 1

        val_recon      /= max(n_val_batches, 1)
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

    checkpoint_name = f"best_harmonization_multi_target-{TARGET_COHORT or 'none'}.pt"

    model = train_harmonization_multi(
        model,
        cohort_train=cohort_train,
        cohort_val=cohort_val,
        n_epochs=1000,
        lambda_mmd=1,
        decoder_freeze_epoch=100,
        target_cohort=TARGET_COHORT,
        checkpoint_path=str(models_dir / checkpoint_name),
    )

    for name in COHORT_NAMES:
        save_harmonized_reconstructions(model, cohort_all[name], name)