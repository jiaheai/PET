from __future__ import annotations
import copy
import itertools
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
DATA_PATH          = "CUBES-Labelled-COHORTS"
COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "SWISS"]


TARGET_COHORT = "SWISS"  # Target-touching pairs use pooled MMD.


class Encoder3D(nn.Module):

    def __init__(self, latent_dim: int = 16):
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
        self.dropout = nn.Dropout3d(p=0.2)
        self.fc = nn.Linear(64 * 4 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.dropout(h)
        h = h.flatten(start_dim=1)
        return self.fc(h)


class Decoder3D(nn.Module):

    def __init__(self, latent_dim: int = 16):
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

    def __init__(self, cohort_names: list[str], latent_dim: int = 16):
        super().__init__()
        self.cohort_names = list(cohort_names)
        self.encoders = nn.ModuleDict({name: Encoder3D(latent_dim) for name in cohort_names})
        self.decoder = Decoder3D(latent_dim)
        self.aux_clf = nn.Linear(latent_dim, 1)

    def encode(self, cohort_name: str, x: torch.Tensor) -> torch.Tensor:
        return self.encoders[cohort_name](x)

    def reconstruct(self, cohort_name: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoders[cohort_name](x)
        x_hat = self.decoder(z)
        return x_hat, z

    def classify(self, z: torch.Tensor) -> torch.Tensor:


        return self.aux_clf(z).squeeze(-1)

    def set_decoder_trainable(self, trainable: bool) -> None:
        for p in self.decoder.parameters():
            p.requires_grad = trainable


def compute_mmd(x: torch.Tensor, y: torch.Tensor, gamma: float, unbiased: bool = False) -> torch.Tensor:
    """RBF MMD; unbiased=True uses the off-diagonal U-statistic when possible."""

    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-gamma * torch.cdist(a, b).pow(2))
    n, m = x.size(0), y.size(0)
    use_unbiased = unbiased and n >= 2 and m >= 2
    if use_unbiased:
        kxx_full, kyy_full = rbf(x, x), rbf(y, y)
        kxx = (kxx_full.sum() - kxx_full.diagonal().sum()) / (n * (n - 1))
        kyy = (kyy_full.sum() - kyy_full.diagonal().sum()) / (m * (m - 1))
    else:
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
    out_stats: dict | None = None,
) -> torch.Tensor:


    common_classes = sorted(set(x_labels.tolist()) & set(y_labels.tolist()))
    if not common_classes:
        return compute_mmd(x, y, gamma)
    terms = []
    for c in common_classes:
        xc = x[x_labels == c]
        yc = y[y_labels == c]
        n, m = xc.size(0), yc.size(0)
        if out_stats is not None:
            out_stats.setdefault("per_class_n", []).append(min(n, m))
            if n < 2 or m < 2:
                out_stats["n_below_2"] = out_stats.get("n_below_2", 0) + 1
        terms.append(compute_mmd(xc, yc, gamma, unbiased=True))
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
    target_pair_weight: float = 1.0,
    pair_weights: dict[tuple[str, str], float] | None = None,
    stratified_stats: dict | None = None,
) -> tuple[torch.Tensor, dict[tuple[str, str], float]]:


    names = list(zs.keys())
    terms = {}
    weights = {}
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
            terms[(a, b)] = compute_mmd_stratified(
                zs[a], zs[b], labels[a], labels[b], gamma=gamma, out_stats=stratified_stats
            )
        else:
            terms[(a, b)] = compute_mmd(zs[a], zs[b], gamma=gamma)
        if pair_weights is not None:
            weights[(a, b)] = pair_weights[(a, b)]
        else:
            weights[(a, b)] = target_pair_weight if touches_target else 1.0
    total_weight = sum(weights.values())
    avg = sum(weights[pair] * terms[pair] for pair in terms) / total_weight
    terms_logged = {k: v.item() for k, v in terms.items()}
    return avg, terms_logged


def _adaptive_pair_weights(
    pair_mmd_ema: dict[tuple[str, str], float] | None,
    pair_weighting: str,
    weighting_temperature: float,
    weighting_floor: float,
    weighting_ceil: float,
) -> dict[tuple[str, str], float] | None:
    if pair_weighting != "adaptive" or pair_mmd_ema is None:
        return None


    eps = 1e-8
    raw = {
        pair: max(ema_val, eps) ** (1.0 / weighting_temperature)
        for pair, ema_val in pair_mmd_ema.items()
    }


    mean_raw = sum(raw.values()) / len(raw)

    return {
        pair: min(max(w / mean_raw, weighting_floor), weighting_ceil)
        for pair, w in raw.items()
    }


class VolumeDataset(Dataset):

    def __init__(self, patients: list):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        patient = self.patients[idx]
        vol = patient.pet_masked.astype("float32")
        return torch.from_numpy(vol).unsqueeze(0), int(patient.label)


def _oversampling_sampler(
    patients: list, num_samples: int, weighted: bool, replacement: bool
) -> WeightedRandomSampler:
    """Build the sampler used to match the driver cohort's epoch size."""
    if weighted:
        labels = np.array([p.label for p in patients])
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
    else:
        weights = np.ones(len(patients))
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=replacement)


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
    batch_size: int = 16,
    lr: float = 1e-3,
    lambda_mmd: float = 1.0,
    decoder_freeze_epoch: int | None = None,
    latent_gamma_mode: str = "adaptive",
    target_cohort: str | None = None,
    target_pair_weight: float = 1.0,
    pair_weighting: str = "static",
    weighting_ema_beta: float = 0.9,
    weighting_temperature: float = 1.0,
    weighting_floor: float = 1e-3,
    weighting_ceil: float = 10.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 100,
    checkpoint_path: str | None = "best_harmonization_multi.pt",
) -> HarmonizationModel:
    """Stage 1: reconstruction. Stage 2: reconstruction + latent MMD after decoder freeze."""
    if latent_gamma_mode not in ("adaptive", "fixed"):
        raise ValueError(f"latent_gamma_mode must be 'adaptive' or 'fixed', got {latent_gamma_mode!r}")
    if pair_weighting not in ("static", "adaptive"):
        raise ValueError(f"pair_weighting must be 'static' or 'adaptive', got {pair_weighting!r}")
    if weighting_temperature == 0:
        raise ValueError("weighting_temperature must be nonzero")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    decoder_frozen = False
    fixed_latent_gammas: dict | None = None
    cohort_names = list(cohort_train.keys())
    n_pairs = len(list(itertools.combinations(cohort_names, 2)))
    stratified_pairs = [
        (a, b) for a, b in itertools.combinations(cohort_names, 2)
        if target_cohort is not None and target_cohort not in (a, b)
    ]


    driver_name = max(cohort_names, key=lambda n: len(cohort_train[n]))
    driver_size = len(cohort_train[driver_name])

    print(f"cohorts      : {cohort_names}  ({n_pairs} pairs)")
    print(f"target       : {target_cohort!r}  (pairs touching it stay pooled)")
    print(f"stratified   : {stratified_pairs}")
    print(f"pair weighting: {pair_weighting}" + (
        f"  (target_pair_weight={target_pair_weight})" if pair_weighting == "static"
        else f"  (ema_beta={weighting_ema_beta}, temperature={weighting_temperature}, "
             f"floor={weighting_floor}, ceil={weighting_ceil})"
    ))
    print(f"driver cohort: {driver_name!r} ({driver_size} patients) -- others oversampled to match")
    print(f"latent gamma : {latent_gamma_mode}")
    if decoder_freeze_epoch is not None:
        print(f"decoder freeze at epoch {decoder_freeze_epoch} (stage-2 encoder-only training)")

    train_loaders = {}
    for name in cohort_names:
        patients = cohort_train[name]
        dataset = VolumeDataset(patients)
        is_source = target_cohort is not None and name != target_cohort
        is_driver = name == driver_name
        num_samples = len(patients) if is_driver else driver_size
        train_loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=_oversampling_sampler(
                patients,
                num_samples=num_samples,
                weighted=is_source,
                replacement=not is_driver,
            ),
        )

    val_loaders = {
        name: DataLoader(VolumeDataset(cohort_val[name]), batch_size=batch_size, shuffle=False)
        for name in cohort_names
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_improvement = 0


    pair_mmd_ema: dict[tuple[str, str], float] | None = None

    for epoch in range(n_epochs):
        if decoder_freeze_epoch is not None and epoch == decoder_freeze_epoch and not decoder_frozen:
            model.set_decoder_trainable(False)
            decoder_frozen = True
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            old_state = {p: optimizer.state[p] for p in trainable_params if p in optimizer.state}
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
            optimizer.state.update(old_state)
            epochs_since_improvement = 0
            best_val_loss = float("inf")
            print(
                f"epoch {epoch:3d}  -- decoder frozen, mmd turned on "
                f"(lambda_mmd={lambda_mmd}), optimizing encoder only from here on"
            )

        lambda_mmd_t = (
            0.0
            if decoder_freeze_epoch is not None and epoch < decoder_freeze_epoch
            else lambda_mmd
        )

        model.train()
        if decoder_frozen:
            model.decoder.eval()

        train_recon = 0.0
        train_mmd_latent = 0.0
        n_train_batches = 0

        for batch_dict in _cycling_batches(train_loaders):
            optimizer.zero_grad()
            zs, batch_labels = {}, {}
            recon = torch.zeros((), device=device)

            for name, (vol, label) in batch_dict.items():
                vol = vol.to(device)
                x_hat, z = model.reconstruct(name, vol)
                recon = recon + F.mse_loss(x_hat, vol)
                zs[name] = z


                # Target labels never enter training.
                if target_cohort is not None and name != target_cohort:
                    batch_labels[name] = label.to(device)

            if latent_gamma_mode == "fixed" and decoder_frozen and fixed_latent_gammas is None:
                fixed_latent_gammas = {
                    pair: median_heuristic_gamma(zs[pair[0]], zs[pair[1]])
                    for pair in itertools.combinations(cohort_names, 2)
                }
                print(f"epoch {epoch:3d}  -- latent gamma frozen at stage-2 start: {fixed_latent_gammas}")

            mmd_latent, mmd_terms_this_step = pairwise_mmd(
                zs,
                fixed_gammas=fixed_latent_gammas,
                target_cohort=target_cohort,
                labels=batch_labels,
                target_pair_weight=target_pair_weight,
                pair_weights=_adaptive_pair_weights(
                    pair_mmd_ema,
                    pair_weighting,
                    weighting_temperature,
                    weighting_floor,
                    weighting_ceil,
                ),
            )

            loss = recon + lambda_mmd_t * mmd_latent
            loss.backward()
            optimizer.step()

            if pair_weighting == "adaptive" and lambda_mmd_t != 0.0:
                if pair_mmd_ema is None:
                    pair_mmd_ema = dict(mmd_terms_this_step)
                else:
                    for pair, val in mmd_terms_this_step.items():
                        pair_mmd_ema[pair] = (
                            weighting_ema_beta * pair_mmd_ema[pair]
                            + (1 - weighting_ema_beta) * val
                        )

            train_recon += recon.item()
            train_mmd_latent += mmd_latent.item()
            n_train_batches += 1

        train_recon /= max(n_train_batches, 1)
        train_mmd_latent /= max(n_train_batches, 1)
        train_loss = train_recon + lambda_mmd_t * train_mmd_latent

        model.eval()
        val_recon = 0.0
        val_mmd_latent = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch_dict in _cycling_batches(val_loaders):
                zs, batch_labels = {}, {}
                recon = torch.zeros((), device=device)

                for name, (vol, label) in batch_dict.items():
                    vol = vol.to(device)
                    x_hat, z = model.reconstruct(name, vol)
                    recon = recon + F.mse_loss(x_hat, vol)
                    zs[name] = z

                    if target_cohort is not None and name != target_cohort:
                        batch_labels[name] = label.to(device)

                mmd_latent, _ = pairwise_mmd(
                    zs,
                    fixed_gammas=fixed_latent_gammas,
                    target_cohort=target_cohort,
                    labels=batch_labels,
                    target_pair_weight=target_pair_weight,
                    pair_weights=_adaptive_pair_weights(
                        pair_mmd_ema,
                        pair_weighting,
                        weighting_temperature,
                        weighting_floor,
                        weighting_ceil,
                    ),
                )

                val_recon += recon.item()
                val_mmd_latent += mmd_latent.item()
                n_val_batches += 1

        val_recon /= max(n_val_batches, 1)
        val_mmd_latent /= max(n_val_batches, 1)
        val_loss = val_recon + lambda_mmd_t * val_mmd_latent

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:3d}  lambda_mmd {lambda_mmd_t:.4f}  "
                f"train_loss {train_loss:.5f}  "
                f"(recon {train_recon:.5f}  mmd_z {train_mmd_latent:.6f})  "
                f"val_loss {val_loss:.5f}  "
                f"(recon {val_recon:.5f}  mmd_z {val_mmd_latent:.6f})"
            )
            if pair_weighting == "adaptive" and pair_mmd_ema is not None:
                w = _adaptive_pair_weights(
                    pair_mmd_ema,
                    pair_weighting,
                    weighting_temperature,
                    weighting_floor,
                    weighting_ceil,
                )
                print(f"              adaptive weights: { {k: round(v, 4) for k, v in w.items()} }")


        patience_active = decoder_freeze_epoch is None or decoder_frozen
        if patience_active:
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

    print(f"training done -- best val_loss {best_val_loss:.5f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def alternating_classifier_finetune(
    model: HarmonizationModel,
    cohort_train: dict[str, list],
    cohort_val: dict[str, list],
    target_cohort: str,
    n_rounds: int = 5,
    epochs_per_round: int = 10,
    lambda_mmd: float = 1.0,
    batch_size: int = 16,
    lr: float = 1e-4,
    target_pair_weight: float = 1.0,
    pair_weighting: str = "static",
    weighting_ema_beta: float = 0.9,
    weighting_temperature: float = 1.0,
    weighting_floor: float = 1e-3,
    weighting_ceil: float = 10.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path: str | None = None,
) -> HarmonizationModel:
    """Fit source LR on current latents, then train encoders with recon + MMD + frozen-LR BCE."""
    if pair_weighting not in ("static", "adaptive"):
        raise ValueError(f"pair_weighting must be 'static' or 'adaptive', got {pair_weighting!r}")
    if weighting_temperature == 0:
        raise ValueError("weighting_temperature must be nonzero")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    model = model.to(device)
    cohort_names = list(cohort_train.keys())
    source_cohorts = [c for c in cohort_names if c != target_cohort]
    source_train_patients = [p for name in source_cohorts for p in cohort_train[name]]
    val_source_patients = [p for name in source_cohorts for p in cohort_val[name]]
    n_pos = sum(p.label for p in source_train_patients)
    n_neg = len(source_train_patients) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    def _encode_all(patients: list) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        zs, ys = [], []
        with torch.no_grad():
            for p in patients:
                vol = torch.from_numpy(p.pet_masked.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
                z = model.encode(p.cohort, vol).squeeze(0).cpu().numpy()
                zs.append(z)
                ys.append(p.label)
        return np.stack(zs), np.array(ys)

    def _fit_and_load_classifier() -> None:
        Z, y = _encode_all(source_train_patients)
        scaler = StandardScaler().fit(Z)
        lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(Z), y)


        w = lr_clf.coef_[0]
        b = lr_clf.intercept_[0]
        w_eff = w / scaler.scale_
        b_eff = b - np.dot(w_eff, scaler.mean_)
        with torch.no_grad():
            model.aux_clf.weight.copy_(torch.from_numpy(w_eff).float().unsqueeze(0).to(device))
            model.aux_clf.bias.copy_(torch.tensor([b_eff], dtype=torch.float32, device=device))
    driver_name = max(cohort_names, key=lambda n: len(cohort_train[n]))
    driver_size = len(cohort_train[driver_name])
    train_loaders = {}
    for name in cohort_names:
        patients = cohort_train[name]
        dataset = VolumeDataset(patients)
        is_source = name != target_cohort
        is_driver = name == driver_name
        num_samples = len(patients) if is_driver else driver_size
        train_loaders[name] = DataLoader(
            dataset, batch_size=batch_size,
            sampler=_oversampling_sampler(
                patients, num_samples=num_samples, weighted=is_source, replacement=not is_driver
            ),
        )
    print(f"alternating fine-tune: {n_rounds} rounds x {epochs_per_round} epochs  "
          f"source={source_cohorts}  target={target_cohort!r} (gets MMD every round, never clf_loss)")
    print(f"pair weighting: {pair_weighting}" + (
        f"  (target_pair_weight={target_pair_weight})" if pair_weighting == "static"
        else f"  (ema_beta={weighting_ema_beta}, temperature={weighting_temperature}, reset every round)"
    ))
    best_val_clf = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_round = -1


    _fit_and_load_classifier()
    model.aux_clf.weight.requires_grad = False
    model.aux_clf.bias.requires_grad = False
    for round_idx in range(n_rounds):
        print(f"round {round_idx:2d}  -- training encoders for {epochs_per_round} epochs against frozen classifier")
        trainable_params = []
        for name in cohort_names:
            trainable_params += list(model.encoders[name].parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)

        pair_mmd_ema: dict[tuple[str, str], float] | None = None  # Reset each round.
        for epoch in range(epochs_per_round):
            model.train()
            model.decoder.eval()
            for batch_dict in _cycling_batches(train_loaders):
                optimizer.zero_grad()
                zs, batch_labels = {}, {}
                recon = torch.zeros((), device=device)
                clf_loss = torch.zeros((), device=device)
                for name, (vol, label) in batch_dict.items():
                    vol = vol.to(device)
                    x_hat, z = model.reconstruct(name, vol)
                    recon = recon + F.mse_loss(x_hat, vol)
                    zs[name] = z
                    if name != target_cohort:
                        label = label.to(device)
                        batch_labels[name] = label
                        logit = model.classify(z)
                        clf_loss = clf_loss + F.binary_cross_entropy_with_logits(
                            logit.float(), label.float(), pos_weight=pos_weight
                        )
                mmd_latent, mmd_terms_this_step = pairwise_mmd(
                    zs, target_cohort=target_cohort, labels=batch_labels,
                    target_pair_weight=target_pair_weight,
                    pair_weights=_adaptive_pair_weights(
                        pair_mmd_ema, pair_weighting, weighting_temperature,
                        weighting_floor, weighting_ceil,
                    ),
                )
                loss = recon + lambda_mmd * mmd_latent + clf_loss
                loss.backward()
                optimizer.step()
                if pair_weighting == "adaptive":
                    if pair_mmd_ema is None:
                        pair_mmd_ema = dict(mmd_terms_this_step)
                    else:
                        for pair, val in mmd_terms_this_step.items():
                            pair_mmd_ema[pair] = (
                                weighting_ema_beta * pair_mmd_ema[pair] + (1 - weighting_ema_beta) * val
                            )


        _fit_and_load_classifier()
        model.aux_clf.weight.requires_grad = False
        model.aux_clf.bias.requires_grad = False


        model.eval()
        val_clf = 0.0
        n_val = 0
        with torch.no_grad():
            for p in val_source_patients:
                vol = torch.from_numpy(p.pet_masked.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
                z = model.encode(p.cohort, vol)
                logit = model.classify(z)
                label = torch.tensor([float(p.label)], device=device)
                val_clf += F.binary_cross_entropy_with_logits(
                    logit.float(), label, pos_weight=pos_weight
                ).item()
                n_val += 1
        val_clf /= max(n_val, 1)
        print(f"round {round_idx:2d}  -- val_clf (held-out source patients, freshly-refit classifier): {val_clf:.5f}")
        if val_clf < best_val_clf:
            best_val_clf = val_clf
            best_round = round_idx
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
    print(f"alternating fine-tune done -- best val_clf {best_val_clf:.5f} at round {best_round}")
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
