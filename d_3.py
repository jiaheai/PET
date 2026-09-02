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
        self.aux_clf = nn.Linear(latent_dim, 1)

    def encode(self, cohort_name: str, x: torch.Tensor) -> torch.Tensor:
        return self.encoders[cohort_name](x)

    def reconstruct(self, cohort_name: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoders[cohort_name](x)
        x_hat = self.decoder(z)
        return x_hat, z

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        # side-by-side supervision, not sequential fine-tuning: this head
        # trains ON THE SAME z, in the SAME backward() call, as recon/MMD --
        # never after them in a separate phase. That's what keeps alignment
        # pressure (MMD) and separability pressure (this) both active at
        # once, so the encoder can't satisfy one by sacrificing the other.
        return self.aux_clf(z).squeeze(-1)

    def set_decoder_trainable(self, trainable: bool) -> None:
        for p in self.decoder.parameters():
            p.requires_grad = trainable


def compute_mmd(x: torch.Tensor, y: torch.Tensor, gamma: float, unbiased: bool = False) -> torch.Tensor:
    """unbiased=True uses the U-statistic (excludes self-comparisons from the
    kxx/kyy means) instead of the V-statistic. The V-statistic's diagonal
    terms are always exactly 1 (self-kernel), which biases kxx/kyy upward by
    O(1/n) -- negligible at large n, but severe at the small per-class n the
    stratified path can hit (see compute_mmd_stratified). Falls back to
    biased automatically if n<2 on either side -- a U-statistic needs at
    least 2 samples to form an off-diagonal average at all.
    """
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
    # source-source only -- never call this on a pair touching an unlabeled target.
    # Always requests the unbiased U-statistic per class (see compute_mmd) --
    # splitting an already-small batch by class routinely lands at n=1-3 per
    # class, where the biased V-statistic overestimates MMD badly (empirically
    # ~1.5 at n=1 for two IDENTICAL distributions, vs the correct ~0).
    # out_stats, if given, accumulates diagnostics: 'per_class_n' (list of
    # min(n,m) seen per class this call) and 'n_below_2' (count of classes
    # that had to fall back to biased because n<2 on one side).
    common_classes = sorted(set(x_labels.tolist()) & set(y_labels.tolist()))
    if not common_classes:
        return compute_mmd(x, y, gamma)  # no shared class this batch -- fall back to pooled

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
    # pairs touching target_cohort stay pooled; source-source pairs stratify
    # when labels are present for both sides. Caller must omit target's
    # labels -- this function only enforces the target_cohort side of it.
    #
    # weighting precedence: pair_weights (explicit dict, used by adaptive
    # mode) overrides target_pair_weight entirely when given. Neither given
    # -> plain uniform mean (old behavior). Weights are plain floats, not
    # tensors -- no gradient flows through the weighting itself, only
    # through the underlying MMD terms.
    #
    # stratified_stats, if given, is passed straight through to
    # compute_mmd_stratified's out_stats -- see there for what it collects.
    # Optional and unused by default; no caller currently passes it.
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
    """Sampler drawing num_samples indices from patients each epoch.
    replacement=True (non-driver cohorts): random-WITH-replacement, so a
    smaller cohort seen multiple times per epoch gets a fresh random draw
    each pass instead of itertools.cycle's fixed, identical, repeated
    batch sequence -- less repetitive input for BatchNorm running stats.
    replacement=False (driver cohort): every patient drawn exactly once,
    same guarantee plain shuffle=True gave the driver before -- weights
    still influence draw ORDER, not which patients get included, so this
    stays a full, unbiased epoch even when weighted=True.

    weighted=True: inverse-class-frequency, for source cohorts -- applied
    regardless of whether that cohort is also the driver, since the driver
    can be a source cohort once target_cohort shrinks its own training set
    (target's harmonization half is only 50% of that cohort). weighted=False:
    uniform, for target -- its labels must never factor into sampling either,
    not just the loss.
    """
    if weighted:
        labels = np.array([p.label for p in patients])
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
    else:
        weights = np.ones(len(patients))
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=replacement)


def _cycling_batches(loaders: dict[str, DataLoader]):
    # train loaders are now all sized to match the driver, so this cycle()
    # degrades to zip() for them; val loaders are still genuinely different
    # lengths per cohort and rely on real cycling here.
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
    lambda_clf: float = 1.0,
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
    """
    lambda_clf : weight on model.aux_clf's BCE loss, gated the same way as
                 lambda_mmd -- 0 during stage 1, lambda_clf from
                 decoder_freeze_epoch on. Only ever computed on SOURCE
                 (non-target) cohorts' patients -- same guard that already
                 keeps target's labels out of batch_labels for stratified
                 MMD covers this too, since clf_loss is built from that
                 same dict. Runs alongside recon/mmd in one loss.backward()
                 call every step (not a separate post-training fine-tune
                 phase), so MMD's alignment pressure is never absent while
                 this pushes for separability -- see HarmonizationModel.classify.
    target_pair_weight : static multiplier (>1 upweights, <1 downweights)
                 applied to any cohort pair touching target_cohort. Only
                 used when pair_weighting="static" (the default).
    pair_weighting : "static" uses target_pair_weight, fixed for the whole
                 run. "adaptive" instead tracks an exponential moving
                 average of each pair's own MMD magnitude and weights pairs
                 proportionally to their (smoothed) current difficulty -- a
                 pair with more residual misalignment right now gets more
                 gradient, full stop. No reference point, no warmup window:
                 weight tracks raw magnitude directly, not progress relative
                 to where a pair started. Simpler than a reference-ratio
                 scheme, and doesn't carry that scheme's failure mode of a
                 single bad early snapshot permanently distorting every
                 later weight (this EMA has no permanent anchor -- a noisy
                 seed's influence decays away over subsequent steps instead
                 of being locked in). Tradeoff: can't distinguish "made real
                 progress but is still numerically bigger" from "never
                 moved at all" -- weight is driven by current scale, not
                 trajectory. gamma_fn's per-pair median-heuristic bandwidth
                 already roughly scale-normalizes different pairs' MMD
                 values before this sees them, which partially (not fully)
                 offsets that.
                 The EMA is always ONE STEP LAGGED: a batch's weights come
                 from the EMA as it stood BEFORE that batch, and only get
                 updated with that batch's results afterward. This is
                 purely a variance-reduction measure -- MMD at small batch
                 sizes is a high-variance estimator, and the lag just keeps
                 a single noisy batch from being both the thing optimized
                 and its own weight in the same step. It is NOT preventing
                 anything "gameable": these are detached floats with no
                 gradient path through the weighting itself, so there was
                 never a self-referential-optimization risk to guard
                 against. Still noisy: weighting_ema_beta close to 1 is the
                 main defense against chasing single-batch noise -- higher
                 beta (e.g. 0.95+) smooths harder at the cost of reacting
                 slower to genuine, sustained shifts in which pair is
                 hardest.
    weighting_ema_beta : EMA decay for pair_weighting="adaptive". new_ema =
                 beta*old_ema + (1-beta)*this_batch_value.
    weighting_temperature : exponent applied to each pair's clamped EMA
                 value (ema ** (1/temperature)) before use as a weight.
                 1.0 = weight directly proportional to smoothed MMD
                 magnitude. <1 sharpens toward whichever pair is currently
                 hardest; >1 flattens back toward uniform. Only used in
                 adaptive mode.
    weighting_floor, weighting_ceil : clamp bounds applied to each pair's
                 weight TWICE -- once before exponentiation and once after.
                 Both are required, not redundant: clamping the EMA value
                 to weighting_ceil does not bound the exponentiated result
                 (e.g. a value clamped to 10.0 becomes 100.0 at
                 temperature=0.5), so the post-exponent clamp is what
                 actually stops one pair from dominating the weighted
                 average. weighting_floor also guards against the
                 underlying MMD estimator occasionally landing slightly
                 negative at small batch sizes (kxx+kyy-2kxy is not
                 clipped at 0), which would otherwise send a fractional
                 exponent to nan.
    """
    if latent_gamma_mode not in ("adaptive", "fixed"):
        raise ValueError(f"latent_gamma_mode must be 'adaptive' or 'fixed', got {latent_gamma_mode!r}")
    if pair_weighting not in ("static", "adaptive"):
        raise ValueError(f"pair_weighting must be 'static' or 'adaptive', got {pair_weighting!r}")
    if weighting_temperature == 0:
        raise ValueError("weighting_temperature must be nonzero")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    decoder_frozen = False
    fixed_latent_gammas: dict | None = None
    cohort_names = list(cohort_train.keys())
    n_pairs = len(list(itertools.combinations(cohort_names, 2)))
    stratified_pairs = [
        (a, b) for a, b in itertools.combinations(cohort_names, 2)
        if target_cohort is not None and target_cohort not in (a, b)
    ]

    # cohort with the most training patients paces the epoch; every other
    # cohort's train loader oversamples (with replacement) up to that count.
    # Source-cohort class-balance weighting applies regardless of whether
    # that cohort happens to be the driver.
    driver_name = max(cohort_names, key=lambda n: len(cohort_train[n]))
    driver_size = len(cohort_train[driver_name])

    # class-imbalance correction for aux_clf, matching cnn.py's convention --
    # computed from TRAIN patients of source (non-target) cohorts only.
    # Empty (and pos_weight=None, lambda_clf inert) when target_cohort=None --
    # mirrors stratified MMD's own "None disables it everywhere" behavior;
    # aux_clf never fires without a designated target_cohort either.
    source_cohorts = [n for n in cohort_names if target_cohort is not None and n != target_cohort]
    source_train_patients = [p for n in source_cohorts for p in cohort_train[n]]
    if source_train_patients:
        n_pos = sum(p.label for p in source_train_patients)
        n_neg = len(source_train_patients) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    else:
        pos_weight = None   # lambda_clf effectively inert -- no source cohort to draw labels from

    print(f"cohorts      : {cohort_names}  ({n_pairs} pairs)")
    print(f"target       : {target_cohort!r}  (pairs touching it stay pooled)")
    print(f"stratified   : {stratified_pairs}")
    print(f"pair weighting: {pair_weighting}" + (
        f"  (target_pair_weight={target_pair_weight})" if pair_weighting == "static"
        else f"  (ema_beta={weighting_ema_beta}, temperature={weighting_temperature}, "
             f"floor={weighting_floor}, ceil={weighting_ceil})"
    ))
    print(f"aux clf on   : {source_cohorts}  (lambda_clf={lambda_clf}, pos_weight={pos_weight})")
    print(f"driver cohort: {driver_name!r} ({driver_size} patients) -- others oversampled to match")
    print(f"latent gamma : {latent_gamma_mode}")
    print(f"gamma        : {DOMAIN_SHIFT_GAMMA:.4e}   (image-space, informational only -- not used in the loss)")
    print(f"baseline MMD : 0.096446")
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
            dataset, batch_size=batch_size,
            sampler=_oversampling_sampler(
                patients, num_samples=num_samples, weighted=is_source, replacement=not is_driver
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

    # adaptive weighting state -- one EMA value per pair, lazily initialized
    # on the first stage-2 batch (no smoothing needed for a single sample).
    pair_mmd_ema: dict[tuple[str, str], float] | None = None

    def _current_pair_weights() -> dict[tuple[str, str], float] | None:
        if pair_weighting != "adaptive" or pair_mmd_ema is None:
            return None  # static mode, or adaptive but not yet observed a batch -- caller falls back
        weights = {}
        for pair, ema_val in pair_mmd_ema.items():
            clamped = min(max(ema_val, weighting_floor), weighting_ceil)
            w = clamped ** (1.0 / weighting_temperature)
            weights[pair] = min(max(w, weighting_floor), weighting_ceil)  # re-clamp post-exponent
        return weights

    for epoch in range(n_epochs):
        if decoder_freeze_epoch is not None and epoch == decoder_freeze_epoch and not decoder_frozen:
            model.set_decoder_trainable(False)
            decoder_frozen = True
            trainable_params = [p for p in model.parameters() if p.requires_grad]

            old_state = {p: optimizer.state[p] for p in trainable_params if p in optimizer.state}
            optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)
            optimizer.state.update(old_state)

            epochs_since_improvement = 0
            best_val_loss = float("inf")
            print(f"epoch {epoch:3d}  -- decoder frozen, mmd turned on (lambda_mmd={lambda_mmd}), optimizing encoder only from here on")

        if decoder_freeze_epoch is not None and epoch < decoder_freeze_epoch:
            lambda_mmd_t = 0.0
            lambda_clf_t = 0.0
        else:
            lambda_mmd_t = lambda_mmd
            lambda_clf_t = lambda_clf

        model.train()
        if decoder_frozen:
            model.decoder.eval()

        train_recon      = 0.0
        train_mmd_latent = 0.0
        train_mmd_image  = 0.0
        train_clf        = 0.0
        n_train_batches = 0

        for batch_dict in _cycling_batches(train_loaders):
            optimizer.zero_grad()

            zs, x_hats, batch_labels = {}, {}, {}
            recon = torch.zeros((), device=device)
            clf_loss = torch.zeros((), device=device)
            for name, (vol, label) in batch_dict.items():
                vol = vol.to(device)
                x_hat, z = model.reconstruct(name, vol)
                recon = recon + F.mse_loss(x_hat, vol)
                zs[name] = z
                x_hats[name] = x_hat.flatten(start_dim=1)
                if target_cohort is not None and name != target_cohort:
                    label = label.to(device)
                    batch_labels[name] = label
                    if lambda_clf_t != 0.0:
                        logit = model.classify(z)
                        clf_loss = clf_loss + F.binary_cross_entropy_with_logits(
                            logit.float(), label.float(), pos_weight=pos_weight
                        )

            if latent_gamma_mode == "fixed" and decoder_frozen and fixed_latent_gammas is None:
                fixed_latent_gammas = {
                    pair: median_heuristic_gamma(zs[pair[0]], zs[pair[1]])
                    for pair in itertools.combinations(cohort_names, 2)
                }
                print(f"epoch {epoch:3d}  -- latent gamma frozen at stage-2 start: {fixed_latent_gammas}")

            mmd_latent, mmd_terms_this_step = pairwise_mmd(
                zs, fixed_gammas=fixed_latent_gammas,
                target_cohort=target_cohort, labels=batch_labels,
                target_pair_weight=target_pair_weight,
                pair_weights=_current_pair_weights(),
            )
            loss = recon + lambda_mmd_t * mmd_latent + lambda_clf_t * clf_loss

            loss.backward()
            optimizer.step()

            if pair_weighting == "adaptive" and lambda_mmd_t != 0.0:
                # lagged update: this step's WEIGHTS came from the EMA as it
                # stood before this step; only now does this step's own
                # result get folded in, for the NEXT step to use.
                if pair_mmd_ema is None:
                    pair_mmd_ema = dict(mmd_terms_this_step)
                else:
                    for pair, val in mmd_terms_this_step.items():
                        pair_mmd_ema[pair] = (
                            weighting_ema_beta * pair_mmd_ema[pair] + (1 - weighting_ema_beta) * val
                        )

            with torch.no_grad():
                mmd_image, _ = pairwise_mmd(x_hats, fixed_gamma=DOMAIN_SHIFT_GAMMA)

            train_recon      += recon.item()
            train_mmd_latent += mmd_latent.item()
            train_mmd_image  += mmd_image.item()
            train_clf        += clf_loss.item()
            n_train_batches += 1

        train_recon      /= max(n_train_batches, 1)
        train_mmd_latent /= max(n_train_batches, 1)
        train_mmd_image  /= max(n_train_batches, 1)
        train_clf        /= max(n_train_batches, 1)
        train_loss = train_recon + lambda_mmd_t * train_mmd_latent + lambda_clf_t * train_clf

        model.eval()
        val_recon      = 0.0
        val_mmd_latent = 0.0
        val_mmd_image  = 0.0
        val_clf        = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch_dict in _cycling_batches(val_loaders):
                zs, x_hats, batch_labels = {}, {}, {}
                recon = torch.zeros((), device=device)
                clf_loss = torch.zeros((), device=device)
                for name, (vol, label) in batch_dict.items():
                    vol = vol.to(device)
                    x_hat, z = model.reconstruct(name, vol)
                    recon = recon + F.mse_loss(x_hat, vol)
                    zs[name] = z
                    x_hats[name] = x_hat.flatten(start_dim=1)
                    if target_cohort is not None and name != target_cohort:
                        label = label.to(device)
                        batch_labels[name] = label
                        if lambda_clf_t != 0.0:
                            logit = model.classify(z)
                            clf_loss = clf_loss + F.binary_cross_entropy_with_logits(
                                logit.float(), label.float(), pos_weight=pos_weight
                            )

                mmd_latent, _ = pairwise_mmd(
                    zs, fixed_gammas=fixed_latent_gammas,
                    target_cohort=target_cohort, labels=batch_labels,
                    target_pair_weight=target_pair_weight,
                    pair_weights=_current_pair_weights(),  # frozen for val -- never updated from val batches
                )
                mmd_image, _ = pairwise_mmd(x_hats, fixed_gamma=DOMAIN_SHIFT_GAMMA)

                val_recon      += recon.item()
                val_mmd_latent += mmd_latent.item()
                val_mmd_image  += mmd_image.item()
                val_clf        += clf_loss.item()
                n_val_batches += 1

        val_recon      /= max(n_val_batches, 1)
        val_mmd_latent /= max(n_val_batches, 1)
        val_mmd_image  /= max(n_val_batches, 1)
        val_clf        /= max(n_val_batches, 1)
        val_loss = val_recon + lambda_mmd_t * val_mmd_latent + lambda_clf_t * val_clf

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:3d}  lambda_mmd {lambda_mmd_t:.4f}  lambda_clf {lambda_clf_t:.4f}  "
                f"train_loss {train_loss:.5f}  (recon {train_recon:.5f}  "
                f"mmd_z {train_mmd_latent:.6f}  mmd_img {train_mmd_image:.6f}  clf {train_clf:.5f})  "
                f"val_loss {val_loss:.5f}  (recon {val_recon:.5f}  "
                f"mmd_z {val_mmd_latent:.6f}  mmd_img {val_mmd_image:.6f}  clf {val_clf:.5f})"
            )
            if pair_weighting == "adaptive" and pair_mmd_ema is not None:
                w = _current_pair_weights()
                print(f"              adaptive weights: { {k: round(v, 4) for k, v in w.items()} }")

        # Early stopping only activates once the decoder is frozen (or from
        # epoch 0 if decoder_freeze_epoch=None, i.e. no staging at all).
        # Without this gate, stage 1's val_loss (pure recon, MMD/clf both
        # forced to 0) can plateau and trigger patience on its own --
        # stopping training before decoder_freeze_epoch is ever reached,
        # regardless of what decoder_freeze_epoch is set to. The reset at
        # freeze time (best_val_loss=inf, epochs_since_improvement=0, a few
        # lines up) only helps AFTER the freeze happens; it does nothing to
        # stop stage 1 from independently ending the run first.
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
                print(f"no val_loss improvement for {patience} epochs "
                      f"(best was {best_val_loss:.5f} at epoch {best_epoch}) -- stopping early")
                break

    print(f"training done -- best val_loss {best_val_loss:.5f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def finetune_classifier_only(
    model: HarmonizationModel,
    cohort_train: dict[str, list],
    cohort_val: dict[str, list],
    target_cohort: str,
    n_epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 8,
    patience: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path: str | None = None,
) -> HarmonizationModel:
    """Stage 3: classification-only fine-tune, source cohorts only, run
    AFTER train_harmonization_multi (stages 1+2) has already produced a
    harmonized model -- continues from wherever that left off.

    Loss is ONLY aux_clf's BCE -- no recon, no MMD, on purpose. Goal is
    classifier accuracy directly, not distribution alignment, so nothing
    in this loss holds source-cohort encoders to their post-stage-2
    position except however much stage 2's alignment already "stuck."

    target_cohort's encoder is NEVER touched here -- not in the optimizer's
    param list, and no target images are ever fed through it in this
    function, so it stays frozen at exactly wherever stage 2 left it while
    source encoders keep moving. Decoder is likewise never used (already
    frozen from stage 2; reconstruction isn't part of this loss at all).

    This is a genuine experiment, not a hedged version of lambda_clf: unlike
    train_harmonization_multi's aux_clf (which runs simultaneously with MMD,
    every step, so alignment pressure never turns off), this has ZERO
    alignment pressure once it starts. If source encoders drift away from
    where target's (frozen) encoder sits, nothing here detects or prevents
    it. Judge this purely by the downstream held-out target_cohort_raw /
    target_cohort_corrected metrics it produces afterward -- not by MMD or
    any other proxy computed in this function, since MMD isn't even
    computed here at all.
    """
    model = model.to(device)
    source_cohorts = [c for c in cohort_train.keys() if c != target_cohort]

    trainable_params = list(model.aux_clf.parameters())
    for name in source_cohorts:
        trainable_params += list(model.encoders[name].parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)

    source_train_patients = [p for name in source_cohorts for p in cohort_train[name]]
    n_pos = sum(p.label for p in source_train_patients)
    n_neg = len(source_train_patients) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    train_loaders = {
        name: DataLoader(VolumeDataset(cohort_train[name]), batch_size=batch_size, shuffle=True)
        for name in source_cohorts
    }
    val_loaders = {
        name: DataLoader(VolumeDataset(cohort_val[name]), batch_size=batch_size, shuffle=False)
        for name in source_cohorts
    }

    print(f"stage 3      : classification-only fine-tune on {source_cohorts}  "
          f"(target={target_cohort!r} untouched -- encoder stays exactly as stage 2 left it)")
    print(f"stage 3 pos_weight: {pos_weight}")

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())   # fallback: stage-2 state if stage 3 never improves
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        train_clf = 0.0
        n_train_batches = 0
        for batch_dict in _cycling_batches(train_loaders):
            optimizer.zero_grad()
            clf_loss = torch.zeros((), device=device)
            for name, (vol, label) in batch_dict.items():
                vol = vol.to(device)
                label = label.to(device)
                z = model.encode(name, vol)
                logit = model.classify(z)
                clf_loss = clf_loss + F.binary_cross_entropy_with_logits(
                    logit.float(), label.float(), pos_weight=pos_weight
                )
            clf_loss.backward()
            optimizer.step()
            train_clf += clf_loss.item()
            n_train_batches += 1
        train_clf /= max(n_train_batches, 1)

        model.eval()
        val_clf = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch_dict in _cycling_batches(val_loaders):
                clf_loss = torch.zeros((), device=device)
                for name, (vol, label) in batch_dict.items():
                    vol = vol.to(device)
                    label = label.to(device)
                    z = model.encode(name, vol)
                    logit = model.classify(z)
                    clf_loss = clf_loss + F.binary_cross_entropy_with_logits(
                        logit.float(), label.float(), pos_weight=pos_weight
                    )
                val_clf += clf_loss.item()
                n_val_batches += 1
        val_clf /= max(n_val_batches, 1)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"stage3 epoch {epoch:3d}  train_clf {train_clf:.5f}  val_clf {val_clf:.5f}")

        if val_clf < best_val_loss:
            best_val_loss = val_clf
            best_epoch = epoch
            epochs_since_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"stage3: no val_clf improvement for {patience} epochs "
                  f"(best was {best_val_loss:.5f} at epoch {best_epoch}) -- stopping early")
            break

    print(f"stage3 done -- best val_clf {best_val_loss:.5f} at epoch {best_epoch}")
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
    batch_size: int = 8,
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
    """Alternating block-coordinate fine-tune, run AFTER train_harmonization_multi
    (stages 1-2) has already produced a harmonized model with its decoder frozen.

    Each round:
      1. Encode all SOURCE (non-target) cohorts' TRAIN patients with the
         CURRENT encoders.
      2. Fit StandardScaler + LogisticRegression on that z -- a full,
         properly-converged fit, same pipeline used for real scoring
         elsewhere in this project. Fold the scaler into the LR's
         coefficients (w_eff = w/scale, b_eff = b - dot(w_eff, mean)) so
         the combined pipeline becomes one equivalent linear layer, loaded
         directly into model.aux_clf.
      3. Freeze aux_clf (requires_grad=False on its weight/bias) and train
         the ENCODERS for epochs_per_round epochs on
         recon + lambda_mmd*mmd + BCE(aux_clf(z), label). MMD stays fully
         active the whole time, same alignment pressure as always -- only
         aux_clf's OWN weights are frozen; gradient still flows THROUGH it
         back into z (freezing requires_grad blocks a layer's weights from
         being updated, not gradient from passing through its forward
         computation to earlier tensors -- same principle as the frozen
         decoder in stage 2).
      4. Repeat: re-encode with the now-shifted encoders, fit a NEW optimal
         classifier for wherever z landed, freeze it, train again.

    Unlike finetune_classifier_only (sequential, zero MMD -- see that
    function's docstring for the target-transfer collapse it produced),
    MMD never leaves the loss here, every round. Unlike
    train_harmonization_multi's lambda_clf (aux_clf jointly trained by
    Adam, sharing the encoder's optimizer, never fully converged), the
    classifier supplying gradient each round is always a complete sklearn
    fit -- stronger, standardized, converged -- not a proxy for one.

    target_cohort's encoder IS still trained here (unlike
    finetune_classifier_only, which excludes it entirely) -- it receives
    gradient via MMD every round, same as stages 1-2, but NEVER via
    clf_loss: target's patients never enter the label-guarded branch below,
    same invariant enforced everywhere else in this file.

    target_pair_weight / pair_weighting / weighting_* : same adaptive
    pair-weighting scheme as train_harmonization_multi -- raw-magnitude EMA,
    no reference/warmup -- see that function's docstring for the full
    mechanics. Here, the EMA state is RESET at the start of every round
    (each round's optimizer and encoder-training loop is independent), not
    carried across rounds.

    Each round is scored on cohort_val's source patients using that
    round's OWN end-of-round refit (fit on that round's post-training z,
    not the round-start classifier it trained against) -- see the refit
    placement in the loop below for why. Only the best-scoring round's
    (encoders, classifier) pair is kept; the classifier in model.aux_clf
    after this function returns IS the classifier to use for eval -- no
    separate downstream refit needed, since it was already synced to its
    own encoders at the moment it was saved as best.
    """
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

        # fold StandardScaler into LogisticRegression's coefficients --
        # w_eff = w/scale, b_eff = b - dot(w_eff, mean) -- see derivation
        # in prior discussion for why this makes the combined pipeline
        # equivalent to one nn.Linear taking raw z directly.
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
    best_state = copy.deepcopy(model.state_dict())   # fallback: pre-fine-tune state if no round improves
    best_round = -1

    # initial fit, once, on the pre-fine-tune z -- this is round 0's frozen
    # target. Each subsequent round's frozen target is set by the PREVIOUS
    # round's end-of-round refit below, so this call only ever happens once
    # here, not once per round -- avoids fitting twice in a row on the same
    # z (once as "end of round N", once as "start of round N+1", which
    # would just be the same deterministic sklearn fit run redundantly).
    _fit_and_load_classifier()
    model.aux_clf.weight.requires_grad = False
    model.aux_clf.bias.requires_grad = False

    for round_idx in range(n_rounds):
        print(f"round {round_idx:2d}  -- training encoders for {epochs_per_round} epochs against frozen classifier")

        trainable_params = []
        for name in cohort_names:
            trainable_params += list(model.encoders[name].parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)

        # adaptive weighting state -- fresh per round, see docstring
        pair_mmd_ema: dict[tuple[str, str], float] | None = None

        def _current_pair_weights() -> dict[tuple[str, str], float] | None:
            if pair_weighting != "adaptive" or pair_mmd_ema is None:
                return None
            weights = {}
            for pair, ema_val in pair_mmd_ema.items():
                clamped = min(max(ema_val, weighting_floor), weighting_ceil)
                w = clamped ** (1.0 / weighting_temperature)
                weights[pair] = min(max(w, weighting_floor), weighting_ceil)
            return weights

        for epoch in range(epochs_per_round):
            model.train()
            model.decoder.eval()   # decoder assumed already frozen from stage 2 -- keep BN stats fixed too
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
                    pair_weights=_current_pair_weights(),
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

        # -- refit on THIS round's post-training z before scoring. Scores
        # "how good is this round's z under its own best-achievable
        # classifier" rather than "how well does this round's now-stale
        # round-start classifier generalize" -- the two are different
        # questions, and the round-start classifier was fit on OLD (pre-
        # this-round) z, so scoring against it conflates "did the encoder
        # improve" with "did the encoder stay compatible with an
        # increasingly stale boundary". This refit also becomes the frozen
        # target for the NEXT round's encoder training, so it does double
        # duty -- no separate fit needed at the top of the next iteration.
        _fit_and_load_classifier()
        model.aux_clf.weight.requires_grad = False
        model.aux_clf.bias.requires_grad = False

        # -- validate THIS round's (freshly-refit classifier + its own z)
        # combo on held-out source patients (cohort_val, never seen by
        # either the encoder training above or _fit_and_load_classifier's
        # sklearn fit, which only used cohort_train). This is the safety
        # net the docstring's own "genuine experiment, real drift risk"
        # framing calls for -- without it, a late round that's actually
        # worse than an earlier one would silently overwrite it with no
        # way to tell and no fallback.
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
    # No separate "final fit" step needed anymore: best_state was saved
    # right after that round's own end-of-round refit (see the loop above),
    # so its classifier is already exactly synced to its own encoders --
    # refitting again here would just reproduce the same weights via the
    # same deterministic sklearn solver on the same z. This used to matter
    # (there was a real final_fit_only toggle) before the round loop was
    # reordered to refit-then-save instead of save-then-refit; now it's
    # dead weight, so it's gone rather than kept as a no-op flag.

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