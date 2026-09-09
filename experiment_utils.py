"""Shared fold-safe preprocessing and experiment provenance helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from copy import copy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from nifti_loader import PatientVolumes


REFERENCE_COHORT = "SWISS"


def seed_everything(seed: int) -> None:
    """Seed all RNGs while retaining fast CUDA/cuDNN execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ["numpy", "scikit-learn", "torch", "nibabel"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "missing"
    return versions


def append_jsonl_record(path: Path, record_type: str, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps({"record_type": record_type, **record}) + "\n")


def load_matching_runs(path: Path, experiment_id: str) -> tuple[list[dict], int]:
    if not path.exists():
        return [], 0

    matching: list[dict] = []
    stale_count = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}: {exc}"
                ) from exc
            if record.get("record_type") != "run":
                continue
            record.pop("record_type", None)
            if record.get("experiment_id") == experiment_id:
                matching.append(record)
            else:
                stale_count += 1
    return matching, stale_count


def prepare_results_file(
    *,
    path: Path,
    fresh: bool,
    experiment_id: str,
    cohort_names: list[str],
    torch_seeds: list[int],
    experiment: dict | None = None,
) -> tuple[list[dict], list[int]]:
    if experiment is not None and experiment.get("experiment_id") != experiment_id:
        raise ValueError("Experiment manifest ID does not match experiment_id.")
    runs, stale_count = (
        ([], 0)
        if fresh
        else load_matching_runs(path, experiment_id)
    )
    expected_targets = set(cohort_names)
    by_seed: dict[int, list[dict]] = {}
    for run in runs:
        by_seed.setdefault(run.get("torch_seed"), []).append(run)

    done_seeds = set()
    for seed, seed_runs in by_seed.items():
        targets = [run.get("target_cohort") for run in seed_runs]
        if (
            seed in torch_seeds
            and len(targets) == len(expected_targets)
            and set(targets) == expected_targets
            and all(run.get("target_heldout") is not None for run in seed_runs)
        ):
            done_seeds.add(seed)

    kept = [run for run in runs if run.get("torch_seed") in done_seeds]
    discarded = len(runs) - len(kept) + stale_count
    if discarded:
        print(
            f"discarding {discarded} stale, duplicate, or incomplete "
            f"run record(s) from {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    if experiment is not None:
        append_jsonl_record(path, "experiment", experiment)
    for run in kept:
        append_jsonl_record(path, "run", run)

    return kept, [seed for seed in torch_seeds if seed not in done_seeds]


def pooled_masked_stats(
    patients: list[PatientVolumes],
) -> tuple[float, float]:
    if not patients:
        raise ValueError("Cannot fit normalization on an empty patient list.")

    values = [patient.pet[patient.mask] for patient in patients]

    if any(part.size == 0 for part in values):
        raise ValueError("Cannot fit normalization with an empty patient mask.")

    pooled = np.concatenate(values)

    if not np.isfinite(pooled).all():
        raise ValueError("Cannot fit normalization with non-finite PET values.")

    mean = float(pooled.mean())
    std = float(pooled.std())

    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid pooled standard deviation: {std}")

    return mean, std


def _match_patient(
    patient: PatientVolumes,
    own_stats: tuple[float, float],
    reference_stats: tuple[float, float],
) -> PatientVolumes:
    own_mean, own_std = own_stats
    reference_mean, reference_std = reference_stats
    matched = (
        (patient.pet - own_mean) / own_std * reference_std
        + reference_mean
    ).astype(np.float32)

    result = copy(patient)
    result.pet = matched
    result.pet_masked = (matched * patient.mask).astype(np.float32)
    return result


def normalize_fold(
    all_cohorts: dict[str, list[PatientVolumes]],
    fit_patients: dict[str, list[PatientVolumes]],
    reference_cohort: str = REFERENCE_COHORT,
) -> dict[str, list[PatientVolumes]]:
    """Fit on allowed fold data and transform every patient without mutation."""
    missing = set(all_cohorts) - set(fit_patients)
    if missing:
        raise ValueError(f"Missing normalization fit data for: {sorted(missing)}")
    if reference_cohort not in fit_patients:
        raise ValueError(f"Reference cohort {reference_cohort!r} is unavailable.")

    stats = {
        cohort: pooled_masked_stats(patients)
        for cohort, patients in fit_patients.items()
    }
    reference_stats = stats[reference_cohort]

    return {
        cohort: [
            _match_patient(patient, stats[cohort], reference_stats)
            for patient in patients
        ]
        for cohort, patients in all_cohorts.items()
    }


def remap_patients(
    patients: list[PatientVolumes],
    normalized_cohorts: dict[str, list[PatientVolumes]],
) -> list[PatientVolumes]:
    lookup = {
        (patient.cohort, patient.patient_id): patient
        for cohort in normalized_cohorts.values()
        for patient in cohort
    }
    return [lookup[(patient.cohort, patient.patient_id)] for patient in patients]


def normalize_split(
    *,
    all_cohorts: dict[str, list[PatientVolumes]],
    cohort_train: dict[str, list[PatientVolumes]],
    cohort_val: dict[str, list[PatientVolumes]],
    target_cohort: str,
    target_harmonization: list[PatientVolumes],
    target_heldout: list[PatientVolumes],
) -> tuple[
    dict[str, list[PatientVolumes]],
    dict[str, list[PatientVolumes]],
    dict[str, list[PatientVolumes]],
    list[PatientVolumes],
    list[PatientVolumes],
]:
    """Normalize a completed split without fitting on validation/held-out data."""
    fit_patients = {
        cohort: (
            target_harmonization
            if cohort == target_cohort
            else cohort_train[cohort]
        )
        for cohort in all_cohorts
    }
    normalized = normalize_fold(all_cohorts, fit_patients)
    normalized_train = {
        cohort: remap_patients(patients, normalized)
        for cohort, patients in cohort_train.items()
    }
    normalized_val = {
        cohort: remap_patients(patients, normalized)
        for cohort, patients in cohort_val.items()
    }
    return (
        normalized,
        normalized_train,
        normalized_val,
        remap_patients(target_harmonization, normalized),
        remap_patients(target_heldout, normalized),
    )


def _hash_file(hasher, path: Path) -> None:
    hasher.update(str(path.resolve()).encode())
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)


def experiment_metadata(
    *,
    model: str,
    data_path: Path,
    cohort_names: list[str],
    zscore_correction: bool,
    parameters: dict,
    code_paths: list[Path],
) -> dict:
    """Create a stable fingerprint of data, code, and effective parameters."""
    data_path = data_path.resolve()
    dataset_hasher = hashlib.sha256()
    for cohort_name in cohort_names:
        cohort_path = data_path / cohort_name
        for path in sorted(cohort_path.glob("*.nii.gz")):
            _hash_file(dataset_hasher, path)

    code_hasher = hashlib.sha256()
    shared_paths = [Path(__file__), Path(__file__).with_name("nifti_loader.py")]
    unique_code_paths = {path.resolve() for path in [*code_paths, *shared_paths]}
    for path in sorted(unique_code_paths, key=str):
        _hash_file(code_hasher, path)

    config = {
        "schema_version": 1,
        "model": model,
        "data_path": str(data_path),
        "dataset_sha256": dataset_hasher.hexdigest(),
        "code_sha256": code_hasher.hexdigest(),
        "cohorts": cohort_names,
        "zscore_correction": zscore_correction,
        "parameters": parameters,
        "determinism": {
            "enabled": False,
            "rng_seeded": True,
            "cudnn_benchmark": True,
            "allow_tf32": True,
        },
        "runtime_versions": _package_versions(),
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        **config,
        "experiment_id": hashlib.sha256(encoded).hexdigest(),
    }
