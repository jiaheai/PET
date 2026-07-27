"""
Z-score normalisation for PET volumes.

Statistics (mean, std) are computed from within-prostate voxels only,
then applied to the full volume. Outside-mask voxels remain zero after
re-masking.

Usage
-----
    from normalize import zscore_patient, zscore_cohort

    normed = zscore_cohort(patients)

    # As a script: compare feature distributions before/after
    python normalize.py
"""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from nifti_loader import PatientVolumes, load_all_cohorts


def zscore_patient(patient: PatientVolumes) -> PatientVolumes:
    """Return a new PatientVolumes with per-patient z-score normalised PET.

    μ and σ are estimated from within-mask voxels only.
    """
    voxels = patient.pet[patient.mask]
    mu = float(voxels.mean())
    sigma = float(voxels.std())

    if sigma < 1e-8:
        raise ValueError(
            f"Patient {patient.patient_id}: near-zero std ({sigma:.2e}), cannot normalise"
        )

    pet_norm = (patient.pet - mu) / sigma
    pet_masked_norm = pet_norm * patient.mask

    normed = copy(patient)
    normed.pet = pet_norm.astype(np.float32)
    normed.pet_masked = pet_masked_norm.astype(np.float32)
    return normed


def zscore_cohort(patients: list[PatientVolumes]) -> list[PatientVolumes]:
    return [zscore_patient(p) for p in patients]


def zscore_all_cohorts(
    all_cohorts: dict[str, list[PatientVolumes]],
) -> dict[str, list[PatientVolumes]]:
    return {name: zscore_cohort(plist) for name, plist in all_cohorts.items()}


# ── saving volumes ────────────────────────────────────────────────────────────
#
# Output layout mirrors save_reconstructions() in the autoencoder script:
# out_root/<cohort>/{patient_id}_PET_zscore.nii.gz + {patient_id}_prostate_mask_res.nii.gz

def save_normalized_cohort(
    patients: list[PatientVolumes],
    out_root: str | Path,
) -> None:
    """Write each patient's z-score normalised PET *and* its prostate mask
    into out_root/<cohort>/, mirroring save_reconstructions()'s layout:

        out_root/{cohort}/{patient_id}_PET_zscore.nii.gz
        out_root/{cohort}/{patient_id}_prostate_mask_res.nii.gz
    """
    out_root = Path(out_root)

    for p in patients:
        out_dir = out_root / p.cohort
        out_dir.mkdir(parents=True, exist_ok=True)

        nib.save(
            nib.Nifti1Image(p.pet_masked, p.affine),
            str(out_dir / f"{p.patient_id}_PET_zscore.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(p.mask.astype(np.float32), p.affine),
            str(out_dir / f"{p.patient_id}_prostate_mask_res.nii.gz"),
        )


def save_normalized_all_cohorts(
    all_cohorts: dict[str, list[PatientVolumes]], out_root: str | Path
) -> None:
    for plist in all_cohorts.values():
        save_normalized_cohort(plist, out_root)


def save_original_cohort(patients: list[PatientVolumes], output_dir: str | Path) -> None:
    """Write each patient's original (un-normalised) PET into output_dir/<cohort>/."""
    output_dir = Path(output_dir)
    for p in patients:
        cohort_dir = output_dir / p.cohort
        cohort_dir.mkdir(parents=True, exist_ok=True)
        img = nib.Nifti1Image(p.pet, p.affine)
        nib.save(img, cohort_dir / f"{p.patient_id}_PET_res.nii.gz")


def save_original_all_cohorts(
    all_cohorts: dict[str, list[PatientVolumes]], output_dir: str | Path
) -> None:
    for plist in all_cohorts.values():
        save_original_cohort(plist, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Z-score normalisation demo")
    parser.add_argument("--data-root", default="CUBES-Labelled-COHORTS")
    parser.add_argument("--output-dir", default="CUBES-Labelled-COHORTS-ZSCORE")
    args = parser.parse_args()

    root = Path(args.data_root)
    print(f"Loading cohorts from {root} …")
    all_cohorts = load_all_cohorts(root)
    for name, plist in all_cohorts.items():
        print(f"  {name}: {len(plist)} patients")

    print("\nApplying per-patient z-score normalisation …")
    normed_cohorts = zscore_all_cohorts(all_cohorts)

    print(f"\nSaving normalised volumes (+ masks) to {args.output_dir} …")
    save_normalized_all_cohorts(normed_cohorts, args.output_dir)


if __name__ == "__main__":
    main()