"""
Cohort-level histogram matching for PET volumes.

Unlike normalize.py's per-patient z-score normalization (which
destroyed each patient's absolute intensity information and, with it,
apparently the classification signal itself -- see CNN results after
per-patient normalization: probabilities collapsed to a ~0.02-wide
band, AUC dropped below chance), this operates at the COHORT level:

  - Pool all within-mask voxels from every AUGSBURG patient together
    into one big reference distribution.
  - Pool all within-mask voxels from every PRE-RAPID patient together.
  - Map PRE-RAPID's pooled distribution onto AUGSBURG's, percentile by
    percentile (histogram matching), using ONE shared mapping function
    for every PRE-RAPID patient.

This corrects the cohort-level intensity shift (the actual target)
while preserving each patient's relative intensity differences from
other patients in their own cohort -- unlike per-patient normalization,
no single patient's own distribution is used to normalize themselves,
so absolute intensity differences between patients within a cohort are
NOT erased.

AUGSBURG's own volumes are saved UNCHANGED (they define the reference
distribution; there's nothing to correct them against). Only PRE-RAPID
is transformed.

Usage
-----
    python histogram_match.py --data-root CUBES-Labelled-COHORTS \\
        --output-dir CUBES-Labelled-COHORTS-HISTMATCH
"""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

import nibabel as nib
import numpy as np

from nifti_loader import PatientVolumes, load_all_cohorts

REFERENCE_COHORT = "AUGSBURG"
TARGET_COHORT    = "PRE-RAPID"


def build_reference_distribution(reference_patients: list[PatientVolumes]) -> np.ndarray:
    """Pool all within-mask voxel intensities from every patient in the
    reference cohort into one big sorted array -- this defines the
    target shape everything else gets mapped onto."""
    all_voxels = []
    for p in reference_patients:
        all_voxels.append(p.pet[p.mask])
    pooled = np.concatenate(all_voxels)
    return np.sort(pooled)


def histogram_match_patient(
    patient: PatientVolumes,
    target_own_cohort_voxels: np.ndarray,   # sorted, pooled voxels from patient's OWN cohort
    reference_sorted: np.ndarray,            # sorted, pooled voxels from the reference cohort
) -> PatientVolumes:
    """Map this patient's PET intensities onto the reference cohort's
    distribution, using percentile rank within their OWN cohort's
    pooled distribution (not their own individual distribution --
    that's the per-patient version that failed).
    """
    voxels = patient.pet[patient.mask]

    # percentile (0-1) of each of this patient's voxels within their
    # OWN COHORT's pooled distribution (not the patient's own values only)
    percentiles = np.searchsorted(target_own_cohort_voxels, voxels, side="right") / len(target_own_cohort_voxels)
    percentiles = np.clip(percentiles, 0.0, 1.0)

    # map those percentiles onto the reference cohort's pooled distribution
    reference_positions = np.linspace(0, 1, len(reference_sorted))
    matched_voxels = np.interp(percentiles, reference_positions, reference_sorted)

    pet_matched = patient.pet.copy()
    pet_matched[patient.mask] = matched_voxels
    pet_masked_matched = pet_matched * patient.mask

    result = copy(patient)
    result.pet = pet_matched.astype(np.float32)
    result.pet_masked = pet_masked_matched.astype(np.float32)
    return result


def histogram_match_cohort(
    target_patients: list[PatientVolumes],
    reference_patients: list[PatientVolumes],
) -> list[PatientVolumes]:
    """Histogram-match every patient in target_patients onto the pooled
    distribution of reference_patients."""
    reference_sorted = build_reference_distribution(reference_patients)
    target_own_sorted = build_reference_distribution(target_patients)   # target's own pooled shape

    return [
        histogram_match_patient(p, target_own_sorted, reference_sorted)
        for p in target_patients
    ]


# -- saving volumes ------------------------------------------------------------
# Same filename convention as normalize.py / save_harmonized_reconstructions,
# so load_all_cohorts() can find and parse these files unchanged:
#   out_root/<cohort>/{patient_id}_PET_res_{label}.nii.gz
#   out_root/<cohort>/{patient_id}_prostate_mask_res.nii.gz

def save_cohort(patients: list[PatientVolumes], out_root: str | Path) -> None:
    out_root = Path(out_root)
    for p in patients:
        out_dir = out_root / p.cohort
        out_dir.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(p.pet_masked, p.affine),
            str(out_dir / f"{p.patient_id}_PET_res_{p.label}.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(p.mask.astype(np.float32), p.affine),
            str(out_dir / f"{p.patient_id}_prostate_mask_res.nii.gz"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cohort-level histogram matching demo")
    parser.add_argument("--data-root", default="CUBES-Labelled-COHORTS")
    parser.add_argument("--output-dir", default="CUBES-Labelled-COHORTS-HISTMATCH")
    args = parser.parse_args()

    root = Path(args.data_root)
    print(f"Loading cohorts from {root} ...")
    all_cohorts = load_all_cohorts(root)
    for name, plist in all_cohorts.items():
        print(f"  {name}: {len(plist)} patients")

    reference_patients = all_cohorts[REFERENCE_COHORT]
    target_patients    = all_cohorts[TARGET_COHORT]

    print(f"\nHistogram-matching {TARGET_COHORT} onto {REFERENCE_COHORT}'s pooled distribution ...")
    matched_target = histogram_match_cohort(target_patients, reference_patients)

    print(f"\nSaving {REFERENCE_COHORT} (unchanged) and {TARGET_COHORT} (matched) to {args.output_dir} ...")
    save_cohort(reference_patients, args.output_dir)   # AUGSBURG saved as-is, unchanged
    save_cohort(matched_target, args.output_dir)        # PRE-RAPID saved matched


if __name__ == "__main__":
    main()