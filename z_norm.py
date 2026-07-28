"""
Cohort-level z-score normalisation for PET volumes.

Unlike a per-patient version (which computes each patient's own
mean/std and normalises them against themselves -- this erases
absolute intensity differences between patients, including whatever
part of that absolute level carries real classification signal), this
version pools voxels ACROSS an entire cohort to compute one shared
mean/std, then applies that single correction to every patient in the
other cohort. This corrects the cohort-level (site/scanner) intensity
shift while preserving each patient's intensity relative to their own
cohort's overall distribution.

AUGSBURG is treated as the reference cohort and saved UNCHANGED.
PRE-RAPID's pooled voxel distribution is z-score matched onto
AUGSBURG's pooled mean/std, then every PRE-RAPID patient's volume is
transformed using that single shared (mu, sigma) pair -- not their own
individual statistics.

Usage
-----
    python normalize.py --data-root CUBES-Labelled-COHORTS \
        --output-dir CUBES-Labelled-COHORTS-ZSCORE
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


def cohort_pooled_stats(patients: list[PatientVolumes]) -> tuple[float, float]:
    """Pool all within-mask voxel intensities across every patient in
    this cohort, return (mean, std) of the pooled distribution."""
    all_voxels = [p.pet[p.mask] for p in patients]
    pooled = np.concatenate(all_voxels)
    return float(pooled.mean()), float(pooled.std())


def zscore_match_patient(
    patient: PatientVolumes,
    own_cohort_mu: float,
    own_cohort_sigma: float,
    reference_mu: float,
    reference_sigma: float,
) -> PatientVolumes:
    """Transform this patient's PET using their OWN COHORT's pooled
    (mu, sigma) -- not their own individual statistics -- then rescale
    to the reference cohort's pooled (mu, sigma).

    This is: standardize against the cohort, then match to the
    reference cohort -- the same "standardize, then match" pattern
    used for the probability-level correction elsewhere in this
    project, just applied to voxel intensities and using cohort-wide
    (not per-patient) statistics.
    """
    z = (patient.pet - own_cohort_mu) / own_cohort_sigma
    pet_matched = z * reference_sigma + reference_mu
    pet_masked_matched = pet_matched * patient.mask

    result = copy(patient)
    result.pet = pet_matched.astype(np.float32)
    result.pet_masked = pet_masked_matched.astype(np.float32)
    return result


def zscore_match_cohort(
    target_patients: list[PatientVolumes],
    reference_patients: list[PatientVolumes],
) -> list[PatientVolumes]:
    """z-score match every patient in target_patients onto the pooled
    distribution of reference_patients, using cohort-level (not
    per-patient) statistics for both sides."""
    ref_mu, ref_sigma = cohort_pooled_stats(reference_patients)
    own_mu, own_sigma = cohort_pooled_stats(target_patients)

    print(f"reference ({reference_patients[0].cohort}) pooled stats: mu={ref_mu:.4f}  sigma={ref_sigma:.4f}")
    print(f"target ({target_patients[0].cohort}) pooled stats:    mu={own_mu:.4f}  sigma={own_sigma:.4f}")

    return [
        zscore_match_patient(p, own_mu, own_sigma, ref_mu, ref_sigma)
        for p in target_patients
    ]


# -- saving volumes ------------------------------------------------------------
# Same filename convention as elsewhere in this project, so
# load_all_cohorts() can find and parse these files unchanged:
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
    parser = argparse.ArgumentParser(description="Cohort-level z-score normalisation demo")
    parser.add_argument("--data-root", default="CUBES-Labelled-COHORTS")
    parser.add_argument("--output-dir", default="CUBES-Labelled-COHORTS-ZSCORE")
    args = parser.parse_args()

    root = Path(args.data_root)
    print(f"Loading cohorts from {root} ...")
    all_cohorts = load_all_cohorts(root)
    for name, plist in all_cohorts.items():
        print(f"  {name}: {len(plist)} patients")

    reference_patients = all_cohorts[REFERENCE_COHORT]
    target_patients    = all_cohorts[TARGET_COHORT]

    print(f"\nz-score matching {TARGET_COHORT} onto {REFERENCE_COHORT}'s pooled cohort statistics ...")
    matched_target = zscore_match_cohort(target_patients, reference_patients)

    print(f"\nSaving {REFERENCE_COHORT} (unchanged) and {TARGET_COHORT} (matched) to {args.output_dir} ...")
    save_cohort(reference_patients, args.output_dir)   # AUGSBURG saved as-is, unchanged
    save_cohort(matched_target, args.output_dir)        # PRE-RAPID saved matched


if __name__ == "__main__":
    main()