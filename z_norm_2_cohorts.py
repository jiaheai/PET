from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

import nibabel as nib
import numpy as np

from nifti_loader import PatientVolumes, load_all_cohorts


REFERENCE_COHORT = "SWISS"
COHORT_NAMES = ["AUGSBURG", "SWISS"]


def cohort_pooled_stats(
    patients: list[PatientVolumes],
) -> tuple[float, float]:
    all_voxels = [p.pet[p.mask] for p in patients]
    pooled = np.concatenate(all_voxels)

    mu = float(pooled.mean())
    sigma = float(pooled.std())

    if sigma == 0:
        raise ValueError(
            f"pooled standard deviation is zero for cohort "
            f"{patients[0].cohort!r}"
        )

    return mu, sigma


def zscore_match_patient(
    patient: PatientVolumes,
    own_mu: float,
    own_sigma: float,
    reference_mu: float,
    reference_sigma: float,
) -> PatientVolumes:
    z = (patient.pet - own_mu) / own_sigma
    pet_matched = z * reference_sigma + reference_mu
    pet_masked_matched = pet_matched * patient.mask

    result = copy(patient)
    result.pet = pet_matched.astype(np.float32)
    result.pet_masked = pet_masked_matched.astype(np.float32)
    return result


def transform_cohort(
    patients: list[PatientVolumes],
    own_mu: float,
    own_sigma: float,
    reference_mu: float,
    reference_sigma: float,
) -> list[PatientVolumes]:
    return [
        zscore_match_patient(
            patient,
            own_mu,
            own_sigma,
            reference_mu,
            reference_sigma,
        )
        for patient in patients
    ]


def save_cohort(
    patients: list[PatientVolumes],
    out_root: str | Path,
) -> None:
    out_root = Path(out_root)

    for patient in patients:
        out_dir = out_root / patient.cohort
        out_dir.mkdir(parents=True, exist_ok=True)

        nib.save(
            nib.Nifti1Image(
                patient.pet_masked,
                patient.affine,
            ),
            str(
                out_dir
                / f"{patient.patient_id}_PET_res_{patient.label}.nii.gz"
            ),
        )

        nib.save(
            nib.Nifti1Image(
                patient.mask.astype(np.float32),
                patient.affine,
            ),
            str(
                out_dir
                / f"{patient.patient_id}_prostate_mask_res.nii.gz"
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Two-cohort AUGSBURG/SWISS z-score matching from one original "
            "dataset directory. PRE-RAPID is ignored."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="CUBES-Labelled-COHORTS_2",
    )
    parser.add_argument(
        "--output-dir",
        default="CUBES-Labelled-COHORTS_ZSCORE_2COHORT",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    output_root = Path(args.output_dir)

    print(f"Loading original dataset from {data_root} ...")
    all_cohorts = load_all_cohorts(data_root)

    cohorts = {
        name: all_cohorts[name]
        for name in COHORT_NAMES
        if name in all_cohorts
    }

    for cohort_name in COHORT_NAMES:
        if cohort_name not in cohorts:
            raise ValueError(
                f"Cohort {cohort_name!r} missing from {data_root}"
            )

    print("\nUsing ONLY these cohorts:")
    for cohort_name in COHORT_NAMES:
        print(f"  {cohort_name}")
    print("PRE-RAPID is not used for statistics, transformation, or output.")

    stats: dict[str, tuple[float, float]] = {}

    print("\nFitting statistics from each full cohort:")
    for cohort_name in COHORT_NAMES:
        mu, sigma = cohort_pooled_stats(cohorts[cohort_name])
        stats[cohort_name] = (mu, sigma)

        print(
            f"  {cohort_name:12s} "
            f"mu={mu:.6f}  sigma={sigma:.6f}"
        )

    reference_mu, reference_sigma = stats[REFERENCE_COHORT]

    print(f"\nReference cohort: {REFERENCE_COHORT}")

    for cohort_name in COHORT_NAMES:
        own_mu, own_sigma = stats[cohort_name]

        transformed = transform_cohort(
            cohorts[cohort_name],
            own_mu,
            own_sigma,
            reference_mu,
            reference_sigma,
        )

        save_cohort(
            transformed,
            output_root,
        )

        print(
            f"{cohort_name:12s}  "
            f"saved={len(transformed)}"
        )

    print(f"\nNormalized dataset: {output_root}")


if __name__ == "__main__":
    main()
