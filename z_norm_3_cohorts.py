from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

from experiment_utils import normalize_fold, pooled_masked_stats
from nifti_loader import PatientVolumes, load_all_cohorts


REFERENCE_COHORT = "SWISS"
COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "SWISS"]
VAL_SPLIT_SEED = 40
TARGET_HOLDOUT_SEED = 123


def cohort_pooled_stats(
    patients: list[PatientVolumes],
) -> tuple[float, float]:
    return pooled_masked_stats(patients)


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
            "Build one leakage-safe three-cohort normalization fold. Statistics "
            "are fitted on source-training data and the target harmonization half."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="CUBES-Labelled-COHORTS_3",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: derived from --target-cohort).",
    )
    parser.add_argument(
        "--target-cohort",
        required=True,
        choices=COHORT_NAMES,
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    output_root = Path(
        args.output_dir
        or f"CUBES-Labelled-COHORTS_ZSCORE_3COHORT_target-{args.target_cohort}"
    )

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

    print("\nUsing these cohorts:")
    for cohort_name in COHORT_NAMES:
        print(f"  {cohort_name}")

    fit_patients: dict[str, list[PatientVolumes]] = {}
    print("\nFitting statistics from allowed fold data:")
    for cohort_name in COHORT_NAMES:
        patients = cohorts[cohort_name]
        if cohort_name == args.target_cohort:
            fit, _ = train_test_split(
                patients,
                test_size=0.5,
                random_state=TARGET_HOLDOUT_SEED,
                stratify=[patient.label for patient in patients],
            )
        else:
            fit, _ = train_test_split(
                patients,
                test_size=0.2,
                random_state=VAL_SPLIT_SEED,
                stratify=[patient.label for patient in patients],
            )
        fit_patients[cohort_name] = fit
        mu, sigma = cohort_pooled_stats(fit)

        print(
            f"  {cohort_name:12s} "
            f"n_fit={len(fit):3d}  mu={mu:.6f}  sigma={sigma:.6f}"
        )

    normalized = normalize_fold(cohorts, fit_patients, REFERENCE_COHORT)
    print(f"\nReference cohort: {REFERENCE_COHORT}")

    for cohort_name in COHORT_NAMES:
        print(
            f"\nTransforming {cohort_name} ..."
        )

        save_cohort(
            normalized[cohort_name],
            output_root,
        )

        print(
            f"  saved={len(normalized[cohort_name])}"
        )

    print(
        f"\nNormalized dataset: {output_root}"
    )


if __name__ == "__main__":
    main()
