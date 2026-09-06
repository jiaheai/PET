from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from nifti_loader import load_all_cohorts


COHORT_NAMES = ["AUGSBURG", "PRE-RAPID", "SWISS"]
SPLIT_SEED = 123


def copy_patient(patient, source_root: Path, output_root: Path) -> None:
    source_dir = source_root / patient.cohort
    output_dir = output_root / patient.cohort
    output_dir.mkdir(parents=True, exist_ok=True)

    pet_name = f"{patient.patient_id}_PET_res_{patient.label}.nii.gz"
    mask_name = f"{patient.patient_id}_prostate_mask_res.nii.gz"

    pet_src = source_dir / pet_name
    mask_src = source_dir / mask_name

    if not pet_src.exists():
        raise FileNotFoundError(f"missing PET file: {pet_src}")
    if not mask_src.exists():
        raise FileNotFoundError(f"missing mask file: {mask_src}")

    shutil.copy2(pet_src, output_dir / pet_name)
    shutil.copy2(mask_src, output_dir / mask_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split each cohort 50/50 into harmonization and held-out datasets."
    )
    parser.add_argument("--data-root", default="CUBES-Labelled-COHORTS")
    parser.add_argument(
        "--harmonization-dir",
        default="CUBES-Labelled-COHORTS_Harmonization",
    )
    parser.add_argument(
        "--heldout-dir",
        default="CUBES-Labelled-COHORTS_Heldout",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into existing output directories.",
    )
    args = parser.parse_args()

    source_root = Path(args.data_root)
    harmonization_root = Path(args.harmonization_dir)
    heldout_root = Path(args.heldout_dir)

    for output_root in (harmonization_root, heldout_root):
        if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"{output_root} already exists and is not empty. "
                f"Use --overwrite if you want to write into it."
            )
        output_root.mkdir(parents=True, exist_ok=True)

    all_cohorts = load_all_cohorts(source_root)

    for cohort_name in COHORT_NAMES:
        if cohort_name not in all_cohorts:
            raise ValueError(
                f"Cohort {cohort_name!r} not found. "
                f"Available cohorts: {list(all_cohorts.keys())}"
            )

        patients = all_cohorts[cohort_name]

        harmonization, heldout = train_test_split(
            patients,
            test_size=0.5,
            random_state=SPLIT_SEED,
            stratify=[p.label for p in patients],
        )

        for patient in harmonization:
            copy_patient(patient, source_root, harmonization_root)

        for patient in heldout:
            copy_patient(patient, source_root, heldout_root)

        harm_pos = sum(p.label for p in harmonization)
        held_pos = sum(p.label for p in heldout)

        print(
            f"{cohort_name:12s}  "
            f"harmonization={len(harmonization):3d} "
            f"(pos={harm_pos:3d}, neg={len(harmonization) - harm_pos:3d})  "
            f"heldout={len(heldout):3d} "
            f"(pos={held_pos:3d}, neg={len(heldout) - held_pos:3d})"
        )

    print(f"\nharmonization dataset: {harmonization_root}")
    print(f"held-out dataset      : {heldout_root}")


if __name__ == "__main__":
    main()
