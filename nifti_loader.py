"""
NIfTI loader for PET + prostate-mask image pairs.

Directory layout expected:
  <cohort_dir>/
    {patient_id}_PET_res_{N}.nii.gz
    {patient_id}_prostate_mask_res.nii.gz

Usage
-----
    from nifti_loader import load_cohort, load_patient

    patients = load_cohort("CUBES-Labelled-COHORTS/AUGSBURG")
    for p in patients:
        print(p.patient_id, p.pet_masked.shape)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass
class PatientVolumes:
    patient_id: str
    cohort: str
    pet: np.ndarray          # raw PET, shape (X, Y, Z)
    mask: np.ndarray         # binary prostate mask, shape (X, Y, Z)
    pet_masked: np.ndarray   # PET with outside-mask voxels zeroed
    affine: np.ndarray       # voxel-to-world affine from the PET image


def load_patient(pet_path: Path, mask_path: Path) -> PatientVolumes:
    pet_img = nib.load(pet_path)
    mask_img = nib.load(mask_path)

    pet_data = np.asarray(pet_img.dataobj, dtype=np.float32)
    mask_data = np.asarray(mask_img.dataobj, dtype=np.float32)

    # Binarise mask (tolerant of soft/probabilistic masks)
    binary_mask = mask_data > 0.5

    pet_masked = pet_data * binary_mask

    patient_id = _patient_id_from_path(pet_path)
    cohort = pet_path.parent.name

    return PatientVolumes(
        patient_id=patient_id,
        cohort=cohort,
        pet=pet_data,
        mask=binary_mask,
        pet_masked=pet_masked,
        affine=pet_img.affine,
    )


def load_cohort(cohort_dir: str | Path) -> list[PatientVolumes]:
    cohort_dir = Path(cohort_dir)
    if not cohort_dir.is_dir():
        raise FileNotFoundError(f"Cohort directory not found: {cohort_dir}")

    pet_files = sorted(cohort_dir.glob("*_PET_res_*.nii.gz"))
    if not pet_files:
        raise FileNotFoundError(f"No PET files found in {cohort_dir}")

    patients: list[PatientVolumes] = []
    missing: list[str] = []

    for pet_path in pet_files:
        pid = _patient_id_from_path(pet_path)
        mask_path = cohort_dir / f"{pid}_prostate_mask_res.nii.gz"
        if not mask_path.exists():
            missing.append(pid)
            continue
        patients.append(load_patient(pet_path, mask_path))

    if missing:
        print(f"Warning: no mask found for patient(s): {', '.join(missing)}")

    return patients


def load_all_cohorts(root_dir: str | Path) -> dict[str, list[PatientVolumes]]:
    root_dir = Path(root_dir)
    result: dict[str, list[PatientVolumes]] = {}
    for cohort_dir in sorted(root_dir.iterdir()):
        if cohort_dir.is_dir():
            try:
                result[cohort_dir.name] = load_cohort(cohort_dir)
            except FileNotFoundError as e:
                print(f"Skipping {cohort_dir.name}: {e}")
    return result


# ── helpers ──────────────────────────────────────────────────────────────────

_PET_ID_RE = re.compile(r"^(\d+)_PET_res_")


def _patient_id_from_path(pet_path: Path) -> str:
    m = _PET_ID_RE.match(pet_path.name)
    if m:
        return m.group(1)
    # Fallback: strip known suffixes
    return pet_path.name.split("_PET")[0]
