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


# ── comparison report ─────────────────────────────────────────────────────────

def _summarise(all_cohorts: dict[str, list[PatientVolumes]]) -> pd.DataFrame:
    rows = []
    for cohort, patients in all_cohorts.items():
        for p in patients:
            voxels = p.pet_masked[p.mask].astype(np.float64)
            if voxels.size == 0:
                continue
            rows.append({
                "cohort":    cohort,
                "patient_id": p.patient_id,
                "mean":      voxels.mean(),
                "std":       voxels.std(),
                "min":       voxels.min(),
                "max":       voxels.max(),
            })
    return pd.DataFrame(rows)


def _cohort_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("cohort")[["mean", "std", "min", "max"]]
        .agg(["mean", "std"])
        .round(4)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Z-score normalisation demo")
    parser.add_argument("--data-root", default="CUBES-Labelled-COHORTS")
    args = parser.parse_args()

    root = Path(args.data_root)
    print(f"Loading cohorts from {root} …")
    all_cohorts = load_all_cohorts(root)
    for name, plist in all_cohorts.items():
        print(f"  {name}: {len(plist)} patients")

    print("\nApplying per-patient z-score normalisation …")
    normed_cohorts = zscore_all_cohorts(all_cohorts)

    before = _summarise(all_cohorts)
    after  = _summarise(normed_cohorts)

    print("\n── Before normalisation ─────────────────────────────────────────")
    print(_cohort_stats(before).to_string())

    print("\n── After normalisation ──────────────────────────────────────────")
    print(_cohort_stats(after).to_string())

    print("\n── Per-patient mean/std after normalisation (should be ≈ 0 / 1) ─")
    check = after.groupby("cohort")[["mean", "std"]].agg(["min", "max"]).round(6)
    print(check.to_string())


if __name__ == "__main__":
    main()
