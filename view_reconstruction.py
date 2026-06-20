"""
View a reconstructed PET volume using the interactive slice viewer.

Usage
-----
    python view_reconstruction.py
        -- first patient in AUGSBURG reconstructions

    python view_reconstruction.py --cohort PRE-RAPID --patient 763 --axis z

    python view_reconstruction.py --compare
        -- show original and reconstruction side-by-side
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from nifti_loader import PatientVolumes, load_cohort
from visualizer import AXIS_KEYS, SliceViewer

RECON_ROOT = Path(__file__).parent / "reconstructions"
DATA_ROOT = Path(__file__).parent / "CUBES-Labelled-COHORTS"


def load_reconstruction(cohort: str, patient_id: str) -> PatientVolumes:
    path = RECON_ROOT / cohort / f"{patient_id}_PET_reconstructed.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Reconstruction not found: {path}")
    img = nib.load(path)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    return PatientVolumes(
        patient_id=patient_id,
        cohort=cohort,
        pet=vol,
        mask=None,
        pet_masked=vol,
        affine=img.affine,
    )


def available_patients(cohort: str) -> list[str]:
    cohort_dir = RECON_ROOT / cohort
    return sorted(p.name.replace("_PET_reconstructed.nii.gz", "")
                  for p in cohort_dir.glob("*_PET_reconstructed.nii.gz"))


def compare_side_by_side(original: PatientVolumes, recon: PatientVolumes, axis: int) -> None:
    """Show original and reconstruction as linked slice viewers."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"Patient {original.patient_id} | {original.cohort} | "
        f"Left: original   Right: reconstruction"
    )

    vmax = float(np.nanpercentile(original.pet, 99.5)) or 1.0
    state = {"idx": original.pet_masked.shape[axis] // 2, "axis": axis}

    def get_slice(vol, idx, ax):
        slices = [slice(None)] * 3
        slices[ax] = idx
        return vol[tuple(slices)].T

    def draw():
        for ax_obj, vol, label in [
            (axes[0], original.pet_masked, "original"),
            (axes[1], recon.pet_masked, "reconstruction"),
        ]:
            ax_obj.clear()
            ax_obj.imshow(
                get_slice(vol, state["idx"], state["axis"]),
                origin="lower", cmap="hot", vmin=0, vmax=vmax, aspect="equal",
            )
            ax_obj.set_title(f"{label} | slice {state['idx']}")
            ax_obj.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        n = original.pet_masked.shape[state["axis"]]
        if event.key in ("right", "d"):
            state["idx"] = min(state["idx"] + 1, n - 1)
        elif event.key in ("left", "a"):
            state["idx"] = max(state["idx"] - 1, 0)
        elif event.key in AXIS_KEYS:
            state["axis"] = AXIS_KEYS[event.key]
            state["idx"] = original.pet_masked.shape[state["axis"]] // 2
        elif event.key in ("q", "escape"):
            plt.close(fig)
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="View reconstructed PET volumes")
    parser.add_argument("--cohort", default="AUGSBURG", help="Cohort name (default: AUGSBURG)")
    parser.add_argument("--patient", default=None, help="Patient ID (default: first available)")
    parser.add_argument("--axis", default="z", choices=["x", "y", "z"])
    parser.add_argument("--compare", action="store_true",
                        help="Show original and reconstruction side-by-side")
    args = parser.parse_args()

    cohort_dir = RECON_ROOT / args.cohort
    if not cohort_dir.exists():
        available = [d.name for d in RECON_ROOT.iterdir() if d.is_dir()]
        print(f"Cohort '{args.cohort}' not found. Available: {available}", file=sys.stderr)
        sys.exit(1)

    patients = available_patients(args.cohort)
    if not patients:
        print("No reconstructions found.", file=sys.stderr)
        sys.exit(1)

    patient_id = args.patient or patients[0]
    if patient_id not in patients:
        print(f"Patient '{patient_id}' not found. Available: {patients}", file=sys.stderr)
        sys.exit(1)

    recon = load_reconstruction(args.cohort, patient_id)
    print(f"Loaded reconstruction for {patient_id} ({args.cohort})")
    print(f"  shape: {recon.pet.shape}  range: [{recon.pet.min():.2f}, {recon.pet.max():.2f}]")

    if args.compare:
        orig_patients = load_cohort(DATA_ROOT / args.cohort)
        match = [p for p in orig_patients if p.patient_id == patient_id]
        if not match:
            print(f"Original data for '{patient_id}' not found.", file=sys.stderr)
            sys.exit(1)
        compare_side_by_side(match[0], recon, axis=AXIS_KEYS[args.axis])
    else:
        SliceViewer(recon, axis=AXIS_KEYS[args.axis]).show()


if __name__ == "__main__":
    main()
