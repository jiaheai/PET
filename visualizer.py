"""
Interactive slice viewer for 3-D PET NIfTI volumes.

Single view (original):
    python visualizer.py
    python visualizer.py --cohort PRE-RAPID --patient 763 --axis z
    python visualizer.py --raw

Side-by-side comparison (original vs reconstruction):
    python visualizer.py --recon
    python visualizer.py --cohort PRE-RAPID --patient 763 --recon

Side-by-side comparison (original vs harmonized reconstruction):
    python visualizer.py --recon --harmonized
    python visualizer.py --cohort PRE-RAPID --patient 763 --recon --harmonized

Keyboard controls
-----------------
  Left / Right arrows (or A / D) : previous / next slice
  X / Y / Z                      : switch axis
  M                               : toggle mask overlay (single view only)
  Q or Escape                     : quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from nifti_loader import PatientVolumes, load_cohort

DATA_ROOT       = Path(__file__).parent / "CUBES-Labelled-COHORTS"
RECON_ROOT      = Path(__file__).parent / "reconstructions"
HARMONIZED_ROOT = Path(__file__).parent / "harmonized_reconstructions"

AXIS_LABELS = {0: "X (sagittal)", 1: "Y (coronal)", 2: "Z (axial)"}
AXIS_KEYS = {"x": 0, "y": 1, "z": 2}


# ── single-volume viewer ──────────────────────────────────────────────────────

class SliceViewer:
    def __init__(self, patient: PatientVolumes, axis: int = 2, show_masked: bool = True):
        self.patient = patient
        self.axis = axis
        self.show_masked = show_masked
        self._update_volume(patient.pet_masked if show_masked else patient.pet)

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()
        plt.tight_layout()

    def _update_volume(self, vol: np.ndarray) -> None:
        self.vol = vol
        self.idx = vol.shape[self.axis] // 2

    def _current_slice(self) -> np.ndarray:
        slices = [slice(None)] * 3
        slices[self.axis] = self.idx
        return self.vol[tuple(slices)]

    def _draw(self) -> None:
        self.ax.clear()
        vmax = float(np.nanpercentile(self.patient.pet, 99.5)) or 1.0
        self.ax.imshow(
            self._current_slice().T,
            origin="lower", cmap="hot", vmin=0, vmax=vmax, aspect="equal",
        )
        self.ax.set_title(
            f"Patient {self.patient.patient_id} | {self.patient.cohort} | "
            f"{AXIS_LABELS[self.axis]} | slice {self.idx}/{self.vol.shape[self.axis]-1} | "
            f"{'masked' if self.show_masked else 'raw PET'}"
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        n = self.vol.shape[self.axis]
        key = event.key
        if key in ("right", "d"):
            self.idx = min(self.idx + 1, n - 1)
        elif key in ("left", "a"):
            self.idx = max(self.idx - 1, 0)
        elif key in AXIS_KEYS:
            self.axis = AXIS_KEYS[key]
            self._update_volume(self.patient.pet_masked if self.show_masked else self.patient.pet)
        elif key == "m":
            self.show_masked = not self.show_masked
            self._update_volume(self.patient.pet_masked if self.show_masked else self.patient.pet)
        elif key in ("q", "escape"):
            plt.close(self.fig)
            return
        self._draw()

    def show(self) -> None:
        plt.show()


# ── side-by-side comparison viewer ───────────────────────────────────────────

class CompareViewer:
    """Original (left) vs reconstruction (right) with a shared vmax.

    vmax is taken from the original's 99.5th percentile so both panels use
    the same colour scale and brightness differences are meaningful.
    """

    def __init__(self, original: PatientVolumes, recon: PatientVolumes, axis: int = 2,
                 recon_label: str = "reconstruction"):
        self.original = original
        self.recon = recon
        self.axis = axis
        self.recon_label = recon_label
        self.idx = original.pet_masked.shape[axis] // 2
        self.vmax = float(np.nanpercentile(original.pet, 99.5)) or 1.0

        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()
        plt.tight_layout()

    def _get_slice(self, vol: np.ndarray) -> np.ndarray:
        slices = [slice(None)] * 3
        slices[self.axis] = self.idx
        return vol[tuple(slices)].T

    def _draw(self) -> None:
        n = self.original.pet_masked.shape[self.axis]
        for ax_obj, vol, label in [
            (self.axes[0], self.original.pet_masked, "original"),
            (self.axes[1], self.recon.pet_masked,    self.recon_label),
        ]:
            ax_obj.clear()
            ax_obj.imshow(
                self._get_slice(vol),
                origin="lower", cmap="hot", vmin=0, vmax=self.vmax, aspect="equal",
            )
            ax_obj.set_title(f"{label} | {AXIS_LABELS[self.axis]} | slice {self.idx}/{n - 1}")
            ax_obj.axis("off")
        self.fig.suptitle(f"Patient {self.original.patient_id} | {self.original.cohort}")
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        n = self.original.pet_masked.shape[self.axis]
        key = event.key
        if key in ("right", "d"):
            self.idx = min(self.idx + 1, n - 1)
        elif key in ("left", "a"):
            self.idx = max(self.idx - 1, 0)
        elif key in AXIS_KEYS:
            self.axis = AXIS_KEYS[key]
            self.idx = self.original.pet_masked.shape[self.axis] // 2
        elif key in ("q", "escape"):
            plt.close(self.fig)
            return
        self._draw()

    def show(self) -> None:
        plt.show()


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_reconstruction(cohort: str, patient_id: str, harmonized: bool = False) -> PatientVolumes:
    if harmonized:
        path = HARMONIZED_ROOT / cohort / f"{patient_id}_PET_harmonized.nii.gz"
    else:
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PET slice viewer")
    parser.add_argument("--cohort",     default="AUGSBURG")
    parser.add_argument("--patient",    default=None, help="Patient ID (default: first in cohort)")
    parser.add_argument("--axis",       default="z", choices=["x", "y", "z"])
    parser.add_argument("--raw",        action="store_true", help="Show raw PET instead of masked")
    parser.add_argument("--recon",      action="store_true",
                        help="Show original and reconstruction side-by-side")
    parser.add_argument("--harmonized", action="store_true",
                        help="Use harmonized_reconstructions instead of reconstructions (requires --recon)")
    args = parser.parse_args()

    if args.harmonized and not args.recon:
        parser.error("--harmonized requires --recon")

    cohort_dir = DATA_ROOT / args.cohort
    if not cohort_dir.exists():
        available = [d.name for d in DATA_ROOT.iterdir() if d.is_dir()]
        print(f"Cohort '{args.cohort}' not found. Available: {available}", file=sys.stderr)
        sys.exit(1)

    patients = load_cohort(cohort_dir)
    if not patients:
        print("No patients loaded.", file=sys.stderr)
        sys.exit(1)

    if args.patient:
        match = [p for p in patients if p.patient_id == args.patient]
        if not match:
            ids = [p.patient_id for p in patients]
            print(f"Patient '{args.patient}' not found. Available: {ids}", file=sys.stderr)
            sys.exit(1)
        patient = match[0]
    else:
        patient = patients[0]

    print(f"Loaded patient {patient.patient_id} from {patient.cohort}")
    print(f"  PET shape : {patient.pet.shape}  range: [{patient.pet.min():.2f}, {patient.pet.max():.2f}]")

    axis = AXIS_KEYS[args.axis]

    if args.recon:
        try:
            recon = _load_reconstruction(args.cohort, patient.patient_id, harmonized=args.harmonized)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        recon_label = "harmonized" if args.harmonized else "reconstruction"
        print(f"  Recon range: [{recon.pet.min():.2f}, {recon.pet.max():.2f}]")
        CompareViewer(patient, recon, axis=axis, recon_label=recon_label).show()
    else:
        SliceViewer(patient, axis=axis, show_masked=not args.raw).show()


if __name__ == "__main__":
    main()