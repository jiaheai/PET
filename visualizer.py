"""
Simple interactive slice viewer for 3-D NIfTI volumes.

Keyboard controls
-----------------
  Left / Right arrows (or A / D) : previous / next slice
  X / Y / Z                      : switch axis
  M                               : toggle mask overlay
  Q or Escape                     : quit

Usage
-----
    python visualizer.py
        -- loads first patient from AUGSBURG cohort

    python visualizer.py --cohort PRE-RAPID --patient 763 --axis z
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nifti_loader import PatientVolumes, load_cohort, load_patient

AXIS_LABELS = {0: "X (sagittal)", 1: "Y (coronal)", 2: "Z (axial)"}
AXIS_KEYS = {"x": 0, "y": 1, "z": 2}


class SliceViewer:
    def __init__(self, patient: PatientVolumes, axis: int = 2, show_masked: bool = True):
        self.patient = patient
        self.axis = axis
        self.show_masked = show_masked

        vol = patient.pet_masked if show_masked else patient.pet
        self._update_volume(vol)

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw()
        plt.tight_layout()

    # ── private ──────────────────────────────────────────────────────────────

    def _update_volume(self, vol: np.ndarray) -> None:
        self.vol = vol
        n_slices = vol.shape[self.axis]
        self.idx = n_slices // 2

    def _current_slice(self) -> np.ndarray:
        slices = [slice(None)] * 3
        slices[self.axis] = self.idx
        return self.vol[tuple(slices)]

    def _draw(self) -> None:
        self.ax.clear()
        img = self._current_slice().T  # transpose so rows=Y, cols=X
        vmax = np.nanpercentile(self.patient.pet, 99.5) or 1.0

        self.ax.imshow(img, origin="lower", cmap="hot", vmin=0, vmax=vmax, aspect="equal")
        self.ax.set_title(
            f"Patient {self.patient.patient_id} | {self.patient.cohort} | "
            f"Axis {AXIS_LABELS[self.axis]} | Slice {self.idx}/{self.vol.shape[self.axis]-1} | "
            f"{'masked' if self.show_masked else 'raw PET'}"
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        key = event.key
        n = self.vol.shape[self.axis]

        if key in ("right", "d"):
            self.idx = min(self.idx + 1, n - 1)
        elif key in ("left", "a"):
            self.idx = max(self.idx - 1, 0)
        elif key in AXIS_KEYS:
            self.axis = AXIS_KEYS[key]
            self._update_volume(self.patient.pet_masked if self.show_masked else self.patient.pet)
        elif key == "m":
            self.show_masked = not self.show_masked
            vol = self.patient.pet_masked if self.show_masked else self.patient.pet
            self._update_volume(vol)
        elif key in ("q", "escape"):
            plt.close(self.fig)
            return

        self._draw()

    def show(self) -> None:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).parent / "CUBES-Labelled-COHORTS"

    parser = argparse.ArgumentParser(description="NIfTI PET slice viewer")
    parser.add_argument("--cohort", default="AUGSBURG", help="Cohort name (default: AUGSBURG)")
    parser.add_argument("--patient", default=None, help="Patient ID (default: first in cohort)")
    parser.add_argument("--axis", default="z", choices=["x", "y", "z"], help="Initial scroll axis")
    parser.add_argument("--raw", action="store_true", help="Show raw PET instead of masked")
    args = parser.parse_args()

    cohort_dir = root / args.cohort
    if not cohort_dir.exists():
        available = [d.name for d in root.iterdir() if d.is_dir()]
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
    print(f"  PET shape : {patient.pet.shape}")
    print(f"  Mask shape: {patient.mask.shape}")
    print(f"  PET range : [{patient.pet.min():.2f}, {patient.pet.max():.2f}]")

    viewer = SliceViewer(patient, axis=AXIS_KEYS[args.axis], show_masked=not args.raw)
    viewer.show()


if __name__ == "__main__":
    main()
