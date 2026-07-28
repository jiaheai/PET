"""
Runs every (approach x image-correction) combination in the project's
comparison matrix and writes one consolidated CSV, matching the table
format used for the results spreadsheet:

    Approach, Image correction, Prob correction, AUC, Bal Acc, Recall(+), Recall(-)

All four sweep scripts (a_sweep_norm.py, b_sweep_norm.py, c_sweep_norm.py,
cnn_sweep.py) now share the same result schema -- each seed's result has
"train_cohort" (in-sample, reference only), "holdout_cohort" (raw),
"holdout_cohort_corrected" (z-score corrected), each a dict with
acc/auc/recall_pos/recall_neg/balanced_acc -- so one parsing function
handles all of them.

Usage
-----
    python run_all_and_export_csv.py                  # run everything, then export
    python run_all_and_export_csv.py --skip-run        # only re-export from
                                                          existing .jsonl files
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# -- comparison matrix -----------------------------------------------------

DATA_VARIANTS = [
    ("none",      "CUBES-Labelled-COHORTS"),
    ("z-score",   "CUBES-Labelled-COHORTS-ZSCORE"),
    ("Histogram", "CUBES-Labelled-COHORTS-HISTMATCH"),
]

# (Approach label, sweep script, results-path prefix)
APPROACHES = [
    ("CNN", "cnn_sweep.py",     "cnn_sweep"),
    ("A",   "a_sweep_norm.py",  "a_sweep_norm"),
    ("B",   "b_sweep_norm.py",  "b_sweep_norm"),
    ("C",   "c_sweep_norm.py",  "c_sweep_norm"),
]

CSV_PATH = Path("results_summary.csv")


def results_path_for(prefix: str, image_correction: str) -> Path:
    suffix = {"none": "raw", "z-score": "zscore", "Histogram": "histmatch"}[image_correction]
    return Path(f"{prefix}_{suffix}_results.jsonl")


def run_all() -> None:
    for approach_label, script, prefix in APPROACHES:
        for image_correction, data_path in DATA_VARIANTS:
            results_path = results_path_for(prefix, image_correction)
            print(f"\n{'='*70}\nRunning {approach_label}  (image correction: {image_correction})\n"
                  f"  data:    {data_path}\n  results: {results_path}\n{'='*70}")
            subprocess.run(
                [sys.executable, script,
                 "--data-path", data_path,
                 "--results-path", str(results_path)],
                check=True,
            )


# -- parsing / summarizing --------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"WARNING: {path} not found -- skipping")
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Shared schema across all four sweep scripts: 'holdout_cohort' (raw)
    and 'holdout_cohort_corrected' (z-score corrected) sub-dicts, each
    with 'auc', 'balanced_acc', 'recall_pos', 'recall_neg'."""
    out = {}
    for sub_key, prob_label in [("holdout_cohort", "none"), ("holdout_cohort_corrected", "z-score")]:
        valid = [r[sub_key] for r in rows if r.get(sub_key) is not None]
        auc_mean, auc_std = mean_std([v["auc"] for v in valid])
        bacc_mean, bacc_std = mean_std([v["balanced_acc"] for v in valid])
        rp_mean, rp_std = mean_std([v["recall_pos"] for v in valid])
        rn_mean, rn_std = mean_std([v["recall_neg"] for v in valid])
        out[prob_label] = {
            "auc": (auc_mean, auc_std),
            "bacc": (bacc_mean, bacc_std),
            "recall_pos": (rp_mean, rp_std),
            "recall_neg": (rn_mean, rn_std),
        }
    return out


def export_csv() -> None:
    rows_out = [["Approach", "Image correction", "Prob correction", "AUC", "Bal Acc", "Recall (+)", "Recall (-)"]]

    for approach_label, script, prefix in APPROACHES:
        for image_correction, data_path in DATA_VARIANTS:
            results_path = results_path_for(prefix, image_correction)
            raw_rows = load_jsonl(results_path)
            if not raw_rows:
                continue

            summary = summarize(raw_rows)

            for prob_correction in ["none", "z-score"]:
                s = summary[prob_correction]
                rows_out.append([
                    approach_label,
                    image_correction,
                    prob_correction,
                    fmt(*s["auc"]),
                    fmt(*s["bacc"]),
                    fmt(*s["recall_pos"]),
                    fmt(*s["recall_neg"]),
                ])

    with open(CSV_PATH, "w") as f:
        for row in rows_out:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nWrote {len(rows_out) - 1} result rows to {CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full comparison matrix and export results_summary.csv")
    parser.add_argument("--skip-run", action="store_true",
                         help="Skip running the sweeps -- only re-export the CSV from existing .jsonl files")
    args = parser.parse_args()

    if not args.skip_run:
        run_all()

    export_csv()