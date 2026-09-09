"""Run all 2-cohort and 3-cohort experiments and build final CSV tables.

Runs CNN/A/B/C on raw images and with fold-fitted cohort z-score matching.
Correction statistics use source-training patients and the target harmonization
half; held-out patients never contribute to fitted preprocessing.

Final CSVs:
  results_2cohort.csv
  results_3cohort.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

TORCH_SEEDS = list(range(10))

COHORTS = {
    2: ["AUGSBURG", "SWISS"],
    3: ["AUGSBURG", "PRE-RAPID", "SWISS"],
}

DATASETS = {
    2: {
        "off": "CUBES-Labelled-COHORTS_2",
        "on": "CUBES-Labelled-COHORTS_2",
    },
    3: {
        "off": "CUBES-Labelled-COHORTS_3",
        "on": "CUBES-Labelled-COHORTS_3",
    },
}

SWEEPS = {
    2: {
        "CNN": "cnn_2_sweep.py",
        "A": "a_2_sweep.py",
        "B": "b_2_sweep.py",
        "C": "c_2_sweep.py",
    },
    3: {
        "CNN": "cnn_3_sweep.py",
        "A": "a_3_sweep.py",
        "B": "b_3_sweep.py",
        "C": "c_3_sweep.py",
    },
}

SUPPORTS_FRESH = {
    "CNN": True,
    "A": True,
    "B": True,
    "C": True,
}

OUTPUT_MODEL_ORDER = ["CNN", "A", "B", "C"]

METRIC_NAMES = [
    "balanced_acc",
    "auc",
    "recall_pos",
    "recall_neg",
]

# Existing 3-cohort CNN baseline results supplied by the user.
FIXED_RESULTS = {
    (3, "CNN", "off", "AUGSBURG"): {
        "balanced_acc": 0.739, "auc": 0.799,
        "recall_pos": 0.570, "recall_neg": 0.908,
    },
    (3, "CNN", "off", "PRE-RAPID"): {
        "balanced_acc": 0.626, "auc": 0.690,
        "recall_pos": 0.585, "recall_neg": 0.667,
    },
    (3, "CNN", "off", "SWISS"): {
        "balanced_acc": 0.632, "auc": 0.686,
        "recall_pos": 0.539, "recall_neg": 0.725,
    },
    (3, "CNN", "off", "POOLED"): {
        "balanced_acc": 0.666, "auc": 0.725,
        "recall_pos": 0.565, "recall_neg": 0.766,
    },
}

CSV_FIELDS = [
    "model",
    "cohort",
    "z score correction",
    "balanced acc",
    "auc",
    "recall_pos",
    "recall_neg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw + fold-normalized 2/3-cohort sweeps and build final CSVs"
    )
    parser.add_argument(
        "--results-dir",
        default=".",
        help="Directory for JSONL sweep outputs and final CSVs (default: current directory).",
    )
    parser.add_argument(
        "--data-2",
        default=DATASETS[2]["off"],
        help=f"Raw 2-cohort dataset (default: {DATASETS[2]['off']}).",
    )
    parser.add_argument(
        "--data-3",
        default=DATASETS[3]["off"],
        help=f"Raw 3-cohort dataset (default: {DATASETS[3]['off']}).",
    )
    parser.add_argument(
        "--cohort-version",
        choices=["2", "3", "both"],
        default="both",
        help="Run/aggregate 2-cohort, 3-cohort, or both (default: both).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Rerun all selected experiments from scratch.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not run training; only build CSVs from existing JSONL files.",
    )
    return parser.parse_args()


def result_jsonl_path(
    results_dir: Path,
    model: str,
    n_cohorts: int,
    correction: str,
) -> Path:
    suffix = "" if correction == "off" else "_z_scored"
    return results_dir / f"{model.lower()}_{n_cohorts}{suffix}.jsonl"


def final_csv_path(
    results_dir: Path,
    n_cohorts: int,
) -> Path:
    return results_dir / f"results_{n_cohorts}cohort.csv"


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []

    runs: list[dict] = []

    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}: {exc}"
                ) from exc

            if rec.get("record_type") == "summary":
                continue

            if rec.get("record_type") not in (None, "run"):
                continue

            if (
                "torch_seed" in rec
                and "target_cohort" in rec
                and "target_heldout" in rec
            ):
                runs.append(rec)

    return runs


def expected_run_keys(
    n_cohorts: int,
) -> set[tuple[int, str]]:
    return {
        (seed, cohort)
        for seed in TORCH_SEEDS
        for cohort in COHORTS[n_cohorts]
    }


def run_has_metrics(run: dict) -> bool:
    metrics = run.get("target_heldout")

    if not isinstance(metrics, dict):
        return False

    try:
        for name in METRIC_NAMES:
            if name == "balanced_acc":
                value = metrics.get(
                    "balanced_acc",
                    metrics.get("bacc"),
                )
            else:
                value = metrics[name]

            float(value)
    except (KeyError, TypeError, ValueError):
        return False

    return True


def is_complete(
    path: Path,
    n_cohorts: int,
) -> bool:
    runs = load_runs(path)
    expected = expected_run_keys(n_cohorts)
    actual = [
        (int(run["torch_seed"]), run["target_cohort"])
        for run in runs
    ]

    # Comparing both the keys and the record count rejects duplicate rows,
    # unexpected seeds/cohorts, and missing runs. Otherwise those records would
    # be silently included in the final averages.
    return (
        len(actual) == len(expected)
        and set(actual) == expected
        and all(run_has_metrics(run) for run in runs)
        and len({run.get("experiment_id") for run in runs}) == 1
        and runs[0].get("experiment_id") is not None
    )


def run_sweep(
    model: str,
    n_cohorts: int,
    correction: str,
    data_path: Path,
    results_path: Path,
    fresh: bool,
) -> None:
    sweep_path = SCRIPT_DIR / SWEEPS[n_cohorts][model]

    if not sweep_path.exists():
        raise FileNotFoundError(
            f"Missing sweep script: {sweep_path}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing dataset directory: {data_path}"
        )

    if fresh and results_path.exists():
        results_path.unlink()

    cmd = [
        sys.executable,
        str(sweep_path),
        "--data-path",
        str(data_path),
        "--results-path",
        str(results_path),
    ]

    if fresh and SUPPORTS_FRESH[model]:
        cmd.append("--fresh")

    if correction == "on":
        cmd.append("--zscore-correction")

    print(
        f"\n{'=' * 72}\n"
        f"RUN {model}{n_cohorts}  "
        f"z score correction={correction}\n"
        f"data    : {data_path}\n"
        f"results : {results_path}\n"
        f"{'=' * 72}"
    )

    subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        check=True,
    )

    if not is_complete(
        results_path,
        n_cohorts,
    ):
        raise RuntimeError(
            f"{model}{n_cohorts} zscore={correction} finished, "
            f"but {results_path} does not contain complete "
            f"0-9 seed x target-cohort results."
        )


def metric_value(
    metrics: dict,
    name: str,
) -> float:
    if name == "balanced_acc":
        if "balanced_acc" in metrics:
            return float(metrics["balanced_acc"])
        if "bacc" in metrics:
            return float(metrics["bacc"])

    return float(metrics[name])


def aggregate_runs(
    runs: list[dict],
) -> dict[str, float]:
    if not runs:
        raise ValueError(
            "Cannot aggregate an empty run list."
        )

    out: dict[str, float] = {}

    for metric_name in METRIC_NAMES:
        values = [
            metric_value(
                run["target_heldout"],
                metric_name,
            )
            for run in runs
            if run.get("target_heldout") is not None
        ]

        out[metric_name] = float(
            np.nanmean(values)
        )

    return out


def build_rows(
    n_cohorts: int,
    results_dir: Path,
) -> list[dict]:
    rows: list[dict] = []

    for model in OUTPUT_MODEL_ORDER:
        for correction in ["off", "on"]:
            results_path = result_jsonl_path(
                results_dir,
                model,
                n_cohorts,
                correction,
            )

            runs = load_runs(results_path)
            complete = is_complete(
                results_path,
                n_cohorts,
            )

            for cohort in [*COHORTS[n_cohorts], "POOLED"]:
                fixed = FIXED_RESULTS.get(
                    (
                        n_cohorts,
                        model,
                        correction,
                        cohort,
                    )
                )

                if complete:
                    subset = (
                        runs
                        if cohort == "POOLED"
                        else [
                            run
                            for run in runs
                            if run["target_cohort"] == cohort
                        ]
                    )
                    metrics = aggregate_runs(subset)
                elif fixed is not None:
                    # Supplied baselines are a fallback for absent results.
                    # A completed sweep, especially one produced by --fresh,
                    # must always take precedence.
                    metrics = fixed
                else:
                    raise RuntimeError(
                        f"Cannot build final table: incomplete/missing "
                        f"results for {model}{n_cohorts}, "
                        f"zscore={correction}: {results_path}"
                    )

                rows.append(
                    {
                        "model": model,
                        "cohort": cohort,
                        "z score correction": correction,
                        "balanced acc": (
                            None
                            if metrics is None
                            else metrics["balanced_acc"]
                        ),
                        "auc": (
                            None
                            if metrics is None
                            else metrics["auc"]
                        ),
                        "recall_pos": (
                            None
                            if metrics is None
                            else metrics["recall_pos"]
                        ),
                        "recall_neg": (
                            None
                            if metrics is None
                            else metrics["recall_neg"]
                        ),
                    }
                )

    return rows

def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            formatted = dict(row)

            for key in [
                "balanced acc",
                "auc",
                "recall_pos",
                "recall_neg",
            ]:
                value = formatted[key]
                formatted[key] = (
                    ""
                    if value is None
                    else f"{float(value):.3f}"
                )

            writer.writerow(
                formatted
            )


def selected_versions(
    value: str,
) -> list[int]:
    if value == "2":
        return [2]

    if value == "3":
        return [3]

    return [2, 3]


def validate_sweep_inputs(
    versions: list[int],
    data_paths: dict[int, dict[str, Path]],
) -> None:
    missing: list[str] = []

    for n_cohorts in versions:
        for model in OUTPUT_MODEL_ORDER:
            sweep_path = SCRIPT_DIR / SWEEPS[n_cohorts][model]

            if not sweep_path.is_file():
                missing.append(f"sweep script: {sweep_path}")

        for correction in ["off", "on"]:
            data_path = data_paths[n_cohorts][correction]

            if not data_path.is_dir():
                missing.append(f"dataset directory: {data_path}")

    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            f"Cannot start sweeps; required inputs are missing:\n{details}"
        )


def main() -> None:
    args = parse_args()

    # Resolve paths before launching children with cwd=SCRIPT_DIR. This keeps a
    # relative CLI path tied to the caller's working directory in both parent
    # and child processes.
    results_dir = Path(
        args.results_dir
    ).expanduser().resolve()
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_paths = {
        2: {
            "off": Path(args.data_2).expanduser().resolve(),
            "on": Path(args.data_2).expanduser().resolve(),
        },
        3: {
            "off": Path(args.data_3).expanduser().resolve(),
            "on": Path(args.data_3).expanduser().resolve(),
        },
    }

    versions = selected_versions(
        args.cohort_version
    )

    if not args.aggregate_only:
        validate_sweep_inputs(
            versions,
            data_paths,
        )

        for n_cohorts in versions:
            for model in [
                "CNN",
                "A",
                "B",
                "C",
            ]:
                for correction in [
                    "off",
                    "on",
                ]:
                    run_sweep(
                        model=model,
                        n_cohorts=n_cohorts,
                        correction=correction,
                        data_path=data_paths[
                            n_cohorts
                        ][correction],
                        results_path=result_jsonl_path(
                            results_dir,
                            model,
                            n_cohorts,
                            correction,
                        ),
                        fresh=args.fresh,
                    )

    for n_cohorts in versions:
        rows = build_rows(
            n_cohorts,
            results_dir,
        )

        csv_path = final_csv_path(
            results_dir,
            n_cohorts,
        )

        write_csv(
            csv_path,
            rows,
        )

        print(
            f"\nwrote final table -> "
            f"{csv_path}"
        )


if __name__ == "__main__":
    main()
