"""
Domain-shift analysis between PET cohorts in image space.

Three complementary analyses:

1. Per-patient feature extraction (15 statistics from within-prostate voxels)
   + pairwise Kolmogorov-Smirnov tests with Bonferroni correction

2. Random Forest centre-identity classifier (5-fold stratified CV,
   balanced accuracy) as a proxy for cohort separability

3. Pairwise Maximum Mean Discrepancy (MMD, RBF kernel) on the full
   flattened 32x32x32 VOI vectors

Usage
-----
    python domain_shift.py
    python domain_shift.py --seed 99 --mmd-subsample 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from nifti_loader import PatientVolumes, load_all_cohorts
from normalize import zscore_all_cohorts

# ── feature extraction ────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "mean", "std", "median", "skewness", "kurtosis",
    "p05", "p25", "p75", "p95",
    "iqr", "entropy",
    "min", "max",
]  # 13 features


def _entropy(v: np.ndarray, n_bins: int = 64) -> float:
    counts, _ = np.histogram(v, bins=n_bins, density=False)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def extract_features(patient: PatientVolumes) -> dict[str, float]:
    voxels = patient.pet_masked[patient.mask].astype(np.float64)
    if voxels.size == 0:
        return {k: np.nan for k in FEATURE_NAMES}

    p05, p25, p75, p95 = np.percentile(voxels, [5, 25, 75, 95])
    iqr = p75 - p25
    mean = float(voxels.mean())
    std = float(voxels.std())

    return {
        "mean":     mean,
        "std":      std,
        "median":   float(np.median(voxels)),
        "skewness": float(stats.skew(voxels)),
        "kurtosis": float(stats.kurtosis(voxels)),
        "p05":      float(p05),
        "p25":      float(p25),
        "p75":      float(p75),
        "p95":      float(p95),
        "iqr":      float(iqr),
        "entropy":  _entropy(voxels),
        "min":      float(voxels.min()),
        "max":      float(voxels.max()),
    }


def build_feature_df(all_cohorts: dict[str, list[PatientVolumes]]) -> pd.DataFrame:
    rows = []
    for cohort_name, patients in all_cohorts.items():
        for p in patients:
            feat = extract_features(p)
            feat["patient_id"] = p.patient_id
            feat["cohort"] = cohort_name
            rows.append(feat)
    df = pd.DataFrame(rows)
    df = df.set_index(["cohort", "patient_id"])
    return df


# ── KS tests ─────────────────────────────────────────────────────────────────

def ks_analysis(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:  # α/15
    cohorts = df.index.get_level_values("cohort").unique().tolist()
    pairs = [(cohorts[i], cohorts[j])
             for i in range(len(cohorts)) for j in range(i + 1, len(cohorts))]
    n_features = len(FEATURE_NAMES)
    alpha_bonf = alpha / n_features

    records = []
    for c1, c2 in pairs:
        g1 = df.loc[c1]
        g2 = df.loc[c2]
        for feat in FEATURE_NAMES:
            x1 = g1[feat].dropna().values
            x2 = g2[feat].dropna().values
            stat, p = ks_2samp(x1, x2)
            records.append({
                "pair":      f"{c1} vs {c2}",
                "feature":   feat,
                "ks_stat":   stat,
                "p_value":   p,
                "significant": p < alpha_bonf,
            })

    result = pd.DataFrame(records)
    result["alpha_bonf"] = alpha_bonf
    return result


# ── Random Forest separability ────────────────────────────────────────────────

def rf_separability(df: pd.DataFrame, n_trees: int = 300, max_depth: int = 5,
                    n_folds: int = 5, seed: int = 42) -> dict:
    X = df[FEATURE_NAMES].values
    cohort_labels = df.index.get_level_values("cohort").values

    le = LabelEncoder()
    y = le.fit_transform(cohort_labels)
    n_classes = len(le.classes_)
    chance = 1.0 / n_classes

    clf = RandomForestClassifier(
        n_estimators=n_trees, max_depth=max_depth,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_validate(clf, X, y, cv=cv,
                            scoring="balanced_accuracy", return_train_score=False)
    ba_scores = scores["test_score"]

    # Feature importances on full data
    clf.fit(X, y)
    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "classes":          list(le.classes_),
        "n_folds":          n_folds,
        "balanced_acc_per_fold": ba_scores.tolist(),
        "balanced_acc_mean": float(ba_scores.mean()),
        "balanced_acc_std":  float(ba_scores.std()),
        "chance_level":      chance,
        "feature_importance": importance_df,
    }


# ── MMD ───────────────────────────────────────────────────────────────────────

def _rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    # ||x - y||^2 via broadcasting; memory-efficient for moderate sizes
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    D2 = XX + YY.T - 2.0 * (X @ Y.T)
    return np.exp(-gamma * D2)


def median_heuristic_gamma(X: np.ndarray, Y: np.ndarray, subsample: int = 200) -> float:
    """Estimate RBF gamma via the median heuristic on a joint subsample."""
    sub = np.vstack([X[:subsample], Y[:subsample]])
    diffs = sub[:, None, :] - sub[None, :, :]
    median_sq = np.median(np.sum(diffs ** 2, axis=-1))
    return 1.0 / (2.0 * max(float(median_sq), 1e-8))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased MMD^2 estimate with RBF kernel."""
    n, m = len(X), len(Y)
    if gamma is None:
        gamma = median_heuristic_gamma(X, Y)

    Kxx = _rbf_kernel_matrix(X, X, gamma)
    Kyy = _rbf_kernel_matrix(Y, Y, gamma)
    Kxy = _rbf_kernel_matrix(X, Y, gamma)

    # Unbiased: zero out diagonals of same-set kernels
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term1 = Kxx.sum() / (n * (n - 1))
    term2 = Kyy.sum() / (m * (m - 1))
    term3 = 2.0 * Kxy.mean()
    return float(term1 + term2 - term3)


def mmd_analysis(all_cohorts: dict[str, list[PatientVolumes]],
                 subsample: int | None = None, seed: int = 42,
                 gamma: float | None = None) -> tuple[pd.DataFrame, float]:
    """Returns (results_df, gamma_used). Pass gamma to reuse a fixed bandwidth."""
    rng = np.random.default_rng(seed)
    cohort_names = list(all_cohorts.keys())

    def _voxel_matrix(patients: list[PatientVolumes]) -> np.ndarray:
        vols = [p.pet_masked.ravel().astype(np.float32) for p in patients]
        M = np.vstack(vols)
        if subsample is not None and len(M) > subsample:
            idx = rng.choice(len(M), subsample, replace=False)
            M = M[idx]
        return M

    matrices = {name: _voxel_matrix(plist) for name, plist in all_cohorts.items()}

    # Estimate gamma once from all pairs if not provided
    if gamma is None:
        all_pairs = [
            (cohort_names[i], cohort_names[j])
            for i in range(len(cohort_names))
            for j in range(i + 1, len(cohort_names))
        ]
        c1, c2 = all_pairs[0]
        gamma = median_heuristic_gamma(matrices[c1], matrices[c2])

    records = []
    for i in range(len(cohort_names)):
        for j in range(i + 1, len(cohort_names)):
            c1, c2 = cohort_names[i], cohort_names[j]
            val = mmd_rbf(matrices[c1], matrices[c2], gamma=gamma)
            records.append({"pair": f"{c1} vs {c2}", "mmd2": val})

    return pd.DataFrame(records), gamma


# ── reporting ─────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    w = 72
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def report(df_feat: pd.DataFrame,
           ks_df: pd.DataFrame,
           rf: dict,
           mmd_df: pd.DataFrame) -> None:

    # 1. Per-cohort feature summary
    print_section("1. Per-cohort summary statistics (median ± IQR)")
    cohorts = df_feat.index.get_level_values("cohort").unique()
    summary_rows = []
    for feat in FEATURE_NAMES:
        row = {"feature": feat}
        for c in cohorts:
            vals = df_feat.loc[c, feat].dropna()
            row[c] = f"{vals.median():.3g}  [{vals.quantile(.25):.3g}–{vals.quantile(.75):.3g}]"
        summary_rows.append(row)
    print(pd.DataFrame(summary_rows).to_string(index=False))

    # 2. KS tests
    print_section("2. KS tests (Bonferroni-corrected α = {:.4f})".format(
        ks_df["alpha_bonf"].iloc[0]))
    sig = ks_df[ks_df["significant"]]
    all_ns = ks_df[~ks_df["significant"]]

    for pair in ks_df["pair"].unique():
        sub = ks_df[ks_df["pair"] == pair].copy()
        sub = sub.sort_values("p_value")
        print(f"\n  Pair: {pair}")
        print(sub[["feature", "ks_stat", "p_value", "significant"]].to_string(index=False))

    sig_features = sig["feature"].unique().tolist()
    print(f"\n  Significant features ({len(sig_features)}): {sig_features}")
    ns_features = all_ns["feature"].unique().tolist()
    print(f"  Non-significant features ({len(ns_features)}): {ns_features}")

    # 3. RF
    print_section("3. Random Forest cohort separability")
    print(f"  Classes : {rf['classes']}")
    print(f"  Chance  : {rf['chance_level']:.1%}")
    folds_str = "  ".join(f"{s:.3f}" for s in rf["balanced_acc_per_fold"])
    print(f"  Folds   : {folds_str}")
    print(f"  Mean BA : {rf['balanced_acc_mean']:.3f}  ± {rf['balanced_acc_std']:.3f}")
    print(f"\n  Feature importance (full-data fit):")
    print(rf["feature_importance"].to_string(index=False))

    # 4. MMD
    print_section("4. Maximum Mean Discrepancy (MMD², RBF kernel)")
    print(f"  gamma (RBF bandwidth): {mmd_df.attrs.get('gamma', 'N/A'):.6g}")
    print(mmd_df[["pair", "mmd2"]].to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-shift analysis for PET cohorts")
    parser.add_argument("--data-root", default="CUBES-Labelled-COHORTS",
                        help="Root directory with cohort subdirectories")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply per-patient z-score normalisation before analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmd-subsample", type=int, default=None,
                        help="Subsample N patients per cohort for MMD (default: use all)")
    args = parser.parse_args()

    root = Path(args.data_root)
    print(f"Loading cohorts from {root} …")
    all_cohorts = load_all_cohorts(root)
    for name, plist in all_cohorts.items():
        print(f"  {name}: {len(plist)} patients")

    if args.normalize:
        print("Applying per-patient z-score normalisation …")
        all_cohorts = zscore_all_cohorts(all_cohorts)

    print("Extracting features …")
    df_feat = build_feature_df(all_cohorts)

    print("Running KS tests …")
    ks_df = ks_analysis(df_feat)

    print("Training Random Forest …")
    rf = rf_separability(df_feat, seed=args.seed)

    print("Computing MMD …")
    # Gamma is estimated from the raw cohorts so it is identical whether or not
    # normalisation is applied, making MMD² values directly comparable.
    raw_cohorts = load_all_cohorts(root)
    _, gamma = mmd_analysis(raw_cohorts, subsample=args.mmd_subsample, seed=args.seed)
    print(f"  RBF gamma (from raw data): {gamma:.6g}")

    mmd_df, _ = mmd_analysis(all_cohorts, subsample=args.mmd_subsample,
                              seed=args.seed, gamma=gamma)
    mmd_df.attrs["gamma"] = gamma

    report(df_feat, ks_df, rf, mmd_df)


if __name__ == "__main__":
    main()
