from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, norm, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = REPO_ROOT / "data" / "traits.csv"
DICT_FILE = REPO_ROOT / "data" / "traits_dictionary.csv"
RELIABILITY_FILE = REPO_ROOT / "data" / "trait_reliability_scores.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "variety_core_fingerprint"
ID_COL = "Field Plot No."
GROUP_COL = "Endosperm Type"
REDUNDANCY_THRESHOLD = 0.85
AVAILABILITY_HARD_THRESHOLD = 0.70
LOW_VARIATION_THRESHOLD = 0.90
CONTINUOUS_BIN_OPTIONS = [5, 7, 10]
TRAIT_NUMBER_COMPARISON_RANGE = [4, 5, 6, 7, 8]
TARGET_TRAIT_FAMILIES = ["Color", "Size", "Shape", "Count"]


def trait_col_from_no(trait_no: int) -> str:
    return f"Trait_{trait_no}_Predict"


def short_trait_name(trait_no: int) -> str:
    return f"T{trait_no}"


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def bh_fdr(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return pd.Series(np.nan, index=p_values.index, dtype=float)

    order = valid.sort_values().index.tolist()
    ranked = valid.loc[order].to_numpy()
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = min(prev, ranked[i] * n / rank)
        adjusted[i] = value
        prev = value

    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    result.loc[order] = adjusted
    return result


def normalize_series_to_unit(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    min_value = float(valid.min())
    max_value = float(valid.max())
    if max_value == min_value:
        return pd.Series(1.0, index=series.index, dtype=float)
    return (series - min_value) / (max_value - min_value)


def parse_code_numbers(code_range: object) -> list[float]:
    if pd.isna(code_range):
        return []
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(code_range))]


def parse_code_minmax(code_range: object) -> tuple[float, float] | None:
    numbers = parse_code_numbers(code_range)
    if not numbers:
        return None
    return min(numbers), max(numbers)


def load_trait_dictionary() -> pd.DataFrame:
    meta = pd.read_csv(DICT_FILE)
    meta["Trait No."] = pd.to_numeric(meta["Trait No."], errors="coerce").astype(int)
    meta["Trait column"] = meta["Trait No."].map(trait_col_from_no)
    meta["Trait short"] = meta["Trait No."].map(short_trait_name)
    return meta


def load_trait_data(meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    trait_cols = meta["Trait column"].tolist()
    df[trait_cols] = df[trait_cols].apply(pd.to_numeric, errors="coerce")
    return df


def load_reliability(meta: pd.DataFrame) -> pd.DataFrame:
    template = meta[
        ["Trait No.", "Trait short", "Trait column", "Trait name", "Observation Method", "Analysis Type", "Data type"]
    ].copy()
    template["Reliability score"] = np.nan
    template["Reliability metric"] = ""
    template["Reliability source"] = ""
    template.to_csv(OUTPUT_DIR / "trait_reliability_template.csv", index=False, encoding="utf-8-sig")

    if not RELIABILITY_FILE.exists():
        return template

    raw = pd.read_csv(RELIABILITY_FILE)
    rel = raw.copy()
    if "Trait No." not in rel.columns:
        raise ValueError("trait_reliability_scores.csv must contain a 'Trait No.' column.")
    rel["Trait No."] = pd.to_numeric(rel["Trait No."], errors="coerce")
    rel = rel.dropna(subset=["Trait No."]).copy()
    rel["Trait No."] = rel["Trait No."].astype(int)
    if "Reliability score" not in rel.columns:
        for candidate in ["R2", "Weighted Kappa", "weighted_kappa", "r2"]:
            if candidate in rel.columns:
                rel["Reliability score"] = pd.to_numeric(rel[candidate], errors="coerce")
                break
    rel = template.drop(columns=["Reliability score", "Reliability metric", "Reliability source"]).merge(
        rel,
        on="Trait No.",
        how="left",
        suffixes=("", "_y"),
    )
    if "Reliability score" not in rel.columns:
        rel["Reliability score"] = np.nan
    if "Reliability metric" not in rel.columns:
        rel["Reliability metric"] = ""
    if "Reliability source" not in rel.columns:
        rel["Reliability source"] = ""
    return rel


def build_scoring_base(meta: pd.DataFrame, df: pd.DataFrame, reliability_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    merged_reliability = reliability_df.set_index("Trait No.")

    for _, row in meta.iterrows():
        trait_no = int(row["Trait No."])
        trait_col = str(row["Trait column"])
        values = pd.to_numeric(df[trait_col], errors="coerce")
        missing_rate = float(values.isna().mean())
        availability = 1.0 - missing_rate
        non_missing = values.dropna()
        dominant_share = float(non_missing.value_counts(normalize=True).iloc[0]) if not non_missing.empty else np.nan
        low_variation_flag = bool(dominant_share >= LOW_VARIATION_THRESHOLD) if not np.isnan(dominant_share) else True
        rel_score = pd.to_numeric(
            pd.Series([merged_reliability.loc[trait_no, "Reliability score"] if trait_no in merged_reliability.index else np.nan]),
            errors="coerce",
        ).iloc[0]
        rel_metric = (
            str(merged_reliability.loc[trait_no, "Reliability metric"])
            if trait_no in merged_reliability.index and "Reliability metric" in merged_reliability.columns
            else ""
        )
        rel_source = (
            str(merged_reliability.loc[trait_no, "Reliability source"])
            if trait_no in merged_reliability.index and "Reliability source" in merged_reliability.columns
            else ""
        )
        rows.append(
            {
                "Trait column": trait_col,
                "Trait no.": trait_no,
                "Trait short": row["Trait short"],
                "Trait name": row["Trait name"],
                "Trait family": row["Analysis Type"],
                "Observation method": row["Observation Method"],
                "Data type": row["Data type"],
                "Code range": row["Code range"],
                "Missing rate": missing_rate,
                "Availability score": availability,
                "Reliability score": rel_score,
                "Reliability metric": rel_metric,
                "Reliability source": rel_source,
                "Dominant state share": dominant_share,
                "Low variation flag": low_variation_flag,
            }
        )

    return pd.DataFrame(rows)


def epsilon_squared_from_kruskal(h_stat: float, n_total: int, n_groups: int) -> float:
    denominator = n_total - n_groups
    if denominator <= 0:
        return np.nan
    value = (h_stat - n_groups + 1) / denominator
    return float(max(0.0, min(1.0, value)))


def subgroup_discriminative_analysis(scoring_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    result = scoring_df.copy()
    p_values: list[float] = []
    effect_sizes: list[float] = []
    h_values: list[float] = []
    valid_samples: list[int] = []
    group_counts: list[int] = []
    skip_flags: list[str] = []

    for _, row in result.iterrows():
        trait_col = str(row["Trait column"])
        sub = df[[GROUP_COL, trait_col]].copy()
        sub[trait_col] = pd.to_numeric(sub[trait_col], errors="coerce")
        sub = sub.dropna(subset=[GROUP_COL, trait_col]).copy()
        comparable = len(sub)
        groups = [g[trait_col].to_numpy(dtype=float) for _, g in sub.groupby(GROUP_COL) if len(g[trait_col].dropna()) > 0]
        n_groups = len(groups)
        valid_samples.append(comparable)
        group_counts.append(n_groups)

        if comparable < 10 or n_groups < 2 or any(len(g) < 3 for g in groups):
            p_values.append(np.nan)
            effect_sizes.append(np.nan)
            h_values.append(np.nan)
            skip_flags.append("insufficient_valid_samples")
            continue

        try:
            h_stat, p_value = kruskal(*groups)
        except ValueError:
            h_stat, p_value = np.nan, np.nan

        p_values.append(p_value)
        h_values.append(h_stat)
        effect_sizes.append(epsilon_squared_from_kruskal(h_stat, comparable, n_groups) if pd.notna(h_stat) else np.nan)
        skip_flags.append("")

    result["Valid sample count"] = valid_samples
    result["Valid group count"] = group_counts
    result["Kruskal H"] = h_values
    result["Kruskal p"] = p_values
    result["FDR p"] = bh_fdr(result["Kruskal p"])
    result["Effect size"] = effect_sizes
    result["Discriminative status"] = skip_flags
    result["Subgroup discriminative score"] = normalize_series_to_unit(result["Effect size"])
    return result


def choose_cv_strategy(y: pd.Series) -> tuple[object, str]:
    counts = y.value_counts()
    min_count = int(counts.min())
    if min_count >= 4:
        n_splits = min(5, min_count)
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=42), f"RepeatedStratifiedKFold({n_splits}x5)"
    if min_count >= 2:
        n_splits = min(3, min_count)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), f"StratifiedKFold({n_splits})"
    if len(y) <= 30:
        return LeaveOneOut(), "LeaveOneOut"
    return KFold(n_splits=5, shuffle=True, random_state=42), "KFold(5)"


def build_model_matrix(meta: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    trait_cols = meta["Trait column"].tolist()
    x = df[trait_cols].apply(pd.to_numeric, errors="coerce").copy()
    y = df[GROUP_COL].astype(str).copy()
    mask = y.notna()
    return x.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def subgroup_model_analysis(scoring_df: pd.DataFrame, meta: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, y = build_model_matrix(meta, df)
    trait_cols = meta["Trait column"].tolist()
    cv, cv_name = choose_cv_strategy(y)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                trait_cols,
            )
        ]
    )
    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    bal_scores: list[float] = []
    f1_scores: list[float] = []
    for train_idx, test_idx in cv.split(x, y):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        bal_scores.append(balanced_accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

    pipeline.fit(x, y)
    rf_importance = pipeline.named_steps["model"].feature_importances_
    perm = permutation_importance(
        pipeline,
        x,
        y,
        n_repeats=30,
        random_state=42,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "cv_strategy": cv_name,
                "n_samples": len(y),
                "n_classes": y.nunique(),
                "balanced_accuracy_mean": float(np.mean(bal_scores)),
                "balanced_accuracy_std": float(np.std(bal_scores, ddof=0)),
                "macro_f1_mean": float(np.mean(f1_scores)),
                "macro_f1_std": float(np.std(f1_scores, ddof=0)),
            }
        ]
    )

    result = scoring_df.copy()
    result["RF importance"] = result["Trait column"].map(dict(zip(trait_cols, rf_importance)))
    result["Permutation importance mean"] = result["Trait column"].map(dict(zip(trait_cols, perm.importances_mean)))
    result["Permutation importance std"] = result["Trait column"].map(dict(zip(trait_cols, perm.importances_std)))

    positive_perm = result["Permutation importance mean"].clip(lower=0)
    result["Model importance score"] = normalize_series_to_unit(positive_perm)
    return result, metrics_df


def redundancy_analysis(scoring_df: pd.DataFrame, meta: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trait_cols = meta["Trait column"].tolist()
    matrix = df[trait_cols].apply(pd.to_numeric, errors="coerce").copy()
    median_imputed = matrix.fillna(matrix.median())

    corr = median_imputed.corr(method="spearman")
    corr.index = meta["Trait short"].tolist()
    corr.columns = meta["Trait short"].tolist()

    redundant_rows: list[dict[str, object]] = []
    penalty_counts: dict[str, int] = {col: 0 for col in trait_cols}
    for left_idx in range(len(trait_cols)):
        for right_idx in range(left_idx + 1, len(trait_cols)):
            left_col = trait_cols[left_idx]
            right_col = trait_cols[right_idx]
            left_short = short_trait_name(int(meta.iloc[left_idx]["Trait No."]))
            right_short = short_trait_name(int(meta.iloc[right_idx]["Trait No."]))
            rho = float(corr.loc[left_short, right_short])
            if abs(rho) > REDUNDANCY_THRESHOLD:
                penalty_counts[left_col] += 1
                penalty_counts[right_col] += 1
                redundant_rows.append(
                    {
                        "Trait A": left_short,
                        "Trait B": right_short,
                        "Trait A column": left_col,
                        "Trait B column": right_col,
                        "Spearman rho": rho,
                    }
                )

    result = scoring_df.copy()
    max_links = max(1, len(trait_cols) - 1)
    result["Redundancy penalty count"] = result["Trait column"].map(penalty_counts)
    result["Redundancy score"] = 1.0 - (result["Redundancy penalty count"] / max_links)
    redundant_df = pd.DataFrame(redundant_rows)
    return result, corr, redundant_df


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / std


def build_zscore_bin_edges(bin_count: int) -> list[float]:
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2.")
    quantiles = np.linspace(0, 1, bin_count + 1)[1:-1]
    inner_edges = norm.ppf(quantiles).tolist()
    return [-np.inf] + inner_edges + [np.inf]


def bin_zscore_to_levels(series: pd.Series, bin_count: int) -> pd.Series:
    z = zscore(series)
    edges = build_zscore_bin_edges(bin_count)
    bins = pd.cut(
        z,
        bins=edges,
        labels=list(range(1, bin_count + 1)),
    )
    return pd.to_numeric(bins.astype("float"), errors="coerce")


def encode_trait_for_id(values: pd.Series, data_type: str, continuous_bin_count: int = 5) -> pd.Series:
    data_type = str(data_type)
    numeric = pd.to_numeric(values, errors="coerce")
    if data_type in {"continuous", "continuous_ratio"}:
        return bin_zscore_to_levels(numeric, continuous_bin_count)
    if data_type == "integer_count":
        return numeric.round()
    return numeric.round()


def normalized_pairwise_distance(values: pd.Series, scale_min: float, scale_max: float) -> tuple[float, float]:
    comparable_pairs = 0
    different_pairs = 0
    distances: list[float] = []
    denominator = scale_max - scale_min
    if denominator <= 0:
        return np.nan, np.nan

    arr = values.to_numpy(dtype=float)
    for left_idx, right_idx in combinations(range(len(arr)), 2):
        left = arr[left_idx]
        right = arr[right_idx]
        if np.isnan(left) or np.isnan(right):
            continue
        comparable_pairs += 1
        if left != right:
            different_pairs += 1
        distances.append(abs(left - right) / denominator)

    if comparable_pairs == 0:
        return np.nan, np.nan
    distinctness_rate = different_pairs / comparable_pairs
    distance_mean = float(np.mean(distances)) if distances else np.nan
    return distinctness_rate, distance_mean


def build_encoded_id_matrix(meta: pd.DataFrame, df: pd.DataFrame, continuous_bin_count: int = 5) -> pd.DataFrame:
    encoded = df[[ID_COL, GROUP_COL]].copy()
    for _, row in meta.iterrows():
        trait_col = str(row["Trait column"])
        encoded[trait_col] = encode_trait_for_id(df[trait_col], str(row["Data type"]), continuous_bin_count=continuous_bin_count)
    return encoded


def variety_distinctness_analysis(
    scoring_df: pd.DataFrame,
    meta: pd.DataFrame,
    df: pd.DataFrame,
    continuous_bin_count: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded = build_encoded_id_matrix(meta, df, continuous_bin_count=continuous_bin_count)
    result = scoring_df.copy()

    pair_rates: list[float] = []
    distance_means: list[float] = []
    code_ranges: list[str] = []
    for _, row in result.iterrows():
        trait_col = str(row["Trait column"])
        data_type = str(row["Data type"])
        values = encoded[trait_col]
        if data_type in {"continuous", "continuous_ratio"}:
            scale_min, scale_max = 1.0, float(continuous_bin_count)
            code_ranges.append(f"1-{continuous_bin_count} bins")
        elif data_type == "integer_count":
            valid = values.dropna()
            if valid.empty:
                scale_min, scale_max = 0.0, 0.0
            else:
                scale_min, scale_max = float(valid.min()), float(valid.max())
            code_ranges.append(f"{scale_min:g}-{scale_max:g}")
        else:
            parsed = parse_code_minmax(row["Code range"])
            if parsed is None:
                valid = values.dropna()
                scale_min, scale_max = (float(valid.min()), float(valid.max())) if not valid.empty else (0.0, 0.0)
            else:
                scale_min, scale_max = parsed
            code_ranges.append(f"{scale_min:g}-{scale_max:g}")

        pair_rate, distance_mean = normalized_pairwise_distance(values, scale_min, scale_max)
        pair_rates.append(pair_rate)
        distance_means.append(distance_mean)

    result["Pairwise distinctness rate"] = pair_rates
    result["Distance contribution"] = distance_means
    result["Encoded ID range"] = code_ranges
    result["Variety distinctness score"] = 0.5 * result["Pairwise distinctness rate"] + 0.5 * result["Distance contribution"]
    return result, encoded


def build_pairwise_distance_matrix(
    encoded: pd.DataFrame,
    selected_trait_cols: list[str],
    meta_map: dict[str, tuple[str, object]],
    continuous_bin_count: int = 5,
) -> pd.DataFrame:
    samples = encoded[ID_COL].astype(str).tolist()
    n_samples = len(samples)
    matrix = np.full((n_samples, n_samples), np.nan, dtype=float)
    np.fill_diagonal(matrix, 0.0)

    trait_scale_map: dict[str, tuple[float, float]] = {}
    for trait_col in selected_trait_cols:
        data_type, code_range = meta_map[trait_col]
        values = encoded[trait_col]
        if data_type in {"continuous", "continuous_ratio"}:
            trait_scale_map[trait_col] = (1.0, float(continuous_bin_count))
        elif data_type == "integer_count":
            valid = values.dropna()
            trait_scale_map[trait_col] = (float(valid.min()), float(valid.max())) if not valid.empty else (0.0, 0.0)
        else:
            parsed = parse_code_minmax(code_range)
            if parsed is None:
                valid = values.dropna()
                trait_scale_map[trait_col] = (float(valid.min()), float(valid.max())) if not valid.empty else (0.0, 0.0)
            else:
                trait_scale_map[trait_col] = parsed

    for left_idx, right_idx in combinations(range(n_samples), 2):
        distances = []
        for trait_col in selected_trait_cols:
            left = encoded.iloc[left_idx][trait_col]
            right = encoded.iloc[right_idx][trait_col]
            if pd.isna(left) or pd.isna(right):
                continue
            scale_min, scale_max = trait_scale_map[trait_col]
            denominator = scale_max - scale_min
            if denominator <= 0:
                continue
            distances.append(abs(float(left) - float(right)) / denominator)
        if distances:
            pair_distance = float(np.mean(distances))
            matrix[left_idx, right_idx] = pair_distance
            matrix[right_idx, left_idx] = pair_distance

    return pd.DataFrame(matrix, index=samples, columns=samples)


def exact_code_matrix(encoded: pd.DataFrame, selected_trait_cols: list[str]) -> pd.DataFrame:
    code_df = encoded[[ID_COL, GROUP_COL] + selected_trait_cols].copy()
    for trait_col in selected_trait_cols:
        code_df[trait_col] = code_df[trait_col].apply(lambda x: "X" if pd.isna(x) else str(int(x)) if float(x).is_integer() else str(x))
    return code_df


def summarize_selected_traits(
    encoded: pd.DataFrame,
    selected_trait_cols: list[str],
    meta_map: dict[str, tuple[str, object]],
    continuous_bin_count: int = 5,
) -> dict[str, float]:
    code_df = exact_code_matrix(encoded, selected_trait_cols)
    code_only = code_df[selected_trait_cols]
    combined = code_only.astype(str).agg("|".join, axis=1)
    unique_prop = float(combined.nunique() / len(combined))
    distance_df = build_pairwise_distance_matrix(
        encoded,
        selected_trait_cols,
        meta_map,
        continuous_bin_count=continuous_bin_count,
    )
    tri = distance_df.where(np.triu(np.ones(distance_df.shape), 1).astype(bool))
    pair_values = tri.stack().astype(float)
    if pair_values.empty:
        mean_distance = 0.0
        zero_pairs = 0
        total_pairs = 0
    else:
        valid_pairs = pair_values.dropna()
        mean_distance = float(valid_pairs.mean()) if not valid_pairs.empty else 0.0
        zero_pairs = int((valid_pairs == 0).sum()) if not valid_pairs.empty else 0
        total_pairs = int(len(valid_pairs))
    zero_pair_rate = zero_pairs / total_pairs if total_pairs > 0 else 0.0
    return {
        "unique_pattern_proportion": unique_prop,
        "mean_pairwise_distance": mean_distance,
        "zero_distance_pairs": zero_pairs,
        "comparable_pairs": total_pairs,
        "zero_pair_rate": zero_pair_rate,
    }


def forward_select_core_traits(
    scoring_df: pd.DataFrame,
    meta: pd.DataFrame,
    encoded: pd.DataFrame,
    continuous_bin_count: int = 5,
    stop_on_perfect: bool = True,
    max_steps: int | None = None,
) -> tuple[list[str], pd.DataFrame]:
    eligible = scoring_df[
        (scoring_df["Availability score"] >= AVAILABILITY_HARD_THRESHOLD)
        & (~scoring_df["Low variation flag"])
    ].copy()
    eligible = eligible.sort_values(["Final score", "Variety distinctness score", "Reliability score"], ascending=[False, False, False])

    selected: list[str] = []
    trace_rows: list[dict[str, object]] = []
    meta_map = meta.set_index("Trait column")[["Data type", "Code range"]].apply(tuple, axis=1).to_dict()

    current_summary = {
        "unique_pattern_proportion": 0.0,
        "mean_pairwise_distance": 0.0,
        "zero_distance_pairs": int(len(encoded) * (len(encoded) - 1) / 2),
        "comparable_pairs": int(len(encoded) * (len(encoded) - 1) / 2),
        "zero_pair_rate": 1.0,
    }

    remaining = eligible["Trait column"].tolist()
    while remaining:
        best_trait = None
        best_summary = None
        best_objective = None
        for trait_col in remaining:
            candidate_traits = selected + [trait_col]
            summary = summarize_selected_traits(
                encoded,
                candidate_traits,
                meta_map,
                continuous_bin_count=continuous_bin_count,
            )
            objective = (
                summary["unique_pattern_proportion"] * 5.0
                + (1.0 - summary["zero_pair_rate"]) * 3.0
                + summary["mean_pairwise_distance"]
                + float(eligible.loc[eligible["Trait column"] == trait_col, "Final score"].iloc[0]) * 0.1
            )
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_trait = trait_col
                best_summary = summary

        if best_trait is None or best_summary is None:
            break

        selected.append(best_trait)
        remaining.remove(best_trait)
        current_summary = best_summary
        trace_rows.append(
            {
                "step": len(selected),
                "selected_trait": best_trait,
                "selected_trait_short": short_trait_name(int(meta.loc[meta["Trait column"] == best_trait, "Trait No."].iloc[0])),
                "unique_pattern_proportion": current_summary["unique_pattern_proportion"],
                "mean_pairwise_distance": current_summary["mean_pairwise_distance"],
                "zero_distance_pairs": current_summary["zero_distance_pairs"],
                "comparable_pairs": current_summary["comparable_pairs"],
            }
        )
        if max_steps is not None and len(selected) >= max_steps:
            break
        if stop_on_perfect and current_summary["unique_pattern_proportion"] >= 1.0 and current_summary["zero_distance_pairs"] == 0:
            break

    return selected, pd.DataFrame(trace_rows)


def build_trait_maps(meta: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    short_to_col = {str(row["Trait short"]): str(row["Trait column"]) for _, row in meta.iterrows()}
    col_to_short = {str(row["Trait column"]): str(row["Trait short"]) for _, row in meta.iterrows()}
    short_to_name = {str(row["Trait short"]): str(row["Trait name"]) for _, row in meta.iterrows()}
    return short_to_col, col_to_short, short_to_name


def eligible_trait_pool(scoring_df: pd.DataFrame) -> pd.DataFrame:
    eligible = scoring_df[
        (scoring_df["Availability score"] >= AVAILABILITY_HARD_THRESHOLD)
        & (~scoring_df["Low variation flag"])
    ].copy()
    eligible["Trait no."] = pd.to_numeric(eligible["Trait no."], errors="coerce").astype(int)
    return eligible.sort_values("Trait no.").reset_index(drop=True)


def family_balance_summary(selected_trait_cols: list[str], meta: pd.DataFrame) -> dict[str, object]:
    family_by_col = meta.set_index("Trait column")["Analysis Type"].to_dict()
    families = [str(family_by_col[col]) for col in selected_trait_cols]
    counts = {family: families.count(family) for family in TARGET_TRAIT_FAMILIES}
    trait_number = len(selected_trait_cols)
    represented = sum(1 for value in counts.values() if value > 0)
    coverage = represented / len(TARGET_TRAIT_FAMILIES) if TARGET_TRAIT_FAMILIES else 0.0
    if trait_number == 0:
        imbalance = 0.0
    else:
        expected = 1.0 / len(TARGET_TRAIT_FAMILIES)
        imbalance = float(sum(abs((counts[family] / trait_number) - expected) for family in TARGET_TRAIT_FAMILIES))
    return {
        "family_coverage_count": represented,
        "family_coverage_proportion": coverage,
        "family_imbalance": imbalance,
        "trait_family_composition": "; ".join(f"{family}:{counts[family]}" for family in TARGET_TRAIT_FAMILIES),
    }


def _trait_scale(meta_map: dict[str, tuple[str, object]], encoded: pd.DataFrame, trait_col: str, continuous_bin_count: int) -> tuple[float, float]:
    data_type, code_range = meta_map[trait_col]
    values = pd.to_numeric(encoded[trait_col], errors="coerce")
    if data_type in {"continuous", "continuous_ratio"}:
        return 1.0, float(continuous_bin_count)
    if data_type == "integer_count":
        valid = values.dropna()
        return (float(valid.min()), float(valid.max())) if not valid.empty else (0.0, 0.0)
    parsed = parse_code_minmax(code_range)
    if parsed is not None:
        return parsed
    valid = values.dropna()
    return (float(valid.min()), float(valid.max())) if not valid.empty else (0.0, 0.0)


def precompute_subset_metrics(
    encoded: pd.DataFrame,
    trait_cols: list[str],
    meta: pd.DataFrame,
    continuous_bin_count: int = 5,
) -> dict[str, object]:
    meta_map = meta.set_index("Trait column")[["Data type", "Code range"]].apply(tuple, axis=1).to_dict()
    row_left, row_right = np.triu_indices(len(encoded), 1)
    distance_vectors: list[np.ndarray] = []
    code_columns: list[list[str]] = []

    for trait_col in trait_cols:
        values = pd.to_numeric(encoded[trait_col], errors="coerce").to_numpy(dtype=float)
        scale_min, scale_max = _trait_scale(meta_map, encoded, trait_col, continuous_bin_count)
        denominator = scale_max - scale_min
        distances = np.full(len(row_left), np.nan, dtype=float)
        if denominator > 0:
            left_values = values[row_left]
            right_values = values[row_right]
            valid = ~np.isnan(left_values) & ~np.isnan(right_values)
            distances[valid] = np.abs(left_values[valid] - right_values[valid]) / denominator
        distance_vectors.append(distances)

        code_columns.append([
            "X" if pd.isna(value) else str(int(value)) if float(value).is_integer() else str(value)
            for value in values
        ])

    code_matrix = np.array(code_columns, dtype=object).T if code_columns else np.empty((len(encoded), 0), dtype=object)
    return {
        "distance_vectors": distance_vectors,
        "code_matrix": code_matrix,
        "trait_to_index": {trait_col: idx for idx, trait_col in enumerate(trait_cols)},
        "pair_count": len(row_left),
        "sample_count": len(encoded),
    }


def summarize_trait_subset_fast(selected_trait_cols: list[str], precomputed: dict[str, object]) -> dict[str, float]:
    trait_to_index = precomputed["trait_to_index"]
    selected_indices = [trait_to_index[col] for col in selected_trait_cols]
    sample_count = int(precomputed["sample_count"])
    pair_count = int(precomputed["pair_count"])

    if not selected_indices:
        return {
            "unique_pattern_proportion": 0.0,
            "mean_pairwise_distance": 0.0,
            "zero_distance_pairs": pair_count,
            "comparable_pairs": pair_count,
            "zero_pair_rate": 1.0,
        }

    code_matrix = precomputed["code_matrix"][:, selected_indices]
    unique_prop = float(len({tuple(row) for row in code_matrix.tolist()}) / sample_count)

    selected_distances = [precomputed["distance_vectors"][idx] for idx in selected_indices]
    distance_stack = np.vstack(selected_distances)
    valid_stack = ~np.isnan(distance_stack)
    valid_counts = valid_stack.sum(axis=0)
    comparable = valid_counts > 0
    if not comparable.any():
        mean_distance = 0.0
        zero_pairs = 0
        total_pairs = 0
    else:
        distance_sums = np.nansum(distance_stack, axis=0)
        pair_values = np.full(distance_sums.shape, np.nan, dtype=float)
        pair_values[comparable] = distance_sums[comparable] / valid_counts[comparable]
        valid_pair_values = pair_values[comparable]
        mean_distance = float(valid_pair_values.mean()) if len(valid_pair_values) else 0.0
        zero_pairs = int(np.isclose(valid_pair_values, 0.0).sum()) if len(valid_pair_values) else 0
        total_pairs = int(len(valid_pair_values))
    zero_pair_rate = zero_pairs / total_pairs if total_pairs > 0 else 0.0
    return {
        "unique_pattern_proportion": unique_prop,
        "mean_pairwise_distance": mean_distance,
        "zero_distance_pairs": zero_pairs,
        "comparable_pairs": total_pairs,
        "zero_pair_rate": zero_pair_rate,
    }


def _sort_objective_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(
        [
            "trait_number",
            "sufficient_for_unique_id",
            "zero_distance_pairs",
            "unique_pattern_proportion",
            "family_coverage_count",
            "family_imbalance",
            "mean_pairwise_distance",
            "mean_reliability",
            "mean_final_score",
            "traits",
        ],
        ascending=[True, False, True, False, False, True, False, False, False, True],
    ).reset_index(drop=True)


def exhaustive_select_core_traits(
    scoring_df: pd.DataFrame,
    meta: pd.DataFrame,
    encoded: pd.DataFrame,
    continuous_bin_count: int = 5,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = eligible_trait_pool(scoring_df)
    eligible_cols = eligible["Trait column"].astype(str).tolist()
    precomputed = precompute_subset_metrics(encoded, eligible_cols, meta, continuous_bin_count=continuous_bin_count)
    col_to_short = meta.set_index("Trait column")["Trait short"].astype(str).to_dict()
    final_score = eligible.set_index("Trait column")["Final score"].astype(float).to_dict()
    reliability = eligible.set_index("Trait column")["Reliability score"].astype(float).to_dict()

    rows: list[dict[str, object]] = []
    selected_row: pd.Series | None = None
    for trait_number in range(1, len(eligible_cols) + 1):
        level_rows: list[dict[str, object]] = []
        for combo in combinations(eligible_cols, trait_number):
            selected_cols = list(combo)
            summary = summarize_trait_subset_fast(selected_cols, precomputed)
            balance = family_balance_summary(selected_cols, meta)
            row = {
                "trait_number": trait_number,
                "traits": ", ".join(col_to_short[col] for col in selected_cols),
                "trait_columns": ", ".join(selected_cols),
                "mean_final_score": float(np.mean([final_score[col] for col in selected_cols])),
                "mean_reliability": float(np.mean([reliability[col] for col in selected_cols])),
                "sufficient_for_unique_id": bool(
                    summary["unique_pattern_proportion"] >= 1.0 and summary["zero_distance_pairs"] == 0
                ),
                **balance,
                **summary,
            }
            level_rows.append(row)
        rows.extend(level_rows)
        level_df = _sort_objective_rows(pd.DataFrame(level_rows))
        sufficient = level_df[level_df["sufficient_for_unique_id"]]
        if not sufficient.empty:
            selected_row = sufficient.iloc[0]
            break

    search_df = _sort_objective_rows(pd.DataFrame(rows))
    if selected_row is None:
        selected_row = search_df.iloc[0] if not search_df.empty else None

    best_by_trait_number = (
        search_df.groupby("trait_number", group_keys=False)
        .head(1)
        .reset_index(drop=True)
        if not search_df.empty
        else pd.DataFrame()
    )

    full_cols = eligible_cols
    full_summary = summarize_trait_subset_fast(full_cols, precomputed)
    full_balance = family_balance_summary(full_cols, meta)
    full_row = {
        "scheme": "full_eligible_set",
        "trait_number": len(full_cols),
        "traits": ", ".join(col_to_short[col] for col in full_cols),
        "trait_columns": ", ".join(full_cols),
        "mean_final_score": float(np.mean([final_score[col] for col in full_cols])) if full_cols else 0.0,
        "mean_reliability": float(np.mean([reliability[col] for col in full_cols])) if full_cols else 0.0,
        "sufficient_for_unique_id": bool(full_summary["unique_pattern_proportion"] >= 1.0 and full_summary["zero_distance_pairs"] == 0),
        **full_balance,
        **full_summary,
    }

    selected_cols = []
    scheme_rows: list[dict[str, object]] = []
    if selected_row is not None:
        selected_cols = str(selected_row["trait_columns"]).split(", ")
        selected_dict = selected_row.to_dict()
        selected_dict["scheme"] = "objective_minimal_balanced_set"
        scheme_rows.append(selected_dict)
    scheme_rows.append(full_row)
    scheme_df = pd.DataFrame(scheme_rows)
    return selected_cols, search_df, best_by_trait_number, scheme_df


def compare_trait_numbers(
    selection_order_cols: list[str],
    encoded: pd.DataFrame,
    meta: pd.DataFrame,
    trait_numbers: list[int],
    continuous_bin_count: int = 5,
) -> pd.DataFrame:
    meta_map = meta.set_index("Trait column")[["Data type", "Code range"]].apply(tuple, axis=1).to_dict()
    col_to_short = {str(row["Trait column"]): str(row["Trait short"]) for _, row in meta.iterrows()}
    rows: list[dict[str, object]] = []
    for trait_number in trait_numbers:
        if trait_number > len(selection_order_cols):
            continue
        selected_cols = selection_order_cols[:trait_number]
        summary = summarize_selected_traits(
            encoded,
            selected_cols,
            meta_map,
            continuous_bin_count=continuous_bin_count,
        )
        rows.append(
            {
                "trait_number": trait_number,
                "traits": ", ".join(col_to_short[col] for col in selected_cols),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def compare_trait_schemes(
    scheme_to_traits: dict[str, list[str]],
    encoded: pd.DataFrame,
    meta: pd.DataFrame,
    continuous_bin_count: int = 5,
) -> pd.DataFrame:
    short_to_col, _, _ = build_trait_maps(meta)
    meta_map = meta.set_index("Trait column")[["Data type", "Code range"]].apply(tuple, axis=1).to_dict()
    rows: list[dict[str, object]] = []
    for scheme_name, shorts in scheme_to_traits.items():
        selected_cols = [short_to_col[trait] for trait in shorts if trait in short_to_col]
        if len(selected_cols) != len(shorts):
            continue
        summary = summarize_selected_traits(
            encoded,
            selected_cols,
            meta_map,
            continuous_bin_count=continuous_bin_count,
        )
        family_counts = (
            meta.set_index("Trait short").loc[shorts, "Analysis Type"].value_counts().sort_index().to_dict()
        )
        rows.append(
            {
                "scheme": scheme_name,
                "traits": ", ".join(shorts),
                "trait_number": len(shorts),
                "trait_family_composition": "; ".join(f"{k}:{v}" for k, v in sorted(family_counts.items())),
                **family_balance_summary(selected_cols, meta),
                **summary,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["unique_pattern_proportion", "zero_distance_pairs", "mean_pairwise_distance"],
        ascending=[False, True, False],
    )


def compare_six_trait_schemes(
    scheme_to_traits: dict[str, list[str]],
    encoded: pd.DataFrame,
    meta: pd.DataFrame,
    continuous_bin_count: int = 5,
) -> pd.DataFrame:
    return compare_trait_schemes(
        scheme_to_traits,
        encoded,
        meta,
        continuous_bin_count=continuous_bin_count,
    )


def compare_binning_sensitivity(
    scheme_to_traits: dict[str, list[str]],
    meta: pd.DataFrame,
    df: pd.DataFrame,
    bin_options: list[int],
) -> pd.DataFrame:
    short_to_col, _, _ = build_trait_maps(meta)
    meta_map = meta.set_index("Trait column")[["Data type", "Code range"]].apply(tuple, axis=1).to_dict()
    rows: list[dict[str, object]] = []
    for bin_count in bin_options:
        encoded = build_encoded_id_matrix(meta, df, continuous_bin_count=bin_count)
        for scheme_name, shorts in scheme_to_traits.items():
            selected_cols = [short_to_col[trait] for trait in shorts if trait in short_to_col]
            if len(selected_cols) != len(shorts):
                continue
            summary = summarize_selected_traits(
                encoded,
                selected_cols,
                meta_map,
                continuous_bin_count=bin_count,
            )
            rows.append(
                {
                    "continuous_bin_count": bin_count,
                    "scheme": scheme_name,
                    "traits": ", ".join(shorts),
                    **summary,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scheme", "continuous_bin_count"]
    ).reset_index(drop=True)


def plot_trait_number_comparison(trait_number_df: pd.DataFrame, output_path: Path) -> None:
    if trait_number_df.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    sns.lineplot(data=trait_number_df, x="trait_number", y="unique_pattern_proportion", marker="o", ax=axes[0], color="#1b9e77")
    sns.lineplot(data=trait_number_df, x="trait_number", y="mean_pairwise_distance", marker="o", ax=axes[1], color="#d95f02")
    sns.lineplot(data=trait_number_df, x="trait_number", y="zero_distance_pairs", marker="o", ax=axes[2], color="#7570b3")
    axes[0].set_ylabel("Unique ID proportion")
    axes[1].set_ylabel("Mean pairwise distance")
    axes[2].set_ylabel("Zero-distance pairs")
    axes[2].set_xlabel("Number of selected traits")
    axes[0].set_title("Trait-number optimization for variety-level core fingerprint")
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scheme_comparison(scheme_df: pd.DataFrame, output_path: Path) -> None:
    if scheme_df.empty:
        return
    plot_df = scheme_df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.barplot(data=plot_df, x="scheme", y="unique_pattern_proportion", ax=axes[0], color="#1b9e77")
    sns.barplot(data=plot_df, x="scheme", y="mean_pairwise_distance", ax=axes[1], color="#d95f02")
    sns.barplot(data=plot_df, x="scheme", y="zero_distance_pairs", ax=axes[2], color="#7570b3")
    axes[0].set_ylabel("Unique ID proportion")
    axes[1].set_ylabel("Mean pairwise distance")
    axes[2].set_ylabel("Zero-distance pairs")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    axes[0].set_title("Objective trait-set comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_six_trait_scheme_comparison(scheme_df: pd.DataFrame, output_path: Path) -> None:
    plot_scheme_comparison(scheme_df, output_path)


def plot_binning_sensitivity(sensitivity_df: pd.DataFrame, output_path: Path) -> None:
    if sensitivity_df.empty:
        return
    heatmap_df = sensitivity_df.pivot(index="scheme", columns="continuous_bin_count", values="unique_pattern_proportion")
    plt.figure(figsize=(8, max(4, heatmap_df.shape[0] * 0.6)))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "Unique ID proportion"})
    plt.title("Sensitivity of variety-level ID performance to continuous-trait binning")
    plt.xlabel("Continuous-trait bin count")
    plt.ylabel("Scheme")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    meta = load_trait_dictionary()
    df = load_trait_data(meta)
    reliability_df = load_reliability(meta)

    scoring_df = build_scoring_base(meta, df, reliability_df)
    scoring_df = subgroup_discriminative_analysis(scoring_df, df)
    scoring_df, rf_metrics_df = subgroup_model_analysis(scoring_df, meta, df)
    scoring_df, corr_df, redundant_pairs_df = redundancy_analysis(scoring_df, meta, df)
    base_bin_count = 5
    scoring_df, encoded_id_df = variety_distinctness_analysis(scoring_df, meta, df, continuous_bin_count=base_bin_count)

    scoring_df["Reliability score"] = pd.to_numeric(scoring_df["Reliability score"], errors="coerce")
    scoring_df["Availability score"] = scoring_df["Availability score"].clip(0, 1)
    scoring_df["Redundancy score"] = scoring_df["Redundancy score"].clip(0, 1)
    scoring_df["Subgroup discriminative score"] = scoring_df["Subgroup discriminative score"].clip(0, 1)
    scoring_df["Model importance score"] = scoring_df["Model importance score"].clip(0, 1)
    scoring_df["Variety distinctness score"] = scoring_df["Variety distinctness score"].clip(0, 1)

    scoring_df["Quality pass"] = (
        (scoring_df["Availability score"] >= AVAILABILITY_HARD_THRESHOLD)
        & (~scoring_df["Low variation flag"])
    )
    scoring_df["Final score"] = (
        0.20 * scoring_df["Availability score"].fillna(0)
        + 0.25 * scoring_df["Reliability score"].fillna(0)
        + 0.10 * scoring_df["Subgroup discriminative score"].fillna(0)
        + 0.05 * scoring_df["Model importance score"].fillna(0)
        + 0.10 * scoring_df["Redundancy score"].fillna(0)
        + 0.20 * scoring_df["Pairwise distinctness rate"].fillna(0)
        + 0.10 * scoring_df["Distance contribution"].fillna(0)
    )

    selected_trait_cols, exhaustive_search_df, objective_trait_number_df, objective_scheme_df = exhaustive_select_core_traits(
        scoring_df,
        meta,
        encoded_id_df,
        continuous_bin_count=base_bin_count,
    )
    scoring_df["Selected or not"] = scoring_df["Trait column"].isin(selected_trait_cols)
    scoring_df = scoring_df.sort_values(["Selected or not", "Final score", "Variety distinctness score"], ascending=[False, False, False]).reset_index(drop=True)

    core_meta = meta[meta["Trait column"].isin(selected_trait_cols)].copy()
    core_meta["selection_order"] = core_meta["Trait column"].map({trait: i + 1 for i, trait in enumerate(selected_trait_cols)})
    core_meta = core_meta.sort_values("selection_order")

    core_code_df = exact_code_matrix(encoded_id_df, selected_trait_cols)
    if selected_trait_cols:
        core_code_df["Phenotypic ID code"] = core_code_df[selected_trait_cols].astype(str).agg("|".join, axis=1)
    else:
        core_code_df["Phenotypic ID code"] = ""

    if selected_trait_cols:
        selected_distance_df = build_pairwise_distance_matrix(
            encoded_id_df,
            selected_trait_cols,
            meta.set_index("Trait column")[["Data type", "Code range"]].apply(tuple, axis=1).to_dict(),
            continuous_bin_count=base_bin_count,
        )
        selected_distance_df.to_csv(OUTPUT_DIR / "selected_core_traits_gower_like_distance.csv", encoding="utf-8-sig")

    rf_importance_df = scoring_df[
        [
            "Trait column",
            "Trait no.",
            "Trait short",
            "Trait name",
            "RF importance",
            "Permutation importance mean",
            "Permutation importance std",
            "Model importance score",
        ]
    ].copy()

    corr_df.to_csv(OUTPUT_DIR / "trait_spearman_correlation.csv", encoding="utf-8-sig")
    redundant_pairs_df.to_csv(OUTPUT_DIR / "redundant_trait_pairs.csv", index=False, encoding="utf-8-sig")
    rf_importance_df.to_csv(OUTPUT_DIR / "rf_importance.csv", index=False, encoding="utf-8-sig")
    rf_metrics_df.to_csv(OUTPUT_DIR / "rf_classification_metrics.csv", index=False, encoding="utf-8-sig")
    scoring_df.to_csv(OUTPUT_DIR / "core_trait_scoring_table.csv", index=False, encoding="utf-8-sig")
    core_meta.to_csv(OUTPUT_DIR / "selected_core_traits.csv", index=False, encoding="utf-8-sig")
    objective_trait_number_df.to_csv(OUTPUT_DIR / "objective_trait_number_comparison.csv", index=False, encoding="utf-8-sig")
    objective_trait_number_df.to_csv(OUTPUT_DIR / "forward_selection_trace.csv", index=False, encoding="utf-8-sig")
    exhaustive_search_df.to_csv(OUTPUT_DIR / "exhaustive_trait_subset_search.csv", index=False, encoding="utf-8-sig")
    exhaustive_search_df.to_csv(OUTPUT_DIR / "forward_selection_trace_full.csv", index=False, encoding="utf-8-sig")
    encoded_id_df.to_csv(OUTPUT_DIR / "encoded_trait_matrix_for_id.csv", index=False, encoding="utf-8-sig")
    core_code_df.to_csv(OUTPUT_DIR / "phenotypic_id_codes.csv", index=False, encoding="utf-8-sig")

    trait_number_df = objective_trait_number_df.copy()
    trait_number_df.to_csv(OUTPUT_DIR / "trait_number_comparison.csv", index=False, encoding="utf-8-sig")

    objective_scheme_df.to_csv(OUTPUT_DIR / "objective_scheme_comparison.csv", index=False, encoding="utf-8-sig")
    objective_scheme_df.to_csv(OUTPUT_DIR / "six_trait_scheme_comparison.csv", index=False, encoding="utf-8-sig")

    sensitivity_scheme_set = {
        str(row["scheme"]): [trait.strip() for trait in str(row["traits"]).split(",") if trait.strip()]
        for _, row in objective_scheme_df.iterrows()
    }
    sensitivity_df = compare_binning_sensitivity(
        sensitivity_scheme_set,
        meta,
        df,
        CONTINUOUS_BIN_OPTIONS,
    )
    sensitivity_df.to_csv(OUTPUT_DIR / "binning_sensitivity_comparison.csv", index=False, encoding="utf-8-sig")

    plot_trait_number_comparison(trait_number_df, OUTPUT_DIR / "trait_number_comparison.png")
    plot_scheme_comparison(objective_scheme_df, OUTPUT_DIR / "objective_scheme_comparison.png")
    plot_six_trait_scheme_comparison(objective_scheme_df, OUTPUT_DIR / "six_trait_scheme_comparison.png")
    plot_binning_sensitivity(sensitivity_df, OUTPUT_DIR / "binning_sensitivity_heatmap.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build variety-level core fingerprint outputs.")
    parser.add_argument("--input-file", default=str(INPUT_FILE), help="Input trait matrix CSV.")
    parser.add_argument("--dict-file", default=str(DICT_FILE), help="Trait dictionary CSV.")
    parser.add_argument("--reliability-file", default=str(RELIABILITY_FILE), help="Trait reliability CSV.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    INPUT_FILE = Path(args.input_file)
    DICT_FILE = Path(args.dict_file)
    RELIABILITY_FILE = Path(args.reliability_file)
    OUTPUT_DIR = Path(args.output_dir)
    main()
