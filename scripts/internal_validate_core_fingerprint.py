from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import build_variety_core_fingerprint as mod


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "variety_core_fingerprint"
SCORING_FILE = OUTPUT_DIR / "core_trait_scoring_table.csv"
SUPPLEMENTARY_XLSX = REPO_ROOT / "data" / "Supplementary_Table1-9.xlsx"

BOOTSTRAP_REPEATS = 1000
HOLDOUT_REPEATS = 100
PERTURBATION_REPEATS = 200
BASE_BIN_COUNT = 5


def build_trait_scale_map(
    meta: pd.DataFrame,
    encoded: pd.DataFrame,
    selected_trait_cols: list[str],
    continuous_bin_count: int,
) -> dict[str, tuple[float, float]]:
    meta_indexed = meta.set_index("Trait column")
    scale_map: dict[str, tuple[float, float]] = {}
    for trait_col in selected_trait_cols:
        data_type = str(meta_indexed.loc[trait_col, "Data type"])
        code_range = meta_indexed.loc[trait_col, "Code range"]
        if data_type in {"continuous", "continuous_ratio"}:
            scale_map[trait_col] = (1.0, float(continuous_bin_count))
        elif data_type == "integer_count":
            valid = pd.to_numeric(encoded[trait_col], errors="coerce").dropna()
            scale_map[trait_col] = (float(valid.min()), float(valid.max())) if not valid.empty else (0.0, 0.0)
        else:
            parsed = mod.parse_code_minmax(code_range)
            if parsed is None:
                scale_map[trait_col] = (0.0, 0.0)
            else:
                scale_map[trait_col] = parsed
    return scale_map


def build_pattern_strings(values: np.ndarray) -> np.ndarray:
    rows = []
    for row in values:
        tokens = []
        for value in row:
            if np.isnan(value):
                tokens.append("X")
            elif float(value).is_integer():
                tokens.append(str(int(value)))
            else:
                tokens.append(f"{value:g}")
        rows.append("|".join(tokens))
    return np.asarray(rows, dtype=object)


def summarize_selected_traits_fast(
    encoded: pd.DataFrame,
    selected_trait_cols: list[str],
    meta: pd.DataFrame,
    continuous_bin_count: int = BASE_BIN_COUNT,
) -> dict[str, float]:
    if not selected_trait_cols:
        return {
            "unique_pattern_proportion": 0.0,
            "mean_pairwise_distance": 0.0,
            "zero_distance_pairs": 0,
            "comparable_pairs": 0,
            "zero_pair_rate": 0.0,
        }

    x = encoded[selected_trait_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n_samples = x.shape[0]
    pattern_strings = build_pattern_strings(x)
    unique_pattern_proportion = float(pd.Series(pattern_strings).nunique() / max(1, n_samples))

    sum_distance = np.zeros((n_samples, n_samples), dtype=float)
    count_matrix = np.zeros((n_samples, n_samples), dtype=np.int16)
    scale_map = build_trait_scale_map(meta, encoded, selected_trait_cols, continuous_bin_count)

    for col_idx, trait_col in enumerate(selected_trait_cols):
        scale_min, scale_max = scale_map[trait_col]
        denominator = scale_max - scale_min
        if denominator <= 0:
            continue
        values = x[:, col_idx]
        valid = np.isfinite(values)
        valid_pairs = valid[:, None] & valid[None, :]
        distances = np.abs(values[:, None] - values[None, :]) / denominator
        sum_distance += np.where(valid_pairs, distances, 0.0)
        count_matrix += valid_pairs.astype(np.int16)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_distance_matrix = np.divide(
            sum_distance,
            count_matrix,
            out=np.full_like(sum_distance, np.nan, dtype=float),
            where=count_matrix > 0,
        )

    upper = np.triu_indices(n_samples, k=1)
    pair_values = mean_distance_matrix[upper]
    valid_pair_values = pair_values[np.isfinite(pair_values)]
    comparable_pairs = int(valid_pair_values.size)
    if comparable_pairs == 0:
        return {
            "unique_pattern_proportion": unique_pattern_proportion,
            "mean_pairwise_distance": 0.0,
            "zero_distance_pairs": 0,
            "comparable_pairs": 0,
            "zero_pair_rate": 0.0,
        }

    zero_distance_pairs = int(np.sum(valid_pair_values == 0))
    return {
        "unique_pattern_proportion": unique_pattern_proportion,
        "mean_pairwise_distance": float(valid_pair_values.mean()),
        "zero_distance_pairs": zero_distance_pairs,
        "comparable_pairs": comparable_pairs,
        "zero_pair_rate": float(zero_distance_pairs / comparable_pairs),
    }


def load_rrmse_map() -> dict[str, float]:
    measured = pd.read_excel(SUPPLEMENTARY_XLSX, sheet_name="Supplementary Table 4", header=1)
    measured["Trait No."] = pd.to_numeric(measured["Trait No."], errors="coerce")
    measured["rRMSE/%"] = pd.to_numeric(measured["rRMSE/%"], errors="coerce")
    return {
        mod.trait_col_from_no(int(row["Trait No."])): float(row["rRMSE/%"]) / 100.0
        for _, row in measured.dropna(subset=["Trait No.", "rRMSE/%"]).iterrows()
    }


def load_ordinal_disagreement_map(meta: pd.DataFrame, df: pd.DataFrame) -> dict[str, float]:
    rates: dict[str, float] = {}
    for _, row in meta.iterrows():
        trait_col = str(row["Trait column"])
        data_type = str(row["Data type"])
        if not data_type.startswith("ordinal_"):
            continue
        actual_col = trait_col.replace("_Predict", "_Actual")
        if actual_col not in df.columns:
            rates[trait_col] = 0.0
            continue
        actual = pd.to_numeric(df[actual_col], errors="coerce")
        pred = pd.to_numeric(df[trait_col], errors="coerce")
        valid = actual.notna() & pred.notna()
        if not valid.any():
            rates[trait_col] = 0.0
            continue
        mismatch_rate = float((actual[valid] != pred[valid]).mean())
        rates[trait_col] = mismatch_rate
    return rates


def apply_noise_perturbation(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    mode: str,
    magnitude: float,
    rng: np.random.Generator,
    rrmse_map: dict[str, float],
    ordinal_disagreement_map: dict[str, float],
) -> pd.DataFrame:
    perturbed = df.copy()
    for _, row in meta.iterrows():
        trait_col = str(row["Trait column"])
        data_type = str(row["Data type"])
        values = pd.to_numeric(perturbed[trait_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        if not np.any(valid):
            continue

        if mode in {"uniform", "rrmse"}:
            if data_type not in {"continuous", "continuous_ratio"}:
                continue
            if mode == "uniform":
                noise = rng.uniform(-magnitude, magnitude, size=valid.sum())
                values[valid] = values[valid] * (1.0 + noise)
            elif mode == "rrmse":
                sigma = rrmse_map.get(trait_col, 0.0)
                noise = rng.normal(0.0, sigma * magnitude, size=valid.sum())
                values[valid] = values[valid] * (1.0 + noise)
            values[valid] = np.clip(values[valid], a_min=0.0, a_max=None)
            perturbed[trait_col] = values
        elif mode == "ordinal_adjacent":
            if not data_type.startswith("ordinal_"):
                continue
            mismatch_rate = ordinal_disagreement_map.get(trait_col, 0.0)
            if mismatch_rate <= 0:
                continue
            code_range = row["Code range"]
            parsed = mod.parse_code_minmax(code_range)
            if parsed is None:
                continue
            code_min, code_max = parsed
            valid_idx = np.where(valid)[0]
            move_mask = rng.random(valid_idx.size) < mismatch_rate
            if np.any(move_mask):
                move_idx = valid_idx[move_mask]
                direction = rng.choice(np.array([-1.0, 1.0]), size=move_idx.size)
                values[move_idx] = np.clip(values[move_idx] + direction, code_min, code_max)
            perturbed[trait_col] = values
        else:
            raise ValueError(f"Unsupported perturbation mode: {mode}")
    return perturbed


def recompute_subset_model_importance(meta: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    x, y = mod.build_model_matrix(meta, train_df)
    trait_cols = meta["Trait column"].tolist()
    preprocessor = mod.ColumnTransformer(
        transformers=[
            (
                "numeric",
                mod.Pipeline(
                    [
                        ("imputer", mod.SimpleImputer(strategy="median")),
                        ("scaler", mod.StandardScaler()),
                    ]
                ),
                trait_cols,
            )
        ]
    )
    model = mod.RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = mod.Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(x, y)
    rf_importance = pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame(
        {
            "Trait column": trait_cols,
            "Model importance score": mod.normalize_series_to_unit(pd.Series(rf_importance)).to_numpy(),
        }
    )
    importance_df["Model importance score"] = importance_df["Model importance score"].fillna(0.0)
    return importance_df


def finalize_scoring_without_refit_rf(
    meta: pd.DataFrame,
    train_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    global_model_importance: pd.DataFrame,
    continuous_bin_count: int = BASE_BIN_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoring_df = mod.build_scoring_base(meta, train_df, reliability_df)
    scoring_df = mod.subgroup_discriminative_analysis(scoring_df, train_df)
    scoring_df, _, _ = mod.redundancy_analysis(scoring_df, meta, train_df)
    scoring_df, encoded_df = mod.variety_distinctness_analysis(
        scoring_df,
        meta,
        train_df,
        continuous_bin_count=continuous_bin_count,
    )

    scoring_df = scoring_df.merge(
        global_model_importance,
        on="Trait column",
        how="left",
    )
    scoring_df["Availability score"] = scoring_df["Availability score"].clip(0, 1)
    scoring_df["Reliability score"] = pd.to_numeric(scoring_df["Reliability score"], errors="coerce")
    scoring_df["Subgroup discriminative score"] = scoring_df["Subgroup discriminative score"].clip(0, 1)
    scoring_df["Model importance score"] = scoring_df["Model importance score"].fillna(0).clip(0, 1)
    scoring_df["Redundancy score"] = scoring_df["Redundancy score"].clip(0, 1)
    scoring_df["Pairwise distinctness rate"] = scoring_df["Pairwise distinctness rate"].fillna(0).clip(0, 1)
    scoring_df["Distance contribution"] = scoring_df["Distance contribution"].fillna(0).clip(0, 1)
    scoring_df["Quality pass"] = (
        (scoring_df["Availability score"] >= mod.AVAILABILITY_HARD_THRESHOLD)
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
    return scoring_df, encoded_df


def summarize_replicates(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "sample_size",
        "unique_pattern_proportion",
        "mean_pairwise_distance",
        "zero_distance_pairs",
        "zero_pair_rate",
    ]
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_repeats"] = len(sub)
        for metric in metric_cols:
            values = pd.to_numeric(sub[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=0))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_q025"] = float(values.quantile(0.025))
            row[f"{metric}_q975"] = float(values.quantile(0.975))
        row["perfect_uniqueness_rate"] = float((sub["zero_distance_pairs"] == 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_objective_schemes(meta: pd.DataFrame, scoring_df: pd.DataFrame, encoded: pd.DataFrame) -> dict[str, list[str]]:
    _, _, _, objective_scheme_df = mod.exhaustive_select_core_traits(
        scoring_df,
        meta,
        encoded,
        continuous_bin_count=BASE_BIN_COUNT,
    )
    return {
        str(row["scheme"]): [trait.strip() for trait in str(row["traits"]).split(",") if trait.strip()]
        for _, row in objective_scheme_df.iterrows()
    }


def evaluate_fixed_schemes(
    encoded: pd.DataFrame,
    meta: pd.DataFrame,
    scheme_to_traits: dict[str, list[str]],
    short_to_col: dict[str, str],
    continuous_bin_count: int = BASE_BIN_COUNT,
) -> list[dict[str, object]]:
    rows = []
    for scheme_name, shorts in scheme_to_traits.items():
        selected_cols = [short_to_col[s] for s in shorts]
        summary = summarize_selected_traits_fast(
            encoded,
            selected_cols,
            meta,
            continuous_bin_count=continuous_bin_count,
        )
        rows.append(
            {
                "scheme": scheme_name,
                "trait_set": ", ".join(shorts),
                "trait_number": len(shorts),
                "sample_size": len(encoded),
                **summary,
            }
        )
    return rows


def run_bootstrap_validation(
    meta: pd.DataFrame,
    df: pd.DataFrame,
    scheme_to_traits: dict[str, list[str]],
    short_to_col: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for replicate in range(1, BOOTSTRAP_REPEATS + 1):
        sampled_indices = rng.choice(df.index.to_numpy(), size=len(df), replace=True)
        sampled = df.loc[sampled_indices].drop_duplicates(subset=[mod.ID_COL]).reset_index(drop=True)
        encoded = mod.build_encoded_id_matrix(meta, sampled, continuous_bin_count=BASE_BIN_COUNT)
        for row in evaluate_fixed_schemes(encoded, meta, scheme_to_traits, short_to_col, BASE_BIN_COUNT):
            row.update({"validation_type": "bootstrap", "replicate": replicate})
            rows.append(row)
    raw_df = pd.DataFrame(rows)
    summary_df = summarize_replicates(raw_df, ["validation_type", "scheme", "trait_set", "trait_number"])
    return raw_df, summary_df


def run_holdout_validation(
    meta: pd.DataFrame,
    df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    global_scoring_df: pd.DataFrame,
    scheme_to_traits: dict[str, list[str]],
    short_to_col: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=HOLDOUT_REPEATS, test_size=0.2, random_state=42)
    y = df[mod.GROUP_COL].astype(str)
    rows: list[dict[str, object]] = []
    for replicate, (train_idx, test_idx) in enumerate(splitter.split(df, y), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        subset_model_importance = recompute_subset_model_importance(meta, train_df)

        train_scoring_df, train_encoded_df = finalize_scoring_without_refit_rf(
            meta,
            train_df,
            reliability_df,
            subset_model_importance,
            continuous_bin_count=BASE_BIN_COUNT,
        )
        selected_cols, _, _, train_objective_scheme_df = mod.exhaustive_select_core_traits(
            train_scoring_df,
            meta,
            train_encoded_df,
            continuous_bin_count=BASE_BIN_COUNT,
        )

        col_to_short = {str(row["Trait column"]): str(row["Trait short"]) for _, row in meta.iterrows()}
        train_minimal_shorts = [col_to_short[col] for col in selected_cols]
        train_objective_schemes = {
            f"train_{row['scheme']}": [trait.strip() for trait in str(row["traits"]).split(",") if trait.strip()]
            for _, row in train_objective_scheme_df.iterrows()
        }
        train_objective_schemes["train_objective_minimal_balanced_set"] = train_minimal_shorts

        test_encoded_df = mod.build_encoded_id_matrix(meta, test_df, continuous_bin_count=BASE_BIN_COUNT)
        holdout_schemes = {
            **train_objective_schemes,
            **scheme_to_traits,
        }

        for row in evaluate_fixed_schemes(test_encoded_df, meta, holdout_schemes, short_to_col, BASE_BIN_COUNT):
            row.update(
                {
                    "validation_type": "holdout",
                    "replicate": replicate,
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                }
            )
            rows.append(row)
    raw_df = pd.DataFrame(rows)
    summary_df = summarize_replicates(raw_df, ["validation_type", "scheme", "trait_set", "trait_number"])
    return raw_df, summary_df


def run_noise_validation(
    meta: pd.DataFrame,
    df: pd.DataFrame,
    scheme_to_traits: dict[str, list[str]],
    short_to_col: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(2024)
    rows: list[dict[str, object]] = []
    rrmse_map = load_rrmse_map()
    ordinal_disagreement_map = load_ordinal_disagreement_map(meta, df)
    scenarios = [
        ("uniform", 0.05, "uniform_pm5pct"),
        ("uniform", 0.10, "uniform_pm10pct"),
        ("rrmse", 1.0, "gaussian_rrmse_x1"),
        ("ordinal_adjacent", 1.0, "ordinal_adjacent_empirical"),
    ]

    for scenario_mode, magnitude, scenario_name in scenarios:
        for replicate in range(1, PERTURBATION_REPEATS + 1):
            perturbed_df = apply_noise_perturbation(
                df,
                meta,
                scenario_mode,
                magnitude,
                rng,
                rrmse_map,
                ordinal_disagreement_map,
            )
            encoded = mod.build_encoded_id_matrix(meta, perturbed_df, continuous_bin_count=BASE_BIN_COUNT)
            for row in evaluate_fixed_schemes(encoded, meta, scheme_to_traits, short_to_col, BASE_BIN_COUNT):
                row.update(
                    {
                        "validation_type": "noise",
                        "scenario": scenario_name,
                        "replicate": replicate,
                    }
                )
                rows.append(row)
    raw_df = pd.DataFrame(rows)
    summary_df = summarize_replicates(raw_df, ["validation_type", "scenario", "scheme", "trait_set", "trait_number"])
    return raw_df, summary_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = mod.load_trait_dictionary()
    df = mod.load_trait_data(meta)
    reliability_df = mod.load_reliability(meta)
    global_scoring_df = pd.read_csv(SCORING_FILE)
    global_encoded = mod.build_encoded_id_matrix(meta, df, continuous_bin_count=BASE_BIN_COUNT)
    scheme_to_traits = build_objective_schemes(meta, global_scoring_df, global_encoded)
    short_to_col, _, _ = mod.build_trait_maps(meta)

    bootstrap_raw, bootstrap_summary = run_bootstrap_validation(meta, df, scheme_to_traits, short_to_col)
    holdout_raw, holdout_summary = run_holdout_validation(
        meta,
        df,
        reliability_df,
        global_scoring_df,
        scheme_to_traits,
        short_to_col,
    )
    noise_raw, noise_summary = run_noise_validation(meta, df, scheme_to_traits, short_to_col)

    bootstrap_raw.to_csv(OUTPUT_DIR / "internal_validation_bootstrap_raw.csv", index=False, encoding="utf-8-sig")
    bootstrap_summary.to_csv(OUTPUT_DIR / "internal_validation_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    holdout_raw.to_csv(OUTPUT_DIR / "internal_validation_holdout_raw.csv", index=False, encoding="utf-8-sig")
    holdout_summary.to_csv(OUTPUT_DIR / "internal_validation_holdout_summary.csv", index=False, encoding="utf-8-sig")
    noise_raw.to_csv(OUTPUT_DIR / "internal_validation_noise_raw.csv", index=False, encoding="utf-8-sig")
    noise_summary.to_csv(OUTPUT_DIR / "internal_validation_noise_summary.csv", index=False, encoding="utf-8-sig")

    holdout_scheme_summary = summarize_replicates(
        holdout_raw,
        ["validation_type", "scheme"],
    )
    holdout_scheme_summary.to_csv(
        OUTPUT_DIR / "internal_validation_holdout_scheme_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overview = pd.concat(
        [
            bootstrap_summary.assign(section="bootstrap"),
            holdout_summary.assign(section="holdout"),
            noise_summary.assign(section="noise"),
        ],
        ignore_index=True,
        sort=False,
    )
    overview.to_csv(OUTPUT_DIR / "internal_validation_overview.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run internal validation for variety-level core fingerprint schemes.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Core fingerprint output directory.")
    parser.add_argument("--supplementary-xlsx", default=str(SUPPLEMENTARY_XLSX), help="Supplementary workbook path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    SCORING_FILE = OUTPUT_DIR / "core_trait_scoring_table.csv"
    SUPPLEMENTARY_XLSX = Path(args.supplementary_xlsx)
    main()
