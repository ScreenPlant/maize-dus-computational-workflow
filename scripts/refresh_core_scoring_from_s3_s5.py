from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

import build_variety_core_fingerprint as mod


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "variety_core_fingerprint"
S3_PATH = ROOT / "outputs" / "supplementary_tables" / "Supplementary_Table_S3_trait_availability_preprocessing_agreement.csv"
S5_PATH = ROOT / "outputs" / "supplementary_tables" / "Supplementary_Table_S5_biological_discrimination_statistics.csv"
RELIABILITY_PATH = ROOT / "trait_reliability_scores.csv"
SCORING_PATH = OUTPUT_DIR / "core_trait_scoring_table.csv"
SUMMARY_PATH = OUTPUT_DIR / "core_scoring_refresh_summary.csv"
BASE_BIN_COUNT = 5


def parse_metric_value(text: object) -> float | None:
    if pd.isna(text):
        return None
    match = re.search(r"=\s*([-+]?\d*\.?\d+)", str(text))
    if not match:
        return None
    return float(match.group(1))


def load_reliability_from_s3(meta: pd.DataFrame) -> pd.DataFrame:
    s3 = pd.read_csv(S3_PATH)
    rows: list[dict[str, object]] = []
    meta_map = meta.set_index("Trait short")[
        ["Trait No.", "Trait column", "Trait name", "Observation Method", "Analysis Type", "Data type"]
    ].to_dict("index")

    for _, row in s3.iterrows():
        trait_code = str(row["Trait code"])
        if trait_code not in meta_map:
            continue
        meta_row = meta_map[trait_code]
        reference_type = str(row["Expert reference type"]).strip().lower()
        if reference_type == "manual measurement":
            reliability_score = parse_metric_value(row["Agreement metric 1"])
            reliability_metric = "R2"
        else:
            reliability_score = parse_metric_value(row["Agreement metric 2"])
            reliability_metric = "weighted_kappa"
        rows.append(
            {
                "Trait No.": int(meta_row["Trait No."]),
                "Trait short": trait_code,
                "Trait column": str(meta_row["Trait column"]),
                "Trait name": str(meta_row["Trait name"]),
                "Observation Method": str(meta_row["Observation Method"]),
                "Analysis Type": str(meta_row["Analysis Type"]),
                "Data type": str(meta_row["Data type"]),
                "Reliability score": reliability_score,
                "Reliability metric": reliability_metric,
                "Reliability source": "supplementary_table_s3",
            }
        )

    reliability_df = pd.DataFrame(rows).sort_values("Trait No.").reset_index(drop=True)
    reliability_df.to_csv(RELIABILITY_PATH, index=False, encoding="utf-8-sig")
    return reliability_df


def load_endosperm_eta_from_s5() -> pd.DataFrame:
    s5 = pd.read_csv(S5_PATH)
    subset = s5[
        (s5["Grouping factor"].astype(str) == "endosperm type")
        & (s5["Used in composite score?"].astype(str) == "Yes")
    ].copy()
    subset["Eta-squared η²"] = pd.to_numeric(subset["Eta-squared η²"], errors="coerce")
    eta_df = (
        subset.groupby("Trait code", as_index=False)["Eta-squared η²"]
        .first()
        .rename(columns={"Trait code": "Trait short", "Eta-squared η²": "Effect size"})
    )
    return eta_df


def rebuild_scoring_table(
    meta: pd.DataFrame,
    df: pd.DataFrame,
    scoring_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    eta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scoring = scoring_df.copy()

    reliability_map = reliability_df.set_index("Trait short")[
        ["Reliability score", "Reliability metric", "Reliability source"]
    ].to_dict("index")
    eta_map = eta_df.set_index("Trait short")["Effect size"].to_dict()

    scoring["Reliability score"] = scoring["Trait short"].map(
        lambda x: reliability_map.get(str(x), {}).get("Reliability score")
    )
    scoring["Reliability metric"] = scoring["Trait short"].map(
        lambda x: reliability_map.get(str(x), {}).get("Reliability metric", "")
    )
    scoring["Reliability source"] = scoring["Trait short"].map(
        lambda x: reliability_map.get(str(x), {}).get("Reliability source", "")
    )
    scoring["Effect size"] = scoring["Trait short"].map(lambda x: eta_map.get(str(x)))
    scoring["Subgroup discriminative score"] = mod.normalize_series_to_unit(scoring["Effect size"])

    scoring["Reliability score"] = pd.to_numeric(scoring["Reliability score"], errors="coerce").clip(0, 1)
    scoring["Availability score"] = pd.to_numeric(scoring["Availability score"], errors="coerce").clip(0, 1)
    scoring["Redundancy score"] = pd.to_numeric(scoring["Redundancy score"], errors="coerce").clip(0, 1)
    scoring["Subgroup discriminative score"] = pd.to_numeric(
        scoring["Subgroup discriminative score"], errors="coerce"
    ).clip(0, 1)
    scoring["Model importance score"] = pd.to_numeric(scoring["Model importance score"], errors="coerce").clip(0, 1)
    scoring["Pairwise distinctness rate"] = pd.to_numeric(
        scoring["Pairwise distinctness rate"], errors="coerce"
    ).clip(0, 1)
    scoring["Distance contribution"] = pd.to_numeric(scoring["Distance contribution"], errors="coerce").clip(0, 1)

    scoring["Quality pass"] = (
        (scoring["Availability score"] >= mod.AVAILABILITY_HARD_THRESHOLD)
        & (~scoring["Low variation flag"].fillna(False).astype(bool))
    )
    scoring["Final score"] = (
        0.20 * scoring["Availability score"].fillna(0)
        + 0.25 * scoring["Reliability score"].fillna(0)
        + 0.10 * scoring["Subgroup discriminative score"].fillna(0)
        + 0.05 * scoring["Model importance score"].fillna(0)
        + 0.10 * scoring["Redundancy score"].fillna(0)
        + 0.20 * scoring["Pairwise distinctness rate"].fillna(0)
        + 0.10 * scoring["Distance contribution"].fillna(0)
    )

    encoded_id_df = mod.build_encoded_id_matrix(meta, df, continuous_bin_count=BASE_BIN_COUNT)
    selected_trait_cols, exhaustive_search_df, objective_trait_number_df, objective_scheme_df = (
        mod.exhaustive_select_core_traits(
            scoring,
            meta,
            encoded_id_df,
            continuous_bin_count=BASE_BIN_COUNT,
        )
    )
    scoring["Selected or not"] = scoring["Trait column"].isin(selected_trait_cols)
    scoring = scoring.sort_values(
        ["Selected or not", "Final score", "Variety distinctness score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return scoring, selected_trait_cols, exhaustive_search_df, objective_trait_number_df, objective_scheme_df


def save_outputs(
    meta: pd.DataFrame,
    encoded_id_df: pd.DataFrame,
    scoring_df: pd.DataFrame,
    selected_trait_cols: list[str],
    exhaustive_search_df: pd.DataFrame,
    objective_trait_number_df: pd.DataFrame,
    objective_scheme_df: pd.DataFrame,
) -> None:
    scoring_df.to_csv(SCORING_PATH, index=False, encoding="utf-8-sig")

    core_meta = meta[meta["Trait column"].isin(selected_trait_cols)].copy()
    core_meta["selection_order"] = core_meta["Trait column"].map({trait: i + 1 for i, trait in enumerate(selected_trait_cols)})
    core_meta = core_meta.sort_values("selection_order")
    core_meta.to_csv(OUTPUT_DIR / "selected_core_traits.csv", index=False, encoding="utf-8-sig")

    objective_trait_number_df.to_csv(
        OUTPUT_DIR / "objective_trait_number_comparison.csv", index=False, encoding="utf-8-sig"
    )
    objective_trait_number_df.to_csv(
        OUTPUT_DIR / "forward_selection_trace.csv", index=False, encoding="utf-8-sig"
    )
    exhaustive_search_df.to_csv(
        OUTPUT_DIR / "exhaustive_trait_subset_search.csv", index=False, encoding="utf-8-sig"
    )
    exhaustive_search_df.to_csv(
        OUTPUT_DIR / "forward_selection_trace_full.csv", index=False, encoding="utf-8-sig"
    )
    objective_scheme_df.to_csv(
        OUTPUT_DIR / "objective_scheme_comparison.csv", index=False, encoding="utf-8-sig"
    )
    objective_scheme_df.to_csv(
        OUTPUT_DIR / "six_trait_scheme_comparison.csv", index=False, encoding="utf-8-sig"
    )
    encoded_id_df.to_csv(OUTPUT_DIR / "encoded_trait_matrix_for_id.csv", index=False, encoding="utf-8-sig")

    core_code_df = mod.exact_code_matrix(encoded_id_df, selected_trait_cols)
    if selected_trait_cols:
        core_code_df["Phenotypic ID code"] = core_code_df[selected_trait_cols].astype(str).agg("|".join, axis=1)
    else:
        core_code_df["Phenotypic ID code"] = ""
    core_code_df.to_csv(OUTPUT_DIR / "phenotypic_id_codes.csv", index=False, encoding="utf-8-sig")

    if selected_trait_cols:
        selected_distance_df = mod.build_pairwise_distance_matrix(
            encoded_id_df,
            selected_trait_cols,
            meta.set_index("Trait column")[["Data type", "Code range"]].apply(tuple, axis=1).to_dict(),
            continuous_bin_count=BASE_BIN_COUNT,
        )
        selected_distance_df.to_csv(
            OUTPUT_DIR / "selected_core_traits_gower_like_distance.csv", encoding="utf-8-sig"
        )


def main() -> None:
    meta = mod.load_trait_dictionary()
    df = mod.load_trait_data(meta)
    before = pd.read_csv(SCORING_PATH)

    reliability_df = load_reliability_from_s3(meta)
    eta_df = load_endosperm_eta_from_s5()

    scoring_df, selected_trait_cols, exhaustive_search_df, objective_trait_number_df, objective_scheme_df = (
        rebuild_scoring_table(meta, df, before, reliability_df, eta_df)
    )
    encoded_id_df = mod.build_encoded_id_matrix(meta, df, continuous_bin_count=BASE_BIN_COUNT)
    save_outputs(
        meta,
        encoded_id_df,
        scoring_df,
        selected_trait_cols,
        exhaustive_search_df,
        objective_trait_number_df,
        objective_scheme_df,
    )

    after_subset = ",".join(scoring_df.loc[scoring_df["Selected or not"], "Trait short"].astype(str).tolist())
    before_subset = ",".join(before.loc[before["Selected or not"], "Trait short"].astype(str).tolist())
    comparison = scoring_df[["Trait short", "Reliability score", "Effect size", "Subgroup discriminative score", "Final score"]].merge(
        before[["Trait short", "Reliability score", "Effect size", "Subgroup discriminative score", "Final score"]],
        on="Trait short",
        how="left",
        suffixes=("_new", "_old"),
    )
    comparison["Reliability delta"] = comparison["Reliability score_new"] - comparison["Reliability score_old"]
    comparison["Effect size delta"] = comparison["Effect size_new"] - comparison["Effect size_old"]
    comparison["D_j delta"] = (
        comparison["Subgroup discriminative score_new"] - comparison["Subgroup discriminative score_old"]
    )
    comparison["S_j delta"] = comparison["Final score_new"] - comparison["Final score_old"]
    comparison.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(f"Updated reliability file: {RELIABILITY_PATH}")
    print(f"Updated scoring table: {SCORING_PATH}")
    print(f"Updated comparison summary: {SUMMARY_PATH}")
    print(f"Selected subset before: {before_subset}")
    print(f"Selected subset after:  {after_subset}")
    print(f"Selected subset changed: {before_subset != after_subset}")


if __name__ == "__main__":
    main()
