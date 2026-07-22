from __future__ import annotations

from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font

import build_variety_core_fingerprint as mod


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "supplementary_tables"
VARIETY_OUTPUT_DIR = ROOT / "outputs" / "variety_core_fingerprint"

SCORING_PATH = VARIETY_OUTPUT_DIR / "core_trait_scoring_table.csv"

FULL_CSV = OUTPUT_DIR / "Supplementary_Table_S6_exhaustive_subset_search_results.csv"
FULL_XLSX = OUTPUT_DIR / "Supplementary_Table_S6_exhaustive_subset_search_results.xlsx"
FULL_XLSX_FALLBACK = OUTPUT_DIR / "Supplementary_Table_S6_exhaustive_subset_search_results_latest.xlsx"
FULL_MD = OUTPUT_DIR / "Supplementary_Table_S6_exhaustive_subset_search_results.md"
SUMMARY_CSV = OUTPUT_DIR / "Supplementary_Table_S6_best_by_k_summary.csv"

MAX_K = 6
BIN_COUNT = 5
FAMILY_ORDER = ["Color", "Size", "Count", "Shape"]


def trait_no_from_short(short_code: str) -> int:
    return int(str(short_code).replace("T", ""))


def format_trait_subset(short_codes: list[str]) -> str:
    ordered = sorted(short_codes, key=trait_no_from_short)
    return ";".join(ordered)


def format_family_composition(selected_cols: list[str], meta: pd.DataFrame) -> tuple[str, int, float]:
    family_by_col = meta.set_index("Trait column")["Analysis Type"].to_dict()
    counts = {family: 0 for family in FAMILY_ORDER}
    for col in selected_cols:
        family = str(family_by_col[col])
        if family not in counts:
            counts[family] = 0
        counts[family] += 1

    represented = sum(1 for family in FAMILY_ORDER if counts.get(family, 0) > 0)
    trait_number = len(selected_cols)
    if trait_number == 0:
        imbalance = 0.0
    else:
        expected = 1.0 / len(FAMILY_ORDER)
        imbalance = float(
            sum(abs((counts.get(family, 0) / trait_number) - expected) for family in FAMILY_ORDER)
        )
    composition = "; ".join(f"{family}:{counts.get(family, 0)}" for family in FAMILY_ORDER)
    return composition, represented, imbalance


def build_candidate_subset_table(
    meta: pd.DataFrame,
    scoring_df: pd.DataFrame,
    encoded: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    eligible = mod.eligible_trait_pool(scoring_df)
    eligible_cols = eligible["Trait column"].astype(str).tolist()
    col_to_short = meta.set_index("Trait column")["Trait short"].astype(str).to_dict()
    final_score = eligible.set_index("Trait column")["Final score"].astype(float).to_dict()
    reliability = eligible.set_index("Trait column")["Reliability score"].astype(float).to_dict()
    precomputed = mod.precompute_subset_metrics(
        encoded,
        eligible_cols,
        meta,
        continuous_bin_count=BIN_COUNT,
    )

    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    subset_global_id = 1
    final_core_subset = ""

    for k in range(1, MAX_K + 1):
        level_rows: list[dict[str, object]] = []
        for lex_rank, combo in enumerate(combinations(eligible_cols, k), start=1):
            selected_cols = list(combo)
            selected_shorts = [col_to_short[col] for col in selected_cols]
            trait_subset = format_trait_subset(selected_shorts)
            composition, coverage, imbalance = format_family_composition(selected_cols, meta)
            summary = mod.summarize_trait_subset_fast(selected_cols, precomputed)
            sufficient = bool(summary["unique_pattern_proportion"] >= 1.0 and summary["zero_distance_pairs"] == 0)
            row = {
                "Subset size k": k,
                "Subset ID": f"subset_{subset_global_id:05d}",
                "Trait subset": trait_subset,
                "Trait-family composition": composition,
                "Family coverage": int(coverage),
                "Family imbalance FI": float(imbalance),
                "Unique pattern proportion": float(summary["unique_pattern_proportion"]),
                "Zero-distance pairs": int(summary["zero_distance_pairs"]),
                "Zero-pair rate": float(summary["zero_pair_rate"]),
                "Mean pairwise distance": float(summary["mean_pairwise_distance"]),
                "Mean reliability": float(np.mean([reliability[col] for col in selected_cols])),
                "Mean composite score": float(np.mean([final_score[col] for col in selected_cols])),
                "Complete uniqueness?": "Yes" if sufficient else "No",
                "Lexicographic rank within k": lex_rank,
                "Selected as best subset for k?": "No",
                "Selected final core fingerprint?": "No",
                "Configuration type": "candidate subset",
            }
            subset_global_id += 1
            level_rows.append(row)

        level_df = pd.DataFrame(level_rows)
        objective_df = level_df.copy()
        objective_df["sufficient_for_unique_id"] = objective_df["Complete uniqueness?"].eq("Yes")
        objective_df["trait_number"] = objective_df["Subset size k"]
        objective_df["family_coverage_count"] = objective_df["Family coverage"]
        objective_df["family_imbalance"] = objective_df["Family imbalance FI"]
        objective_df["unique_pattern_proportion"] = objective_df["Unique pattern proportion"]
        objective_df["zero_distance_pairs"] = objective_df["Zero-distance pairs"]
        objective_df["mean_pairwise_distance"] = objective_df["Mean pairwise distance"]
        objective_df["mean_reliability"] = objective_df["Mean reliability"]
        objective_df["mean_final_score"] = objective_df["Mean composite score"]
        objective_df["traits"] = objective_df["Trait subset"].str.replace(";", ", ", regex=False)

        ranked_level = mod._sort_objective_rows(
            objective_df[
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
                ]
            ].join(level_df[["Subset ID", "Trait subset"]])
        )
        best_subset_id = str(ranked_level.iloc[0]["Subset ID"])
        level_df.loc[level_df["Subset ID"] == best_subset_id, "Selected as best subset for k?"] = "Yes"

        if final_core_subset == "":
            sufficient_ranked = ranked_level[ranked_level["sufficient_for_unique_id"]]
            if not sufficient_ranked.empty:
                final_core_subset = str(sufficient_ranked.iloc[0]["Trait subset"])
                final_subset_id = str(sufficient_ranked.iloc[0]["Subset ID"])
                level_df.loc[level_df["Subset ID"] == final_subset_id, "Selected final core fingerprint?"] = "Yes"

        best_row = level_df.loc[level_df["Subset ID"] == best_subset_id].iloc[0].to_dict()
        summary_rows.append(best_row)
        all_rows.extend(level_df.to_dict("records"))

    full_df = pd.DataFrame(all_rows)
    if final_core_subset:
        full_df.loc[full_df["Trait subset"] == final_core_subset, "Selected final core fingerprint?"] = "Yes"

    eligible_cols_full = eligible["Trait column"].astype(str).tolist()
    full_shorts = [col_to_short[col] for col in eligible_cols_full]
    composition, coverage, imbalance = format_family_composition(eligible_cols_full, meta)
    full_summary = mod.summarize_trait_subset_fast(eligible_cols_full, precomputed)
    full_reference_row = {
        "Subset size k": len(eligible_cols_full),
        "Subset ID": "reference_eligible_19",
        "Trait subset": format_trait_subset(full_shorts),
        "Trait-family composition": composition,
        "Family coverage": int(coverage),
        "Family imbalance FI": float(imbalance),
        "Unique pattern proportion": float(full_summary["unique_pattern_proportion"]),
        "Zero-distance pairs": int(full_summary["zero_distance_pairs"]),
        "Zero-pair rate": float(full_summary["zero_pair_rate"]),
        "Mean pairwise distance": float(full_summary["mean_pairwise_distance"]),
        "Mean reliability": float(np.mean([reliability[col] for col in eligible_cols_full])),
        "Mean composite score": float(np.mean([final_score[col] for col in eligible_cols_full])),
        "Complete uniqueness?": "Yes" if full_summary["unique_pattern_proportion"] >= 1.0 and full_summary["zero_distance_pairs"] == 0 else "No",
        "Lexicographic rank within k": "",
        "Selected as best subset for k?": "No",
        "Selected final core fingerprint?": "No",
        "Configuration type": "upper-bound reference",
    }
    full_df = pd.concat([full_df, pd.DataFrame([full_reference_row])], ignore_index=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = pd.concat([summary_df, pd.DataFrame([full_reference_row])], ignore_index=True)
    return full_df, summary_df, final_core_subset


def finalize_tables(full_df: pd.DataFrame, summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = [
        "Family imbalance FI",
        "Unique pattern proportion",
        "Zero-pair rate",
        "Mean pairwise distance",
        "Mean reliability",
        "Mean composite score",
    ]
    for df in (full_df, summary_df):
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    full_df = full_df.sort_values(
        by=["Subset size k", "Configuration type", "Lexicographic rank within k", "Trait subset"],
        key=lambda s: pd.to_numeric(s, errors="coerce") if s.name in {"Subset size k", "Lexicographic rank within k"} else s,
        ignore_index=True,
    )
    summary_df = summary_df.sort_values(
        by=["Subset size k", "Configuration type"],
        ignore_index=True,
    )
    return full_df, summary_df


def write_excel(full_df: pd.DataFrame, summary_df: pd.DataFrame, xlsx_path: Path) -> None:
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        full_df.to_excel(writer, index=False, sheet_name="Supplementary Table S6", startrow=1)
        summary_df.to_excel(writer, index=False, sheet_name="Best-by-k summary", startrow=1)

        ws = writer.book["Supplementary Table S6"]
        ws["A1"] = "Supplementary Table S6 | Exhaustive subset-search results and performance comparison"
        ws["A1"].font = Font(bold=True)
        ws.freeze_panes = "A3"
        for cell in ws[2]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        widths = {
            "A": 12,
            "B": 16,
            "C": 36,
            "D": 28,
            "E": 14,
            "F": 16,
            "G": 18,
            "H": 18,
            "I": 14,
            "J": 20,
            "K": 16,
            "L": 18,
            "M": 20,
            "N": 22,
            "O": 24,
            "P": 26,
            "Q": 20,
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width

        sum_ws = writer.book["Best-by-k summary"]
        sum_ws["A1"] = "Best-by-k summary | Top-performing subset at each subset-size level plus the 19-trait reference"
        sum_ws["A1"].font = Font(bold=True)
        sum_ws.freeze_panes = "A3"
        for cell in sum_ws[2]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for col, width in widths.items():
            sum_ws.column_dimensions[col].width = width


def write_markdown(full_df: pd.DataFrame, summary_df: pd.DataFrame, final_core_subset: str) -> None:
    candidate_rows = full_df[full_df["Configuration type"] == "candidate subset"]
    sufficient_count = int((candidate_rows["Complete uniqueness?"] == "Yes").sum())
    expected_count = sum(comb(19, k) for k in range(1, MAX_K + 1))
    lines = [
        "# Supplementary Table S6",
        "",
        "Exhaustive subset-search results and performance comparison",
        "",
        f"- Eligible traits searched: 19",
        f"- Exhaustive subset-size range: k = 1-{MAX_K}",
        f"- Candidate subsets evaluated: {len(candidate_rows):,}",
        f"- Expected combination count for k = 1-{MAX_K}: {expected_count:,}",
        f"- Complete-uniqueness subsets within k = 1-{MAX_K}: {sufficient_count:,}",
        f"- Final core fingerprint selected by exhaustive search: {final_core_subset}",
        "- `Selected as best subset for k?` follows the same objective priority order used in the core-selection pipeline.",
        "- `Selected final core fingerprint?` marks the first subset-size level achieving complete uniqueness under that same objective priority order.",
        "- The appended 19-trait row is the upper-bound internal reference rather than a member of the k = 1-6 candidate search space.",
        "",
        "## Best-by-k Summary",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Full Table Preview",
        "",
        full_df.head(20).to_markdown(index=False),
    ]
    FULL_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = mod.load_trait_dictionary()
    df = mod.load_trait_data(meta)
    scoring_df = pd.read_csv(SCORING_PATH)
    encoded = mod.build_encoded_id_matrix(meta, df, continuous_bin_count=BIN_COUNT)

    full_df, summary_df, final_core_subset = build_candidate_subset_table(meta, scoring_df, encoded)
    full_df, summary_df = finalize_tables(full_df, summary_df)

    full_df.to_csv(FULL_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    try:
        write_excel(full_df, summary_df, FULL_XLSX)
        written_xlsx = FULL_XLSX
    except PermissionError:
        write_excel(full_df, summary_df, FULL_XLSX_FALLBACK)
        written_xlsx = FULL_XLSX_FALLBACK

    write_markdown(full_df, summary_df, final_core_subset)

    candidate_count = int((full_df["Configuration type"] == "candidate subset").sum())
    sufficient_count = int(
        (
            (full_df["Configuration type"] == "candidate subset")
            & (full_df["Complete uniqueness?"] == "Yes")
        ).sum()
    )
    print(f"Wrote S6 full table: {FULL_CSV}")
    print(f"Wrote S6 workbook: {written_xlsx}")
    print(f"Wrote S6 summary: {SUMMARY_CSV}")
    print(f"Candidate subsets evaluated: {candidate_count}")
    print(f"Complete-uniqueness subsets in k=1-{MAX_K}: {sufficient_count}")
    print(f"Final core fingerprint: {final_core_subset}")


if __name__ == "__main__":
    main()
