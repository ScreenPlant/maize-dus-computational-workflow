from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns

import build_variety_core_fingerprint as mod


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "variety_core_fingerprint"
SCORING_FILE = OUTPUT_DIR / "core_trait_scoring_table.csv"


def main() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    meta = mod.load_trait_dictionary()
    df = mod.load_trait_data(meta)
    scoring_df = pd.read_csv(SCORING_FILE)

    encoded_5 = mod.build_encoded_id_matrix(meta, df, continuous_bin_count=5)
    selected_trait_cols, exhaustive_search_df, trait_number_df, objective_scheme_df = mod.exhaustive_select_core_traits(
        scoring_df,
        meta,
        encoded_5,
        continuous_bin_count=5,
    )

    scoring_df["Selected or not"] = scoring_df["Trait column"].isin(selected_trait_cols)
    scoring_df.to_csv(SCORING_FILE, index=False, encoding="utf-8-sig")
    core_meta = meta[meta["Trait column"].isin(selected_trait_cols)].copy()
    core_meta["selection_order"] = core_meta["Trait column"].map({trait: i + 1 for i, trait in enumerate(selected_trait_cols)})
    core_meta = core_meta.sort_values("selection_order")
    core_meta.to_csv(OUTPUT_DIR / "selected_core_traits.csv", index=False, encoding="utf-8-sig")

    trait_number_df.to_csv(OUTPUT_DIR / "objective_trait_number_comparison.csv", index=False, encoding="utf-8-sig")
    trait_number_df.to_csv(OUTPUT_DIR / "trait_number_comparison.csv", index=False, encoding="utf-8-sig")
    trait_number_df.to_csv(OUTPUT_DIR / "forward_selection_trace.csv", index=False, encoding="utf-8-sig")
    exhaustive_search_df.to_csv(OUTPUT_DIR / "exhaustive_trait_subset_search.csv", index=False, encoding="utf-8-sig")
    exhaustive_search_df.to_csv(OUTPUT_DIR / "forward_selection_trace_full.csv", index=False, encoding="utf-8-sig")

    objective_scheme_df.to_csv(OUTPUT_DIR / "objective_scheme_comparison.csv", index=False, encoding="utf-8-sig")
    objective_scheme_df.to_csv(OUTPUT_DIR / "six_trait_scheme_comparison.csv", index=False, encoding="utf-8-sig")

    sensitivity_scheme_set = {
        str(row["scheme"]): [trait.strip() for trait in str(row["traits"]).split(",") if trait.strip()]
        for _, row in objective_scheme_df.iterrows()
    }
    sensitivity_df = mod.compare_binning_sensitivity(
        sensitivity_scheme_set,
        meta,
        df,
        mod.CONTINUOUS_BIN_OPTIONS,
    )
    sensitivity_df.to_csv(OUTPUT_DIR / "binning_sensitivity_comparison.csv", index=False, encoding="utf-8-sig")

    mod.plot_trait_number_comparison(trait_number_df, OUTPUT_DIR / "trait_number_comparison.png")
    mod.plot_scheme_comparison(objective_scheme_df, OUTPUT_DIR / "objective_scheme_comparison.png")
    mod.plot_six_trait_scheme_comparison(objective_scheme_df, OUTPUT_DIR / "six_trait_scheme_comparison.png")
    mod.plot_binning_sensitivity(sensitivity_df, OUTPUT_DIR / "binning_sensitivity_heatmap.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare core fingerprint schemes.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory containing core fingerprint outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    SCORING_FILE = OUTPUT_DIR / "core_trait_scoring_table.csv"
    main()
