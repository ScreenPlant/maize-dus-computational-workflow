from __future__ import annotations

import argparse
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font
from scipy.stats import studentized_range


PANEL_A_TRAITS = ["T30", "T31", "T36", "T37"]
PANEL_C_TRAITS = ["T10", "T40", "T41", "T43"]
GROUP_ORDER = [
    "Normal type",
    "Sweet type",
    "Waxy type",
    "Sweet-waxy hybrid type",
    "Popcorn type",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export real-data source tables for Fig. 3 panels A and C.")
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--effect-stats", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_trait_metadata(workbook: Path) -> pd.DataFrame:
    metadata = pd.read_excel(workbook, sheet_name="Supplementary Table 1", header=1)
    metadata = metadata.dropna(subset=["Trait No.", "Trait name"]).copy()
    metadata["Trait No."] = pd.to_numeric(metadata["Trait No."], errors="coerce").astype(int)
    metadata["Trait ID"] = metadata["Trait No."].map(lambda value: f"T{value}")
    return metadata.set_index("Trait ID")


def effect_row(effect_stats: pd.DataFrame, trait_id: str) -> pd.Series:
    selected = effect_stats[
        effect_stats["Grouping_Factor"].eq("Endosperm Type")
        & effect_stats["Trait_ID"].eq(trait_id)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one endosperm-type effect-statistics row for {trait_id}, found {len(selected)}.")
    return selected.iloc[0]


def build_raw_table(
    table3: pd.DataFrame,
    metadata: pd.DataFrame,
    trait_ids: list[str],
    panel: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for trait_id in trait_ids:
        trait_no = int(trait_id[1:])
        meta = metadata.loc[trait_id]
        frame = pd.DataFrame(
            {
                "Panel": panel,
                "Serial No.": table3["Serial No."],
                "Field Plot No.": table3["Field Plot No."],
                "Cultivar Name": table3["Cultivar Name"],
                "Endosperm Type": table3["Endosperm Type"],
                "Trait ID": trait_id,
                "Trait name": str(meta["Trait name"]).strip(),
                "Trait family": str(meta["Analysis Type"]).strip(),
                "Expert reference": pd.to_numeric(table3[f"Trait_{trait_no}_Actual"], errors="coerce"),
                "Image-derived value": pd.to_numeric(table3[f"Trait_{trait_no}_Predict"], errors="coerce"),
                "Valid image-derived value": table3[f"Trait_{trait_no}_Predict"].notna().map({True: "Yes", False: "No"}),
                "Actual source column": f"Trait_{trait_no}_Actual",
                "Image-derived source column": f"Trait_{trait_no}_Predict",
            }
        )
        frame["Endosperm Type"] = pd.Categorical(
            frame["Endosperm Type"],
            categories=GROUP_ORDER,
            ordered=True,
        )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["Trait ID", "Endosperm Type", "Serial No."]).reset_index(drop=True)


def build_panel_a_summary(raw: pd.DataFrame, effect_stats: pd.DataFrame) -> pd.DataFrame:
    valid = raw.dropna(subset=["Image-derived value"]).copy()
    summary = (
        valid.groupby(["Trait ID", "Trait name", "Endosperm Type"], observed=True)["Image-derived value"]
        .agg(
            n="count",
            Mean="mean",
            SD="std",
            Minimum="min",
            Q1=lambda values: values.quantile(0.25),
            Median="median",
            Q3=lambda values: values.quantile(0.75),
            Maximum="max",
        )
        .reset_index()
    )
    summary.insert(3, "Unit", "cm")
    stats_rows = []
    for trait_id in PANEL_A_TRAITS:
        effect = effect_row(effect_stats, trait_id)
        stats_rows.append(
            {
                "Trait ID": trait_id,
                "Eta-squared": float(effect["Eta_Squared"]),
                "ANOVA P": float(effect["ANOVA_p"]),
                "ANOVA FDR": float(effect["ANOVA_FDR"]),
                "Kruskal-Wallis P": float(effect["Kruskal_p"]),
                "Kruskal-Wallis FDR": float(effect["Kruskal_FDR"]),
            }
        )
    return summary.merge(pd.DataFrame(stats_rows), on="Trait ID", how="left")


def games_howell_for_trait(trait_data: pd.DataFrame, trait_id: str) -> pd.DataFrame:
    groups = {
        group: values["Image-derived value"].dropna().to_numpy(float)
        for group, values in trait_data.groupby("Endosperm Type", observed=True)
    }
    rows: list[dict[str, object]] = []
    group_count = len(groups)
    for group_1, group_2 in combinations(GROUP_ORDER, 2):
        values_1 = groups[group_1]
        values_2 = groups[group_2]
        n_1, n_2 = len(values_1), len(values_2)
        mean_1, mean_2 = float(values_1.mean()), float(values_2.mean())
        variance_1 = float(values_1.var(ddof=1))
        variance_2 = float(values_2.var(ddof=1))
        component_1 = variance_1 / n_1
        component_2 = variance_2 / n_2
        standard_error = float(np.sqrt(component_1 + component_2))
        mean_difference = mean_1 - mean_2
        if standard_error == 0:
            q_value = 0.0 if mean_difference == 0 else np.inf
            degrees_freedom = np.inf
            p_value = 1.0 if mean_difference == 0 else 0.0
        else:
            q_value = abs(mean_difference) / standard_error * np.sqrt(2.0)
            denominator = (
                component_1**2 / (n_1 - 1)
                + component_2**2 / (n_2 - 1)
            )
            degrees_freedom = (component_1 + component_2) ** 2 / denominator
            p_value = float(studentized_range.sf(q_value, group_count, degrees_freedom))
        rows.append(
            {
                "Trait ID": trait_id,
                "Trait name": str(trait_data["Trait name"].iloc[0]),
                "Group 1": group_1,
                "Group 2": group_2,
                "n1": n_1,
                "n2": n_2,
                "Mean 1": mean_1,
                "Mean 2": mean_2,
                "Mean difference (1-2)": mean_difference,
                "Games-Howell q": q_value,
                "Degrees of freedom": degrees_freedom,
                "Games-Howell adjusted P": p_value,
                "Significant at 0.05": "Yes" if p_value < 0.05 else "No",
            }
        )
    return pd.DataFrame(rows)


def compact_letter_display(
    trait_data: pd.DataFrame,
    pairwise: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    means = (
        trait_data.groupby("Endosperm Type", observed=True)["Image-derived value"]
        .mean()
        .reindex(GROUP_ORDER)
    )
    ranked_groups = means.sort_values(ascending=False).index.tolist()
    nonsignificant: dict[frozenset[str], bool] = {}
    for _, row in pairwise.iterrows():
        nonsignificant[frozenset([str(row["Group 1"]), str(row["Group 2"])])] = (
            float(row["Games-Howell adjusted P"]) >= alpha
        )

    valid_cliques: list[set[str]] = []
    for size in range(1, len(ranked_groups) + 1):
        for subset_tuple in combinations(ranked_groups, size):
            subset = set(subset_tuple)
            if all(
                nonsignificant.get(frozenset([group_1, group_2]), False)
                for group_1, group_2 in combinations(subset, 2)
            ):
                valid_cliques.append(subset)
    maximal_cliques = [
        clique
        for clique in valid_cliques
        if not any(clique < other for other in valid_cliques)
    ]
    rank = {group: index for index, group in enumerate(ranked_groups)}
    maximal_cliques.sort(key=lambda clique: (min(rank[group] for group in clique), -len(clique)))

    letters = {group: "" for group in ranked_groups}
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if len(maximal_cliques) > len(alphabet):
        raise ValueError("Too many compact-letter groups for the available alphabet.")
    for index, clique in enumerate(maximal_cliques):
        letter = alphabet[index]
        for group in clique:
            letters[group] += letter

    return pd.DataFrame(
        {
            "Endosperm Type": GROUP_ORDER,
            "Games-Howell letter": [letters[group] for group in GROUP_ORDER],
        }
    )


def build_panel_a_posthoc(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_frames: list[pd.DataFrame] = []
    letter_frames: list[pd.DataFrame] = []
    for trait_id in PANEL_A_TRAITS:
        trait_data = raw[raw["Trait ID"].eq(trait_id)].dropna(subset=["Image-derived value"]).copy()
        pairwise = games_howell_for_trait(trait_data, trait_id)
        letters = compact_letter_display(trait_data, pairwise)
        letters.insert(0, "Trait ID", trait_id)
        letters.insert(1, "Trait name", str(trait_data["Trait name"].iloc[0]))
        pairwise_frames.append(pairwise)
        letter_frames.append(letters)
    return pd.concat(pairwise_frames, ignore_index=True), pd.concat(letter_frames, ignore_index=True)


def build_panel_c_proportions(raw: pd.DataFrame, effect_stats: pd.DataFrame) -> pd.DataFrame:
    valid = raw.dropna(subset=["Image-derived value"]).copy()
    valid["Image-derived code"] = valid["Image-derived value"].astype(int)
    rows: list[pd.DataFrame] = []

    for trait_id in PANEL_C_TRAITS:
        trait_data = valid[valid["Trait ID"].eq(trait_id)].copy()
        trait_name = str(trait_data["Trait name"].iloc[0])
        codes = sorted(trait_data["Image-derived code"].unique().tolist())
        complete_index = pd.MultiIndex.from_tuples(
            list(product(GROUP_ORDER, codes)),
            names=["Endosperm Type", "Image-derived code"],
        )
        counts = (
            trait_data.groupby(["Endosperm Type", "Image-derived code"], observed=True)
            .size()
            .reindex(complete_index, fill_value=0)
            .rename("Count")
            .reset_index()
        )
        group_n = (
            trait_data.groupby("Endosperm Type", observed=True)
            .size()
            .reindex(GROUP_ORDER, fill_value=0)
            .rename("Group n")
        )
        counts = counts.merge(group_n, left_on="Endosperm Type", right_index=True, how="left")
        counts["Percentage (%)"] = np.where(
            counts["Group n"] > 0,
            counts["Count"] / counts["Group n"] * 100.0,
            np.nan,
        )
        effect = effect_row(effect_stats, trait_id)
        counts.insert(0, "Trait ID", trait_id)
        counts.insert(1, "Trait name", trait_name)
        counts["Eta-squared"] = float(effect["Eta_Squared"])
        counts["ANOVA P"] = float(effect["ANOVA_p"])
        counts["ANOVA FDR"] = float(effect["ANOVA_FDR"])
        counts["Kruskal-Wallis P"] = float(effect["Kruskal_p"])
        counts["Kruskal-Wallis FDR"] = float(effect["Kruskal_FDR"])
        rows.append(counts)

    return pd.concat(rows, ignore_index=True)


def build_data_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Panel A raw", "One row per cultivar and trait; Image-derived value is used for boxplots."],
            ["Panel A summary", "Five-group descriptive statistics and endosperm-type effect statistics."],
            ["Panel A posthoc", "Games-Howell pairwise comparisons; adjusted P values control all pairwise comparisons within each trait."],
            ["Panel C raw", "One row per cultivar and ordinal trait; numeric DUS codes are retained unchanged."],
            ["Panel C proportions", "Zero-completed counts and percentages for each trait, endosperm type, and observed image-derived code."],
            ["Expert reference", "Original Trait_x_Actual value retained for audit; not used to draw panels A or C."],
            ["Image-derived value", "Original Trait_x_Predict value used in the biological-relevance panels."],
            ["Percentage (%)", "Count divided by the number of valid image-derived values within the corresponding endosperm-type group."],
        ],
        columns=["Sheet/field", "Definition"],
    )


def write_workbook(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
            worksheet = writer.book[sheet_name]
            worksheet["A1"] = {
                "Data dictionary": "Fig. 3 panels A and C | Data dictionary and provenance",
                "Panel A raw": "Fig. 3A | Image-derived measured traits across endosperm types (raw cultivar-level data)",
                "Panel A summary": "Fig. 3A | Group-level boxplot statistics and biological-effect tests",
                "Panel A posthoc": "Fig. 3A | Games-Howell pairwise comparisons for endosperm types",
                "Panel C raw": "Fig. 3C | Image-derived ordinal traits across endosperm types (raw cultivar-level data)",
                "Panel C proportions": "Fig. 3C | Endosperm-type grade counts and stacked percentages",
            }[sheet_name]
            worksheet["A1"].font = Font(bold=True)
            worksheet.freeze_panes = "A3"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[2]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in list(column_cells)[:250]]
                width = min(max(max((len(value) for value in values), default=0) + 2, 11), 34)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def write_csv_with_fallback(table: pd.DataFrame, path: Path) -> Path:
    try:
        table.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_latest{path.suffix}")
        table.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    workbook = (args.workbook or root / "Supplementary Table1-9.xlsx").resolve()
    effect_path = (
        args.effect_stats
        or root / "outputs" / "supplementary_figures" / "Supplementary_Fig_S3_effect_sizes.csv"
    ).resolve()
    output_dir = (args.output_dir or root / "outputs" / "main_figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    table3 = pd.read_excel(workbook, sheet_name="Supplementary Table 3", header=1)
    metadata = load_trait_metadata(workbook)
    effect_stats = pd.read_csv(effect_path)
    if len(table3) != 150:
        raise ValueError(f"Expected 150 cultivars, found {len(table3)}.")

    panel_a_raw = build_raw_table(table3, metadata, PANEL_A_TRAITS, "A")
    panel_c_raw = build_raw_table(table3, metadata, PANEL_C_TRAITS, "C")
    panel_a_summary = build_panel_a_summary(panel_a_raw, effect_stats)
    panel_a_posthoc, panel_a_letters = build_panel_a_posthoc(panel_a_raw)
    panel_a_summary = panel_a_summary.merge(
        panel_a_letters[["Trait ID", "Endosperm Type", "Games-Howell letter"]],
        on=["Trait ID", "Endosperm Type"],
        how="left",
    )
    panel_c_proportions = build_panel_c_proportions(panel_c_raw, effect_stats)

    if len(panel_a_raw) != 600 or len(panel_c_raw) != 600:
        raise ValueError("Expected 600 raw rows for each panel (4 traits x 150 cultivars).")
    if not panel_a_raw["Image-derived value"].notna().all():
        raise ValueError("Panel A contains missing image-derived values.")
    if not panel_c_raw["Image-derived value"].notna().all():
        raise ValueError("Panel C contains missing image-derived values.")
    if panel_a_summary["Games-Howell letter"].isna().any():
        raise ValueError("One or more panel A groups are missing a Games-Howell display letter.")
    percentage_sums = panel_c_proportions.groupby(["Trait ID", "Endosperm Type"])["Percentage (%)"].sum()
    if not np.allclose(percentage_sums.to_numpy(float), 100.0):
        raise ValueError("One or more panel C stacked distributions do not sum to 100%.")

    tables = {
        "Data dictionary": build_data_dictionary(),
        "Panel A raw": panel_a_raw,
        "Panel A summary": panel_a_summary,
        "Panel A posthoc": panel_a_posthoc,
        "Panel C raw": panel_c_raw,
        "Panel C proportions": panel_c_proportions,
    }
    output_xlsx = output_dir / "Fig3_panels_A_C_real_data.xlsx"
    write_workbook(output_xlsx, tables)

    csv_paths = [
        write_csv_with_fallback(panel_a_raw, output_dir / "Fig3_panel_A_raw.csv"),
        write_csv_with_fallback(panel_a_summary, output_dir / "Fig3_panel_A_boxplot_summary.csv"),
        write_csv_with_fallback(panel_a_posthoc, output_dir / "Fig3_panel_A_games_howell_posthoc.csv"),
        write_csv_with_fallback(panel_c_raw, output_dir / "Fig3_panel_C_raw.csv"),
        write_csv_with_fallback(panel_c_proportions, output_dir / "Fig3_panel_C_stacked_proportions.csv"),
    ]

    print(f"Workbook: {output_xlsx}")
    print(f"Panel A raw rows: {len(panel_a_raw)}")
    print(f"Panel C raw rows: {len(panel_c_raw)}")
    print(f"Panel C proportion rows: {len(panel_c_proportions)}")
    print("All panel C trait-by-group percentages sum to 100%.")
    print("CSV outputs:")
    for csv_path in csv_paths:
        print(f"  {csv_path}")


if __name__ == "__main__":
    main()
