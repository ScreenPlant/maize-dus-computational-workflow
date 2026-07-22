from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font


TRAIT_ORDER = ["T10", "T40", "T41", "T43"]
GROUP_ORDER = [
    "Normal type",
    "Sweet type",
    "Waxy type",
    "Sweet-waxy hybrid type",
    "Popcorn type",
]
GROUP_LABELS = {
    "Normal type": "Normal (n=127)",
    "Sweet type": "Sweet (n=8)",
    "Waxy type": "Waxy (n=6)",
    "Sweet-waxy hybrid type": "Sweet-waxy (n=3)",
    "Popcorn type": "Popcorn (n=6)",
}

GRADE_COLORS = {
    1: "#FFF7D6",
    2: "#FEE8A8",
    3: "#FDD77A",
    4: "#F8BD4B",
    5: "#E89B2E",
    6: "#CF741F",
    7: "#A8521B",
    8: "#7E3718",
    9: "#552314",
    10: "#2E160D",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export Origin-ready 100% stacked-bar data for Fig. 3C.")
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def pivot_trait(data: pd.DataFrame, trait_id: str, value_column: str, suffix: str) -> pd.DataFrame:
    sub = data[data["Trait ID"].eq(trait_id)].copy()
    if sub.empty:
        raise ValueError(f"No stacked-distribution data found for {trait_id}.")
    trait_name = str(sub["Trait name"].iloc[0])
    codes = sorted(sub["Image-derived code"].astype(int).unique().tolist())
    pivoted = sub.pivot(
        index="Endosperm Type",
        columns="Image-derived code",
        values=value_column,
    ).reindex(GROUP_ORDER)
    pivoted.columns = [f"Code {int(code)} {suffix}" for code in pivoted.columns]
    pivoted = pivoted.reset_index()
    group_n = sub.groupby("Endosperm Type")["Group n"].first().reindex(GROUP_ORDER)
    pivoted.insert(1, "Origin X label", [GROUP_LABELS[group] for group in GROUP_ORDER])
    pivoted.insert(2, "Group n", group_n.to_numpy(int))
    pivoted.insert(0, "Trait ID", trait_id)
    pivoted.insert(1, "Trait name", trait_name)
    pivoted["ANOVA FDR"] = float(sub["ANOVA FDR"].iloc[0])
    pivoted["Kruskal-Wallis FDR"] = float(sub["Kruskal-Wallis FDR"].iloc[0])
    anova_fdr = float(sub["ANOVA FDR"].iloc[0])
    pivoted["FDR label"] = "FDR = ns" if anova_fdr >= 0.05 else f"FDR = {anova_fdr:.3g}"

    if value_column == "Percentage (%)":
        percentage_columns = [f"Code {code} (%)" for code in codes]
        if not np.allclose(pivoted[percentage_columns].sum(axis=1).to_numpy(float), 100.0):
            raise ValueError(f"Percentages do not sum to 100% for {trait_id}.")
    return pivoted


def palette_table() -> pd.DataFrame:
    rows = []
    for code, hex_color in GRADE_COLORS.items():
        rgb = tuple(int(hex_color[index : index + 2], 16) for index in (1, 3, 5))
        rows.append(
            {
                "Image-derived code": code,
                "Hex color": hex_color,
                "RGB red": rgb[0],
                "RGB green": rgb[1],
                "RGB blue": rgb[2],
            }
        )
    return pd.DataFrame(rows)


def readme_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Plot type", "Origin: Plot > Column/Bar/Pie > 100% Stacked Column."],
            ["X column", "Use 'Origin X label' and set it as categorical X."],
            ["Y columns", "Select all columns named 'Code x (%)'; do not include Group n or FDR columns."],
            ["Rows", "Five endosperm types in the manuscript order; labels already include group sample size."],
            ["Percentages", "Each row sums to 100%; percentages use valid image-derived values within that group."],
            ["Codes", "Original image-derived DUS codes are retained; absent codes are not invented."],
            ["Colors", "Assign the same code-specific color across all four traits using the Color palette sheet."],
            ["FDR", "Use ANOVA FDR once per trait; T10 is non-significant and should be labelled ns."],
            ["Count sheets", "Count tables are supplied for auditing and are not required for percentage plotting."],
        ],
        columns=["Item", "Instruction"],
    )


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, table in sheets.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
                width = min(max(max((len(value) for value in values), default=0) + 2, 11), 36)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    input_path = (
        args.input
        or root / "outputs" / "main_figures" / "Fig3_panel_C_stacked_proportions.csv"
    ).resolve()
    output_dir = (args.output_dir or root / "outputs" / "main_figures" / "origin").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    sheets: dict[str, pd.DataFrame] = {
        "README": readme_table(),
        "Color palette": palette_table(),
    }
    csv_paths: list[Path] = []
    for trait_id in TRAIT_ORDER:
        percent = pivot_trait(data, trait_id, "Percentage (%)", "(%)")
        counts = pivot_trait(data, trait_id, "Count", "count")
        sheets[f"{trait_id}_percent"] = percent
        sheets[f"{trait_id}_count"] = counts
        csv_path = output_dir / f"Fig3_panel_C_{trait_id}_Origin_percent.csv"
        percent.to_csv(csv_path, index=False, encoding="utf-8-sig")
        csv_paths.append(csv_path)

    workbook_path = output_dir / "Fig3_panel_C_Origin_100pct_stacked_data.xlsx"
    write_workbook(workbook_path, sheets)

    print(f"Workbook: {workbook_path}")
    for csv_path in csv_paths:
        print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
