from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = REPO_ROOT / "data" / "Supplementary_Table1-9.xlsx"
OUTPUT_DIR = REPO_ROOT / "outputs" / "mixed_fingerprint_analysis"
DEFAULT_CATEGORY_TRAITS_FILE = REPO_ROOT / "outputs" / "category_fingerprint_analysis" / "fingerprint_candidate_traits.csv"
DEFAULT_MS_STATS_FILE = REPO_ROOT / "outputs" / "ms_trait_analysis" / "predict_endosperm_type_difference_stats.csv"
SHEET1 = "Supplementary Table 1"
SHEET2 = "Supplementary Table 2"
SHEET3 = "Supplementary Table 3"


def load_rebuilt_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    columns = raw.iloc[1].tolist()
    df = raw.iloc[2:].copy().reset_index(drop=True)
    df.columns = columns
    return df


def short_trait_id(trait_no: int) -> str:
    return f"T{trait_no}"


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / std


def parse_class_range(class_value: object) -> tuple[float, float] | None:
    if pd.isna(class_value):
        return None
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(class_value))]
    if len(numbers) < 2:
        return None
    return min(numbers), max(numbers)


def minmax_vg(series: pd.Series, min_score: float, max_score: float) -> pd.Series:
    denominator = max_score - min_score
    if denominator == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - min_score) / denominator


def choose_traits(trait_meta: pd.DataFrame) -> pd.DataFrame:
    category_stats = pd.read_csv(CATEGORY_TRAITS_FILE)
    ms_stats = pd.read_csv(MS_STATS_FILE)

    vg_selected = category_stats["Trait_ID"].tolist()
    ms_selected = (
        ms_stats[ms_stats["ANOVA_FDR"] <= 0.10]
        .sort_values(["ANOVA_FDR", "Eta_Squared"], ascending=[True, False])
        ["Trait_ID"]
        .head(4)
        .tolist()
    )

    chosen_ids = []
    for trait_id in vg_selected + ms_selected:
        if trait_id not in chosen_ids:
            chosen_ids.append(trait_id)

    selected = trait_meta[trait_meta["Trait_ID"].isin(chosen_ids)].copy()
    selected["Selection_Order"] = selected["Trait_ID"].map({trait_id: i for i, trait_id in enumerate(chosen_ids)})
    selected = selected.sort_values("Selection_Order").reset_index(drop=True)
    return selected


def build_matrix(
    merged: pd.DataFrame,
    selected_traits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = merged[["Cultivar Name", "Endosperm Type", "Breeding Type"]].copy()
    method_rows: list[dict[str, str]] = []

    for _, row in selected_traits.iterrows():
        trait_no = int(row["Trait No."])
        trait_id = row["Trait_ID"]
        predict_col = f"Trait_{trait_no}_Predict"
        obs_method = str(row["Observation Method"])

        values = pd.to_numeric(merged[predict_col], errors="coerce")
        if obs_method == "MS":
            scaled = zscore(values)
            norm_method = "z-score"
            class_range_text = ""
        else:
            class_range = parse_class_range(row.get("class"))
            if class_range is None:
                observed_min = float(values.min(skipna=True))
                observed_max = float(values.max(skipna=True))
                class_range = (observed_min, observed_max)
                norm_method = f"ordinal min-max (observed {observed_min:g}-{observed_max:g} -> 0-1)"
            else:
                norm_method = f"ordinal min-max (class {class_range[0]:g}-{class_range[1]:g} -> 0-1)"
            scaled = minmax_vg(values, class_range[0], class_range[1])
            class_range_text = f"{class_range[0]:g}-{class_range[1]:g}"

        matrix[trait_id] = scaled
        method_rows.append(
            {
                "Trait_ID": trait_id,
                "Trait_Name": str(row["Trait name"]),
                "Observation_Method": obs_method,
                "Analysis_Type": str(row["Analysis Type"]),
                "Class_Range": class_range_text,
                "Normalization": norm_method,
            }
        )

    return matrix, pd.DataFrame(method_rows)


def plot_heatmap(
    plot_df: pd.DataFrame,
    title: str,
    output_path: Path,
    center: float | None = None,
) -> None:
    plt.figure(figsize=(max(12, plot_df.shape[1] * 0.65), max(6, plot_df.shape[0] * 0.18)))
    sns.heatmap(
        plot_df,
        cmap="YlGnBu",
        linewidths=0.0,
        cbar_kws={"label": "Normalized trait value"},
        center=center,
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=6)
    plt.title(title)
    plt.xlabel("Fingerprint traits")
    plt.ylabel("Cultivars")
    plt.gcf().subplots_adjust(left=0.18, right=0.96, bottom=0.22, top=0.92)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_category_heatmap(
    category_means: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(max(10, category_means.shape[1] * 0.8), max(3.5, category_means.shape[0] * 0.8)))
    sns.heatmap(
        category_means,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Mean normalized value"},
        center=0,
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=9)
    plt.title("Category-Level Mixed Phenotypic Fingerprints")
    plt.xlabel("Fingerprint traits")
    plt.ylabel("Endosperm Type")
    plt.gcf().subplots_adjust(left=0.18, right=0.96, bottom=0.24, top=0.90)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trait_meta = load_rebuilt_sheet(FILE_PATH, SHEET1)
    trait_meta = trait_meta.dropna(subset=["Trait No.", "Trait name"]).copy()
    trait_meta["Trait No."] = pd.to_numeric(trait_meta["Trait No."], errors="coerce")
    trait_meta = trait_meta[trait_meta["Trait No."].notna()].copy()
    trait_meta["Trait No."] = trait_meta["Trait No."].astype(int)
    trait_meta["Trait_ID"] = trait_meta["Trait No."].map(short_trait_id)

    meta_df = load_rebuilt_sheet(FILE_PATH, SHEET2)[["Field Plot No.", "Cultivar Name", "Breeding Type", "Endosperm Type"]].copy()
    trait_df = load_rebuilt_sheet(FILE_PATH, SHEET3).copy()

    selected_traits = choose_traits(trait_meta)
    predict_cols = [f"Trait_{int(t)}_Predict" for t in selected_traits["Trait No."]]
    for col in predict_cols:
        trait_df[col] = pd.to_numeric(trait_df[col], errors="coerce")

    merged = trait_df.merge(meta_df, on=["Field Plot No.", "Cultivar Name"], how="left")
    matrix, methods_df = build_matrix(merged, selected_traits)
    methods_df.to_csv(OUTPUT_DIR / "fingerprint_trait_normalization_methods.csv", index=False, encoding="utf-8-sig")

    matrix.to_csv(OUTPUT_DIR / "mixed_trait_fingerprint_matrix.csv", index=False, encoding="utf-8-sig")

    trait_ids = methods_df["Trait_ID"].tolist()
    plot_source = matrix[["Cultivar Name", "Endosperm Type"] + trait_ids].copy()
    plot_source = plot_source.sort_values(["Endosperm Type", "Cultivar Name"]).reset_index(drop=True)
    plot_df = plot_source.set_index("Cultivar Name")[trait_ids]
    plot_heatmap(
        plot_df,
        "Mixed MS + VG Phenotypic Fingerprint Matrix",
        OUTPUT_DIR / "mixed_trait_fingerprint_heatmap_all.png",
        center=0,
    )

    category_means = plot_source.groupby("Endosperm Type")[trait_ids].mean()
    category_means.to_csv(OUTPUT_DIR / "mixed_trait_fingerprint_category_means.csv", encoding="utf-8-sig")
    plot_category_heatmap(category_means, OUTPUT_DIR / "mixed_trait_fingerprint_heatmap_by_category.png")

    representative = (
        plot_source.groupby("Endosperm Type", group_keys=False)
        .head(6)
        .copy()
    )
    rep_df = representative.set_index("Cultivar Name")[trait_ids]
    plot_heatmap(
        rep_df,
        "Representative Cultivars: Mixed Phenotypic Fingerprints",
        OUTPUT_DIR / "mixed_trait_fingerprint_heatmap_representatives.png",
        center=0,
    )

    report_lines = [
        "# 混合型表型指纹构建",
        "",
        "## 归一化策略",
        "- MS 性状：使用 z-score 标准化，保留连续型差异尺度。",
        "- VG 性状：依据 Table 1 的 class 列，按每个性状自身的 DUS 分级范围做有序 min-max 归一化。",
        "- 构建混合指纹矩阵时，统一使用 Predict 列作为自动表型值来源。",
        "",
        "## 指纹性状组成",
    ]

    for _, row in methods_df.iterrows():
        report_lines.append(
            f"- {row['Trait_ID']} {row['Trait_Name']} | {row['Observation_Method']} | "
            f"{row['Analysis_Type']} | class={row['Class_Range'] or 'NA'} | {row['Normalization']}"
        )

    report_lines.extend(
        [
            "",
            "## 输出文件",
            "- mixed_trait_fingerprint_matrix.csv：品种层面的混合指纹矩阵。",
            "- mixed_trait_fingerprint_category_means.csv：类别平均指纹矩阵。",
            "- mixed_trait_fingerprint_heatmap_all.png：全部品种热图。",
            "- mixed_trait_fingerprint_heatmap_by_category.png：类别平均热图。",
            "- mixed_trait_fingerprint_heatmap_representatives.png：代表品种热图。",
        ]
    )
    (OUTPUT_DIR / "mixed_fingerprint_report_zh.md").write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed MS + VG phenotypic fingerprints.")
    parser.add_argument("--file-path", default=str(FILE_PATH), help="Supplementary workbook path.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--category-traits-file", default=str(DEFAULT_CATEGORY_TRAITS_FILE), help="Category fingerprint candidate trait file.")
    parser.add_argument("--ms-stats-file", default=str(DEFAULT_MS_STATS_FILE), help="MS trait statistics file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    FILE_PATH = Path(args.file_path)
    OUTPUT_DIR = Path(args.output_dir)
    CATEGORY_TRAITS_FILE = Path(args.category_traits_file)
    MS_STATS_FILE = Path(args.ms_stats_file)
    main()
