from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, kruskal


REPO_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = REPO_ROOT / "data" / "Supplementary_Table1-9.xlsx"
OUTPUT_DIR = REPO_ROOT / "outputs" / "category_fingerprint_analysis"
SHEET1 = "Supplementary Table 1"
SHEET2 = "Supplementary Table 2"
SHEET3 = "Supplementary Table 3"

CORE_TRAIT_IDS = [1, 2, 3, 6, 7, 9, 10, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 40, 41, 42, 43]
GROUP_COLS = ["Endosperm Type", "Breeding Type"]
TOP_TRAIT_N = 12
MAX_FINGERPRINT_TRAITS = 10


def load_rebuilt_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    columns = raw.iloc[1].tolist()
    df = raw.iloc[2:].copy().reset_index(drop=True)
    df.columns = columns
    return df


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


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def short_trait_id(trait_no: int) -> str:
    return f"T{trait_no}"


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def eta_squared_from_anova(groups: list[np.ndarray]) -> float:
    clean_groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) > 0]
    if len(clean_groups) < 2:
        return np.nan

    grand = np.concatenate(clean_groups)
    grand_mean = float(np.mean(grand))
    ss_between = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in clean_groups)
    ss_total = sum(np.sum((g - grand_mean) ** 2) for g in clean_groups)
    if ss_total == 0:
        return np.nan
    return ss_between / ss_total


def plot_group_heatmap(
    means_df: pd.DataFrame,
    trait_display_map: dict[str, str],
    output_path: Path,
    title: str,
) -> None:
    if means_df.empty:
        return

    plot_df = means_df.copy()
    plt.figure(figsize=(max(10, plot_df.shape[1] * 0.52), max(3.6, plot_df.shape[0] * 0.8)))
    sns.heatmap(
        plot_df,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Mean trait value"},
    )
    plt.xticks(
        ticks=np.arange(plot_df.shape[1]) + 0.5,
        labels=[trait_display_map.get(col, col) for col in plot_df.columns],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    plt.yticks(rotation=0, fontsize=9)
    plt.title(title)
    plt.xlabel("Traits")
    plt.ylabel("Category")
    plt.gcf().subplots_adjust(left=0.14, right=0.98, bottom=0.28, top=0.90)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_fingerprint_candidates(stats_df: pd.DataFrame) -> pd.DataFrame:
    eligible = stats_df[
        (stats_df["Missing_Rate"] <= 0.30)
        & (stats_df["ANOVA_FDR"].fillna(1) <= 0.05)
        & (stats_df["Eta_Squared"].fillna(0) >= 0.08)
    ].copy()
    if eligible.empty:
        return eligible

    analysis_priority = {"Color": 0, "Shape": 1, "Count": 2, "Size": 3}
    eligible["Type_Priority"] = eligible["Analysis Type"].map(analysis_priority).fillna(9)
    eligible = eligible.sort_values(
        ["Type_Priority", "Eta_Squared", "Kruskal_H", "Between_Group_Range"],
        ascending=[True, False, False, False],
    )

    selected_rows: list[pd.Series] = []
    used_types: set[str] = set()
    for _, row in eligible.iterrows():
        trait_type = str(row["Analysis Type"])
        if trait_type not in used_types:
            selected_rows.append(row)
            used_types.add(trait_type)
        if len(selected_rows) >= MAX_FINGERPRINT_TRAITS:
            break

    if len(selected_rows) < MAX_FINGERPRINT_TRAITS:
        chosen = {str(row["Trait_ID"]) for row in selected_rows}
        for _, row in eligible.iterrows():
            if str(row["Trait_ID"]) in chosen:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= MAX_FINGERPRINT_TRAITS:
                break

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trait_meta = load_rebuilt_sheet(FILE_PATH, SHEET1)
    trait_meta = trait_meta.dropna(subset=["Trait No.", "Trait name"]).copy()
    trait_meta["Trait No."] = pd.to_numeric(trait_meta["Trait No."], errors="coerce").astype("Int64")
    trait_meta = trait_meta[trait_meta["Trait No."].isin(CORE_TRAIT_IDS)].copy()
    trait_meta["Trait_ID"] = trait_meta["Trait No."].astype(int).map(short_trait_id)
    trait_meta["Trait_Display"] = trait_meta["Trait_ID"] + ": " + trait_meta["Trait name"].astype(str)

    meta_df = load_rebuilt_sheet(FILE_PATH, SHEET2)[["Field Plot No.", "Cultivar Name", "Breeding Type", "Endosperm Type"]].copy()
    trait_df = load_rebuilt_sheet(FILE_PATH, SHEET3).copy()

    actual_cols = [f"Trait_{i}_Actual" for i in CORE_TRAIT_IDS if f"Trait_{i}_Actual" in trait_df.columns]
    predict_cols = [f"Trait_{i}_Predict" for i in CORE_TRAIT_IDS if f"Trait_{i}_Predict" in trait_df.columns]
    safe_numeric(trait_df, actual_cols + predict_cols)

    merged = trait_df.merge(meta_df, on=["Field Plot No.", "Cultivar Name"], how="left", suffixes=("", "_meta"))
    if "Breeding Type_meta" in merged.columns:
        merged["Breeding Type"] = merged["Breeding Type"].fillna(merged["Breeding Type_meta"])
    if "Endosperm Type_meta" in merged.columns:
        merged["Endosperm Type"] = merged["Endosperm Type"].fillna(merged["Endosperm Type_meta"])

    trait_display_map = dict(zip(trait_meta["Trait_ID"], trait_meta["Trait_Display"]))
    trait_type_map = dict(zip(trait_meta["Trait_ID"], trait_meta["Analysis Type"]))
    trait_name_map = dict(zip(trait_meta["Trait_ID"], trait_meta["Trait name"]))

    summary_rows: list[dict[str, object]] = []

    for group_col in GROUP_COLS:
        records: list[dict[str, object]] = []
        group_sizes = merged[group_col].astype(str).value_counts().sort_index()
        group_sizes.rename_axis(group_col).reset_index(name="Count").to_csv(
            OUTPUT_DIR / f"{slugify(group_col)}_group_counts.csv",
            index=False,
            encoding="utf-8-sig",
        )

        for trait_no in CORE_TRAIT_IDS:
            predict_col = f"Trait_{trait_no}_Predict"
            trait_id = short_trait_id(trait_no)
            sub = merged[[group_col, predict_col]].copy()
            sub = sub.dropna(subset=[group_col])
            sub[group_col] = sub[group_col].astype(str)

            grouped_arrays = []
            group_means: dict[str, float] = {}
            for category, cat_df in sub.groupby(group_col):
                values = cat_df[predict_col].dropna().to_numpy(dtype=float)
                if len(values) > 0:
                    grouped_arrays.append(values)
                    group_means[category] = float(np.mean(values))

            anova_f = np.nan
            anova_p = np.nan
            kruskal_h = np.nan
            kruskal_p = np.nan
            if len(grouped_arrays) >= 2 and sum(len(g) for g in grouped_arrays) > len(grouped_arrays):
                try:
                    anova_f, anova_p = f_oneway(*grouped_arrays)
                except ValueError:
                    pass
                try:
                    kruskal_h, kruskal_p = kruskal(*grouped_arrays)
                except ValueError:
                    pass

            ordered_means = sorted(group_means.items(), key=lambda x: x[1])
            lowest_group, lowest_mean = (ordered_means[0] if ordered_means else (np.nan, np.nan))
            highest_group, highest_mean = (ordered_means[-1] if ordered_means else (np.nan, np.nan))

            records.append(
                {
                    "Trait_No": trait_no,
                    "Trait_ID": trait_id,
                    "Trait_Name": trait_name_map.get(trait_id, trait_id),
                    "Trait_Display": trait_display_map.get(trait_id, trait_id),
                    "Analysis Type": trait_type_map.get(trait_id, np.nan),
                    "Missing_Rate": float(merged[predict_col].isna().mean()),
                    "Group_Count": len(group_means),
                    "ANOVA_F": anova_f,
                    "ANOVA_p": anova_p,
                    "Kruskal_H": kruskal_h,
                    "Kruskal_p": kruskal_p,
                    "Eta_Squared": eta_squared_from_anova(grouped_arrays),
                    "Between_Group_Range": (highest_mean - lowest_mean) if ordered_means else np.nan,
                    "Lowest_Group": lowest_group,
                    "Lowest_Group_Mean": lowest_mean,
                    "Highest_Group": highest_group,
                    "Highest_Group_Mean": highest_mean,
                }
            )

        stats_df = pd.DataFrame(records)
        stats_df["ANOVA_FDR"] = bh_fdr(stats_df["ANOVA_p"])
        stats_df["Kruskal_FDR"] = bh_fdr(stats_df["Kruskal_p"])
        stats_df = stats_df.sort_values(
            ["ANOVA_FDR", "Eta_Squared", "Kruskal_H", "Between_Group_Range"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        stats_df.to_csv(
            OUTPUT_DIR / f"{slugify(group_col)}_trait_difference_stats.csv",
            index=False,
            encoding="utf-8-sig",
        )

        top_traits = stats_df["Trait_ID"].head(TOP_TRAIT_N).tolist()
        top_predict_cols = [f"Trait_{trait_id[1:]}_Predict" for trait_id in top_traits]
        top_means = (
            merged[[group_col] + top_predict_cols]
            .groupby(group_col)
            .mean(numeric_only=True)
            .rename(columns={f"Trait_{trait_id[1:]}_Predict": trait_id for trait_id in top_traits})
        )
        top_means = top_means[top_traits]
        top_means.to_csv(
            OUTPUT_DIR / f"{slugify(group_col)}_top_trait_means.csv",
            encoding="utf-8-sig",
        )
        plot_group_heatmap(
            top_means,
            trait_display_map,
            OUTPUT_DIR / f"{slugify(group_col)}_top_trait_heatmap.png",
            f"Top {TOP_TRAIT_N} trait differences by {group_col}",
        )

        summary_rows.append(
            {
                "Grouping": group_col,
                "Sample_Count": len(merged),
                "Category_Count": int(merged[group_col].nunique(dropna=True)),
                "Significant_Traits_FDR_0.05": int((stats_df["ANOVA_FDR"] <= 0.05).sum()),
                "Low_Missing_Significant_Traits": int(
                    ((stats_df["ANOVA_FDR"] <= 0.05) & (stats_df["Missing_Rate"] <= 0.30)).sum()
                ),
                "Top_1_Trait": stats_df.iloc[0]["Trait_ID"],
                "Top_1_Trait_Name": stats_df.iloc[0]["Trait_Name"],
            }
        )

        if group_col == "Endosperm Type":
            fingerprint_df = build_fingerprint_candidates(stats_df)
            fingerprint_df.to_csv(
                OUTPUT_DIR / "fingerprint_candidate_traits.csv",
                index=False,
                encoding="utf-8-sig",
            )

            if not fingerprint_df.empty:
                candidate_ids = fingerprint_df["Trait_ID"].tolist()
                candidate_cols = [f"Trait_{trait_id[1:]}_Predict" for trait_id in candidate_ids]
                zscore_df = merged[["Cultivar Name", "Endosperm Type"] + candidate_cols].copy()
                for col in candidate_cols:
                    std = zscore_df[col].std(ddof=0)
                    if pd.notna(std) and std != 0:
                        zscore_df[col] = (zscore_df[col] - zscore_df[col].mean()) / std
                zscore_df = zscore_df.rename(
                    columns={f"Trait_{trait_id[1:]}_Predict": trait_id for trait_id in candidate_ids}
                )
                zscore_df.to_csv(
                    OUTPUT_DIR / "fingerprint_candidate_trait_zscores.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "analysis_overview.csv",
        index=False,
        encoding="utf-8-sig",
    )

    endo_stats = pd.read_csv(OUTPUT_DIR / "endosperm_type_trait_difference_stats.csv")
    breed_stats = pd.read_csv(OUTPUT_DIR / "breeding_type_trait_difference_stats.csv")
    fingerprint_df = pd.read_csv(OUTPUT_DIR / "fingerprint_candidate_traits.csv")

    endo_sig = endo_stats[endo_stats["ANOVA_FDR"] <= 0.05].copy()
    breed_sig = breed_stats[breed_stats["ANOVA_FDR"] <= 0.05].copy()
    type_summary = (
        endo_sig.groupby("Analysis Type", as_index=False)
        .agg(
            Significant_Trait_Count=("Trait_ID", "count"),
            Mean_Eta_Squared=("Eta_Squared", "mean"),
        )
        .sort_values(["Significant_Trait_Count", "Mean_Eta_Squared"], ascending=[False, False])
    )
    type_summary.to_csv(
        OUTPUT_DIR / "endosperm_type_significant_traits_by_analysis_type.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report_lines = [
        "# 更新后数据的类别差异分析与表型指纹构建建议",
        "",
        "## 数据概况",
        f"- 样本数：{len(merged)}",
        f"- 性状数：{len(CORE_TRAIT_IDS)}",
        f"- 胚乳类型类别数：{merged['Endosperm Type'].nunique(dropna=True)}",
        f"- 育种类型类别数：{merged['Breeding Type'].nunique(dropna=True)}",
        "",
        "## 按品种类别看性状差异",
        f"- 以 `Endosperm Type` 分组时，FDR <= 0.05 的显著差异性状有 {len(endo_sig)} 个。",
        f"- 以 `Breeding Type` 分组时，FDR <= 0.05 的显著差异性状有 {len(breed_sig)} 个。",
        "",
        "### 胚乳类型下最有区分力的前 8 个性状",
    ]

    for _, row in endo_stats.head(8).iterrows():
        report_lines.append(
            f"- {row['Trait_ID']} {row['Trait_Name']} | 类型={row['Analysis Type']} | "
            f"FDR={row['ANOVA_FDR']:.4g} | Eta²={row['Eta_Squared']:.3f} | "
            f"最高={row['Highest_Group']} ({row['Highest_Group_Mean']:.3f}) | "
            f"最低={row['Lowest_Group']} ({row['Lowest_Group_Mean']:.3f})"
        )

    report_lines.extend(
        [
            "",
            "### 从性状类型看",
        ]
    )
    for _, row in type_summary.iterrows():
        report_lines.append(
            f"- {row['Analysis Type']}：显著性状 {int(row['Significant_Trait_Count'])} 个，平均 Eta²={row['Mean_Eta_Squared']:.3f}"
        )

    report_lines.extend(
        [
            "",
            "## 作物品种表型指纹如何构建",
            "- 第一步，先保留缺失率低、类别间差异显著、效应量较高的性状。",
            "- 第二步，在 Color、Shape、Count、Size 四类性状里尽量都保留代表项，避免指纹只由尺寸性状主导。",
            "- 第三步，把入选性状标准化为 z-score，形成“品种 x 指纹性状”的数值矩阵。",
            "- 第四步，用热图、雷达图或 PCA/聚类结果展示，既能表达单品种指纹，也能表达类别级平均指纹。",
            "- 第五步，如果后续要做品种鉴别，可把这组指纹性状直接作为分类模型输入。",
            "",
            "### 本次推荐的指纹候选性状",
        ]
    )
    for _, row in fingerprint_df.iterrows():
        report_lines.append(
            f"- {row['Trait_ID']} {row['Trait_Name']} | 类型={row['Analysis Type']} | FDR={row['ANOVA_FDR']:.4g} | Eta²={row['Eta_Squared']:.3f}"
        )

    report_lines.extend(
        [
            "",
            "## 主要输出文件",
            "- `endosperm_type_trait_difference_stats.csv`：胚乳类型分组下每个性状的统计结果。",
            "- `breeding_type_trait_difference_stats.csv`：育种类型分组下每个性状的统计结果。",
            "- `fingerprint_candidate_traits.csv`：推荐用于构建表型指纹的候选性状。",
            "- `fingerprint_candidate_trait_zscores.csv`：候选性状标准化后的品种指纹矩阵。",
            "- `endosperm_type_top_trait_heatmap.png`：胚乳类型前 12 个差异性状热图。",
            "- `breeding_type_top_trait_heatmap.png`：育种类型前 12 个差异性状热图。",
        ]
    )

    (OUTPUT_DIR / "analysis_report_zh.md").write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze category-level fingerprint candidates.")
    parser.add_argument("--file-path", default=str(FILE_PATH), help="Supplementary workbook path.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    FILE_PATH = Path(args.file_path)
    OUTPUT_DIR = Path(args.output_dir)
    main()
