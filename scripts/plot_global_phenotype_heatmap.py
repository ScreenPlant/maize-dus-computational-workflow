from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = REPO_ROOT / "data" / "traits.csv"
METADATA_FILE = REPO_ROOT / "data" / "traits_metadata.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "global_phenotype_heatmap"
ID_COL = "Field Plot No."
GROUP_COL = "Endosperm Type"
GROUP_ORDER = [
    "Sweet type",
    "Sweet-waxy hybrid type",
    "Waxy type",
    "Popcorn type",
    "Normal type",
]
GROUP_COLORS = {
    "Sweet type": "#f4a261",
    "Sweet-waxy hybrid type": "#e76f51",
    "Waxy type": "#2a9d8f",
    "Popcorn type": "#264653",
    "Normal type": "#8d99ae",
}
FAMILY_ORDER = ["Count", "Size", "Shape", "Color"]
FAMILY_COLORS = {
    "Count": "#7f5539",
    "Size": "#577590",
    "Shape": "#43aa8b",
    "Color": "#f94144",
}


def trait_col_from_no(trait_no: int) -> str:
    return f"Trait_{trait_no}_Predict"


def short_trait_name(column: str) -> str:
    return column.replace("_Predict", "").replace("Trait_", "T")


def zscore_standardize(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=0, skipna=True)
    std = df.std(axis=0, skipna=True, ddof=0).replace(0, np.nan)
    return (df - mean) / std


def build_row_colors(groups: pd.Series) -> pd.Series:
    return groups.map(GROUP_COLORS).fillna("#bdbdbd")


def add_legends(fig: plt.Figure) -> None:
    group_handles = [
        Patch(facecolor=GROUP_COLORS[group], edgecolor="none", label=group)
        for group in GROUP_ORDER
        if group in GROUP_COLORS
    ]
    family_handles = [
        Patch(facecolor=FAMILY_COLORS[family], edgecolor="none", label=family)
        for family in FAMILY_ORDER
        if family in FAMILY_COLORS
    ]
    fig.legend(
        handles=group_handles,
        title="Endosperm Type",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    fig.legend(
        handles=family_handles,
        title="Trait Family",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.82),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_FILE)
    meta["Trait No."] = pd.to_numeric(meta["Trait No."], errors="coerce").astype(int)
    meta["Trait_Column"] = meta["Trait No."].map(trait_col_from_no)
    meta["Trait_Short"] = meta["Trait_Column"].map(short_trait_name)
    meta["Analysis Type"] = meta["Analysis Type"].astype(str)
    meta["family_rank"] = meta["Analysis Type"].map({family: i for i, family in enumerate(FAMILY_ORDER)})
    meta = meta.sort_values(["family_rank", "Trait No."]).reset_index(drop=True)
    return meta


def build_ordered_matrix(df: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    trait_cols = meta["Trait_Column"].tolist()
    work = df[[ID_COL, GROUP_COL] + trait_cols].copy()
    work[trait_cols] = work[trait_cols].apply(pd.to_numeric, errors="coerce")
    work["group_rank"] = work[GROUP_COL].map({group: i for i, group in enumerate(GROUP_ORDER)}).fillna(len(GROUP_ORDER))
    work = work.sort_values(["group_rank", GROUP_COL, ID_COL], kind="stable").reset_index(drop=True)

    scaled_df = zscore_standardize(work[trait_cols])
    scaled_df.index = work[ID_COL].astype(str)
    scaled_df.columns = meta["Trait_Short"].tolist()

    row_groups = work.set_index(work[ID_COL].astype(str))[GROUP_COL]
    col_families = meta.set_index("Trait_Short")["Analysis Type"]
    return scaled_df, row_groups, col_families


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    meta = load_metadata()
    display_df, row_groups, col_families = build_ordered_matrix(df, meta)

    row_colors = build_row_colors(row_groups)
    col_colors = col_families.map(FAMILY_COLORS).fillna("#bdbdbd")

    cmap = sns.color_palette("RdYlBu_r", as_cmap=True)
    cmap.set_bad("#bfbfbf")

    sns.set_theme(style="white", font="DejaVu Sans")
    cluster_grid = sns.clustermap(
        display_df,
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        center=0,
        vmin=-2.5,
        vmax=2.5,
        linewidths=0.12,
        linecolor="#eeeeee",
        xticklabels=True,
        yticklabels=True,
        figsize=(13, 30),
        dendrogram_ratio=(0.01, 0.01),
        colors_ratio=(0.03, 0.03),
        cbar_pos=(0.92, 0.20, 0.02, 0.55),
        cbar_kws={"label": "Predict value (column-wise Z-score)"},
    )

    cluster_grid.ax_heatmap.set_xlabel("23 DUS traits", fontsize=11)
    cluster_grid.ax_heatmap.set_ylabel("150 maize varieties", fontsize=11)
    cluster_grid.ax_heatmap.set_xticklabels(
        cluster_grid.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=9,
    )
    cluster_grid.ax_heatmap.set_yticklabels(
        cluster_grid.ax_heatmap.get_yticklabels(),
        fontsize=6,
    )
    cluster_grid.ax_heatmap.set_title(
        "Full phenotypic fingerprint heatmap of 150 maize varieties",
        fontsize=13,
        pad=20,
    )

    cluster_grid.ax_row_dendrogram.set_visible(False)
    cluster_grid.ax_col_dendrogram.set_visible(False)
    cluster_grid.ax_row_colors.set_xticks([])
    cluster_grid.ax_row_colors.set_yticks([])
    cluster_grid.ax_row_colors.set_title("Type", fontsize=9, pad=8)
    cluster_grid.ax_col_colors.set_xticks([])
    cluster_grid.ax_col_colors.set_yticks([])

    add_legends(cluster_grid.fig)
    cluster_grid.fig.savefig(OUTPUT_DIR / "full_phenotypic_fingerprint_heatmap.png", dpi=300, bbox_inches="tight")
    cluster_grid.fig.savefig(OUTPUT_DIR / "full_phenotypic_fingerprint_heatmap.pdf", bbox_inches="tight")
    plt.close(cluster_grid.fig)

    ordered_samples = pd.DataFrame(
        {
            "Field Plot No.": display_df.index,
            "Endosperm Type": row_groups.reindex(display_df.index).to_numpy(),
        }
    )
    ordered_samples.to_csv(OUTPUT_DIR / "full_phenotypic_fingerprint_sample_order.csv", index=False, encoding="utf-8-sig")

    ordered_traits = meta[["Trait No.", "Trait name", "Abbreviation", "Observation Method", "Analysis Type", "Trait_Short"]].copy()
    ordered_traits.to_csv(OUTPUT_DIR / "full_phenotypic_fingerprint_trait_order.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the global phenotypic fingerprint heatmap.")
    parser.add_argument("--input-file", default=str(INPUT_FILE), help="Input traits CSV.")
    parser.add_argument("--metadata-file", default=str(METADATA_FILE), help="Trait metadata CSV.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    INPUT_FILE = Path(args.input_file)
    METADATA_FILE = Path(args.metadata_file)
    OUTPUT_DIR = Path(args.output_dir)
    main()
