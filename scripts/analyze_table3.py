from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import chi2, f_oneway
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = REPO_ROOT / "data" / "Supplementary_Table1-9.xlsx"
SHEET_NAME = "Supplementary Table 3"
OUTPUT_DIR = REPO_ROOT / "outputs" / "table3_analysis"

META_COLS = ["Serial No.", "Field Plot No.", "Cultivar Name"]
CORE_TRAIT_IDS = [1, 2, 3, 6, 7, 9, 10, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 40, 41, 42, 43]
CLUSTER_K = 5
MAIN_TEXT_MAX_VARIETIES = 20
MISSING_THRESHOLD = 0.30
MAIN_TEXT_TRAIT_MISSING_THRESHOLD = 0.50
MAIN_TEXT_TOP_N_TRAITS = 8
NORMAL_TYPE_TARGET = 10


def load_table3(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    rebuilt_columns = raw.iloc[1].tolist()
    df = raw.iloc[2:].copy().reset_index(drop=True)
    df.columns = rebuilt_columns
    return df


def load_table2_metadata(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Supplementary Table 2", header=None)
    rebuilt_columns = raw.iloc[1].tolist()
    df = raw.iloc[2:].copy().reset_index(drop=True)
    df.columns = rebuilt_columns
    return df[["Field Plot No.", "Cultivar Name", "Breeding Type", "Endosperm Type"]].copy()


def load_table1_trait_metadata(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Supplementary Table 1", header=None)
    rebuilt_columns = raw.iloc[1].tolist()
    df = raw.iloc[2:].copy().reset_index(drop=True)
    df.columns = rebuilt_columns
    df = df.dropna(subset=["Trait No.", "Trait name"]).copy()
    df["Trait No."] = pd.to_numeric(df["Trait No."], errors="coerce").astype("Int64")
    return df[["Trait No.", "Trait name", "Abbreviation"]].copy()


def get_trait_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    actual_cols = [f"Trait_{trait_id}_Actual" for trait_id in CORE_TRAIT_IDS if f"Trait_{trait_id}_Actual" in df.columns]
    predict_cols = [f"Trait_{trait_id}_Predict" for trait_id in CORE_TRAIT_IDS if f"Trait_{trait_id}_Predict" in df.columns]
    return actual_cols, predict_cols


def to_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def short_trait_names(columns: list[str]) -> list[str]:
    return [f"T{column.split('_')[1]}" for column in columns]


def build_trait_label_maps(trait_meta: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    short_to_full: dict[str, str] = {}
    short_to_display: dict[str, str] = {}
    for _, row in trait_meta.iterrows():
        trait_no = int(row["Trait No."])
        short_id = f"T{trait_no}"
        full_name = str(row["Trait name"]).strip()
        if ":" in full_name:
            display_name = full_name.split(":", 1)[1].strip()
        else:
            display_name = full_name
        short_to_full[short_id] = full_name
        short_to_display[short_id] = f"{short_id}\n{display_name}"
    return short_to_full, short_to_display


def display_trait_labels(traits: list[str], trait_display_map: dict[str, str]) -> list[str]:
    return [trait_display_map.get(trait, trait) for trait in traits]


def zscore_by_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(
        lambda col: (col - col.mean()) / col.std(ddof=0) if col.std(ddof=0) not in [0, np.nan] and pd.notna(col.std(ddof=0)) else col,
        axis=0,
    )


def safe_to_csv(df: pd.DataFrame | pd.Series, path: Path, **kwargs: object) -> Path:
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_latest{path.suffix}")
        df.to_csv(fallback, **kwargs)
        return fallback


def save_basic_tables(df_actual: pd.DataFrame, df_predict: pd.DataFrame) -> None:
    safe_to_csv(df_actual, OUTPUT_DIR / "table3_actual.csv", index=False, encoding="utf-8-sig")
    safe_to_csv(df_predict, OUTPUT_DIR / "table3_predict.csv", index=False, encoding="utf-8-sig")


def select_traits_by_missing_rate(df_predict: pd.DataFrame, predict_cols: list[str], threshold: float) -> tuple[list[str], pd.Series]:
    missing_rate = df_predict[predict_cols].isna().mean().sort_values(ascending=False)
    selected = [column for column in predict_cols if missing_rate[column] <= threshold]
    return selected, missing_rate


def select_main_text_varieties(
    cluster_df: pd.DataFrame,
    df_predict: pd.DataFrame,
    main_trait_ids: list[str],
    normal_type_target: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trait_source_cols = []
    for trait_id in main_trait_ids:
        trait_no = trait_id.replace("T", "")
        source_col = f"Trait_{trait_no}_Predict"
        if source_col in df_predict.columns:
            trait_source_cols.append(source_col)

    trait_frame = df_predict[["Cultivar Name"] + trait_source_cols].copy()
    trait_frame = trait_frame.rename(
        columns={source: f"T{source.split('_')[1]}" for source in trait_source_cols}
    )

    merged = cluster_df.merge(
        trait_frame,
        on="Cultivar Name",
        how="left",
    )

    minority = merged[merged["Endosperm Type"] != "Normal type"].copy()
    minority["Selection_Reason"] = "All retained for minority category coverage"

    normal = merged[merged["Endosperm Type"] == "Normal type"].copy()
    normal["PC_Distance_To_Center"] = np.sqrt(
        (normal["PC1"] - normal["PC1"].mean()) ** 2 + (normal["PC2"] - normal["PC2"].mean()) ** 2
    )

    selected_rows: list[pd.DataFrame] = []
    selected_names: set[str] = set()

    def add_candidate(row: pd.Series, reason: str) -> None:
        cultivar = str(row["Cultivar Name"])
        if cultivar in selected_names:
            return
        row_df = row.to_frame().T.copy()
        row_df["Selection_Reason"] = reason
        selected_rows.append(row_df)
        selected_names.add(cultivar)

    if not normal.empty:
        add_candidate(
            normal.sort_values(["PC_Distance_To_Center", "Missing_Rate", "Cultivar Name"]).iloc[0],
            "Normal type PCA center",
        )
        add_candidate(
            normal.sort_values(["PC_Distance_To_Center", "Missing_Rate", "Cultivar Name"], ascending=[False, True, True]).iloc[0],
            "Normal type PCA farthest extreme",
        )

        for axis in ["PC1", "PC2"]:
            add_candidate(
                normal.sort_values([axis, "Missing_Rate", "Cultivar Name"]).iloc[0],
                f"Normal type {axis} minimum",
            )
            add_candidate(
                normal.sort_values([axis, "Missing_Rate", "Cultivar Name"], ascending=[False, True, True]).iloc[0],
                f"Normal type {axis} maximum",
            )

        for trait_id in main_trait_ids:
            non_missing = normal.dropna(subset=[trait_id])
            if non_missing.empty:
                continue
            add_candidate(
                non_missing.sort_values([trait_id, "Missing_Rate", "Cultivar Name"]).iloc[0],
                f"Normal type {trait_id} minimum",
            )
            if len(selected_names) >= normal_type_target:
                break
            add_candidate(
                non_missing.sort_values([trait_id, "Missing_Rate", "Cultivar Name"], ascending=[False, True, True]).iloc[0],
                f"Normal type {trait_id} maximum",
            )
            if len(selected_names) >= normal_type_target:
                break

    normal_selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else normal.head(0).copy()
    if len(normal_selected) > normal_type_target:
        normal_selected = normal_selected.head(normal_type_target).copy()
    elif len(normal_selected) < normal_type_target:
        remainder = normal.loc[~normal["Cultivar Name"].isin(normal_selected["Cultivar Name"])].copy()
        remainder = remainder.sort_values(["PC_Distance_To_Center", "Missing_Rate", "Cultivar Name"])
        fill_count = normal_type_target - len(normal_selected)
        filler = remainder.head(fill_count).copy()
        if not filler.empty:
            filler["Selection_Reason"] = "Normal type PCA coverage filler"
            normal_selected = pd.concat([normal_selected, filler], ignore_index=True)

    selected = pd.concat([normal_selected, minority], ignore_index=True)
    selected = selected.sort_values(["Endosperm Type", "Cluster", "Missing_Rate", "Cultivar Name"]).reset_index(drop=True)
    return selected, normal_selected[["Cultivar Name", "Selection_Reason"]].copy()


def build_endosperm_palette(series: pd.Series) -> dict[str, tuple[float, float, float, float]]:
    categories = sorted(series.dropna().astype(str).unique().tolist())
    cmap = mpl.colormaps["Set2"].resampled(max(len(categories), 3))
    return {category: cmap(i) for i, category in enumerate(categories)}


def get_group_centers(row_types: list[str]) -> list[tuple[str, float]]:
    centers: list[tuple[str, float]] = []
    start = 0
    for idx in range(1, len(row_types) + 1):
        is_boundary = idx == len(row_types) or row_types[idx] != row_types[start]
        if is_boundary:
            centers.append((row_types[start], (start + idx - 1) / 2))
            start = idx
    return centers


def get_group_boundaries(row_types: list[str]) -> list[float]:
    boundaries: list[float] = []
    for idx in range(1, len(row_types)):
        if row_types[idx] != row_types[idx - 1]:
            boundaries.append(idx - 0.5)
    return boundaries


def derive_cluster_type_mapping(cluster_df: pd.DataFrame) -> tuple[dict[int, str], pd.DataFrame]:
    contingency = pd.crosstab(cluster_df["Cluster"], cluster_df["Endosperm Type"])
    mapping_records: list[dict[str, object]] = []
    cluster_label_map: dict[int, str] = {}
    for cluster_id in contingency.index:
        counts = contingency.loc[cluster_id].sort_values(ascending=False)
        main_type = str(counts.index[0])
        main_count = int(counts.iloc[0])
        cluster_total = int(counts.sum())
        enriched = [f"{idx}:{int(val)}" for idx, val in counts.items() if val > 0]
        label = f"Cluster {cluster_id} ({main_type})"
        cluster_label_map[int(cluster_id)] = label
        mapping_records.append(
            {
                "Cluster": int(cluster_id),
                "Assigned_Label": label,
                "Dominant_Endosperm_Type": main_type,
                "Dominant_Count": main_count,
                "Cluster_Size": cluster_total,
                "Dominant_Percentage": main_count / cluster_total if cluster_total else np.nan,
                "Composition": "; ".join(enriched),
            }
        )
    mapping_df = pd.DataFrame(mapping_records)
    return cluster_label_map, mapping_df


def build_cluster_palette(series: pd.Series) -> dict[int, tuple[float, float, float, float]]:
    categories = sorted(pd.Series(series).dropna().astype(int).unique().tolist())
    cmap = mpl.colormaps["tab10"].resampled(max(len(categories), 3))
    return {category: cmap(i) for i, category in enumerate(categories)}


def make_predict_fingerprint_heatmaps(
    df_predict: pd.DataFrame,
    heatmap_predict_cols: list[str],
    cluster_df: pd.DataFrame,
    trait_order: list[str],
    trait_display_map: dict[str, str],
) -> pd.DataFrame:
    trait_data = df_predict[heatmap_predict_cols].copy()
    trait_data.columns = short_trait_names(heatmap_predict_cols)
    variety_names = df_predict["Cultivar Name"].astype(str)

    trait_z = zscore_by_column(trait_data)
    trait_z.index = variety_names
    trait_z = trait_z[trait_order]
    safe_to_csv(trait_z, OUTPUT_DIR / "maize_trait_fingerprint_zscore.csv", encoding="utf-8-sig")

    ordered_df = cluster_df.sort_values(["Endosperm Type", "Cluster", "Missing_Rate", "Cultivar Name"]).copy()
    ordered_names = ordered_df["Cultivar Name"].astype(str)
    supplementary_plot = trait_z.loc[ordered_names]
    ordered_endosperm = ordered_df["Endosperm Type"].astype(str).tolist()

    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad(color="lightgray")
    endosperm_palette = build_endosperm_palette(cluster_df["Endosperm Type"])

    def draw_heatmap(
        plot_data: pd.DataFrame,
        row_types: list[str],
        output_name: str,
        title: str,
        figsize: tuple[float, float],
    ) -> None:
        masked_data = np.ma.masked_invalid(plot_data.values)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.35, 12], wspace=0.05)
        ax_label = fig.add_subplot(gs[0, 0])
        ax_type = fig.add_subplot(gs[0, 1])
        ax = fig.add_subplot(gs[0, 2])

        type_rgb = np.array([endosperm_palette[row_type] for row_type in row_types]).reshape(len(row_types), 1, 4)
        ax_type.imshow(type_rgb, aspect="auto")
        ax_type.set_xticks([])
        ax_type.set_yticks([])
        ax_type.set_title("Endosperm\nType", fontsize=9)
        ax_type.set_ylim(plot_data.shape[0] - 0.5, -0.5)

        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(plot_data.shape[0] - 0.5, -0.5)
        ax_label.axis("off")
        ax_label.set_title("Grouped by\nEndosperm Type", fontsize=9)

        im = ax.imshow(masked_data, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(plot_data.shape[1]))
        ax.set_xticklabels(display_trait_labels(plot_data.columns.tolist(), trait_display_map), rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(plot_data.shape[0]))
        ax.set_yticklabels(plot_data.index, fontsize=7 if plot_data.shape[0] > 30 else 9)
        ax.set_xlabel("DUS Traits")
        ax.set_ylabel("Maize Varieties")
        ax.set_title(title)

        boundaries = get_group_boundaries(row_types)
        for boundary in boundaries:
            ax.axhline(boundary, color="white", linewidth=1.6)
            ax_type.axhline(boundary, color="white", linewidth=1.6)
            ax_label.axhline(boundary, color="#808080", linewidth=1.0)

        for label, center in get_group_centers(row_types):
            ax_label.text(0.98, center, label, ha="right", va="center", fontsize=9)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Standardized Trait Value (z-score)")
        legend_handles = [
            mpl.patches.Patch(color=color, label=label) for label, color in endosperm_palette.items()
        ]
        ax.legend(
            handles=legend_handles,
            title="Endosperm Type",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
        )
        fig.text(
            0.99,
            0.01,
            "Gray cells indicate missing traits. Rows are grouped by Endosperm Type; traits are ordered by discrimination strength.",
            ha="right",
            va="bottom",
            fontsize=9,
        )
        fig.savefig(OUTPUT_DIR / output_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    selected_main, normal_selection_reasons = select_main_text_varieties(
        cluster_df,
        df_predict,
        trait_order[:MAIN_TEXT_TOP_N_TRAITS],
        NORMAL_TYPE_TARGET,
    )
    selected_main_names = selected_main["Cultivar Name"].astype(str).tolist()
    selected_main_types = selected_main["Endosperm Type"].astype(str).tolist()
    main_plot = supplementary_plot.loc[selected_main_names]
    main_plot = main_plot[trait_order[:MAIN_TEXT_TOP_N_TRAITS]]

    draw_heatmap(
        main_plot,
        selected_main_types,
        "digital_phenotypic_fingerprint_heatmap_main_text.png",
        "Multi-Dimensional Digital Phenotypic Fingerprints",
        (14, 8),
    )
    draw_heatmap(
        supplementary_plot,
        ordered_endosperm,
        "digital_phenotypic_fingerprint_heatmap_supplementary.png",
        "Multi-Dimensional Digital Phenotypic Fingerprints (All Varieties)",
        (14, 24),
    )

    safe_to_csv(selected_main, OUTPUT_DIR / "main_text_representative_varieties.csv", index=False, encoding="utf-8-sig")
    safe_to_csv(normal_selection_reasons, OUTPUT_DIR / "normal_type_selection_reasons.csv", index=False, encoding="utf-8-sig")
    return trait_z, selected_main


def make_pca_kmeans(df_predict: pd.DataFrame, predict_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    x = df_predict[predict_cols].copy()
    variety_missing_rate = x.isna().mean(axis=1)

    imputer = SimpleImputer(strategy="median")
    x_imp = imputer.fit_transform(x)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imp)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(x_scaled)

    kmeans = KMeans(n_clusters=CLUSTER_K, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(x_scaled)

    pca_df = pd.DataFrame(
        {
            "Cultivar Name": df_predict["Cultivar Name"].astype(str).values,
            "Field Plot No.": df_predict["Field Plot No."].astype(str).values,
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "Cluster": clusters + 1,
            "Missing_Rate": variety_missing_rate.values,
            "Endosperm Type": df_predict["Endosperm Type"].astype(str).values,
        }
    )
    safe_to_csv(pca_df, OUTPUT_DIR / "maize_pca_clusters.csv", index=False, encoding="utf-8-sig")

    cluster_trait_df = pd.DataFrame(x_scaled, columns=short_trait_names(predict_cols))
    cluster_trait_df["Cluster"] = clusters + 1
    cluster_means = cluster_trait_df.groupby("Cluster").mean().sort_index()
    safe_to_csv(cluster_means, OUTPUT_DIR / "maize_cluster_trait_means.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = mpl.colormaps["tab10"].resampled(CLUSTER_K)
    for cl in sorted(pca_df["Cluster"].unique()):
        sub = pca_df[pca_df["Cluster"] == cl]
        ax.scatter(sub["PC1"], sub["PC2"], label=f"Cluster {cl}", s=40, alpha=0.8, color=colors(cl - 1))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.set_title("PCA of Automated Maize DUS Traits")
    ax.legend(frameon=False, loc="best")
    fig.text(0.99, 0.01, "PC1 + PC2 show broad trait-structure trends; k=5 illustrates example population structure.", ha="right", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_kmeans_clusters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pca_df, cluster_means, pca


def make_pca_by_endosperm_plot(
    pca_df: pd.DataFrame,
    pca: PCA,
    endosperm_palette: dict[str, tuple[float, float, float, float]],
    cluster_palette: dict[int, tuple[float, float, float, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v"]
    cluster_markers = {
        cluster_id: marker_cycle[idx % len(marker_cycle)]
        for idx, cluster_id in enumerate(sorted(pca_df["Cluster"].unique()))
    }

    for endosperm_type in sorted(pca_df["Endosperm Type"].dropna().astype(str).unique()):
        subset_type = pca_df[pca_df["Endosperm Type"].astype(str) == endosperm_type]
        for cluster_id in sorted(subset_type["Cluster"].unique()):
            sub = subset_type[subset_type["Cluster"] == cluster_id]
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                s=55,
                alpha=0.88,
                marker=cluster_markers[int(cluster_id)],
                facecolor=endosperm_palette[endosperm_type],
                edgecolor=cluster_palette[int(cluster_id)],
                linewidth=1.1,
                label=f"{endosperm_type} | C{int(cluster_id)}",
            )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.set_title("PCA of Automated Maize DUS Traits\nColor = Real Variety Category, Marker/Edge = K-means Cluster")

    type_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=8,
            label=label,
        )
        for label, color in endosperm_palette.items()
    ]
    cluster_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker=marker_cycle[idx % len(marker_cycle)],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=cluster_palette[cluster_id],
            markeredgewidth=1.2,
            markersize=8,
            label=f"Cluster {cluster_id}",
        )
        for idx, cluster_id in enumerate(sorted(cluster_palette))
    ]

    legend1 = ax.legend(handles=type_handles, title="Real Variety Category", loc="upper right", frameon=False)
    ax.add_artist(legend1)
    ax.legend(handles=cluster_handles, title="K-means Cluster", loc="lower right", frameon=False)
    fig.text(
        0.99,
        0.01,
        "PC1 + PC2 show broad trait-structure trends. Colors show real categories; marker styles keep the unsupervised cluster structure visible.",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_by_real_category_with_clusters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_confidence_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: tuple[float, float, float, float], label: str) -> bool:
    if len(x) < 3:
        return False
    cov = np.cov(x, y)
    if np.any(~np.isfinite(cov)) or np.linalg.matrix_rank(cov) < 2:
        return False

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    scale = np.sqrt(chi2.ppf(0.95, df=2))

    width = 2 * scale * np.sqrt(max(eigenvalues[0], 0))
    height = 2 * scale * np.sqrt(max(eigenvalues[1], 0))
    if width == 0 or height == 0:
        return False

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.14,
        linewidth=1.4,
    )
    ax.add_patch(ellipse)
    return True


def make_representative_pca_with_ellipses(
    pca_df: pd.DataFrame,
    selected_main: pd.DataFrame,
    pca: PCA,
    endosperm_palette: dict[str, tuple[float, float, float, float]],
) -> None:
    selected_plot = pca_df.merge(
        selected_main[["Cultivar Name", "Selection_Reason"]],
        on="Cultivar Name",
        how="inner",
    ).copy()

    fig, ax = plt.subplots(figsize=(9, 6.8))
    for endosperm_type in sorted(selected_plot["Endosperm Type"].dropna().astype(str).unique()):
        sub = selected_plot[selected_plot["Endosperm Type"].astype(str) == endosperm_type].copy()
        color = endosperm_palette[endosperm_type]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=70,
            alpha=0.9,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label=f"{endosperm_type} (n={len(sub)})",
        )
        add_confidence_ellipse(ax, sub["PC1"].to_numpy(), sub["PC2"].to_numpy(), color, endosperm_type)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.set_title("PCA of 33 Representative Varieties with 95% Confidence Ellipses")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.text(
        0.99,
        0.01,
        "Points show the 33 representative varieties. Ellipses indicate 95% confidence regions by Endosperm Type when covariance estimation is supported.",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_representative_33_with_95ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_representative_trait_summary_plots(
    selected_main: pd.DataFrame,
    trait_order: list[str],
    trait_display_map: dict[str, str],
) -> None:
    top_traits = trait_order[:MAIN_TEXT_TOP_N_TRAITS]
    stats_source = selected_main[["Cultivar Name", "Endosperm Type"] + top_traits].copy()

    summary = (
        stats_source.groupby("Endosperm Type")[top_traits]
        .agg(["mean", "std"])
        .sort_index()
    )
    summary.columns = [f"{trait}_{stat}" for trait, stat in summary.columns]
    safe_to_csv(summary.reset_index(), OUTPUT_DIR / "representative_33_trait_mean_sd_by_endosperm_type.csv", index=False, encoding="utf-8-sig")

    long_df = stats_source.melt(
        id_vars=["Cultivar Name", "Endosperm Type"],
        value_vars=top_traits,
        var_name="Trait_ID",
        value_name="Trait_Value",
    )
    long_df["Trait_Label"] = long_df["Trait_ID"].map(trait_display_map).fillna(long_df["Trait_ID"])
    safe_to_csv(long_df, OUTPUT_DIR / "representative_33_trait_long_format.csv", index=False, encoding="utf-8-sig")

    endosperm_palette = build_endosperm_palette(selected_main["Endosperm Type"])
    ordered_types = sorted(selected_main["Endosperm Type"].dropna().astype(str).unique().tolist())
    ordered_labels = [trait_display_map.get(trait, trait) for trait in top_traits]

    mean_df = (
        long_df.groupby(["Endosperm Type", "Trait_ID", "Trait_Label"], as_index=False)
        .agg(Mean=("Trait_Value", "mean"), SD=("Trait_Value", "std"))
    )

    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = np.arange(len(top_traits))
    width = 0.16
    offsets = np.linspace(-2, 2, num=len(ordered_types)) * width
    for offset, endosperm_type in zip(offsets, ordered_types):
        sub = mean_df[mean_df["Endosperm Type"] == endosperm_type].copy()
        sub = sub.set_index("Trait_ID").reindex(top_traits).reset_index()
        ax.bar(
            x + offset,
            sub["Mean"],
            width=width,
            yerr=sub["SD"],
            capsize=3,
            label=endosperm_type,
            color=endosperm_palette[endosperm_type],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean trait value ± SD")
    ax.set_xlabel("Representative fingerprint traits")
    ax.set_title("Representative 33 Varieties: Mean ± SD by Endosperm Type")
    ax.legend(frameon=False, loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "representative_33_trait_mean_sd_barplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 7))
    sns.violinplot(
        data=long_df,
        x="Trait_Label",
        y="Trait_Value",
        hue="Endosperm Type",
        order=ordered_labels,
        hue_order=ordered_types,
        palette=endosperm_palette,
        cut=0,
        density_norm="width",
        inner="quart",
        ax=ax,
    )
    ax.set_xlabel("Representative fingerprint traits")
    ax.set_ylabel("Trait value")
    ax.set_title("Representative 33 Varieties: Trait Distributions by Endosperm Type")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.legend(title="Endosperm Type", frameon=False, loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "representative_33_trait_violinplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(
        data=long_df,
        x="Trait_Label",
        y="Trait_Value",
        hue="Endosperm Type",
        order=ordered_labels,
        hue_order=ordered_types,
        palette=endosperm_palette,
        ax=ax,
        showfliers=True,
        linewidth=0.9,
    )
    ax.set_xlabel("Representative fingerprint traits")
    ax.set_ylabel("Trait value")
    ax.set_title("Representative 33 Varieties: Box Plot by Endosperm Type")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.legend(title="Endosperm Type", frameon=False, loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "representative_33_trait_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_real_category_trait_heatmap(
    df_predict: pd.DataFrame,
    predict_cols: list[str],
    trait_order: list[str],
    trait_display_map: dict[str, str],
) -> None:
    x = df_predict[predict_cols].copy()
    x.columns = short_trait_names(predict_cols)
    x["Endosperm Type"] = df_predict["Endosperm Type"].astype(str).values

    category_means = x.groupby("Endosperm Type").mean().sort_index()
    category_means = category_means[trait_order]
    safe_to_csv(category_means, OUTPUT_DIR / "real_category_trait_means.csv", encoding="utf-8-sig")

    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad(color="lightgray")
    masked_data = np.ma.masked_invalid(category_means.values)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(masked_data, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(category_means.shape[1]))
    ax.set_xticklabels(display_trait_labels(category_means.columns.tolist(), trait_display_map), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(category_means.shape[0]))
    ax.set_yticklabels(category_means.index, fontsize=10)
    ax.set_xlabel("DUS Traits")
    ax.set_ylabel("Real Variety Category")
    ax.set_title("Real Category-Wise Average Trait Profiles")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean standardized trait value")
    fig.text(
        0.99,
        0.01,
        "Rows correspond to the observed Endosperm Type categories rather than unsupervised clusters.",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_category_wise_trait_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_cluster_trait_heatmap(
    cluster_means: pd.DataFrame,
    cluster_label_map: dict[int, str],
    trait_display_map: dict[str, str],
) -> None:
    plot_data = cluster_means.copy()
    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad(color="lightgray")
    masked_data = np.ma.masked_invalid(plot_data.values)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(masked_data, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(plot_data.shape[1]))
    ax.set_xticklabels(display_trait_labels(plot_data.columns.tolist(), trait_display_map), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(plot_data.shape[0]))
    ax.set_yticklabels([cluster_label_map.get(int(i), f"Cluster {i}") for i in plot_data.index], fontsize=9)
    ax.set_xlabel("DUS Traits")
    ax.set_ylabel("Clusters / Assigned Variety Category")
    ax.set_title("Cluster-Wise Average Trait Profiles")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Standardized Trait Value")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_wise_trait_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_endosperm_trait_views(
    df_predict: pd.DataFrame,
    predict_cols: list[str],
    trait_display_map: dict[str, str],
) -> pd.DataFrame:
    x = df_predict[predict_cols].copy()
    x.columns = short_trait_names(predict_cols)
    x["Endosperm Type"] = df_predict["Endosperm Type"].astype(str).values

    endosperm_means = x.groupby("Endosperm Type").mean().sort_index()
    safe_to_csv(endosperm_means, OUTPUT_DIR / "endosperm_type_trait_means.csv", encoding="utf-8-sig")

    z_means = zscore_by_column(endosperm_means)
    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad(color="lightgray")
    masked_data = np.ma.masked_invalid(z_means.values)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(masked_data, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(z_means.shape[1]))
    ax.set_xticklabels(display_trait_labels(z_means.columns.tolist(), trait_display_map), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(z_means.shape[0]))
    ax.set_yticklabels(z_means.index, fontsize=10)
    ax.set_xlabel("DUS Traits")
    ax.set_ylabel("Endosperm Type")
    ax.set_title("Trait Profiles Grouped by Endosperm Type")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean standardized trait value")
    fig.text(0.99, 0.01, "Rows summarize average trait profiles for each Endosperm Type.", ha="right", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "endosperm_type_trait_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    records = []
    for trait in x.columns[:-1]:
        groups = [group[trait].dropna().values for _, group in x.groupby("Endosperm Type")]
        groups = [group for group in groups if len(group) > 1]
        if len(groups) >= 2:
            f_stat, p_value = f_oneway(*groups)
        else:
            f_stat, p_value = np.nan, np.nan
        mean_by_type = x.groupby("Endosperm Type")[trait].mean()
        records.append(
            {
                "Trait": trait,
                "Missing_Rate": x[trait].isna().mean(),
                "Between_Type_Range": mean_by_type.max() - mean_by_type.min(),
                "Between_Type_STD": mean_by_type.std(ddof=0),
                "ANOVA_F": f_stat,
                "ANOVA_p": p_value,
            }
        )

    discrimination_df = pd.DataFrame(records).sort_values(
        ["ANOVA_F", "Between_Type_Range"], ascending=[False, False]
    )
    safe_to_csv(discrimination_df, OUTPUT_DIR / "trait_discrimination_by_endosperm_type.csv", index=False, encoding="utf-8-sig")
    return discrimination_df


def save_missingness_outputs(df_predict: pd.DataFrame, predict_cols: list[str]) -> tuple[pd.Series, pd.DataFrame, list[str], pd.DataFrame]:
    x = df_predict[predict_cols].copy()
    trait_missing_rate = x.isna().mean().sort_values(ascending=False)
    trait_missing_rate.index = short_trait_names(trait_missing_rate.index.tolist())
    safe_to_csv(trait_missing_rate, OUTPUT_DIR / "maize_trait_missing_rate.csv", encoding="utf-8-sig")

    variety_missing_rate = x.isna().mean(axis=1)
    missing_df = pd.DataFrame(
        {
            "Cultivar_Name": df_predict["Cultivar Name"].astype(str).values,
            "Field_Plot_No": df_predict["Field Plot No."].astype(str).values,
            "Missing_Rate": variety_missing_rate.values,
        }
    ).sort_values("Missing_Rate", ascending=False)
    safe_to_csv(missing_df, OUTPUT_DIR / "maize_variety_missing_rate.csv", index=False, encoding="utf-8-sig")

    raw_trait_missing_rate = x.isna().mean().sort_values(ascending=False)
    keep_traits = raw_trait_missing_rate[raw_trait_missing_rate <= MISSING_THRESHOLD].index.tolist()
    keep_varieties_mask = variety_missing_rate <= MISSING_THRESHOLD
    filtered = df_predict.loc[keep_varieties_mask, META_COLS + keep_traits].copy()
    safe_to_csv(filtered, OUTPUT_DIR / "df_predict_filtered_missing_le_30.csv", index=False, encoding="utf-8-sig")

    safe_to_csv(pd.DataFrame(
        {
            "Retained_Trait_Columns": keep_traits,
        }
    ), OUTPUT_DIR / "retained_traits_missing_le_30.csv", index=False, encoding="utf-8-sig")

    return trait_missing_rate, missing_df, keep_traits, filtered


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_table3(FILE_PATH, SHEET_NAME)
    meta_df = load_table2_metadata(FILE_PATH)
    trait_meta = load_table1_trait_metadata(FILE_PATH)
    trait_full_map, trait_display_map = build_trait_label_maps(trait_meta)
    actual_cols, predict_cols = get_trait_columns(df)
    to_numeric(df, actual_cols + predict_cols)

    df_actual = df[META_COLS + actual_cols].copy()
    df_predict = df[META_COLS + predict_cols].copy()
    df_actual = df_actual.merge(meta_df, on=["Field Plot No.", "Cultivar Name"], how="left")
    df_predict = df_predict.merge(meta_df, on=["Field Plot No.", "Cultivar Name"], how="left")
    save_basic_tables(df_actual, df_predict)

    main_text_predict_cols, trait_missing_rate_all = select_traits_by_missing_rate(
        df_predict, predict_cols, MAIN_TEXT_TRAIT_MISSING_THRESHOLD
    )

    pca_df, cluster_means, pca = make_pca_kmeans(df_predict, main_text_predict_cols)
    cluster_label_map, cluster_mapping_df = derive_cluster_type_mapping(pca_df)
    pca_df["Assigned Cluster Category"] = pca_df["Cluster"].map(cluster_label_map)
    safe_to_csv(pca_df, OUTPUT_DIR / "maize_pca_clusters.csv", index=False, encoding="utf-8-sig")
    safe_to_csv(cluster_mapping_df, OUTPUT_DIR / "cluster_category_mapping.csv", index=False, encoding="utf-8-sig")
    endosperm_palette = build_endosperm_palette(pca_df["Endosperm Type"])
    cluster_palette = build_cluster_palette(pca_df["Cluster"])

    discrimination_df = make_endosperm_trait_views(df_predict, main_text_predict_cols, trait_display_map)
    trait_order = discrimination_df["Trait"].tolist()
    trait_z, selected_main = make_predict_fingerprint_heatmaps(
        df_predict, main_text_predict_cols, pca_df, trait_order, trait_display_map
    )
    cluster_means = cluster_means[trait_order]
    cluster_means.index.name = "Cluster"
    safe_to_csv(cluster_means, OUTPUT_DIR / "maize_cluster_trait_means.csv", encoding="utf-8-sig")
    make_cluster_trait_heatmap(cluster_means, cluster_label_map, trait_display_map)
    make_real_category_trait_heatmap(df_predict, main_text_predict_cols, trait_order, trait_display_map)
    make_representative_pca_with_ellipses(pca_df, selected_main, pca, endosperm_palette)
    make_representative_trait_summary_plots(selected_main, trait_order, trait_display_map)
    trait_missing_rate, missing_df, keep_traits, filtered = save_missingness_outputs(df_predict, predict_cols)

    safe_to_csv(pd.DataFrame(
        {
            "Main_Text_Heatmap_Traits": main_text_predict_cols,
        }
    ), OUTPUT_DIR / "main_text_heatmap_traits.csv", index=False, encoding="utf-8-sig")
    safe_to_csv(
        pd.DataFrame({"Trait_Order_By_Discrimination": trait_order}),
        OUTPUT_DIR / "trait_order_by_discrimination.csv",
        index=False,
        encoding="utf-8-sig",
    )
    safe_to_csv(
        pd.DataFrame({"Main_Text_Top8_Traits": trait_order[:MAIN_TEXT_TOP_N_TRAITS]}),
        OUTPUT_DIR / "main_text_top8_traits.csv",
        index=False,
        encoding="utf-8-sig",
    )
    safe_to_csv(pd.DataFrame(
        {
            "Trait_ID": trait_order,
            "Trait_Full_Name": [trait_full_map.get(trait, trait) for trait in trait_order],
            "Trait_Display_Label": [trait_display_map.get(trait, trait) for trait in trait_order],
        }
    ), OUTPUT_DIR / "trait_name_mapping.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = mpl.colormaps["tab10"].resampled(CLUSTER_K)
    for cl in sorted(pca_df["Cluster"].unique()):
        sub = pca_df[pca_df["Cluster"] == cl]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            label=cluster_label_map.get(int(cl), f"Cluster {cl}"),
            s=40,
            alpha=0.8,
            color=colors(int(cl) - 1),
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.set_title("PCA of Automated Maize DUS Traits")
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.text(
        0.99,
        0.01,
        "Cluster labels are assigned by the dominant Endosperm Type within each current K-means cluster.",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_kmeans_clusters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    make_pca_by_endosperm_plot(pca_df, pca, endosperm_palette, cluster_palette)

    summary = pd.DataFrame(
        {
            "metric": [
                "n_varieties",
                "n_actual_traits",
                "n_predict_traits",
                "main_text_heatmap_traits_le_50_missing",
                "main_text_top8_traits",
                "kmeans_k",
                "pc1_variance_ratio",
                "pc2_variance_ratio",
                "pc1_pc2_cumulative_variance",
                "retained_traits_missing_le_30",
                "retained_varieties_missing_le_30",
            ],
            "value": [
                len(df_predict),
                len(actual_cols),
                len(predict_cols),
                len(main_text_predict_cols),
                MAIN_TEXT_TOP_N_TRAITS,
                CLUSTER_K,
                pca.explained_variance_ratio_[0],
                pca.explained_variance_ratio_[1],
                pca.explained_variance_ratio_.sum(),
                len(keep_traits),
                len(filtered),
            ],
        }
    )
    safe_to_csv(summary, OUTPUT_DIR / "analysis_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Actual traits: {len(actual_cols)}")
    print(f"Predict traits: {len(predict_cols)}")
    print(f"Main-text heatmap/PCA traits (<=50% missing): {len(main_text_predict_cols)}")
    print(f"KMeans k: {CLUSTER_K}")
    print(f"PC1 variance ratio: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"PC2 variance ratio: {pca.explained_variance_ratio_[1]:.4f}")
    print(f"Cumulative variance: {pca.explained_variance_ratio_.sum():.4f}")
    print("Top trait missing rates:")
    print(trait_missing_rate.head(5).to_string())
    print("Top variety missing rates:")
    print(missing_df.head(5).to_string(index=False))
    print(f"Retained traits (<=30% missing): {len(keep_traits)}")
    print(f"Retained varieties (<=30% missing): {len(filtered)}")
    print(f"Fingerprint matrix shape: {trait_z.shape}")
    print("Top discriminative traits by Endosperm Type:")
    print(discrimination_df.head(8).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Supplementary Table 3 and generate manuscript-oriented outputs.")
    parser.add_argument("--file-path", default=str(FILE_PATH), help="Supplementary workbook path.")
    parser.add_argument("--sheet-name", default=SHEET_NAME, help="Worksheet name for Supplementary Table 3.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    FILE_PATH = Path(args.file_path)
    SHEET_NAME = args.sheet_name
    OUTPUT_DIR = Path(args.output_dir)
    main()
