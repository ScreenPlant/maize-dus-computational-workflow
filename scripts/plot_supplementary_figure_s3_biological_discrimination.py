from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


TRAIT_FAMILY_COLORS = {
    "Color": "#F57C00",
    "Size": "#4E86C5",
    "Count": "#59A14F",
    "Shape": "#8064A2",
}

ENDOSPERM_ORDER = [
    "Normal type",
    "Sweet type",
    "Waxy type",
    "Sweet-waxy hybrid type",
    "Popcorn type",
]
ENDOSPERM_COLORS = {
    "Normal type": "#8FA3B8",
    "Sweet type": "#F28E2B",
    "Waxy type": "#59A14F",
    "Sweet-waxy hybrid type": "#8B6BB1",
    "Popcorn type": "#E5B74B",
}

BREEDING_ORDER = ["Inbred line", "Single-cross hybrid"]
BREEDING_COLORS = {
    "Inbred line": "#4E86C5",
    "Single-cross hybrid": "#F28E2B",
}

ENDOSPERM_TRAITS = [
    ("T37", "Grain width", "mm", 10.0),
    ("T7", "Leaf curvature", "Score", 1.0),
    ("T34", "Number of grain colors", "Count", 1.0),
    ("T40", "Color of grain apex", "Score", 1.0),
]

BREEDING_TRAITS = [
    ("T26", "Upper-ear insertion height", "cm", 1.0),
    ("T31", "Ear diameter", "cm", 1.0),
    ("T30", "Ear length", "cm", 1.0),
    ("T27", "Plant length", "cm", 1.0),
    ("T28", "Ear-height / plant-length ratio", "Ratio", 1.0),
]


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Reproduce Supplementary Fig. S3 from the processed 150-cultivar trait matrix."
    )
    parser.add_argument("--project-root", type=Path, default=script_root)
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--stats-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--effect-sizes-only",
        action="store_true",
        help="Draw panels A–B directly from Supplementary Table 5 without requiring cultivar-level sheets.",
    )
    return parser.parse_args()


def load_rebuilt_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    frame = raw.iloc[2:].copy().reset_index(drop=True)
    frame.columns = raw.iloc[1].tolist()
    return frame


def load_analysis_data(workbook: Path) -> pd.DataFrame:
    metadata = load_rebuilt_sheet(workbook, "Supplementary Table 2")
    metadata = metadata[
        ["Field Plot No.", "Cultivar Name", "Endosperm Type", "Breeding Type"]
    ].copy()
    traits = load_rebuilt_sheet(workbook, "Supplementary Table 3")
    merged = traits.merge(
        metadata,
        on=["Field Plot No.", "Cultivar Name"],
        how="left",
        suffixes=("", "_meta"),
    )
    for grouping in ("Endosperm Type", "Breeding Type"):
        meta_col = f"{grouping}_meta"
        if meta_col in merged.columns:
            merged[grouping] = merged[grouping].fillna(merged[meta_col])
    return merged


def _numeric_p_value(value: object) -> float:
    """Parse numeric and '<x' P-value cells from Supplementary Table 5."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.startswith("<"):
        text = text[1:].strip()
    return float(text)


def load_table5_statistics(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read effect sizes and FDR values directly from the published Table 5."""
    table = load_rebuilt_sheet(workbook, "Supplementary Table 5")
    table["Trait code"] = table["Trait code"].ffill()
    table = table.loc[table["Grouping factor"].notna() & table["Test method"].notna()].copy()
    table["Raw P value"] = table["Raw P value"].map(_numeric_p_value)
    table["FDR-adjusted P value"] = table["FDR-adjusted P value"].map(_numeric_p_value)
    table["Eta-squared\nη²"] = pd.to_numeric(table["Eta-squared\nη²"], errors="coerce")

    def build(grouping: str) -> pd.DataFrame:
        subset = table.loc[table["Grouping factor"].astype(str).str.casefold().eq(grouping.casefold())]
        anova = subset.loc[subset["Test method"].astype(str).str.casefold().eq("anova")].copy()
        kruskal = subset.loc[subset["Test method"].astype(str).str.contains("Kruskal", case=False, na=False)].copy()
        anova = anova[
            ["Trait code", "Trait name", "Trait family", "Eta-squared\nη²", "Raw P value", "FDR-adjusted P value"]
        ].rename(
            columns={
                "Trait code": "Trait_ID",
                "Trait name": "Trait_Name",
                "Trait family": "Analysis Type",
                "Eta-squared\nη²": "Eta_Squared",
                "Raw P value": "ANOVA_p",
                "FDR-adjusted P value": "ANOVA_FDR",
            }
        )
        kruskal = kruskal[["Trait code", "Raw P value", "FDR-adjusted P value"]].rename(
            columns={
                "Trait code": "Trait_ID",
                "Raw P value": "Kruskal_p",
                "FDR-adjusted P value": "Kruskal_FDR",
            }
        )
        result = anova.merge(kruskal, on="Trait_ID", how="inner", validate="one_to_one")
        if len(result) != 23:
            raise ValueError(f"Expected 23 traits for {grouping}, found {len(result)}.")
        return result

    return build("endosperm type"), build("breeding type")


def significance_text(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "FDR = NA"
    if p_value < 1e-16:
        return "FDR < 1 × 10⁻¹⁶"
    exponent = int(np.floor(np.log10(p_value)))
    coefficient = p_value / (10**exponent)
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"FDR = {coefficient:.1f} × 10{str(exponent).translate(superscripts)}"


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9AA0A3")
    ax.spines[["left", "bottom"]].set_linewidth(0.7)
    ax.tick_params(axis="both", colors="#343A3D", labelsize=7, width=0.6, length=2.5)
    ax.grid(axis="y", color="#E8EBED", linewidth=0.6, zorder=0)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.115,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "circle,pad=0.28", "fc": "#111517", "ec": "none"},
        clip_on=False,
    )


def plot_effect_sizes(
    ax: plt.Axes,
    stats: pd.DataFrame,
    title: str,
    panel_label: str,
) -> None:
    frame = stats.sort_values(["Eta_Squared", "Trait_ID"], ascending=[False, True]).reset_index(drop=True)
    x = np.arange(len(frame))
    colors = frame["Analysis Type"].map(TRAIT_FAMILY_COLORS).fillna("#B8BABC")
    bars = ax.bar(x, frame["Eta_Squared"], color=colors, width=0.72, edgecolor="white", linewidth=0.35, zorder=2)
    # Red points follow the figure legend: Kruskal–Wallis P after BH-FDR correction.
    significant = frame["Kruskal_FDR"].lt(0.05)
    y_max = max(0.60, float(frame["Eta_Squared"].max()) * 1.19)

    for bar, eta, is_significant in zip(bars, frame["Eta_Squared"], significant):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            eta + y_max * 0.018,
            f"{eta:.3f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=5.4,
            color="#1D2224",
        )
        if is_significant:
            ax.plot(
                bar.get_x() + bar.get_width() / 2,
                eta + y_max * 0.010,
                marker="o",
                markersize=2.2,
                color="#C00000",
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(frame["Trait_ID"], rotation=90, fontsize=6)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Eta-squared (η²)", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10.2, fontweight="bold", pad=19)
    count = int(significant.sum())
    ax.text(
        1.0,
        1.035,
        f"{count} of {len(frame)} traits significant after BH-FDR correction",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#4D5558",
    )
    style_axis(ax)
    add_panel_label(ax, panel_label)


def trait_fdr(stats: pd.DataFrame, trait_id: str) -> float:
    values = stats.loc[stats["Trait_ID"].eq(trait_id), "Kruskal_FDR"]
    if values.empty:
        return np.nan
    return float(values.iloc[0])


def plot_boxplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    grouping: str,
    order: list[str],
    palette: dict[str, str],
    trait_id: str,
    trait_name: str,
    unit: str,
    multiplier: float,
    fdr: float,
) -> None:
    value_col = f"Trait_{trait_id[1:]}_Predict"
    plot_data = data[[grouping, value_col]].copy()
    plot_data[value_col] = pd.to_numeric(plot_data[value_col], errors="coerce") * multiplier
    plot_data = plot_data.dropna()

    sns.boxplot(
        data=plot_data,
        x=grouping,
        y=value_col,
        order=order,
        hue=grouping,
        hue_order=order,
        palette=palette,
        dodge=False,
        width=0.58,
        linewidth=0.7,
        fliersize=1.5,
        ax=ax,
        legend=False,
        boxprops={"edgecolor": "#50585B"},
        medianprops={"color": "#252B2D", "linewidth": 1.0},
        whiskerprops={"color": "#6B7275", "linewidth": 0.7},
        capprops={"color": "#6B7275", "linewidth": 0.7},
    )

    family = "Size"
    if trait_id in {"T7"}:
        family = "Shape"
    elif trait_id in {"T34"}:
        family = "Count"
    elif trait_id in {"T40"}:
        family = "Color"
    ax.set_title(
        f"{trait_id}  {trait_name}",
        fontsize=6.6,
        color=TRAIT_FAMILY_COLORS[family],
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.5,
        1.025,
        significance_text(fdr),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.4,
        color="#424A4D",
    )
    ax.set_xlabel("")
    ax.set_ylabel(unit, fontsize=6.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [label.replace(" type", "").replace(" hybrid", "\nhybrid").replace("-cross ", "-cross\n") for label in order],
        rotation=52 if grouping == "Endosperm Type" else 48,
        ha="right",
        fontsize=5.2,
    )
    style_axis(ax)


def add_section_title(fig: plt.Figure, bbox, title: str) -> None:
    fig.text(
        bbox.x0 + 0.008,
        bbox.y1 - 0.011,
        title,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="#202628",
    )


def add_figure_panel_label(fig: plt.Figure, bbox, label: str) -> None:
    fig.text(
        bbox.x0 - 0.007,
        bbox.y1 - 0.010,
        label,
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "circle,pad=0.28", "fc": "#111517", "ec": "none"},
    )


def add_panel_frame(fig: plt.Figure, bbox) -> None:
    pad_x, pad_y = 0.007, 0.014
    frame = FancyBboxPatch(
        (bbox.x0 - pad_x, bbox.y0 - pad_y),
        bbox.width + 2 * pad_x,
        bbox.height + 2 * pad_y,
        transform=fig.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor="white",
        edgecolor="#D9DDDF",
        linewidth=0.8,
        zorder=-10,
    )
    fig.add_artist(frame)


def export_source_tables(
    data: pd.DataFrame,
    endosperm_stats: pd.DataFrame,
    breeding_stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    effects = pd.concat(
        [
            endosperm_stats.assign(Grouping_Factor="Endosperm Type"),
            breeding_stats.assign(Grouping_Factor="Breeding Type"),
        ],
        ignore_index=True,
    )
    effects["Red_point"] = np.where(effects["Kruskal_FDR"].lt(0.05), "Yes", "No")
    effects["Red_point_basis"] = "Kruskal–Wallis BH-FDR-adjusted P < 0.05"
    effects[
        [
            "Grouping_Factor",
            "Trait_ID",
            "Trait_Name",
            "Analysis Type",
            "Eta_Squared",
            "ANOVA_p",
            "ANOVA_FDR",
            "Kruskal_p",
            "Kruskal_FDR",
            "Red_point",
            "Red_point_basis",
        ]
    ].to_csv(output_dir / "Supplementary_Fig_S3_effect_sizes.csv", index=False, encoding="utf-8-sig")

    long_frames = []
    for grouping, traits in (
        ("Endosperm Type", ENDOSPERM_TRAITS),
        ("Breeding Type", BREEDING_TRAITS),
    ):
        for trait_id, trait_name, unit, multiplier in traits:
            value_col = f"Trait_{trait_id[1:]}_Predict"
            frame = data[["Cultivar Name", grouping, value_col]].copy()
            frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce") * multiplier
            frame = frame.rename(columns={grouping: "Group", value_col: "Value"})
            frame["Grouping_Factor"] = grouping
            frame["Trait_ID"] = trait_id
            frame["Trait_Name"] = trait_name
            frame["Unit"] = unit
            long_frames.append(frame)
    pd.concat(long_frames, ignore_index=True).to_csv(
        output_dir / "Supplementary_Fig_S3_boxplot_data.csv",
        index=False,
        encoding="utf-8-sig",
    )


def export_red_point_audit(
    endosperm_stats: pd.DataFrame,
    breeding_stats: pd.DataFrame,
    output_dir: Path,
) -> Path:
    audit = pd.concat(
        [
            endosperm_stats.assign(Grouping_Factor="Endosperm type"),
            breeding_stats.assign(Grouping_Factor="Breeding type"),
        ],
        ignore_index=True,
    )
    audit["Red_point"] = np.where(audit["Kruskal_FDR"].lt(0.05), "Yes", "No")
    audit["Criterion"] = "Kruskal–Wallis BH-FDR-adjusted P < 0.05"
    audit = audit[
        [
            "Grouping_Factor", "Trait_ID", "Trait_Name", "Analysis Type", "Eta_Squared",
            "Kruskal_p", "Kruskal_FDR", "Red_point", "Criterion",
        ]
    ].sort_values(["Grouping_Factor", "Eta_Squared", "Trait_ID"], ascending=[True, False, True])
    path = output_dir / "Supplementary_Fig_S3_red_point_audit.csv"
    audit.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_effect_size_panels(
    endosperm_stats: pd.DataFrame,
    breeding_stats: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 5.25), facecolor="white")
    plot_effect_sizes(axes[0], endosperm_stats, "Endosperm-type effect sizes", "A")
    plot_effect_sizes(axes[1], breeding_stats, "Breeding-type effect sizes", "B")
    legend_items = [
        Line2D([0], [0], color=color, lw=7, label=family)
        for family, color in TRAIT_FAMILY_COLORS.items()
    ]
    legend_items.append(
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#C00000",
            markeredgecolor="#C00000", markersize=5,
            label="Kruskal–Wallis BH-FDR < 0.05",
        )
    )
    fig.legend(
        handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=5, frameon=False, fontsize=9, handlelength=1.5, columnspacing=1.6,
    )
    fig.text(
        0.5, 0.005,
        "Bar height: one-way ANOVA η²; red points: Kruskal–Wallis P values after Benjamini–Hochberg FDR correction.",
        ha="center", va="bottom", fontsize=7, color="#4D5558",
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.22, wspace=0.14)
    stem = output_dir / "Supplementary_Fig_S3_effect_sizes_from_Table5_260715"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    workbook = (args.workbook or root / "Supplementary Table1-9.xlsx").resolve()
    stats_dir = (args.stats_dir or root / "outputs" / "category_fingerprint_analysis").resolve()
    output_dir = (args.output_dir or root / "outputs" / "supplementary_figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    endosperm_stats, breeding_stats = load_table5_statistics(workbook)
    audit_path = export_red_point_audit(endosperm_stats, breeding_stats, output_dir)
    expected = {
        "endosperm": {"T2", "T6", "T7", "T30", "T31", "T34", "T36", "T37", "T40", "T41", "T43"},
        "breeding": {"T2", "T7", "T23", "T26", "T27", "T28", "T30", "T31", "T32", "T36", "T37", "T42"},
    }
    observed = {
        "endosperm": set(endosperm_stats.loc[endosperm_stats["Kruskal_FDR"].lt(0.05), "Trait_ID"]),
        "breeding": set(breeding_stats.loc[breeding_stats["Kruskal_FDR"].lt(0.05), "Trait_ID"]),
    }
    if observed != expected:
        raise AssertionError(f"Red-point audit mismatch: observed={observed}, expected={expected}")
    if args.effect_sizes_only:
        png, pdf = save_effect_size_panels(endosperm_stats, breeding_stats, output_dir, args.dpi)
        print(f"PNG: {png}")
        print(f"PDF: {pdf}")
        print(f"Red-point audit: {audit_path}")
        return

    data = load_analysis_data(workbook)
    export_source_tables(data, endosperm_stats, breeding_stats, output_dir)

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(16.2, 10.2), facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        left=0.045,
        right=0.985,
        bottom=0.125,
        top=0.925,
        width_ratios=[1, 1],
        height_ratios=[1.02, 1.0],
        wspace=0.095,
        hspace=0.24,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    plot_effect_sizes(ax_a, endosperm_stats, "Endosperm-type effect sizes", "A")
    plot_effect_sizes(ax_b, breeding_stats, "Breeding-type effect sizes", "B")

    c_grid = outer[1, 0].subgridspec(1, 4, wspace=0.34)
    d_grid = outer[1, 1].subgridspec(1, 5, wspace=0.38)
    c_axes = [fig.add_subplot(c_grid[0, i]) for i in range(4)]
    d_axes = [fig.add_subplot(d_grid[0, i]) for i in range(5)]

    # Reserve a clean header band above the lower-panel trait titles.
    for ax in c_axes + d_axes:
        position = ax.get_position()
        ax.set_position(
            [position.x0, position.y0, position.width, position.height - 0.038]
        )

    for ax, trait in zip(c_axes, ENDOSPERM_TRAITS):
        trait_id, trait_name, unit, multiplier = trait
        plot_boxplot(
            ax,
            data,
            "Endosperm Type",
            ENDOSPERM_ORDER,
            ENDOSPERM_COLORS,
            trait_id,
            trait_name,
            unit,
            multiplier,
            trait_fdr(endosperm_stats, trait_id),
        )

    for ax, trait in zip(d_axes, BREEDING_TRAITS):
        trait_id, trait_name, unit, multiplier = trait
        plot_boxplot(
            ax,
            data,
            "Breeding Type",
            BREEDING_ORDER,
            BREEDING_COLORS,
            trait_id,
            trait_name,
            unit,
            multiplier,
            trait_fdr(breeding_stats, trait_id),
        )

    fig.canvas.draw()
    bbox_a = outer[0, 0].get_position(fig)
    bbox_b = outer[0, 1].get_position(fig)
    bbox_c = outer[1, 0].get_position(fig)
    bbox_d = outer[1, 1].get_position(fig)
    for bbox in (bbox_a, bbox_b, bbox_c, bbox_d):
        add_panel_frame(fig, bbox)
    add_section_title(fig, bbox_c, "Representative endosperm-type distributions")
    add_section_title(fig, bbox_d, "Representative breeding-type distributions")
    add_figure_panel_label(fig, bbox_c, "C")
    add_figure_panel_label(fig, bbox_d, "D")

    legend_items = [
        Line2D([0], [0], color=color, lw=7, label=family)
        for family, color in TRAIT_FAMILY_COLORS.items()
    ]
    legend_items.append(
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#C00000",
            markeredgecolor="#C00000", markersize=4,
            label="Kruskal–Wallis BH-FDR < 0.05",
        )
    )
    fig.legend(
        handles=legend_items,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.018),
        ncol=5,
        frameon=False,
        fontsize=7.2,
        handlelength=1.4,
        columnspacing=1.5,
    )

    fig.suptitle(
        "Supplementary Fig. S3 | Biological discrimination",
        x=0.045,
        y=0.98,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#171C1E",
    )
    fig.text(
        0.985,
        0.022,
        "Effect sizes: one-way ANOVA η²; red points: Kruskal–Wallis P values after Benjamini–Hochberg FDR correction. "
        "Boxplots show automated trait values for 150 cultivars; T37 is displayed in mm.",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#4D5558",
    )

    stem = output_dir / "Supplementary_Fig_S3_biological_discrimination"
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"PNG: {stem.with_suffix('.png')}")
    print(f"PDF: {stem.with_suffix('.pdf')}")
    print(f"Source tables: {output_dir}")


if __name__ == "__main__":
    main()
