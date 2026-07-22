from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch


FAMILY_COLORS = {
    "Color": "#E76F51",
    "Count": "#5AA469",
    "Shape": "#8064A2",
}

TRAIT_FAMILY = {
    "T1": "Shape",
    "T2": "Color",
    "T3": "Color",
    "T6": "Shape",
    "T7": "Shape",
    "T9": "Color",
    "T10": "Color",
    "T33": "Shape",
    "T34": "Count",
    "T40": "Color",
    "T41": "Color",
    "T42": "Shape",
    "T43": "Color",
}

SHORT_NAMES = {
    "T1": "Seedling apex shape",
    "T2": "Sheath anthocyanin",
    "T3": "Foliage green intensity",
    "T6": "Leaf angle",
    "T7": "Leaf curvature",
    "T9": "Silk anthocyanin",
    "T10": "Anther anthocyanin",
    "T23": "Leaf length",
    "T24": "Leaf width",
    "T26": "Upper-ear insertion height",
    "T27": "Plant length",
    "T28": "Ear-height / plant-length ratio",
    "T30": "Ear length",
    "T31": "Ear diameter",
    "T32": "Number of grain rows",
    "T33": "Ear shape",
    "T34": "Number of grain colors",
    "T36": "Grain length",
    "T37": "Grain width",
    "T40": "Grain-apex color",
    "T41": "Dorsal grain color",
    "T42": "Grain shape",
    "T43": "Cob-glume anthocyanin",
}

MEASURED_UNITS = {
    "T23": "cm",
    "T24": "cm",
    "T26": "cm",
    "T27": "cm",
    "T28": "ratio",
    "T30": "cm",
    "T31": "cm",
    "T32": "rows",
    "T36": "cm",
    "T37": "cm",
}

# The mother table stores all physical lengths in their original reporting units.
# No display-only conversion is applied so figure values remain identical to Table S3.
RMSE_MULTIPLIER: dict[str, float] = {}

CONFUSION_ORDER = [
    "T2",
    "T3",
    "T9",
    "T10",
    "T40",
    "T41",
    "T43",
    "T34",
    "T1",
    "T6",
    "T7",
    "T33",
    "T42",
]

ORDINAL_ORDER = [
    "T43",
    "T9",
    "T10",
    "T40",
    "T41",
    "T2",
    "T3",
    "T34",
    "T1",
    "T7",
    "T42",
    "T6",
    "T33",
]


def format_half_up(value: float, decimals: int) -> str:
    quantum = Decimal("1").scaleb(-decimals)
    return str(Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot data-audited method-comparison panels B-D."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def panel_frame(fig: plt.Figure, bbox) -> None:
    patch = FancyBboxPatch(
        (bbox.x0 - 0.008, bbox.y0 - 0.012),
        bbox.width + 0.016,
        bbox.height + 0.024,
        transform=fig.transFigure,
        boxstyle="round,pad=0.003,rounding_size=0.012",
        facecolor="white",
        edgecolor="#D7DCDE",
        linewidth=0.8,
        zorder=-10,
    )
    fig.add_artist(patch)


def panel_label(fig: plt.Figure, bbox, label: str) -> None:
    fig.text(
        bbox.x0 - 0.008,
        bbox.y1 + 0.006,
        label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "circle,pad=0.28", "fc": "#151A1C", "ec": "none"},
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#A9B0B3")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", labelsize=6.5, length=0, colors="#343A3D")
    ax.grid(axis="x", color="#E7EAEC", linewidth=0.65, zorder=0)


def measured_label(trait_id: str) -> str:
    unit = MEASURED_UNITS[trait_id]
    return f"{trait_id}  {SHORT_NAMES[trait_id]} ({unit})"


def plot_measured_panel(fig: plt.Figure, spec, measured: pd.DataFrame) -> list[plt.Axes]:
    panel = spec.subgridspec(2, 3, height_ratios=[0.10, 0.90], wspace=0.23, hspace=0.02)
    title_ax = fig.add_subplot(panel[0, :])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.62,
        "Measured / count traits: agreement metrics",
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="center",
    )
    title_ax.text(1, 0.62, "10 traits; n = 150 each", fontsize=7.2, ha="right", va="center", color="#525A5D")

    frame = measured.sort_values("R2", ascending=True).reset_index(drop=True)
    y = np.arange(len(frame))
    axes = [fig.add_subplot(panel[1, i]) for i in range(3)]
    metrics = [
        ("R2", "R²", "#78A1D4"),
        ("MAPE (%)", "MAPE (%)", "#E9A45D"),
        ("RMSE_display", "RMSE (original units)", "#7CB77B"),
    ]

    for ax, (column, title, color) in zip(axes, metrics):
        values = frame[column].to_numpy(float)
        ax.barh(y, values, color=color, edgecolor="white", linewidth=0.35, height=0.72, zorder=2)
        clean_axis(ax)
        ax.set_title(title, fontsize=8.4, fontweight="bold", color=color, loc="left", pad=7)
        ax.set_yticks(y)
        if column == "R2":
            ax.set_yticklabels([measured_label(t) for t in frame["Trait ID"]], fontsize=6.0)
            ax.set_xlim(0, 1.08)
        else:
            ax.set_yticklabels([])
            ax.set_xlim(left=0)

        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        for idx, (trait_id, value) in enumerate(zip(frame["Trait ID"], values)):
            if column == "R2":
                label = format_half_up(value, 3)
            elif column == "MAPE (%)":
                label = format_half_up(value, 1)
            else:
                unit = MEASURED_UNITS[trait_id]
                decimals = 3 if value < 0.1 else 2
                label = f"{format_half_up(value, decimals)} {unit}"
            ax.text(value + span * 0.012, idx, label, va="center", ha="left", fontsize=5.8, color="#262C2E")

    return [title_ax] + axes


def matrix_for_trait(confusion: pd.DataFrame, trait_id: str) -> tuple[pd.DataFrame, int]:
    sub = confusion[confusion["Trait ID"].eq(trait_id)].copy()
    grades = sorted(set(sub["Manual_code"]).union(sub["Image_code"]))
    counts = (
        sub.pivot_table(index="Manual_code", columns="Image_code", values="Count", aggfunc="sum", fill_value=0)
        .reindex(index=grades, columns=grades, fill_value=0)
        .astype(int)
    )
    totals = counts.sum(axis=1).replace(0, np.nan)
    percentages = counts.div(totals, axis=0).fillna(0) * 100
    return percentages, int(counts.to_numpy().sum())


def plot_confusion_panel(
    fig: plt.Figure,
    spec,
    confusion: pd.DataFrame,
    ordinal: pd.DataFrame,
) -> list[plt.Axes]:
    panel = spec.subgridspec(5, 4, height_ratios=[0.10, 1, 1, 1, 1], wspace=0.48, hspace=0.72)
    title_ax = fig.add_subplot(panel[0, :])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.64,
        "Ordinal traits: confusion matrices (image-derived vs expert reference)",
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="center",
    )
    title_ax.text(1, 0.64, "row-normalized percentages", fontsize=7.2, ha="right", va="center", color="#525A5D")

    cmap = LinearSegmentedColormap.from_list("agreement_blue", ["#FFFFFF", "#DCE8F4", "#78A7D2", "#174A86"])
    axes: list[plt.Axes] = [title_ax]
    summary = ordinal.set_index("Trait ID")
    for index, trait_id in enumerate(CONFUSION_ORDER):
        row, col = divmod(index, 4)
        ax = fig.add_subplot(panel[row + 1, col])
        matrix, n_value = matrix_for_trait(confusion, trait_id)
        expected_n = int(summary.loc[trait_id, "n"])
        if n_value != expected_n:
            raise ValueError(f"Confusion-matrix total mismatch for {trait_id}: {n_value} != {expected_n}")
        labels = np.where(matrix.to_numpy() >= 0.5, np.rint(matrix.to_numpy()).astype(int).astype(str), "")
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=100,
            cbar=False,
            square=True,
            linewidths=0.32,
            linecolor="#E3E7E9",
            annot=labels,
            fmt="",
            annot_kws={"fontsize": 4.4},
        )
        family = TRAIT_FAMILY[trait_id]
        ax.set_title(
            f"{trait_id}  {SHORT_NAMES[trait_id]}\n(n={n_value})",
            color=FAMILY_COLORS[family],
            fontsize=6.3,
            fontweight="bold",
            pad=4,
        )
        ax.set_xlabel("Image-derived", fontsize=5.2, labelpad=1)
        ax.set_ylabel("Expert", fontsize=5.2, labelpad=1)
        ax.tick_params(axis="both", labelsize=4.5, rotation=0, length=0)
        axes.append(ax)

    colorbar_host = fig.add_subplot(panel[4, 1:])
    colorbar_host.axis("off")
    colorbar_ax = colorbar_host.inset_axes([0.14, 0.44, 0.72, 0.13])
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=100), cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=[0, 25, 50, 75, 100],
    )
    colorbar.ax.tick_params(labelsize=5.5, length=2)
    colorbar.set_label("Row percentage (%)", fontsize=6, labelpad=2)
    colorbar.outline.set_linewidth(0.5)
    axes.append(colorbar_host)
    return axes


def plot_ordinal_panel(fig: plt.Figure, spec, ordinal: pd.DataFrame) -> list[plt.Axes]:
    panel = spec.subgridspec(2, 3, height_ratios=[0.10, 0.90], wspace=0.23, hspace=0.02)
    title_ax = fig.add_subplot(panel[0, :])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.62,
        "Ordinal traits: agreement metrics",
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="center",
    )
    title_ax.text(1, 0.62, "trait-specific valid n", fontsize=7.2, ha="right", va="center", color="#525A5D")

    frame = ordinal.set_index("Trait ID").loc[ORDINAL_ORDER].reset_index()
    frame = frame.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(frame))
    axes = [fig.add_subplot(panel[1, i]) for i in range(3)]
    metrics = [
        ("weighted kappa", "Weighted κ", "#9B83B7", 1.0),
        ("Spearman rho", "Spearman's ρ", "#61A6A6", 1.0),
        ("exact agreement (%)", "Exact agreement (%)", "#75B876", 100.0),
    ]

    for ax, (column, title, color, upper) in zip(axes, metrics):
        values = frame[column].to_numpy(float)
        bar_colors = [FAMILY_COLORS[TRAIT_FAMILY[t]] for t in frame["Trait ID"]]
        ax.barh(y, values, color=bar_colors, alpha=0.88, edgecolor="white", linewidth=0.35, height=0.72, zorder=2)
        clean_axis(ax)
        ax.axvline(0, color="#7F878A", linewidth=0.65)
        ax.set_title(title, fontsize=8.4, fontweight="bold", color=color, loc="left", pad=7)
        ax.set_yticks(y)
        if column == "weighted kappa":
            labels = [f"{t}  {SHORT_NAMES[t]} (n={int(n)})" for t, n in zip(frame["Trait ID"], frame["n"])]
            ax.set_yticklabels(labels, fontsize=5.65)
        else:
            ax.set_yticklabels([])
        ax.set_xlim(-0.08 if column == "Spearman rho" else 0, upper * 1.12)
        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        for idx, value in enumerate(values):
            label = format_half_up(value, 3) if upper == 1.0 else format_half_up(value, 1)
            ax.text(value + span * 0.012, idx, label, va="center", ha="left", fontsize=5.6, color="#262C2E")

    return [title_ax] + axes


def bounding_box(axes: list[plt.Axes], fig: plt.Figure):
    boxes = [ax.get_position(fig) for ax in axes]
    return boxes[0].union(boxes)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    input_dir = (args.input_dir or root / "outputs" / "method_comparison_analysis").resolve()
    output_dir = (args.output_dir or root / "outputs" / "supplementary_figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    measured = pd.read_csv(input_dir / "measured_traits_agreement_summary.csv")
    ordinal = pd.read_csv(input_dir / "visual_traits_agreement_summary.csv")
    confusion = pd.read_csv(input_dir / "visual_traits_confusion_matrices_long.csv")

    if len(measured) != 10 or not measured["n"].eq(150).all():
        raise ValueError("Expected 10 measured/count traits with n=150 each.")
    if set(ordinal["Trait ID"]) != set(CONFUSION_ORDER):
        raise ValueError("Ordinal trait set does not match the expected 13 traits.")
    confusion_totals = confusion.groupby("Trait ID")["Count"].sum().sort_index()
    ordinal_totals = ordinal.set_index("Trait ID")["n"].sort_index()
    if not confusion_totals.equals(ordinal_totals):
        raise ValueError("Confusion-matrix totals do not match trait-specific valid sample sizes.")

    measured = measured.copy()
    measured["RMSE_display"] = [
        value * RMSE_MULTIPLIER.get(trait_id, 1.0)
        for trait_id, value in zip(measured["Trait ID"], measured["RMSE"])
    ]

    measured.to_csv(output_dir / "Method_comparison_panel_B_source.csv", index=False, encoding="utf-8-sig")
    ordinal.to_csv(output_dir / "Method_comparison_panel_D_source.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(output_dir / "Method_comparison_panel_C_source.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(18.2, 11.2), facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.988,
        bottom=0.065,
        top=0.955,
        width_ratios=[0.83, 1.17],
        height_ratios=[0.48, 0.52],
        wspace=0.075,
        hspace=0.10,
    )

    axes_b = plot_measured_panel(fig, outer[0, 0], measured)
    axes_d = plot_ordinal_panel(fig, outer[1, 0], ordinal)
    axes_c = plot_confusion_panel(fig, outer[:, 1], confusion, ordinal)

    fig.canvas.draw()
    bbox_b = outer[0, 0].get_position(fig)
    bbox_d = outer[1, 0].get_position(fig)
    bbox_c = outer[:, 1].get_position(fig)
    for bbox, label in [(bbox_b, "B"), (bbox_c, "C"), (bbox_d, "D")]:
        panel_frame(fig, bbox)
        panel_label(fig, bbox, label)

    legend_y = 0.024
    x = 0.058
    for family in ["Color", "Count", "Shape"]:
        fig.text(
            x,
            legend_y,
            "■",
            color=FAMILY_COLORS[family],
            fontsize=11,
            va="center",
            ha="left",
        )
        fig.text(x + 0.014, legend_y, family, fontsize=7, va="center", ha="left", color="#363D40")
        x += 0.065

    fig.text(
        0.988,
        legend_y,
        "R²: coefficient of determination; RMSE: root mean square error; MAPE: mean absolute percentage error; "
        "κ: weighted kappa; ρ: Spearman correlation. Confusion matrices are normalized within expert-reference grades.",
        ha="right",
        va="center",
        fontsize=6.4,
        color="#4F575A",
    )

    stem = output_dir / "Supplementary_method_comparison_panels_BCD_real_data"
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"PNG: {stem.with_suffix('.png')}")
    print(f"PDF: {stem.with_suffix('.pdf')}")
    print(f"Panel B traits: {len(measured)}; all n=150")
    print(f"Panel C/D traits: {len(ordinal)}; n range={int(ordinal['n'].min())}-{int(ordinal['n'].max())}")
    print("Confusion totals verified against ordinal summary.")


if __name__ == "__main__":
    main()
