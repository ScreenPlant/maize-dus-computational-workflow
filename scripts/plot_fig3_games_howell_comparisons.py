from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


GROUP_ORDER = [
    "Normal type",
    "Sweet type",
    "Waxy type",
    "Sweet-waxy hybrid type",
    "Popcorn type",
]
GROUP_SHORT = {
    "Normal type": "Normal",
    "Sweet type": "Sweet",
    "Waxy type": "Waxy",
    "Sweet-waxy hybrid type": "Sweet-waxy",
    "Popcorn type": "Popcorn",
}
TRAIT_ORDER = ["T30", "T31", "T36", "T37"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_dir = root / "outputs" / "fig3_games_howell_basis_260715_v3"
    parser = argparse.ArgumentParser(description="Plot all 40 Games–Howell comparisons underlying Fig. 3A.")
    parser.add_argument(
        "--comparisons",
        type=Path,
        default=default_dir / "Fig3_Games_Howell_40_pairwise_comparisons.csv",
    )
    parser.add_argument(
        "--letters",
        type=Path,
        default=default_dir / "Fig3_compact_letter_display.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def format_p(value: float) -> str:
    if value < 0.0001:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def trait_matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p_matrix = np.full((5, 5), np.nan, dtype=float)
    annotations = np.full((5, 5), "", dtype=object)
    group_index = {group: index for index, group in enumerate(GROUP_ORDER)}
    for _, row in frame.iterrows():
        i = group_index[str(row["Group 1"])]
        j = group_index[str(row["Group 2"])]
        value = float(row["Games-Howell adjusted P"])
        # Put each of the 10 unique comparisons in the lower triangle.
        lower_row, lower_col = max(i, j), min(i, j)
        p_matrix[lower_row, lower_col] = value
        marker = " *" if value < 0.05 else ""
        annotations[lower_row, lower_col] = f"P={format_p(value)}{marker}"
    return p_matrix, annotations


def main() -> None:
    args = parse_args()
    comparisons = pd.read_csv(args.comparisons, encoding="utf-8-sig")
    letters = pd.read_csv(args.letters, encoding="utf-8-sig")
    if len(comparisons) != 40 or len(letters) != 20:
        raise ValueError("Expected 40 pairwise comparisons and 20 compact-letter rows.")

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.2), facecolor="white")
    axes = axes.ravel()
    image = None
    for panel_index, (ax, trait_id) in enumerate(zip(axes, TRAIT_ORDER)):
        frame = comparisons.loc[comparisons["Trait ID"].eq(trait_id)].copy()
        if len(frame) != 10:
            raise ValueError(f"{trait_id}: expected 10 comparisons, found {len(frame)}")
        p_matrix, annotations = trait_matrices(frame)
        strength = -np.log10(np.clip(p_matrix, 1e-10, 1.0))
        mask = np.triu(np.ones_like(strength, dtype=bool), k=0) | np.isnan(strength)
        trait_letters = (
            letters.loc[letters["Trait ID"].eq(trait_id)]
            .set_index("Endosperm type")["Games-Howell letter"]
            .to_dict()
        )
        labels = [f"{GROUP_SHORT[group]} ({trait_letters[group]})" for group in GROUP_ORDER]
        sns.heatmap(
            strength,
            mask=mask,
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=10,
            annot=annotations,
            fmt="",
            square=True,
            linewidths=1.0,
            linecolor="#D5D9DD",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            annot_kws={"fontsize": 8.2, "fontweight": "bold"},
        )
        image = ax.collections[0]
        trait_name = str(frame["Trait name"].iloc[0])
        ax.set_title(
            f"{chr(65 + panel_index)}  {trait_id} {trait_name}",
            loc="left",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35, labelsize=8.5)
        ax.tick_params(axis="y", rotation=0, labelsize=8.5)

    if image is None:
        raise RuntimeError("No heatmap was drawn")
    colorbar_axis = fig.add_axes([0.925, 0.23, 0.016, 0.55])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("−log₁₀(Games–Howell adjusted P)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Games–Howell pairwise comparisons underlying the Fig. 3A compact letter displays",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.018,
        "Each lower-triangle cell is one endosperm-type comparison. * Games–Howell studentized-range–adjusted "
        "P < 0.05 within the trait.\nLetters in parentheses are the Fig. 3 compact-letter display; groups sharing "
        "at least one letter were not significantly different.",
        fontsize=8.5,
        ha="left",
    )
    fig.subplots_adjust(left=0.06, right=0.90, top=0.92, bottom=0.15, wspace=0.20, hspace=0.28)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Fig3_Games_Howell_pairwise_comparison_heatmap"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
