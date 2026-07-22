from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress, spearmanr
from sklearn.metrics import cohen_kappa_score


FIT_TRAITS = ["T30", "T31", "T36", "T37"]
CONFUSION_TRAITS = ["T43", "T40", "T41", "T42"]

TRAIT_NAMES = {
    "T30": "Ear length",
    "T31": "Ear diameter",
    "T36": "Grain length",
    "T37": "Grain width",
    "T40": "Color of grain apex",
    "T41": "Dorsal grain color",
    "T42": "Grain shape",
    "T43": "Cob-glume anthocyanin",
}

TRAIT_FAMILIES = {
    "T30": "Size",
    "T31": "Size",
    "T36": "Size",
    "T37": "Size",
    "T40": "Color",
    "T41": "Color",
    "T42": "Shape",
    "T43": "Color",
}

FAMILY_COLORS = {"Size": "#2F6FB0", "Color": "#E76F51", "Shape": "#8064A2"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot manuscript-ready representative agreement panels from raw paired trait data."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--effect-stats", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=500)
    return parser.parse_args()


def paired_values(table3: pd.DataFrame, trait_id: str) -> pd.DataFrame:
    trait_no = int(trait_id[1:])
    actual_col = f"Trait_{trait_no}_Actual"
    predicted_col = f"Trait_{trait_no}_Predict"
    pair = pd.DataFrame(
        {
            "Serial No.": table3["Serial No."],
            "Field Plot No.": table3["Field Plot No."],
            "Cultivar Name": table3["Cultivar Name"],
            "Endosperm Type": table3["Endosperm Type"],
            "Expert reference": pd.to_numeric(table3[actual_col], errors="coerce"),
            "Image-derived": pd.to_numeric(table3[predicted_col], errors="coerce"),
        }
    )
    pair = pair.dropna(subset=["Expert reference", "Image-derived"]).copy()
    pair.insert(4, "Trait ID", trait_id)
    pair.insert(5, "Trait name", TRAIT_NAMES[trait_id])
    pair.insert(6, "Trait family", TRAIT_FAMILIES[trait_id])
    return pair


def measured_metrics(pair: pd.DataFrame) -> dict[str, float]:
    x = pair["Expert reference"].to_numpy(float)
    y = pair["Image-derived"].to_numpy(float)
    fit = linregress(x, y)
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    mape = float(np.mean(np.abs((y - x) / x)) * 100.0)
    return {
        "n": len(pair),
        "R2": float(fit.rvalue**2),
        "RMSE": rmse,
        "MAPE (%)": mape,
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
    }


def ordinal_metrics(pair: pd.DataFrame) -> dict[str, float]:
    actual = pair["Expert reference"].astype(int)
    predicted = pair["Image-derived"].astype(int)
    return {
        "n": len(pair),
        "Spearman rho": float(spearmanr(actual, predicted).statistic),
        "Weighted kappa": float(cohen_kappa_score(actual, predicted, weights="quadratic")),
        "Exact agreement (%)": float((actual.to_numpy() == predicted.to_numpy()).mean() * 100.0),
    }


def confusion_long(pair: pd.DataFrame, trait_id: str) -> pd.DataFrame:
    actual = pair["Expert reference"].astype(int)
    predicted = pair["Image-derived"].astype(int)
    actual_states = sorted(actual.unique().tolist())
    predicted_states = sorted(predicted.unique().tolist())
    counts = pd.crosstab(actual, predicted).reindex(
        index=actual_states,
        columns=predicted_states,
        fill_value=0,
    )
    rows: list[dict[str, object]] = []
    for actual_code in actual_states:
        row_total = int(counts.loc[actual_code].sum())
        for predicted_code in predicted_states:
            count = int(counts.loc[actual_code, predicted_code])
            rows.append(
                {
                    "Trait ID": trait_id,
                    "Trait name": TRAIT_NAMES[trait_id],
                    "Expert reference code": actual_code,
                    "Image-derived code": predicted_code,
                    "Count": count,
                    "Reference-grade total": row_total,
                    "Row percentage (%)": count / row_total * 100.0 if row_total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def selection_table(effect_stats: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_index = metrics.set_index("Trait ID")
    for trait_id in FIT_TRAITS + CONFUSION_TRAITS:
        grouping = "Breeding Type" if trait_id == "T42" else "Endosperm Type"
        effect = effect_stats[
            effect_stats["Trait_ID"].eq(trait_id) & effect_stats["Grouping_Factor"].eq(grouping)
        ].iloc[0]
        metric = metric_index.loc[trait_id]
        rows.append(
            {
                "Panel type": "Fit" if trait_id in FIT_TRAITS else "Confusion matrix",
                "Trait ID": trait_id,
                "Trait name": TRAIT_NAMES[trait_id],
                "Trait family": TRAIT_FAMILIES[trait_id],
                "Biological grouping used for selection": grouping,
                "Eta-squared": float(effect["Eta_Squared"]),
                "ANOVA FDR": float(effect["ANOVA_FDR"]),
                "Valid n": int(metric["n"]),
                "R2": metric.get("R2", np.nan),
                "MAPE (%)": metric.get("MAPE (%)", np.nan),
                "Spearman rho": metric.get("Spearman rho", np.nan),
                "Weighted kappa": metric.get("Weighted kappa", np.nan),
                "Selection rationale": (
                    "Complete measured trait with significant endosperm-type differentiation"
                    if trait_id in FIT_TRAITS
                    else "Ordinal trait spanning high-to-moderate agreement without low-variation exclusion"
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_fit(ax: plt.Axes, pair: pd.DataFrame, trait_id: str, metric: pd.Series) -> None:
    x = pair["Expert reference"].to_numpy(float)
    y = pair["Image-derived"].to_numpy(float)
    values = np.concatenate([x, y])
    low = float(values.min())
    high = float(values.max())
    margin = max((high - low) * 0.07, 0.02)
    low -= margin
    high += margin

    ax.scatter(x, y, s=13, color="#2F78C4", alpha=0.72, edgecolors="white", linewidths=0.25, zorder=2)
    ax.plot([low, high], [low, high], color="#737B7E", linestyle=(0, (4, 3)), linewidth=0.8, label="1:1")
    fit_x = np.array([low, high])
    fit_y = float(metric["slope"]) * fit_x + float(metric["intercept"])
    ax.plot(fit_x, fit_y, color="#D95F32", linewidth=1.15, label="Linear fit")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=6, length=2)
    ax.set_title(f"{trait_id}  {TRAIT_NAMES[trait_id]} (cm)", fontsize=8, fontweight="bold", color="#235B94")
    ax.text(
        0.045,
        0.95,
        f"R² = {float(metric['R2']):.4f}\nRMSE = {float(metric['RMSE']):.4f} cm\nMAPE = {float(metric['MAPE (%)']):.4f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        linespacing=1.25,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#B4B9BC")


def plot_confusion(
    ax: plt.Axes,
    confusion: pd.DataFrame,
    trait_id: str,
    metric: pd.Series,
    cmap,
) -> None:
    sub = confusion[confusion["Trait ID"].eq(trait_id)]
    matrix = sub.pivot(
        index="Expert reference code",
        columns="Image-derived code",
        values="Row percentage (%)",
    )
    rounded = np.rint(matrix.to_numpy()).astype(int)
    annotations = np.where(matrix.to_numpy() >= 0.5, rounded.astype(str), "")
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=100,
        cbar=False,
        square=True,
        linewidths=0.38,
        linecolor="#E4E8EA",
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 5.2},
    )
    ax.set_title(
        f"{trait_id}  {TRAIT_NAMES[trait_id]}\nρ = {float(metric['Spearman rho']):.4f}; κw = {float(metric['Weighted kappa']):.4f}",
        fontsize=7.5,
        fontweight="bold",
        color=FAMILY_COLORS[TRAIT_FAMILIES[trait_id]],
        pad=5,
    )
    ax.set_xlabel("Image-derived code", fontsize=6.5)
    ax.set_ylabel("Expert-reference code", fontsize=6.5)
    ax.tick_params(axis="both", labelsize=5.5, length=0, rotation=0)


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
    effect_stats = pd.read_csv(effect_path)
    if len(table3) != 150:
        raise ValueError(f"Expected 150 varieties, found {len(table3)}.")

    pair_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    confusion_frames: list[pd.DataFrame] = []
    for trait_id in FIT_TRAITS:
        pair = paired_values(table3, trait_id)
        metric_rows.append({"Trait ID": trait_id, **measured_metrics(pair)})
        pair_frames.append(pair)
    for trait_id in CONFUSION_TRAITS:
        pair = paired_values(table3, trait_id)
        metric_rows.append({"Trait ID": trait_id, **ordinal_metrics(pair)})
        pair_frames.append(pair)
        confusion_frames.append(confusion_long(pair, trait_id))

    pairs = pd.concat(pair_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    confusion = pd.concat(confusion_frames, ignore_index=True)
    selection = selection_table(effect_stats, metrics)

    if not metrics["n"].eq(150).all():
        raise ValueError("All selected main-figure traits must have 150 valid paired observations.")
    if int(confusion.groupby("Trait ID")["Count"].sum().min()) != 150:
        raise ValueError("A representative confusion matrix does not contain 150 observations.")

    pairs.to_csv(output_dir / "Fig3_representative_trait_paired_source.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "Fig3_representative_trait_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(output_dir / "Fig3_representative_confusion_source.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(output_dir / "Fig3_representative_trait_selection_rationale.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    cmap = LinearSegmentedColormap.from_list("agreement", ["#FFFFFF", "#DCE9F5", "#79A9D4", "#174A86"])

    fig = plt.figure(figsize=(14.8, 8.6), facecolor="white")
    outer = fig.add_gridspec(2, 1, left=0.065, right=0.985, bottom=0.105, top=0.84, hspace=0.38)
    fit_grid = outer[0].subgridspec(1, 4, wspace=0.34)
    confusion_grid = outer[1].subgridspec(1, 4, wspace=0.42)
    metric_index = metrics.set_index("Trait ID")

    for index, trait_id in enumerate(FIT_TRAITS):
        plot_fit(
            fig.add_subplot(fit_grid[0, index]),
            pairs[pairs["Trait ID"].eq(trait_id)],
            trait_id,
            metric_index.loc[trait_id],
        )
    for index, trait_id in enumerate(CONFUSION_TRAITS):
        plot_confusion(
            fig.add_subplot(confusion_grid[0, index]),
            confusion,
            trait_id,
            metric_index.loc[trait_id],
            cmap,
        )

    fig.suptitle("Representative trait-extraction agreement", x=0.065, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.047, 0.875, "A", color="white", fontsize=9, fontweight="bold", ha="center", va="center", bbox={"boxstyle": "circle,pad=0.25", "fc": "#151A1C", "ec": "none"})
    fig.text(0.067, 0.875, "Measured traits: image-derived vs expert reference", fontsize=10, fontweight="bold", va="center")
    fig.text(0.047, 0.485, "B", color="white", fontsize=9, fontweight="bold", ha="center", va="center", bbox={"boxstyle": "circle,pad=0.25", "fc": "#151A1C", "ec": "none"})
    fig.text(0.067, 0.485, "Representative ordinal traits: row-normalized confusion matrices", fontsize=10, fontweight="bold", va="center")
    fig.text(0.065, 0.055, "Dashed line: 1:1 agreement; orange line: fitted regression. Matrix values are percentages within expert-reference grades.", fontsize=7, color="#4F575A")

    colorbar_ax = fig.add_axes([0.70, 0.035, 0.25, 0.014])
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 100), cmap=cmap), cax=colorbar_ax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    colorbar.ax.tick_params(labelsize=6, length=2)
    colorbar.set_label("Row percentage (%)", fontsize=6.5, labelpad=2)
    colorbar.outline.set_linewidth(0.5)

    stem = output_dir / "Fig3_representative_agreement_real_data"
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"PNG: {stem.with_suffix('.png')}")
    print(f"PDF: {stem.with_suffix('.pdf')}")
    print(f"Paired source rows: {len(pairs)}")
    print("Selected fit traits:", ", ".join(FIT_TRAITS))
    print("Selected confusion traits:", ", ".join(CONFUSION_TRAITS))


if __name__ == "__main__":
    main()
