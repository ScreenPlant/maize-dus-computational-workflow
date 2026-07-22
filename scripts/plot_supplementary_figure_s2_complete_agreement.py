from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress, spearmanr
from sklearn.metrics import cohen_kappa_score

import plot_method_comparison_panels_bcd as agreement_panels


MEASURED_ORDER = ["T23", "T24", "T26", "T27", "T28", "T30", "T31", "T32", "T36", "T37"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot Supplementary Fig. 2 from paired expert-reference and image-derived trait data."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument(
        "--summary-workbook",
        type=Path,
        default=None,
        help="Optional finalized supplementary workbook whose Table S3 values define the displayed publication metrics.",
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def prepare_measured_pairs(
    table3: pd.DataFrame,
    measured: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    cultivar_col = "Cultivar Name"
    if cultivar_col not in table3.columns:
        raise ValueError(f"Missing required column: {cultivar_col}")

    summary = measured.set_index("Trait ID")
    if set(summary.index) != set(MEASURED_ORDER):
        raise ValueError("Measured-trait summary does not contain the expected 10 traits.")

    for trait_id in MEASURED_ORDER:
        trait_no = int(trait_id[1:])
        actual_col = f"Trait_{trait_no}_Actual"
        predict_col = f"Trait_{trait_no}_Predict"
        if actual_col not in table3.columns or predict_col not in table3.columns:
            raise ValueError(f"Missing paired columns for {trait_id}.")

        pair = pd.DataFrame(
            {
                "Cultivar Name": table3[cultivar_col],
                "Manual reference (stored units)": pd.to_numeric(table3[actual_col], errors="coerce"),
                "Image-derived (stored units)": pd.to_numeric(table3[predict_col], errors="coerce"),
            }
        ).dropna(subset=["Manual reference (stored units)", "Image-derived (stored units)"])

        expected_n = int(summary.loc[trait_id, "n"])
        if len(pair) != expected_n:
            raise ValueError(f"Paired sample count mismatch for {trait_id}: {len(pair)} != {expected_n}")

        multiplier = agreement_panels.RMSE_MULTIPLIER.get(trait_id, 1.0)
        unit = agreement_panels.MEASURED_UNITS[trait_id]
        pair.insert(1, "Trait ID", trait_id)
        pair.insert(2, "Trait name", agreement_panels.SHORT_NAMES[trait_id])
        pair.insert(3, "Display unit", unit)
        pair["Manual reference"] = pair["Manual reference (stored units)"] * multiplier
        pair["Image-derived"] = pair["Image-derived (stored units)"] * multiplier
        rows.append(pair)

    return pd.concat(rows, ignore_index=True)


def verify_measured_metrics(pairs: pd.DataFrame, measured: pd.DataFrame) -> None:
    summary = measured.set_index("Trait ID")
    tolerance = {"R2": 5e-4, "RMSE": 5e-4, "MAPE (%)": 5e-4}

    for trait_id, trait_pairs in pairs.groupby("Trait ID", sort=False):
        x = trait_pairs["Manual reference (stored units)"].to_numpy(float)
        y = trait_pairs["Image-derived (stored units)"].to_numpy(float)
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        threshold = max(1e-8, 0.01 * float(np.mean(np.abs(x))))
        safe = np.abs(x) > threshold
        mape = float(np.mean(np.abs((y[safe] - x[safe]) / x[safe])) * 100.0)
        recomputed = {"R2": r2, "RMSE": rmse, "MAPE (%)": mape}

        for metric, value in recomputed.items():
            recorded = float(summary.loc[trait_id, metric])
            if not np.isclose(value, recorded, atol=tolerance[metric], rtol=0):
                raise ValueError(
                    f"Metric mismatch for {trait_id} {metric}: recomputed={value:.6f}, recorded={recorded:.6f}"
                )


def recompute_measured_metrics(pairs: pd.DataFrame, measured: pd.DataFrame) -> pd.DataFrame:
    exact = measured.copy().set_index("Trait ID")
    for trait_id, trait_pairs in pairs.groupby("Trait ID", sort=False):
        x = trait_pairs["Manual reference (stored units)"].to_numpy(float)
        y = trait_pairs["Image-derived (stored units)"].to_numpy(float)
        fit = linregress(x, y)
        threshold = max(1e-8, 0.01 * float(np.mean(np.abs(x))))
        safe = np.abs(x) > threshold
        exact.loc[trait_id, "R2"] = float(fit.rvalue**2)
        exact.loc[trait_id, "RMSE"] = float(np.sqrt(np.mean((y - x) ** 2)))
        exact.loc[trait_id, "MAPE (%)"] = float(np.mean(np.abs((y[safe] - x[safe]) / x[safe])) * 100.0)
        exact.loc[trait_id, "n"] = len(x)
    return exact.reset_index()


def recompute_ordinal_metrics_and_confusion(
    table3: pd.DataFrame,
    ordinal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = ordinal.copy().set_index("Trait ID")
    confusion_rows: list[dict[str, object]] = []

    for trait_id in agreement_panels.CONFUSION_ORDER:
        trait_no = int(trait_id[1:])
        actual = pd.to_numeric(table3[f"Trait_{trait_no}_Actual"], errors="coerce")
        predicted = pd.to_numeric(table3[f"Trait_{trait_no}_Predict"], errors="coerce")
        valid = actual.notna() & predicted.notna()
        actual_pair = actual[valid].astype(int)
        predicted_pair = predicted[valid].astype(int)
        n = len(actual_pair)

        exact.loc[trait_id, "n"] = n
        exact.loc[trait_id, "Spearman rho"] = float(spearmanr(actual_pair, predicted_pair).statistic)
        exact.loc[trait_id, "weighted kappa"] = float(
            cohen_kappa_score(actual_pair, predicted_pair, weights="quadratic")
        )
        exact.loc[trait_id, "exact agreement (%)"] = float(
            (actual_pair.to_numpy() == predicted_pair.to_numpy()).mean() * 100.0
        )

        states = sorted(set(actual_pair.tolist()).union(predicted_pair.tolist()))
        counts = pd.crosstab(actual_pair, predicted_pair).reindex(index=states, columns=states, fill_value=0)
        for actual_code in states:
            for predicted_code in states:
                confusion_rows.append(
                    {
                        "Trait ID": trait_id,
                        "Trait name": agreement_panels.SHORT_NAMES[trait_id],
                        "Manual_code": int(actual_code),
                        "Image_code": int(predicted_code),
                        "Count": int(counts.loc[actual_code, predicted_code]),
                    }
                )

    return exact.reset_index(), pd.DataFrame(confusion_rows)


def apply_published_s3_metrics(
    summary_workbook: Path,
    measured_exact: pd.DataFrame,
    ordinal_exact: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    published = pd.read_excel(summary_workbook, sheet_name="Supplementary Table 3", header=1)
    published = published[published["Trait code"].astype(str).str.fullmatch(r"T\d+")].copy()
    published = published.set_index("Trait code")

    measured_plot = measured_exact.copy().set_index("Trait ID")
    ordinal_plot = ordinal_exact.copy().set_index("Trait ID")
    audit_rows: list[dict[str, object]] = []

    for trait_id, exact_row in measured_plot.iterrows():
        if trait_id not in published.index:
            raise ValueError(f"Finalized Table S3 is missing {trait_id}.")
        report_row = published.loc[trait_id]
        if int(report_row["Valid n"]) != int(exact_row["n"]):
            raise ValueError(f"Finalized Table S3 valid n differs for {trait_id}.")
        metric_map = {"R2": "R2", "RMSE": "RMSE", "MAPE (%)": "MAPE\n(%)"}
        for exact_column, published_column in metric_map.items():
            exact_value = float(exact_row[exact_column])
            published_value = float(report_row[published_column])
            measured_plot.loc[trait_id, exact_column] = published_value
            audit_rows.append(
                {
                    "Trait ID": trait_id,
                    "Metric": exact_column,
                    "Recomputed from mother table": exact_value,
                    "Reported in finalized Table S3": published_value,
                    "Difference (reported - recomputed)": published_value - exact_value,
                }
            )

    ordinal_metric_map = {
        "Spearman rho": "Spearman's ρ",
        "weighted kappa": "Weighted κw",
        "exact agreement (%)": "Exact agreement\n(%)",
    }
    for trait_id, exact_row in ordinal_plot.iterrows():
        if trait_id not in published.index:
            raise ValueError(f"Finalized Table S3 is missing {trait_id}.")
        report_row = published.loc[trait_id]
        if int(report_row["Valid n"]) != int(exact_row["n"]):
            raise ValueError(f"Finalized Table S3 valid n differs for {trait_id}.")
        for exact_column, published_column in ordinal_metric_map.items():
            exact_value = float(exact_row[exact_column])
            published_value = float(report_row[published_column])
            ordinal_plot.loc[trait_id, exact_column] = published_value
            audit_rows.append(
                {
                    "Trait ID": trait_id,
                    "Metric": exact_column,
                    "Recomputed from mother table": exact_value,
                    "Reported in finalized Table S3": published_value,
                    "Difference (reported - recomputed)": published_value - exact_value,
                }
            )

    return measured_plot.reset_index(), ordinal_plot.reset_index(), pd.DataFrame(audit_rows)


def write_alignment_audit(
    output_path: Path,
    measured_exact: pd.DataFrame,
    ordinal_exact: pd.DataFrame,
    metric_audit: pd.DataFrame,
) -> None:
    rows: list[dict[str, object]] = []
    measured_index = measured_exact.set_index("Trait ID")
    ordinal_index = ordinal_exact.set_index("Trait ID")
    all_traits = sorted(
        set(measured_index.index).union(ordinal_index.index),
        key=lambda trait_id: int(trait_id[1:]),
    )
    for trait_id in all_traits:
        trait_no = int(trait_id[1:])
        common = {
            "Trait ID": trait_id,
            "Trait name": agreement_panels.SHORT_NAMES[trait_id],
            "Actual source column": f"Trait_{trait_no}_Actual",
            "Image-derived source column": f"Trait_{trait_no}_Predict",
        }
        if trait_id in measured_index.index:
            row = measured_index.loc[trait_id]
            common.update(
                {
                    "Trait type": "measured/count",
                    "Valid n": int(row["n"]),
                    "R2": float(row["R2"]),
                    "RMSE (original units)": float(row["RMSE"]),
                    "MAPE (%)": float(row["MAPE (%)"]),
                    "Spearman rho": np.nan,
                    "Weighted kappa": np.nan,
                    "Exact agreement (%)": np.nan,
                }
            )
        else:
            row = ordinal_index.loc[trait_id]
            common.update(
                {
                    "Trait type": "ordinal",
                    "Valid n": int(row["n"]),
                    "R2": np.nan,
                    "RMSE (original units)": np.nan,
                    "MAPE (%)": np.nan,
                    "Spearman rho": float(row["Spearman rho"]),
                    "Weighted kappa": float(row["weighted kappa"]),
                    "Exact agreement (%)": float(row["exact agreement (%)"]),
                }
            )
        rows.append(common)
    alignment = pd.DataFrame(rows)
    metric_wide = metric_audit.pivot(
        index="Trait ID",
        columns="Metric",
        values=[
            "Recomputed from mother table",
            "Reported in finalized Table S3",
            "Difference (reported - recomputed)",
        ],
    )
    metric_wide.columns = [f"{metric} | {source}" for source, metric in metric_wide.columns]
    alignment = alignment.merge(metric_wide.reset_index(), on="Trait ID", how="left")
    alignment.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_scatter_panel(
    fig: plt.Figure,
    spec,
    pairs: pd.DataFrame,
    measured: pd.DataFrame,
) -> list[plt.Axes]:
    panel = spec.subgridspec(3, 5, height_ratios=[0.14, 1, 1], wspace=0.28, hspace=0.43)
    title_ax = fig.add_subplot(panel[0, :])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.68,
        "Measured / count traits: expert reference vs image-derived",
        fontsize=11.2,
        fontweight="bold",
        ha="left",
        va="center",
    )
    title_ax.text(
        1,
        0.68,
        "10 traits; 150 paired observations per trait",
        fontsize=7.2,
        ha="right",
        va="center",
        color="#525A5D",
    )

    summary = measured.set_index("Trait ID")
    axes: list[plt.Axes] = [title_ax]
    for index, trait_id in enumerate(MEASURED_ORDER):
        row, col = divmod(index, 5)
        ax = fig.add_subplot(panel[row + 1, col])
        sub = pairs[pairs["Trait ID"].eq(trait_id)]
        x = sub["Manual reference"].to_numpy(float)
        y = sub["Image-derived"].to_numpy(float)

        values = np.concatenate([x, y])
        low = min(0.0, float(np.nanmin(values)))
        high = float(np.nanmax(values))
        span = high - low
        margin = max(span * 0.055, 0.02)
        low_limit = low if low == 0 else low - margin
        high_limit = high + margin

        ax.scatter(
            x,
            y,
            s=9,
            facecolors="white",
            edgecolors="#2E6DB4",
            linewidths=0.55,
            alpha=0.82,
            zorder=2,
        )
        ax.plot(
            [low_limit, high_limit],
            [low_limit, high_limit],
            linestyle=(0, (4, 3)),
            color="#202628",
            linewidth=0.8,
            zorder=1,
        )
        ax.set_xlim(low_limit, high_limit)
        ax.set_ylim(low_limit, high_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="both", labelsize=5.5, length=2, width=0.55, pad=1.5)
        for spine in ax.spines.values():
            spine.set_color("#AEB5B8")
            spine.set_linewidth(0.6)

        unit = agreement_panels.MEASURED_UNITS[trait_id]
        unit_text = "" if unit == "ratio" else f" ({unit})"
        ax.set_title(
            f"{trait_id}  {agreement_panels.SHORT_NAMES[trait_id]}{unit_text}",
            fontsize=6.5,
            color="#1F5A9D",
            fontweight="bold",
            pad=4,
        )
        rmse_display = float(summary.loc[trait_id, "RMSE"]) * agreement_panels.RMSE_MULTIPLIER.get(trait_id, 1.0)
        rmse_decimals = 3 if rmse_display < 0.1 else 2
        ax.text(
            0.045,
            0.95,
            f"R² = {agreement_panels.format_half_up(float(summary.loc[trait_id, 'R2']), 3)}\n"
            f"RMSE = {agreement_panels.format_half_up(rmse_display, rmse_decimals)}{unit_text}\n"
            f"MAPE = {agreement_panels.format_half_up(float(summary.loc[trait_id, 'MAPE (%)']), 1)}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.25,
            linespacing=1.25,
            color="#202628",
        )
        axes.append(ax)

    label_ax = fig.add_subplot(spec)
    label_ax.set_zorder(-20)
    label_ax.patch.set_alpha(0)
    label_ax.axis("off")
    label_ax.text(-0.022, 0.43, "Image-derived", rotation=90, fontsize=7.5, fontweight="bold", ha="center", va="center")
    label_ax.text(0.50, -0.035, "Expert reference", fontsize=7.5, fontweight="bold", ha="center", va="center")
    axes.append(label_ax)
    return axes


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    workbook = (args.workbook or root / "Supplementary Table1-9.xlsx").resolve()
    summary_workbook = args.summary_workbook.resolve() if args.summary_workbook else None
    input_dir = (args.input_dir or root / "outputs" / "method_comparison_analysis").resolve()
    output_dir = (args.output_dir or root / "outputs" / "supplementary_figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    measured_summary = pd.read_csv(input_dir / "measured_traits_agreement_summary.csv")
    ordinal_summary = pd.read_csv(input_dir / "visual_traits_agreement_summary.csv")
    confusion_summary = pd.read_csv(input_dir / "visual_traits_confusion_matrices_long.csv")
    table3 = pd.read_excel(workbook, sheet_name="Supplementary Table 3", header=1)

    if len(table3) != 150:
        raise ValueError(f"Expected 150 varieties in Supplementary Table 3, found {len(table3)}.")
    if len(measured_summary) != 10 or not measured_summary["n"].eq(150).all():
        raise ValueError("Expected 10 measured/count traits with n=150 each.")
    if set(ordinal_summary["Trait ID"]) != set(agreement_panels.CONFUSION_ORDER):
        raise ValueError("Ordinal trait set does not match the expected 13 traits.")
    confusion_totals = confusion_summary.groupby("Trait ID")["Count"].sum().sort_index()
    ordinal_totals = ordinal_summary.set_index("Trait ID")["n"].sort_index()
    if not confusion_totals.equals(ordinal_totals):
        raise ValueError("Confusion-matrix totals do not match trait-specific valid sample sizes.")

    pairs = prepare_measured_pairs(table3, measured_summary)
    verify_measured_metrics(pairs, measured_summary)
    measured_exact = recompute_measured_metrics(pairs, measured_summary)
    ordinal_exact, confusion = recompute_ordinal_metrics_and_confusion(table3, ordinal_summary)
    recomputed_totals = confusion.groupby("Trait ID")["Count"].sum().sort_index()
    if not recomputed_totals.equals(ordinal_exact.set_index("Trait ID")["n"].sort_index()):
        raise ValueError("Recomputed confusion matrices do not match recomputed ordinal sample sizes.")
    original_counts = confusion_summary.sort_values(["Trait ID", "Manual_code", "Image_code"]).reset_index(drop=True)
    recomputed_counts = confusion.sort_values(["Trait ID", "Manual_code", "Image_code"]).reset_index(drop=True)
    count_columns = ["Trait ID", "Manual_code", "Image_code", "Count"]
    if not original_counts[count_columns].equals(recomputed_counts[count_columns]):
        raise ValueError("Recomputed confusion matrices differ from the preserved analysis output.")

    if summary_workbook:
        measured, ordinal, metric_audit = apply_published_s3_metrics(
            summary_workbook,
            measured_exact,
            ordinal_exact,
        )
    else:
        measured = measured_exact.copy()
        ordinal = ordinal_exact.copy()
        metric_audit = pd.DataFrame(
            columns=[
                "Trait ID",
                "Metric",
                "Recomputed from mother table",
                "Reported in finalized Table S3",
                "Difference (reported - recomputed)",
            ]
        )

    measured["RMSE_display"] = [
        value * agreement_panels.RMSE_MULTIPLIER.get(trait_id, 1.0)
        for trait_id, value in zip(measured["Trait ID"], measured["RMSE"])
    ]

    pairs.to_csv(output_dir / "Supplementary_Fig_S2_panel_A_paired_source.csv", index=False, encoding="utf-8-sig")
    measured.to_csv(output_dir / "Supplementary_Fig_S2_panel_B_source.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(output_dir / "Supplementary_Fig_S2_panel_C_source.csv", index=False, encoding="utf-8-sig")
    ordinal.to_csv(output_dir / "Supplementary_Fig_S2_panel_D_source.csv", index=False, encoding="utf-8-sig")
    write_alignment_audit(
        output_dir / "Supplementary_Fig_S2_trait_alignment_audit.csv",
        measured_exact,
        ordinal_exact,
        metric_audit,
    )

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

    fig = plt.figure(figsize=(18.2, 18.0), facecolor="white")
    fig.text(
        0.035,
        0.985,
        "Supplementary Fig. 2 | Complete trait-wise agreement",
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
        color="#151A1C",
    )
    outer = fig.add_gridspec(
        2,
        1,
        left=0.050,
        right=0.988,
        bottom=0.045,
        top=0.955,
        height_ratios=[0.41, 0.59],
        hspace=0.075,
    )
    bottom = outer[1].subgridspec(1, 2, width_ratios=[0.49, 0.51], wspace=0.075)
    bottom_left = bottom[0].subgridspec(2, 1, height_ratios=[0.47, 0.53], hspace=0.11)

    axes_a = plot_scatter_panel(fig, outer[0], pairs, measured)
    axes_b = agreement_panels.plot_measured_panel(fig, bottom_left[0], measured)
    axes_d = agreement_panels.plot_ordinal_panel(fig, bottom_left[1], ordinal)
    axes_c = agreement_panels.plot_confusion_panel(fig, bottom[1], confusion, ordinal)

    fig.canvas.draw()
    panel_specs = [
        (outer[0].get_position(fig), "A"),
        (bottom_left[0].get_position(fig), "B"),
        (bottom[1].get_position(fig), "C"),
        (bottom_left[1].get_position(fig), "D"),
    ]
    for bbox, label in panel_specs:
        agreement_panels.panel_frame(fig, bbox)
        agreement_panels.panel_label(fig, bbox, label)

    fig.text(
        0.050,
        0.018,
        "R²: coefficient of determination; RMSE: root mean square error; MAPE: mean absolute percentage error; "
        "κ: weighted kappa; ρ: Spearman correlation. Confusion matrices are normalized within expert-reference grades.",
        ha="left",
        va="center",
        fontsize=6.6,
        color="#4F575A",
    )

    stem = output_dir / "Supplementary_Fig_S2_complete_trait_wise_agreement_real_data"
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Workbook: {workbook}")
    print(f"Finalized summary workbook: {summary_workbook or 'not supplied; recomputed metrics used'}")
    print(f"PNG: {stem.with_suffix('.png')}")
    print(f"PDF: {stem.with_suffix('.pdf')}")
    print(f"Panel A source rows: {len(pairs)}")
    print("All measured metrics and confusion-matrix totals verified.")


if __name__ == "__main__":
    main()
