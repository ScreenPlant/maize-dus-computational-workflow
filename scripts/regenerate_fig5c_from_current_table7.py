from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from PIL import Image


CONFIG_ORDER = [
    "Four-trait pre-sufficient subset",
    "Five-trait core subset",
    "Six-trait balanced reference subset",
    "Full coding-eligible reference",
]
CONFIG_LABELS = ["4-trait subset", "5-trait subset", "6-trait subset", "19-trait subset"]
TRANSFER_SCENARIOS = ["Bootstrap OOB", "Constrained stratified 80/20"]
PERTURBATION_SCENARIOS = [
    "Empirical count residual",
    "Ordinal adjacent-grade",
    "Uniform ±5%",
    "Uniform ±10%",
    "Gaussian rRMSE",
]
DISPLAY_COLUMNS = [
    "Bootstrap\nOOB",
    "80/20\nsplit",
    "Count\nresidual",
    "Ordinal\nadjacent",
    "±5%",
    "±10%",
    "Gaussian\nrRMSE",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Regenerate Fig. 5C directly from the current Supplementary Table 7."
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        required=True,
        help="Final supplementary workbook containing Supplementary Table S7.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "fig5c_from_table7_260715",
    )
    parser.add_argument(
        "--old-figure",
        type=Path,
        default=root / "mnt" / "data" / "fig5_audit_media" / "media" / "image5.png",
        help="Optional prior full Fig. 5 used only to retain panels A–B in a composite image.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def first_number(value: object) -> float:
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?", str(value))
    if match is None:
        raise ValueError(f"No numeric value found in {value!r}")
    return float(match.group())


def locate_row(raw: pd.DataFrame, first_cell: str, start: int = 0) -> int:
    values = raw.iloc[start:, 0].astype(str).str.strip()
    hits = values.index[values.eq(first_cell)].tolist()
    if len(hits) != 1:
        raise ValueError(f"Expected one row beginning {first_cell!r}; found {hits}")
    return int(hits[0])


def read_block(raw: pd.DataFrame, header_row: int, stop_row: int) -> pd.DataFrame:
    columns = raw.iloc[header_row].tolist()
    block = raw.iloc[header_row + 1 : stop_row].copy()
    block.columns = columns
    block = block.dropna(how="all").reset_index(drop=True)
    return block


def load_current_table7(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
    raw = pd.read_excel(workbook, sheet_name="Supplementary Table 7", header=None)
    row_a = locate_row(raw, "Scenario")
    row_b = locate_row(raw, "Perturbation scenario")
    row_b_title = row_b - 1
    row_c_title = locate_row(raw, "C. Nested exhaustive-selection stability")
    row_metric = locate_row(raw, "Metric", row_c_title)
    row_trait_title = locate_row(raw, "C2. Trait eligibility and selection frequency across 100 nested splits")
    row_trait = locate_row(raw, "Trait code", row_trait_title)
    row_notes = locate_row(raw, next(v for v in raw.iloc[:, 0].dropna().astype(str) if v.startswith("Notes:")), row_trait)

    transfer = read_block(raw, row_a, row_b_title)
    perturbation = read_block(raw, row_b, row_c_title)
    metrics_block = read_block(raw, row_metric, row_trait_title)
    metrics = {str(row["Metric"]): first_number(row["Result"]) for _, row in metrics_block.iterrows()}
    traits = read_block(raw, row_trait, row_notes)

    if set(transfer["Configuration"]) != set(CONFIG_ORDER):
        raise ValueError("Table 7A configuration set differs from the four expected fixed configurations.")
    if set(perturbation["Configuration"]) != set(CONFIG_ORDER):
        raise ValueError("Table 7B configuration set differs from the four expected fixed configurations.")
    return transfer, perturbation, metrics, traits


def matrices_from_table7(
    transfer: pd.DataFrame,
    perturbation: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    upp = np.empty((4, 7), dtype=float)
    lower = np.empty((4, 7), dtype=float)
    audit_rows: list[dict[str, object]] = []
    for row_idx, configuration in enumerate(CONFIG_ORDER):
        for col_idx, scenario in enumerate(TRANSFER_SCENARIOS):
            row = transfer.loc[
                transfer["Configuration"].eq(configuration) & transfer["Scenario"].eq(scenario)
            ]
            if len(row) != 1:
                raise ValueError((configuration, scenario, len(row)))
            row = row.iloc[0]
            upp_value = first_number(row["Mean evaluation UPP [95% empirical interval]"])
            lower_value = float(row["Strict complete transfer (%)"]) / 100.0
            upp[row_idx, col_idx] = upp_value
            lower[row_idx, col_idx] = lower_value
            audit_rows.extend(
                [
                    {"Block": "Reference–evaluation", "Configuration": configuration, "Scenario": scenario,
                     "Metric": "Mean evaluation UPP", "Value": upp_value},
                    {"Block": "Reference–evaluation", "Configuration": configuration, "Scenario": scenario,
                     "Metric": "Strict complete transfer", "Value": lower_value},
                ]
            )
        for offset, scenario in enumerate(PERTURBATION_SCENARIOS, start=2):
            row = perturbation.loc[
                perturbation["Configuration"].eq(configuration)
                & perturbation["Perturbation scenario"].eq(scenario)
            ]
            if len(row) != 1:
                raise ValueError((configuration, scenario, len(row)))
            row = row.iloc[0]
            upp_value = first_number(row["Mean UPP\n[95% empirical interval]"])
            lower_value = float(row["Complete uniqueness (%)"]) / 100.0
            upp[row_idx, offset] = upp_value
            lower[row_idx, offset] = lower_value
            audit_rows.extend(
                [
                    {"Block": "Frozen-code perturbation", "Configuration": configuration, "Scenario": scenario,
                     "Metric": "Mean UPP", "Value": upp_value},
                    {"Block": "Frozen-code perturbation", "Configuration": configuration, "Scenario": scenario,
                     "Metric": "Complete uniqueness", "Value": lower_value},
                ]
            )
    if not (np.isfinite(upp).all() and np.isfinite(lower).all()):
        raise ValueError("Fig. 5C matrix contains missing or non-finite values.")
    if not (((upp >= 0) & (upp <= 1)).all() and ((lower >= 0) & (lower <= 1)).all()):
        raise ValueError("Fig. 5C rates must lie within [0, 1].")
    return upp, lower, pd.DataFrame(audit_rows)


def heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    annotations: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    show_x: bool,
    cbar_label: str,
) -> None:
    sns.heatmap(
        values,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annotations,
        fmt="",
        linewidths=0.8,
        linecolor="#C8CDD2",
        yticklabels=CONFIG_LABELS,
        xticklabels=DISPLAY_COLUMNS if show_x else False,
        cbar=True,
        cbar_kws={"label": cbar_label, "shrink": 0.78, "pad": 0.015},
        annot_kws={"fontsize": 8.4, "fontweight": "bold"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0, labelsize=8.5, length=0)
    ax.tick_params(axis="x", rotation=0, labelsize=8.5, length=0)


def metric_card(ax: plt.Axes, x: float, title: str, value: str) -> None:
    width = 0.225
    patch = FancyBboxPatch(
        (x, 0.07), width, 0.82,
        boxstyle="round,pad=0.012,rounding_size=0.045",
        facecolor="white", edgecolor="#D8DDE2", linewidth=1.0,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, 0.68, title, ha="center", va="center", fontsize=6.2,
            fontweight="bold", transform=ax.transAxes, linespacing=1.05)
    ax.text(x + width / 2, 0.27, value, ha="center", va="center", fontsize=11,
            fontweight="bold", color="#1F4EAA", transform=ax.transAxes)


def draw_panel_c(
    upp: np.ndarray,
    lower: np.ndarray,
    metrics: dict[str, float],
    traits: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(13.3, 5.25), facecolor="white")
    outer = fig.add_gridspec(
        1, 2, width_ratios=[4.2, 1.23], left=0.075, right=0.985, top=0.81, bottom=0.13, wspace=0.14
    )
    left = outer[0, 0].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.43)
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[2.0, 0.85], hspace=0.40)
    ax_upp = fig.add_subplot(left[0, 0])
    ax_lower = fig.add_subplot(left[1, 0])
    ax_frequency = fig.add_subplot(right[0, 0])
    ax_cards = fig.add_subplot(right[1, 0])

    upp_annotations = np.vectorize(lambda x: f"{x:.4f}")(upp)
    lower_annotations = np.vectorize(lambda x: f"{100 * x:.1f}%")(lower)
    heatmap(ax_upp, upp, upp_annotations, "Blues", 0.93, 1.0, True, "Mean UPP")
    heatmap(ax_lower, lower, lower_annotations, "Oranges", 0.0, 1.0, True, "Rate")

    ax_upp.text(-0.105, 0.5, "Mean unique\npattern proportion", transform=ax_upp.transAxes,
                ha="center", va="center", fontsize=9, color="#1F4EAA", fontweight="bold")
    ax_lower.text(-0.105, 0.5, "Strict transfer /\ncomplete uniqueness", transform=ax_lower.transAxes,
                  ha="center", va="center", fontsize=8.5, color="#D9482B", fontweight="bold")
    ax_upp.text(1 / 7, 1.28, "Reference–evaluation", transform=ax_upp.transAxes,
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax_upp.text(4.5 / 7, 1.28, "Frozen-code perturbation", transform=ax_upp.transAxes,
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax_upp.plot([0, 2 / 7], [1.23, 1.23], transform=ax_upp.transAxes, color="#555555", lw=0.8, clip_on=False)
    ax_upp.plot([2 / 7, 1], [1.23, 1.23], transform=ax_upp.transAxes, color="#555555", lw=0.8, clip_on=False)

    selected = traits.loc[pd.to_numeric(traits["Selection frequency"], errors="coerce").gt(0)].copy()
    selected["Selection frequency"] = pd.to_numeric(selected["Selection frequency"], errors="coerce")
    selected = selected.sort_values(["Selection frequency", "Trait code"], ascending=[False, True]).head(6)
    selected = selected.sort_values("Selection frequency", ascending=True)
    ax_frequency.barh(selected["Trait code"], selected["Selection frequency"] * 100, color="#4B77BE", height=0.58)
    for y, value in enumerate(selected["Selection frequency"] * 100):
        ax_frequency.text(min(value + 1.5, 98), y, f"{value:.0f}%", va="center", fontsize=7.5)
    ax_frequency.set_xlim(0, 100)
    ax_frequency.set_xlabel("Selection frequency", fontsize=8.5)
    ax_frequency.set_title("Nested selection stability", loc="left", fontsize=10, fontweight="bold", pad=7)
    ax_frequency.tick_params(labelsize=7.5)
    ax_frequency.spines[["top", "right"]].set_visible(False)
    ax_frequency.grid(axis="x", color="#E6E9EC", linewidth=0.6)

    ax_cards.axis("off")
    metric_card(ax_cards, 0.00, "Median\nminimum k", f"{metrics['Median minimum sufficient subset size']:.0f}")
    metric_card(ax_cards, 0.255, "Exact core\nrecovery", f"{100 * metrics['Exact recovery rate of T10/T23/T32/T37/T42']:.0f}%")
    metric_card(ax_cards, 0.510, "Mean\nJaccard", f"{metrics['Mean Jaccard similarity']:.3f}")
    metric_card(ax_cards, 0.765, "Strict complete\ntransfer", f"{100 * metrics['Complete transfer rate']:.0f}%")

    fig.text(0.016, 0.945, "C", fontsize=17, fontweight="bold", va="top")
    fig.text(
        0.047, 0.945,
        "Conditional internal performance, selection stability, and measurement-error sensitivity",
        fontsize=15, fontweight="bold", va="top",
    )
    fig.text(
        0.075, 0.025,
        "Reference–evaluation lower cells show strict complete transfer; perturbation lower cells show complete uniqueness. "
        "All values were read directly from current Supplementary Table 7.",
        fontsize=7.5, color="#444444", ha="left",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Fig5C_regenerated_from_current_Supplementary_Table7"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def compose_full_figure(old_figure: Path, panel_c: Path, output_dir: Path) -> Path | None:
    if not old_figure.exists():
        return None
    with Image.open(old_figure) as old_image, Image.open(panel_c) as panel_image:
        old = old_image.convert("RGB")
        upper = old.crop((0, 0, old.width, 460))
        panel = panel_image.convert("RGB")
        panel = panel.resize(
            (old.width, round(panel.height * old.width / panel.width)), Image.Resampling.LANCZOS
        )
        canvas = Image.new("RGB", (old.width, upper.height + panel.height), "white")
        canvas.paste(upper, (0, 0))
        canvas.paste(panel, (0, upper.height))
        output = output_dir / "Fig5_full_with_regenerated_panel_C.png"
        canvas.save(output, dpi=(300, 300))
    return output


def main() -> None:
    args = parse_args()
    workbook = args.workbook.resolve()
    output_dir = args.output_dir.resolve()
    transfer, perturbation, metrics, traits = load_current_table7(workbook)
    upp, lower, audit = matrices_from_table7(transfer, perturbation)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit["Source workbook"] = str(workbook)
    audit_path = output_dir / "Fig5C_cell_level_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    nested_path = output_dir / "Fig5C_nested_selection_audit.csv"
    traits.to_csv(nested_path, index=False, encoding="utf-8-sig")
    png, pdf = draw_panel_c(upp, lower, metrics, traits, output_dir, args.dpi)
    composite = compose_full_figure(args.old_figure.resolve(), png, output_dir)
    print(f"Panel C PNG: {png}")
    print(f"Panel C PDF: {pdf}")
    print(f"Cell audit: {audit_path}")
    print(f"Nested audit: {nested_path}")
    if composite is not None:
        print(f"Full Fig. 5 composite: {composite}")


if __name__ == "__main__":
    main()
