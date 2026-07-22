from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAITS = REPO_ROOT / "data" / "traits.csv"
DEFAULT_SCORING = REPO_ROOT / "outputs" / "variety_core_fingerprint" / "core_trait_scoring_table.csv"
DEFAULT_SELECTED = REPO_ROOT / "outputs" / "variety_core_fingerprint" / "selected_core_traits.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "representative_heatmaps"

ID_COL = "Field Plot No."
GROUP_COL = "Endosperm Type"
TARGET_FAMILIES = ["Color", "Size", "Count", "Shape"]
FAMILY_DEFAULTS = {"Color": 3, "Size": 4, "Count": 2, "Shape": 3}
NORMAL_LABEL = "Normal type"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select representative cultivars from image-derived traits and plot fingerprint heatmaps."
    )
    parser.add_argument("--traits-file", type=Path, default=DEFAULT_TRAITS)
    parser.add_argument("--scoring-file", type=Path, default=DEFAULT_SCORING)
    parser.add_argument("--selected-core-file", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--normal-n", type=int, default=5, help="Number of Normal type cultivars for main-text heatmap.")
    parser.add_argument(
        "--non-normal-n",
        type=int,
        default=1,
        help="Number of representative cultivars per non-Normal endosperm type.",
    )
    parser.add_argument(
        "--family-counts",
        nargs="*",
        default=None,
        help="Override trait counts per family, e.g. Color=3 Size=4 Count=2 Shape=3",
    )
    parser.add_argument(
        "--trait-mode",
        choices=["eligible-all", "family-top", "core-only"],
        default="eligible-all",
        help="How to choose heatmap columns.",
    )
    return parser.parse_args()


def parse_family_counts(raw_items: list[str] | None) -> dict[str, int]:
    counts = FAMILY_DEFAULTS.copy()
    if not raw_items:
        return counts
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid family count override: {item}")
        family, value = item.split("=", 1)
        family = family.strip()
        if family not in counts:
            raise ValueError(f"Unknown family: {family}")
        counts[family] = int(value)
    return counts


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(float)
    for col in out.columns:
        vals = out[col]
        mean = vals.mean()
        std = vals.std(ddof=0)
        if pd.isna(std) or std == 0:
            out[col] = 0.0
        else:
            out[col] = (vals - mean) / std
    return out


def pick_trait_columns(
    scoring_df: pd.DataFrame,
    selected_core_df: pd.DataFrame,
    trait_mode: str,
    family_counts: dict[str, int],
) -> pd.DataFrame:
    score = scoring_df.copy()
    score["Trait family"] = score["Trait family"].astype(str)
    score["Trait column"] = score["Trait column"].astype(str)

    if trait_mode == "eligible-all":
        picked = score[score["Quality pass"].fillna(False)].copy()
        family_order = {name: i for i, name in enumerate(TARGET_FAMILIES)}
        picked["family_order"] = picked["Trait family"].map(family_order)
        picked = picked.sort_values(["family_order", "Trait no."]).reset_index(drop=True)
        return picked

    if trait_mode == "core-only":
        core_cols = selected_core_df["Trait column"].astype(str).tolist()
        picked = score[score["Trait column"].isin(core_cols)].copy()
        picked["family_rank"] = picked.groupby("Trait family")["Final score"].rank(method="first", ascending=False)
        return picked.sort_values(["Trait family", "family_rank", "Trait no."]).reset_index(drop=True)

    picked_frames: list[pd.DataFrame] = []
    qualified = score[score["Quality pass"].fillna(False)].copy()
    for family in TARGET_FAMILIES:
        sub = qualified[qualified["Trait family"] == family].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["Selected or not", "Final score", "Trait no."], ascending=[False, False, True])
        n = family_counts[family]
        picked_frames.append(sub.head(n))
    if not picked_frames:
        raise ValueError("No traits selected for heatmap.")
    picked = pd.concat(picked_frames, ignore_index=True).drop_duplicates(subset=["Trait column"])
    family_order = {name: i for i, name in enumerate(TARGET_FAMILIES)}
    picked["family_order"] = picked["Trait family"].map(family_order)
    picked = picked.sort_values(["family_order", "Final score"], ascending=[True, False]).reset_index(drop=True)
    return picked


def choose_group_representatives(
    feature_df: pd.DataFrame,
    group_labels: pd.Series,
    normal_n: int,
    non_normal_n: int,
) -> pd.DataFrame:
    selected_rows: list[dict[str, object]] = []
    all_groups = group_labels.dropna().unique().tolist()
    ordered_groups = [g for g in all_groups if g != NORMAL_LABEL]
    if NORMAL_LABEL in all_groups:
        ordered_groups = ordered_groups + [NORMAL_LABEL]

    for group in ordered_groups:
        idx = group_labels[group_labels == group].index.tolist()
        sub = feature_df.loc[idx]
        if sub.empty:
            continue
        target_n = normal_n if group == NORMAL_LABEL else non_normal_n
        target_n = max(1, min(target_n, len(sub)))

        center = sub.mean(axis=0).to_numpy()
        d_center = np.linalg.norm(sub.to_numpy() - center, axis=1)
        ranked_positions = np.argsort(d_center)[:target_n]
        chosen = [sub.index[i] for i in ranked_positions]

        for order, row_idx in enumerate(chosen, start=1):
            role = "center" if order == 1 else "center-near"
            selected_rows.append(
                {
                    "row_index": row_idx,
                    ID_COL: row_idx,
                    GROUP_COL: group,
                    "selection_order_within_group": order,
                    "selection_role": role,
                }
            )

    return pd.DataFrame(selected_rows)


def hierarchical_order(feature_df: pd.DataFrame) -> list[str]:
    if len(feature_df) <= 2:
        return feature_df.index.tolist()
    dist = pdist(feature_df.to_numpy(), metric="euclidean")
    link = linkage(dist, method="average")
    leaves = leaves_list(link)
    return feature_df.index[leaves].tolist()


def order_rows_within_groups(feature_df: pd.DataFrame, row_meta: pd.DataFrame) -> list[str]:
    order: list[str] = []
    for group, sub_meta in row_meta.groupby(GROUP_COL, sort=False):
        ids = sub_meta[ID_COL].tolist()
        if len(ids) == 1:
            order.extend(ids)
            continue
        sub = feature_df.loc[ids]
        center_ids = sub_meta.loc[sub_meta["selection_role"] == "center", ID_COL].tolist()
        clustered = hierarchical_order(sub)
        if center_ids and center_ids[0] in clustered:
            center_id = center_ids[0]
            pos = clustered.index(center_id)
            clustered = clustered[pos:] + clustered[:pos]
        order.extend(clustered)
    return order


def palette_for_groups(groups: list[str]) -> dict[str, str]:
    fixed = {
        "Normal type": "#F6D37A",              # pale yellow
        "Sweet type": "#F08AA0",               # pink
        "Waxy type": "#8DB6F2",                # light blue
        "Sweet-waxy hybrid type": "#C6A6E8",   # lavender
        "Popcorn type": "#E9C89B",             # light beige/orange
    }
    fallback = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    palette: dict[str, str] = {}
    for i, g in enumerate(groups):
        palette[g] = fixed.get(g, fallback[i % len(fallback)])
    return palette


def family_palette() -> dict[str, str]:
    return {
        "Color": "#F97316",
        "Size": "#2563EB",
        "Count": "#16A34A",
        "Shape": "#7C3AED",
    }


def build_display_heatmap(z_df: pd.DataFrame, col_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    display_cols: list[pd.Series] = []
    display_meta_rows: list[dict[str, object]] = []
    last_family = None

    for _, row in col_meta.iterrows():
        family = row["Trait family"]
        trait_col = row["Trait column"]
        trait_short = row["Trait short"]

        if last_family is not None and family != last_family:
            gap_name = f"__gap__{last_family}_to_{family}"
            display_cols.append(pd.Series(np.nan, index=z_df.index, name=gap_name))
            display_meta_rows.append(
                {
                    "Trait no.": np.nan,
                    "Trait short": "",
                    "Trait name": "",
                    "Trait family": "__gap__",
                    "Trait column": gap_name,
                    "Final score": np.nan,
                }
            )

        display_cols.append(z_df[trait_col].rename(trait_col))
        display_meta_rows.append(
            {
                "Trait no.": row["Trait no."],
                "Trait short": trait_short,
                "Trait name": row["Trait name"],
                "Trait family": family,
                "Trait column": trait_col,
                "Final score": row["Final score"],
            }
        )
        last_family = family

    display_df = pd.concat(display_cols, axis=1)
    display_meta = pd.DataFrame(display_meta_rows)
    return display_df, display_meta


def draw_heatmap(
    z_df: pd.DataFrame,
    row_labels: pd.Series,
    col_meta: pd.DataFrame,
    output_path: Path,
    title: str,
    *,
    width_in: float | None = None,
    height_in: float | None = None,
    top_bar_height: float = 0.12,
    show_title: bool = True,
    show_top_labels: bool = True,
    show_top_bar: bool = True,
    cbar_mode: str = "right",
) -> None:
    sns.set_theme(style="white")
    row_groups = row_labels.loc[z_df.index].tolist()
    unique_groups = []
    for g in row_groups:
        if g not in unique_groups:
            unique_groups.append(g)
    g_palette = palette_for_groups(unique_groups)
    f_palette = family_palette()

    display_df, display_meta = build_display_heatmap(z_df, col_meta)
    n_rows, n_cols = display_df.shape
    if width_in is None:
        width_in = max(8, 0.55 * n_cols + 3.8)
    if height_in is None:
        height_in = max(6, 0.34 * n_rows + 2.8)

    if cbar_mode == "bottom":
        fig = plt.figure(figsize=(width_in, height_in), dpi=300)
        gs = gridspec.GridSpec(
            3,
            2,
            width_ratios=[0.38, 1.0],
            height_ratios=[top_bar_height, 1.0, 0.08],
            wspace=0.02,
            hspace=0.04,
        )
        ax_left = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 1])
        ax = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[2, 1])
    elif cbar_mode == "right":
        fig = plt.figure(figsize=(width_in, height_in), dpi=300)
        gs = gridspec.GridSpec(
            2,
            2,
            width_ratios=[0.32, 1.0],
            height_ratios=[top_bar_height, 1.0],
            wspace=0.02,
            hspace=0.02,
        )
        ax_left = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 1])
        ax = fig.add_subplot(gs[1, 1])
        cax = None
    elif cbar_mode == "none":
        fig = plt.figure(figsize=(width_in, height_in), dpi=300)
        gs = gridspec.GridSpec(
            2,
            2,
            width_ratios=[0.38, 1.0],
            height_ratios=[top_bar_height, 1.0],
            wspace=0.02,
            hspace=0.02,
        )
        ax_left = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 1])
        ax = fig.add_subplot(gs[1, 1])
        cax = None
    else:
        raise ValueError(f"Unsupported cbar_mode: {cbar_mode}")

    cmap = LinearSegmentedColormap.from_list("fingerprint", ["#1E5BBE", "#F7F7F7", "#E3261C"])
    cmap.set_bad("#E6E6E6")
    mask = display_df.isna()
    cbar_kws = {"shrink": 0.55, "pad": 0.02, "label": "Z-score standardized value (column-wise)"}
    if cbar_mode == "bottom":
        cbar_kws = {
            "orientation": "horizontal",
            "pad": 0.02,
            "label": "Z-score standardized value (column-wise)",
            "ticks": [-2, -1, 0, 1, 2],
        }
    sns.heatmap(
        display_df,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-2,
        vmax=2,
        mask=mask,
        cbar=(cbar_mode != "none"),
        cbar_ax=cax,
        linewidths=0.45,
        linecolor="#F2F2F2",
        xticklabels=display_meta["Trait short"].tolist(),
        yticklabels=False,
        cbar_kws=cbar_kws,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    if show_title and title:
        ax.set_title(title, fontsize=13, weight="bold", pad=10)
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_facecolor("white")
    if cbar_mode == "bottom" and cax is not None:
        cax.tick_params(labelsize=7)
        cax.xaxis.label.set_size(8)

    # top family bar
    ax_top.set_xlim(0, n_cols)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")
    if show_top_bar:
        families = display_meta["Trait family"].tolist()
        start = 0
        for i in range(1, n_cols + 1):
            if i == n_cols or families[i] != families[start]:
                fam = families[start]
                if fam != "__gap__":
                    ax_top.add_patch(
                        plt.Rectangle((start, 0), i - start, 1, facecolor=f_palette[fam], edgecolor="white", linewidth=1.2)
                    )
                    real_count = int((display_meta.iloc[start:i]["Trait family"] != "__gap__").sum())
                    if show_top_labels:
                        ax_top.text(
                            (start + i) / 2,
                            0.5,
                            f"{fam} ({real_count})",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=7 if top_bar_height <= 0.06 else 9,
                            weight="bold",
                        )
                start = i

    # left group bar
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, n_rows)
    ax_left.invert_yaxis()
    ax_left.axis("off")
    name_fontsize = 4.2 if n_rows >= 100 else (6.0 if n_rows >= 40 else 8.0)
    for i, group in enumerate(row_groups):
        ax_left.add_patch(
            plt.Rectangle((0.06, i), 0.12, 1, facecolor=g_palette[group], edgecolor="white", linewidth=0.5)
        )
        ax_left.text(0.23, i + 0.5, str(display_df.index[i]), ha="left", va="center", fontsize=name_fontsize)
    start = 0
    for i in range(1, n_rows + 1):
        if i == n_rows or row_groups[i] != row_groups[start]:
            start = i

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    family_counts = parse_family_counts(args.family_counts)

    traits = pd.read_csv(args.traits_file)
    scoring = pd.read_csv(args.scoring_file)
    selected_core = pd.read_csv(args.selected_core_file)

    col_meta = pick_trait_columns(scoring, selected_core, args.trait_mode, family_counts)
    trait_cols = [c for c in col_meta["Trait column"].tolist() if c in traits.columns]
    col_meta = col_meta[col_meta["Trait column"].isin(trait_cols)].copy()
    if not trait_cols:
        raise ValueError("No selected trait columns were found in traits.csv.")

    work = traits[[ID_COL, GROUP_COL] + trait_cols].copy()
    work[trait_cols] = work[trait_cols].apply(pd.to_numeric, errors="coerce")
    work = work.dropna(subset=[ID_COL, GROUP_COL]).copy()
    work[trait_cols] = work[trait_cols].fillna(work[trait_cols].median(numeric_only=True))
    work[ID_COL] = work[ID_COL].astype(str)
    work = work.set_index(ID_COL)

    z = zscore_frame(work[trait_cols])
    row_groups = work[GROUP_COL]

    rep = choose_group_representatives(z, row_groups, args.normal_n, args.non_normal_n)
    rep = rep.drop(columns=["row_index"])
    rep_order = order_rows_within_groups(z, rep)
    rep = rep.set_index(ID_COL).loc[rep_order].reset_index()

    rep_heat = z.loc[rep_order, trait_cols]
    rep_row_labels = work.loc[rep_order, GROUP_COL]

    # full order: within each endosperm type use hierarchical clustering
    full_order: list[str] = []
    group_sequence = row_groups.drop_duplicates().tolist()
    for group in group_sequence:
        ids = row_groups[row_groups == group].index.tolist()
        sub = z.loc[ids, trait_cols]
        full_order.extend(hierarchical_order(sub))
    full_heat = z.loc[full_order, trait_cols]
    full_row_labels = work.loc[full_order, GROUP_COL]

    rep["selection_reason"] = np.where(
        rep["selection_role"] == "center",
        "Representative central cultivar within endosperm type",
        "Representative boundary cultivar within endosperm type",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rep.to_csv(args.output_dir / "main_text_representative_cultivars.csv", index=False, encoding="utf-8-sig")
    col_meta[["Trait no.", "Trait short", "Trait name", "Trait family", "Trait column", "Final score"]].to_csv(
        args.output_dir / "main_text_selected_traits.csv", index=False, encoding="utf-8-sig"
    )

    draw_heatmap(
        rep_heat,
        rep_row_labels,
        col_meta,
        args.output_dir / "main_text_representative_fingerprint_heatmap.png",
        "Representative phenotypic fingerprint heatmap",
    )
    draw_heatmap(
        full_heat,
        full_row_labels,
        col_meta,
        args.output_dir / "supplementary_all_cultivars_fingerprint_heatmap.png",
        "All-cultivar phenotypic fingerprint heatmap",
        width_in=16 / 2.54,
        height_in=24 / 2.54,
        top_bar_height=0.01,
        show_top_bar=False,
        show_title=False,
        show_top_labels=False,
        cbar_mode="none",
    )

    summary = {
        "traits_used": trait_cols,
        "n_traits": len(trait_cols),
        "n_main_text_cultivars": len(rep),
        "main_text_group_counts": rep[GROUP_COL].value_counts().to_dict(),
    }
    pd.Series(summary).to_json(args.output_dir / "heatmap_selection_summary.json", force_ascii=False, indent=2)

    print(f"Saved representative cultivar table to: {args.output_dir / 'main_text_representative_cultivars.csv'}")
    print(f"Saved trait selection table to: {args.output_dir / 'main_text_selected_traits.csv'}")
    print(f"Saved main-text heatmap to: {args.output_dir / 'main_text_representative_fingerprint_heatmap.png'}")
    print(f"Saved supplementary heatmap to: {args.output_dir / 'supplementary_all_cultivars_fingerprint_heatmap.png'}")


if __name__ == "__main__":
    main()
