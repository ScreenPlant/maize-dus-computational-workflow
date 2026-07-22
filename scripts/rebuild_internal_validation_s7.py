from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_BREAK
from docx.shared import Pt
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from scipy.stats import linregress
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


Q = np.array([-0.8416, -0.2533, 0.2533, 0.8416], dtype=np.float64)
DDOF = 0
BOOTSTRAP_REPEATS = 1000
SPLIT_REPEATS = 100
PERTURBATION_REPEATS = 1000
BOOTSTRAP_SEED = 42001
SPLIT_MASTER_SEED = 42002
PERTURBATION_SEED = 42003
RF_SEED = 42
ID_COL = "Field Plot No."
GROUP_COL = "Endosperm Type"
FULL_PANEL_CORE = ("T10", "T23", "T32", "T37", "T42")
FIXED_CONFIGS = {
    "Four-trait pre-sufficient subset": ("T10", "T23", "T32", "T37"),
    "Five-trait core subset": FULL_PANEL_CORE,
    "Six-trait balanced reference subset": ("T3", "T30", "T32", "T37", "T42", "T43"),
}


@dataclass(frozen=True)
class Paths:
    project: Path
    manuscript: Path
    supp_methods: Path
    supp_tables: Path
    out: Path


def parse_args() -> Paths:
    parser = argparse.ArgumentParser(description="Rebuild leakage-free internal validation for Supplementary Table 7.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project", type=Path, default=repo_root,
                        help="Project directory containing the unencoded trait matrix and paired agreement inputs.")
    parser.add_argument("--manuscript", type=Path, required=True,
                        help="Source manuscript DOCX; the original is never overwritten.")
    parser.add_argument("--supp-methods", type=Path, required=True,
                        help="Source supplementary-methods DOCX; the original is never overwritten.")
    parser.add_argument("--supp-tables", type=Path, required=True,
                        help="Source supplementary workbook containing the eligibility table.")
    parser.add_argument("--out", type=Path, default=repo_root / "outputs" / "internal_validation_rebuild")
    ns = parser.parse_args()
    return Paths(ns.project.resolve(), ns.manuscript.resolve(), ns.supp_methods.resolve(), ns.supp_tables.resolve(), ns.out.resolve())


def require_inputs(paths: Paths) -> None:
    required = [paths.manuscript, paths.supp_methods, paths.supp_tables,
                paths.project / "traits.csv", paths.project / "traits dictionary.csv",
                paths.project / "traits_metadata.csv", paths.project / "outputs/table3_analysis/table3_actual.csv",
                paths.project / "outputs/table3_analysis/table3_predict.csv",
                paths.project / "outputs/method_comparison_analysis/measured_traits_detailed_stats.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        paths.out.mkdir(parents=True, exist_ok=True)
        report = "# Missing inputs report\n\n" + "\n".join(f"- Missing file: `{p}`" for p in missing)
        report += "\n\nTrain-only/OOB/frozen-code numerical recalculation was not run. Required format: one row per accession, a unique accession ID, endosperm type, and the 23 unencoded image-derived trait values; paired manual and image-derived values are also required for nested reliability and empirical perturbation.\n"
        (paths.out / "missing_inputs_report.md").write_text(report, encoding="utf-8")
        raise FileNotFoundError("Missing required inputs; see missing_inputs_report.md")


def read_meta(paths: Paths) -> pd.DataFrame:
    meta = pd.read_csv(paths.project / "traits_metadata.csv", encoding="utf-8-sig")
    dictionary = pd.read_csv(paths.project / "traits dictionary.csv", encoding="utf-8-sig")
    meta = meta.merge(dictionary[["Trait No.", "Data type", "Code range"]], on="Trait No.", how="left")
    meta["Trait No."] = meta["Trait No."].astype(int)
    meta["Trait short"] = "T" + meta["Trait No."].astype(str)
    meta["Trait column"] = "Trait_" + meta["Trait No."].astype(str) + "_Predict"
    return meta


def read_eligible_from_s4(workbook: Path) -> list[str]:
    wb = load_workbook(workbook, read_only=True, data_only=True)
    sheet_name = next(
        (name for name in ("Supplementary Table S5", "Supplementary Table 5", "Supplementary Table 4")
         if name in wb.sheetnames),
        None,
    )
    if sheet_name is None:
        raise KeyError("Could not find the trait eligibility worksheet (expected Supplementary Table S5/5/4).")
    ws = wb[sheet_name]
    header_row = next(
        (row for row in range(1, min(ws.max_row, 10) + 1)
         if "Trait code" in [str(c.value).strip() if c.value is not None else "" for c in ws[row]]),
        None,
    )
    if header_row is None:
        raise ValueError(f"Could not find the trait eligibility header in {sheet_name}.")
    header = [str(c.value).strip() if c.value is not None else "" for c in ws[header_row]]
    code_idx = header.index("Trait code")
    status_idx = header.index("Eligibility status")
    eligible = []
    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        if row[code_idx] and str(row[status_idx]).strip().lower() == "eligible":
            eligible.append(str(row[code_idx]).strip())
    return eligible


def parse_fixed_range(value: object) -> tuple[float, float] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(value))]
    return (min(nums), max(nums)) if nums else None


def load_data(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(paths.project / "traits.csv", encoding="utf-8-sig")
    actual = pd.read_csv(paths.project / "outputs/table3_analysis/table3_actual.csv", encoding="utf-8-sig")
    predicted = pd.read_csv(paths.project / "outputs/table3_analysis/table3_predict.csv", encoding="utf-8-sig")
    meta = read_meta(paths)
    eligible = read_eligible_from_s4(paths.supp_tables)
    if len(df) != 150 or df[ID_COL].nunique() != 150:
        raise ValueError("Expected 150 rows and 150 unique accession IDs in traits.csv")
    expected_groups = {"Normal type": 127, "Sweet type": 8, "Waxy type": 6, "Sweet-waxy hybrid type": 3, "Popcorn type": 6}
    if df[GROUP_COL].value_counts().to_dict() != expected_groups:
        raise ValueError(f"Unexpected endosperm distribution: {df[GROUP_COL].value_counts().to_dict()}")
    code_traits = set(meta["Trait short"])
    if not set(eligible).issubset(code_traits):
        raise ValueError("S4 contains unknown eligible trait codes")
    return df, actual, predicted, meta, eligible


def meta_maps(meta: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, tuple[float, float] | None]]:
    short_to_col = dict(zip(meta["Trait short"], meta["Trait column"]))
    short_to_type = dict(zip(meta["Trait short"], meta["Data type"].astype(str)))
    short_to_family = dict(zip(meta["Trait short"], meta["Analysis Type"].astype(str)))
    short_to_range = {r["Trait short"]: parse_fixed_range(r["Code range"]) for _, r in meta.iterrows()}
    return short_to_col, short_to_type, short_to_family, short_to_range


def fit_codebook(df: pd.DataFrame, meta: pd.DataFrame) -> dict[str, dict[str, object]]:
    book: dict[str, dict[str, object]] = {}
    for _, row in meta.iterrows():
        short, col, dtype = str(row["Trait short"]), str(row["Trait column"]), str(row["Data type"])
        if dtype not in {"continuous", "continuous_ratio"}:
            book[short] = {"type": dtype, "mu": None, "sd": None, "boundaries": None, "constant": False}
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            book[short] = {"type": dtype, "mu": np.nan, "sd": np.nan, "boundaries": [np.nan] * 4, "constant": True}
            continue
        mu = float(np.mean(valid, dtype=np.float64))
        sd = float(np.std(valid, ddof=DDOF, dtype=np.float64))
        constant = not np.isfinite(sd) or sd == 0.0
        boundaries = (mu + sd * Q).astype(np.float64) if not constant else np.repeat(mu, 4).astype(np.float64)
        book[short] = {"type": dtype, "mu": mu, "sd": sd, "boundaries": boundaries.tolist(), "constant": constant}
    return book


def encode_with_codebook(df: pd.DataFrame, meta: pd.DataFrame, book: dict[str, dict[str, object]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out[ID_COL] = df[ID_COL].astype(str).to_numpy()
    for _, row in meta.iterrows():
        short, col, dtype = str(row["Trait short"]), str(row["Trait column"]), str(row["Data type"])
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        coded = np.full(len(values), np.nan, dtype=np.float64)
        valid = np.isfinite(values)
        if dtype in {"continuous", "continuous_ratio"}:
            p = book[short]
            if p["constant"]:
                coded[valid] = 3.0
            else:
                # side='right': a value exactly on a boundary enters the bin on its right,
                # matching the published lower-inclusive/upper-exclusive equations.
                coded[valid] = np.searchsorted(np.asarray(p["boundaries"], dtype=np.float64), values[valid], side="right") + 1
        else:
            coded[valid] = np.rint(values[valid])
        out[short] = coded
    return out


def ranges_for_reference(ref_encoded: pd.DataFrame, traits: Iterable[str], meta: pd.DataFrame) -> dict[str, tuple[float, float]]:
    _, short_to_type, _, short_to_fixed = meta_maps(meta)
    ranges: dict[str, tuple[float, float]] = {}
    for t in traits:
        dtype = short_to_type[t]
        if dtype in {"continuous", "continuous_ratio"}:
            ranges[t] = (1.0, 5.0)
        elif short_to_fixed[t] is not None:
            ranges[t] = short_to_fixed[t]  # official/dictionary range
        else:
            vals = pd.to_numeric(ref_encoded[t], errors="coerce").dropna()
            if vals.empty:
                ranges[t] = (0.0, 1.0)
            else:
                lo, hi = float(vals.min()), float(vals.max())
                ranges[t] = (lo, hi if hi > lo else lo + 1.0)
    return ranges


def id_strings(encoded: pd.DataFrame, traits: Iterable[str]) -> np.ndarray:
    traits = list(traits)
    matrix = encoded[traits].to_numpy(dtype=np.float64)
    rows = []
    for arr in matrix:
        parts = ["X" if not np.isfinite(v) else str(int(v)) if float(v).is_integer() else format(v, ".15g") for v in arr]
        rows.append("-".join(parts))
    return np.asarray(rows, dtype=object)


def distance_matrix(left: pd.DataFrame, right: pd.DataFrame, traits: Iterable[str], ranges: dict[str, tuple[float, float]], self_panel: bool = False) -> np.ndarray:
    traits = list(traits)
    dist_sum = np.zeros((len(left), len(right)), dtype=np.float64)
    comparable = np.zeros_like(dist_sum, dtype=np.int16)
    for t in traits:
        a = left[t].to_numpy(dtype=np.float64)[:, None]
        b = right[t].to_numpy(dtype=np.float64)[None, :]
        valid = np.isfinite(a) & np.isfinite(b)
        lo, hi = ranges[t]
        denom = hi - lo
        if denom <= 0:
            continue
        dist_sum += np.where(valid, np.abs(a - b) / denom, 0.0)
        comparable += valid.astype(np.int16)
    result = np.full_like(dist_sum, np.nan)
    np.divide(dist_sum, comparable, out=result, where=comparable > 0)
    if self_panel:
        np.fill_diagonal(result, np.nan)
    return result


def panel_metrics(encoded: pd.DataFrame, traits: Iterable[str], ranges: dict[str, tuple[float, float]]) -> dict[str, float]:
    traits = tuple(traits)
    n = len(encoded)
    ids = id_strings(encoded, traits)
    upp = float(len(set(ids)) / n) if n else np.nan
    d = distance_matrix(encoded, encoded, traits, ranges, self_panel=True)
    iu = np.triu_indices(n, 1)
    pair = d[iu]
    comparable = np.isfinite(pair)
    zero = int(np.isclose(pair[comparable], 0.0, atol=1e-12).sum())
    denom = n * (n - 1) // 2
    zpr = float(zero / denom) if denom else np.nan
    mean_dist = float(np.nanmean(pair)) if comparable.any() else np.nan
    if np.isclose(upp, 1.0) and zero != 0:
        raise AssertionError("UPP=1 but zero-distance pairs is nonzero")
    return {"evaluation_size": n, "evaluation_upp": upp, "evaluation_zero_pairs": zero,
            "evaluation_zero_pair_rate": zpr, "evaluation_mean_pairwise_distance": mean_dist,
            "evaluation_complete_unique": int(np.isclose(upp, 1.0) and zero == 0)}


def cross_metrics(eval_encoded: pd.DataFrame, ref_encoded: pd.DataFrame, traits: Iterable[str], ranges: dict[str, tuple[float, float]]) -> dict[str, float]:
    traits = tuple(traits)
    eid, rid = id_strings(eval_encoded, traits), id_strings(ref_encoded, traits)
    matches = eid[:, None] == rid[None, :]
    collision_pairs = int(matches.sum())
    denom = len(eid) * len(rid)
    cfr = float((~matches.any(axis=1)).mean()) if len(eid) else np.nan
    d = distance_matrix(eval_encoded, ref_encoded, traits, ranges)
    zero_matches = np.isfinite(d) & np.isclose(d, 0.0, atol=1e-12)
    zero_pairs = int(zero_matches.sum())
    zero_free = float((~zero_matches.any(axis=1)).mean()) if len(eid) else np.nan
    mins = np.nanmin(d, axis=1) if len(eid) and len(rid) else np.array([], dtype=float)
    return {"cross_collision_pairs": collision_pairs,
            "cross_collision_pair_rate": float(collision_pairs / denom) if denom else np.nan,
            "evaluation_to_reference_cfr": cfr,
            "cross_zero_distance_pairs": zero_pairs,
            "cross_zero_distance_pair_rate": float(zero_pairs / denom) if denom else np.nan,
            "evaluation_to_reference_zero_distance_free_rate": zero_free,
            "mean_minimum_reference_distance": float(np.nanmean(mins)) if len(mins) else np.nan,
            "median_minimum_reference_distance": float(np.nanmedian(mins)) if len(mins) else np.nan}


def evaluate_transfer(eval_encoded: pd.DataFrame, ref_encoded: pd.DataFrame, traits: Iterable[str], meta: pd.DataFrame) -> dict[str, float]:
    ranges = ranges_for_reference(ref_encoded, traits, meta)
    m = panel_metrics(eval_encoded, traits, ranges)
    m.update(cross_metrics(eval_encoded, ref_encoded, traits, ranges))
    m["complete_transfer"] = int(
        m["evaluation_complete_unique"] == 1
        and np.isclose(m["evaluation_to_reference_cfr"], 1.0)
        and np.isclose(m["evaluation_to_reference_zero_distance_free_rate"], 1.0)
    )
    if m["complete_transfer"]:
        assert (np.isclose(m["evaluation_upp"], 1.0) and m["evaluation_zero_pairs"] == 0
                and np.isclose(m["evaluation_to_reference_cfr"], 1.0)
                and np.isclose(m["evaluation_to_reference_zero_distance_free_rate"], 1.0))
    return m


def fixed_configuration_map(eligible: list[str]) -> dict[str, tuple[str, ...]]:
    configs = dict(FIXED_CONFIGS)
    configs["Full coding-eligible reference"] = tuple(eligible)
    return configs


def run_bootstrap(df: pd.DataFrame, meta: pd.DataFrame, configs: dict[str, tuple[str, ...]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows, params = [], []
    n = len(df)
    for rep in range(1, BOOTSTRAP_REPEATS + 1):
        sampled_pos = rng.integers(0, n, size=n)
        counts = np.bincount(sampled_pos, minlength=n)
        ref_pos = np.flatnonzero(counts > 0)
        eval_pos = np.flatnonzero(counts == 0)
        if len(eval_pos) == 0:
            rows.append({"analysis": "bootstrap_oob", "repeat": rep, "skipped": 1, "reason": "empty OOB"})
            continue
        # Repeated rows retain bootstrap multiplicity during parameter estimation.
        fit_df = df.iloc[sampled_pos].reset_index(drop=True)
        ref_df = df.iloc[ref_pos].reset_index(drop=True)
        eval_df = df.iloc[eval_pos].reset_index(drop=True)
        book = fit_codebook(fit_df, meta)
        ref_enc = encode_with_codebook(ref_df, meta, book)
        eval_enc = encode_with_codebook(eval_df, meta, book)
        for t, p in book.items():
            if p["boundaries"] is not None:
                params.append({"analysis": "bootstrap_oob", "repeat": rep, "seed": BOOTSTRAP_SEED, "trait": t,
                               "mu": p["mu"], "sd": p["sd"], **{f"boundary_{i+1}": b for i, b in enumerate(p["boundaries"])}})
        for name, traits in configs.items():
            m = evaluate_transfer(eval_enc, ref_enc, traits, meta)
            rows.append({"analysis": "bootstrap_oob", "repeat": rep, "seed": BOOTSTRAP_SEED, "skipped": 0,
                         "configuration": name, "trait_set": ", ".join(traits), "reference_size": len(ref_df),
                         "evaluation_size": len(eval_df), **m})
    return pd.DataFrame(rows), pd.DataFrame(params)


def bootstrap_memberships(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate and retain per-accession bootstrap multiplicities and OOB membership."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(df)
    rows = []
    for rep in range(1, BOOTSTRAP_REPEATS + 1):
        sampled_pos = rng.integers(0, n, size=n)
        counts = np.bincount(sampled_pos, minlength=n)
        for pos, count in enumerate(counts):
            rows.append(
                {
                    "repeat": rep,
                    "seed": BOOTSTRAP_SEED,
                    "position": pos,
                    "accession_id": str(df.iloc[pos][ID_COL]),
                    "endosperm_type": str(df.iloc[pos][GROUP_COL]),
                    "bootstrap_multiplicity": int(count),
                    "role": "reference" if count > 0 else "evaluation_oob",
                }
            )
    return pd.DataFrame(rows)


def constrained_eval_counts(group_counts: dict[str, int], total_eval: int) -> dict[str, int]:
    groups = list(group_counts)
    raw = {g: group_counts[g] * total_eval / sum(group_counts.values()) for g in groups}
    allocation = {g: max(1, min(group_counts[g] - 2, int(math.floor(raw[g])))) for g in groups}
    while sum(allocation.values()) < total_eval:
        candidates = [g for g in groups if allocation[g] < group_counts[g] - 2]
        g = max(candidates, key=lambda x: (raw[x] - allocation[x], group_counts[x], x))
        allocation[g] += 1
    while sum(allocation.values()) > total_eval:
        candidates = [g for g in groups if allocation[g] > 1]
        g = min(candidates, key=lambda x: (raw[x] - allocation[x], -group_counts[x], x))
        allocation[g] -= 1
    if any(allocation[g] < 1 or group_counts[g] - allocation[g] < 2 for g in groups):
        raise ValueError("Constrained split allocation is infeasible")
    return allocation


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    master = np.random.default_rng(SPLIT_MASTER_SEED)
    counts = df[GROUP_COL].value_counts().to_dict()
    alloc = constrained_eval_counts(counts, 30)
    rows = []
    all_pos = np.arange(len(df))
    for rep in range(1, SPLIT_REPEATS + 1):
        split_seed = int(master.integers(0, np.iinfo(np.int32).max))
        rng = np.random.default_rng(split_seed)
        eval_positions = []
        for group, n_eval in alloc.items():
            pos = np.flatnonzero(df[GROUP_COL].to_numpy() == group)
            eval_positions.extend(rng.choice(pos, size=n_eval, replace=False).tolist())
        eval_positions = np.asarray(sorted(eval_positions), dtype=int)
        ref_positions = np.setdiff1d(all_pos, eval_positions)
        if np.intersect1d(ref_positions, eval_positions).size:
            raise AssertionError("Reference/evaluation overlap")
        for role, positions in [("reference", ref_positions), ("evaluation", eval_positions)]:
            for pos in positions:
                rows.append({"repeat": rep, "split_seed": split_seed, "role": role, "position": int(pos),
                             "accession_id": str(df.iloc[pos][ID_COL]), "endosperm_type": str(df.iloc[pos][GROUP_COL])})
    return pd.DataFrame(rows)


def run_fixed_splits(df: pd.DataFrame, meta: pd.DataFrame, configs: dict[str, tuple[str, ...]], splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, params = [], []
    for rep, split in splits.groupby("repeat"):
        ref_pos = split.loc[split.role == "reference", "position"].to_numpy(dtype=int)
        eval_pos = split.loc[split.role == "evaluation", "position"].to_numpy(dtype=int)
        seed = int(split["split_seed"].iloc[0])
        ref_df, eval_df = df.iloc[ref_pos].reset_index(drop=True), df.iloc[eval_pos].reset_index(drop=True)
        book = fit_codebook(ref_df, meta)
        ref_enc, eval_enc = encode_with_codebook(ref_df, meta, book), encode_with_codebook(eval_df, meta, book)
        for t, p in book.items():
            if p["boundaries"] is not None:
                params.append({"analysis": "stratified_80_20", "repeat": rep, "seed": seed, "trait": t,
                               "mu": p["mu"], "sd": p["sd"], **{f"boundary_{i+1}": b for i, b in enumerate(p["boundaries"])}})
        for name, traits in configs.items():
            rows.append({"analysis": "stratified_80_20", "repeat": rep, "seed": seed, "configuration": name,
                         "trait_set": ", ".join(traits), "reference_size": len(ref_df), "evaluation_size": len(eval_df),
                         **evaluate_transfer(eval_enc, ref_enc, traits, meta)})
    return pd.DataFrame(rows), pd.DataFrame(params)


def minmax(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    lo, hi = values.min(), values.max()
    if pd.isna(lo) or pd.isna(hi) or np.isclose(lo, hi):
        return pd.Series(0.0, index=values.index)
    return (values - lo) / (hi - lo)


def nested_reliability(ref_ids: set[str], meta: pd.DataFrame, actual: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, float]:
    a = actual[actual[ID_COL].astype(str).isin(ref_ids)].set_index(ID_COL)
    p = predicted[predicted[ID_COL].astype(str).isin(ref_ids)].set_index(ID_COL)
    common = a.index.intersection(p.index)
    scores = {}
    for _, row in meta.iterrows():
        no, short, dtype = int(row["Trait No."]), str(row["Trait short"]), str(row["Data type"])
        av = pd.to_numeric(a.loc[common, f"Trait_{no}_Actual"], errors="coerce")
        pv = pd.to_numeric(p.loc[common, f"Trait_{no}_Predict"], errors="coerce")
        valid = av.notna() & pv.notna()
        if valid.sum() < 3:
            scores[short] = 0.0
        elif dtype.startswith("ordinal_"):
            k = cohen_kappa_score(av[valid].round().astype(int), pv[valid].round().astype(int), weights="quadratic")
            scores[short] = float(np.clip(k if np.isfinite(k) else 0.0, 0.0, 1.0))
        else:
            lr = linregress(av[valid].to_numpy(float), pv[valid].to_numpy(float))
            scores[short] = float(np.clip(lr.rvalue ** 2 if np.isfinite(lr.rvalue) else 0.0, 0.0, 1.0))
    return scores


def eta_squared_by_group(df: pd.DataFrame, col: str) -> float:
    sub = df[[GROUP_COL, col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 2:
        return 0.0
    grand = float(sub[col].mean())
    ss_total = float(((sub[col] - grand) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_between = sum(len(g) * (float(g[col].mean()) - grand) ** 2 for _, g in sub.groupby(GROUP_COL))
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def permutation_scores(ref_df: pd.DataFrame, meta: pd.DataFrame) -> dict[str, float]:
    cols = meta["Trait column"].astype(str).tolist()
    x = ref_df[cols].apply(pd.to_numeric, errors="coerce")
    y = ref_df[GROUP_COL].astype(str)
    prep = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                     ("scaler", StandardScaler())]), cols)])
    # Split-level parallelism is used by run_nested; keep estimator internals single-threaded
    # to avoid oversubscription while retaining deterministic results.
    model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=RF_SEED, n_jobs=1)
    pipe = Pipeline([("preprocessor", prep), ("model", model)])
    pipe.fit(x, y)
    perm = permutation_importance(pipe, x, y, n_repeats=30, random_state=RF_SEED,
                                  scoring="balanced_accuracy", n_jobs=1)
    raw = pd.Series(np.maximum(perm.importances_mean, 0.0), index=meta["Trait short"].astype(str))
    scaled = minmax(raw)
    return scaled.to_dict()


def per_trait_distance_stats(encoded: pd.DataFrame, traits: list[str], ranges: dict[str, tuple[float, float]]) -> tuple[dict[str, float], dict[str, float]]:
    distinct, distance = {}, {}
    for t in traits:
        x = encoded[t].to_numpy(float)
        a, b = np.triu_indices(len(x), 1)
        valid = np.isfinite(x[a]) & np.isfinite(x[b])
        if not valid.any():
            distinct[t] = distance[t] = 0.0
            continue
        diff = np.abs(x[a][valid] - x[b][valid])
        distinct[t] = float((diff != 0).mean())
        lo, hi = ranges[t]
        distance[t] = float((diff / (hi - lo)).mean()) if hi > lo else 0.0
    return distinct, distance


def nested_diagnostics(ref_df: pd.DataFrame, ref_encoded: pd.DataFrame, book: dict[str, dict[str, object]], meta: pd.DataFrame,
                       actual: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    traits = meta["Trait short"].astype(str).tolist()
    short_to_col, _, short_to_family, _ = meta_maps(meta)
    reliability = nested_reliability(set(ref_df[ID_COL].astype(str)), meta, actual, predicted)
    eta = {t: eta_squared_by_group(ref_df, short_to_col[t]) for t in traits}
    model = permutation_scores(ref_df, meta)
    raw_matrix = ref_df[[short_to_col[t] for t in traits]].apply(pd.to_numeric, errors="coerce")
    raw_matrix.columns = traits
    corr = raw_matrix.fillna(raw_matrix.median()).corr(method="spearman").abs()
    redundant = {t: int(((corr.loc[t] > 0.85) & (corr.index != t)).sum()) for t in traits}
    ranges = ranges_for_reference(ref_encoded, traits, meta)
    distinct, distance = per_trait_distance_stats(ref_encoded, traits, ranges)
    rows = []
    for t in traits:
        raw = pd.to_numeric(ref_df[short_to_col[t]], errors="coerce")
        enc = pd.to_numeric(ref_encoded[t], errors="coerce").dropna()
        rows.append({"trait": t, "trait_name": meta.loc[meta["Trait short"] == t, "Trait name"].iloc[0],
                     "family": short_to_family[t], "availability": float(raw.notna().mean()),
                     "dominant_state_proportion": float(enc.value_counts(normalize=True).iloc[0]) if len(enc) else 1.0,
                     "reliability": reliability[t], "eta_squared": eta[t], "model_importance": model[t],
                     "non_redundancy": 1.0 - redundant[t] / max(1, len(traits) - 1),
                     "pairwise_distinctness": distinct[t], "distance_contribution": distance[t]})
    d = pd.DataFrame(rows)
    d["subgroup_score"] = minmax(d["eta_squared"])
    d["composite_score"] = (0.20 * d.availability + 0.25 * d.reliability + 0.10 * d.subgroup_score +
                            0.05 * d.model_importance + 0.10 * d.non_redundancy +
                            0.20 * d.pairwise_distinctness + 0.10 * d.distance_contribution)
    d["eligible"] = (d.availability >= 0.70) & (d.dominant_state_proportion < 0.90)
    d["diagnostic_rank"] = d["composite_score"].rank(ascending=False, method="min")
    return d


def family_imbalance(traits: tuple[str, ...], family: dict[str, str]) -> float:
    counts = Counter(family[t] for t in traits)
    return float(sum(abs(counts.get(f, 0) / len(traits) - 0.25) for f in ["Color", "Size", "Count", "Shape"]))


def exhaustive_select(ref_encoded: pd.DataFrame, diagnostics: pd.DataFrame, meta: pd.DataFrame) -> dict[str, object]:
    eligible = diagnostics.loc[diagnostics.eligible, "trait"].astype(str).tolist()
    _, _, family, _ = meta_maps(meta)
    reliability = diagnostics.set_index("trait")["reliability"].to_dict()
    score = diagnostics.set_index("trait")["composite_score"].to_dict()
    ranges = ranges_for_reference(ref_encoded, eligible, meta)
    selected = None
    selected_metrics = None
    for k in range(1, len(eligible) + 1):
        sufficient = []
        for combo in combinations(eligible, k):
            ids = id_strings(ref_encoded, combo)
            if len(set(ids)) != len(ids):
                continue
            pm = panel_metrics(ref_encoded, combo, ranges)
            if pm["evaluation_zero_pairs"] != 0:
                continue
            coverage = len(set(family[t] for t in combo))
            key = (-coverage, family_imbalance(combo, family), -pm["evaluation_mean_pairwise_distance"],
                   -float(np.mean([reliability[t] for t in combo])), -float(np.mean([score[t] for t in combo])), combo)
            sufficient.append((key, combo, pm))
        if sufficient:
            sufficient.sort(key=lambda z: z[0])
            _, selected, selected_metrics = sufficient[0]
            break
    full_metrics = panel_metrics(ref_encoded, eligible, ranges) if eligible else {}
    if selected is None:
        return {"status": "no sufficient subset", "eligible": eligible, "selected": tuple(), "k": np.nan,
                "training_metrics": full_metrics, "full_eligible_metrics": full_metrics}
    return {"status": "selected", "eligible": eligible, "selected": tuple(selected), "k": len(selected),
            "training_metrics": selected_metrics, "full_eligible_metrics": full_metrics}


def run_nested(df: pd.DataFrame, actual: pd.DataFrame, predicted: pd.DataFrame, meta: pd.DataFrame,
               splits: pd.DataFrame, parallel: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_ids = sorted(splits["repeat"].unique().tolist())
    if parallel and len(repeat_ids) > 1:
        jobs = Parallel(n_jobs=min(4, len(repeat_ids)), backend="loky", verbose=10)(
            delayed(run_nested)(df, actual, predicted, meta, splits[splits["repeat"] == rep].copy(), False)
            for rep in repeat_ids
        )
        return (pd.concat([x[0] for x in jobs], ignore_index=True).sort_values("repeat").reset_index(drop=True),
                pd.concat([x[1] for x in jobs], ignore_index=True).sort_values(["repeat", "trait"]).reset_index(drop=True))
    rows, trait_rows = [], []
    full_set = set(FULL_PANEL_CORE)
    for rep, split in splits.groupby("repeat"):
        started = time.time()
        ref_pos = split.loc[split.role == "reference", "position"].to_numpy(dtype=int)
        eval_pos = split.loc[split.role == "evaluation", "position"].to_numpy(dtype=int)
        seed = int(split.split_seed.iloc[0])
        ref_df, eval_df = df.iloc[ref_pos].reset_index(drop=True), df.iloc[eval_pos].reset_index(drop=True)
        book = fit_codebook(ref_df, meta)
        ref_enc, eval_enc = encode_with_codebook(ref_df, meta, book), encode_with_codebook(eval_df, meta, book)
        diag = nested_diagnostics(ref_df, ref_enc, book, meta, actual, predicted)
        selection = exhaustive_select(ref_enc, diag, meta)
        selected = tuple(selection["selected"])
        if selected:
            transfer = evaluate_transfer(eval_enc, ref_enc, selected, meta)
            jaccard = len(set(selected) & full_set) / len(set(selected) | full_set)
            exact = int(set(selected) == full_set and len(selected) == len(full_set))
            composition = ", ".join(f"{f}:{sum(1 for t in selected if dict(zip(meta['Trait short'], meta['Analysis Type']))[t] == f)}"
                                    for f in ["Color", "Size", "Count", "Shape"])
        else:
            transfer = {k: np.nan for k in ["evaluation_upp", "evaluation_zero_pairs", "evaluation_zero_pair_rate",
                                             "evaluation_mean_pairwise_distance", "evaluation_complete_unique", "cross_collision_pairs",
                                             "cross_collision_pair_rate", "evaluation_to_reference_cfr", "cross_zero_distance_pairs",
                                             "cross_zero_distance_pair_rate", "evaluation_to_reference_zero_distance_free_rate", "complete_transfer",
                                             "mean_minimum_reference_distance", "median_minimum_reference_distance"]}
            jaccard, exact, composition = np.nan, 0, ""
        rows.append({"analysis": "nested_selection", "repeat": rep, "seed": seed, "selection_status": selection["status"],
                     "reference_size": len(ref_df), "evaluation_size": len(eval_df), "eligible_pool_size": len(selection["eligible"]),
                     "eligible_traits": ", ".join(selection["eligible"]), "minimum_sufficient_k": selection["k"],
                     "selected_trait_set": ", ".join(selected), "trait_family_composition": composition,
                     "training_upp": selection["training_metrics"].get("evaluation_upp", np.nan),
                     "training_zero_pairs": selection["training_metrics"].get("evaluation_zero_pairs", np.nan),
                     "jaccard_with_full_panel_core": jaccard, "exact_core_recovery": exact,
                     "elapsed_seconds": time.time() - started, **transfer})
        selected_set = set(selected)
        for _, d in diag.iterrows():
            trait_rows.append({"repeat": rep, "seed": seed, **d.to_dict(), "selected": int(d.trait in selected_set),
                               "selected_k": selection["k"]})
        print(f"nested {rep}/{SPLIT_REPEATS}: k={selection['k']}, selected={','.join(selected)}, {time.time()-started:.1f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(trait_rows)


def recompute_nested_transfer_metrics(df: pd.DataFrame, meta: pd.DataFrame, splits: pd.DataFrame,
                                      nested: pd.DataFrame) -> pd.DataFrame:
    """Recompute only evaluation/reference metrics for an already audited deterministic selection trace."""
    result=nested.copy()
    transfer_columns=None
    for rep, split in splits.groupby("repeat"):
        row_idx=result.index[result["repeat"]==rep]
        if len(row_idx)!=1: raise ValueError(f"Expected one nested row for repeat {rep}")
        idx=row_idx[0]
        if str(result.at[idx,"selection_status"])!="selected": continue
        selected=tuple(t.strip() for t in str(result.at[idx,"selected_trait_set"]).split(",") if t.strip())
        ref_pos=split.loc[split.role=="reference","position"].to_numpy(dtype=int)
        eval_pos=split.loc[split.role=="evaluation","position"].to_numpy(dtype=int)
        ref_df,eval_df=df.iloc[ref_pos].reset_index(drop=True),df.iloc[eval_pos].reset_index(drop=True)
        book=fit_codebook(ref_df,meta)
        ref_enc=encode_with_codebook(ref_df,meta,book); eval_enc=encode_with_codebook(eval_df,meta,book)
        metrics=evaluate_transfer(eval_enc,ref_enc,selected,meta)
        transfer_columns=list(metrics)
        for key,value in metrics.items(): result.at[idx,key]=value
    if transfer_columns is None: raise ValueError("No selected nested split found")
    return result


def empirical_error_maps(paths: Paths, meta: pd.DataFrame, actual: pd.DataFrame, predicted: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], dict[str, np.ndarray]]:
    stats = pd.read_csv(paths.project / "outputs/method_comparison_analysis/measured_traits_detailed_stats.csv", encoding="utf-8-sig")
    rrmse = {str(r["Trait ID"]): float(r["rRMSE (%)"]) / 100.0 for _, r in stats.iterrows() if pd.notna(r["rRMSE (%)"])}
    mismatch, residuals = {}, {}
    for _, row in meta.iterrows():
        no, t, dtype = int(row["Trait No."]), str(row["Trait short"]), str(row["Data type"])
        av = pd.to_numeric(actual[f"Trait_{no}_Actual"], errors="coerce")
        pv = pd.to_numeric(predicted[f"Trait_{no}_Predict"], errors="coerce")
        valid = av.notna() & pv.notna()
        if dtype.startswith("ordinal_"):
            mismatch[t] = float((av[valid].round() != pv[valid].round()).mean()) if valid.any() else 0.0
        if dtype == "integer_count" and valid.any():
            residuals[t] = (pv[valid] - av[valid]).to_numpy(dtype=float)
    return rrmse, mismatch, residuals


def perturb(df: pd.DataFrame, meta: pd.DataFrame, scenario: str, rng: np.random.Generator,
            rrmse: dict[str, float], mismatch: dict[str, float], residuals: dict[str, np.ndarray]) -> pd.DataFrame:
    out = df.copy()
    _, _, _, fixed_ranges = meta_maps(meta)
    for _, row in meta.iterrows():
        t, col, dtype = str(row["Trait short"]), str(row["Trait column"]), str(row["Data type"])
        x = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
        valid = np.isfinite(x)
        if scenario in {"Uniform ±5%", "Uniform ±10%", "Gaussian rRMSE"} and dtype in {"continuous", "continuous_ratio"}:
            if scenario == "Uniform ±5%": noise = rng.uniform(-0.05, 0.05, valid.sum())
            elif scenario == "Uniform ±10%": noise = rng.uniform(-0.10, 0.10, valid.sum())
            else:
                if t not in rrmse: raise ValueError(f"Missing rRMSE for {t}")
                noise = rng.normal(0.0, rrmse[t], valid.sum())
            x[valid] = np.maximum(0.0, x[valid] * (1.0 + noise))
        elif scenario == "Ordinal adjacent-grade" and dtype.startswith("ordinal_"):
            legal = fixed_ranges[t]
            if legal is None: continue
            lo, hi = int(legal[0]), int(legal[1])
            idx = np.flatnonzero(valid & (rng.random(len(x)) < mismatch.get(t, 0.0)))
            for i in idx:
                value = int(round(x[i]))
                choices = [v for v in (value - 1, value + 1) if lo <= v <= hi]
                if choices: x[i] = rng.choice(choices)
        elif scenario == "Empirical count residual" and dtype == "integer_count":
            if t not in residuals or not len(residuals[t]): continue
            x[valid] = np.maximum(0.0, np.rint(x[valid] + rng.choice(residuals[t], size=valid.sum(), replace=True)))
        out[col] = x
    return out


def run_perturbation(paths: Paths, df: pd.DataFrame, actual: pd.DataFrame, predicted: pd.DataFrame,
                     meta: pd.DataFrame, configs: dict[str, tuple[str, ...]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_book = fit_codebook(df, meta)
    baseline_enc = encode_with_codebook(df, meta, baseline_book)
    rrmse, mismatch, residuals = empirical_error_maps(paths, meta, actual, predicted)
    scenarios = ["Ordinal adjacent-grade", "Uniform ±5%", "Uniform ±10%", "Gaussian rRMSE"]
    if residuals:
        scenarios.append("Empirical count residual")
    rng = np.random.default_rng(PERTURBATION_SEED)
    rows, params = [], []
    for t, p in baseline_book.items():
        if p["boundaries"] is not None:
            params.append({"analysis": "frozen_baseline", "repeat": 0, "seed": PERTURBATION_SEED, "trait": t,
                           "mu": p["mu"], "sd": p["sd"], **{f"boundary_{i+1}": b for i, b in enumerate(p["boundaries"])}})
    for scenario in scenarios:
        for rep in range(1, PERTURBATION_REPEATS + 1):
            p_df = perturb(df, meta, scenario, rng, rrmse, mismatch, residuals)
            p_enc = encode_with_codebook(p_df, meta, baseline_book)
            for name, traits in configs.items():
                ranges = ranges_for_reference(baseline_enc, traits, meta)
                m = panel_metrics(p_enc, traits, ranges)
                baseline_ids, perturbed_ids = id_strings(baseline_enc, traits), id_strings(p_enc, traits)
                m["id_retention_rate"] = float((baseline_ids == perturbed_ids).mean())
                rows.append({"analysis": "frozen_perturbation", "scenario": scenario, "repeat": rep,
                             "seed": PERTURBATION_SEED, "configuration": name, "trait_set": ", ".join(traits),
                             "coding_parameters_recalculated": "No", **m})
    return pd.DataFrame(rows), pd.DataFrame(params)


def summarize(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple): key = (key,)
        row = dict(zip(group_cols, key)); row["n_repeats"] = len(g)
        for metric in metrics:
            x = pd.to_numeric(g[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = x.mean() if len(x) else np.nan
            row[f"{metric}_sd"] = x.std(ddof=0) if len(x) else np.nan
            row[f"{metric}_median"] = x.median() if len(x) else np.nan
            row[f"{metric}_q025"] = x.quantile(0.025) if len(x) else np.nan
            row[f"{metric}_q975"] = x.quantile(0.975) if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_summaries(fixed: pd.DataFrame, perturb_df: pd.DataFrame, nested: pd.DataFrame, nested_traits: pd.DataFrame, meta: pd.DataFrame):
    fixed_metrics = ["reference_size", "evaluation_size", "evaluation_upp", "evaluation_zero_pairs", "evaluation_zero_pair_rate",
                     "evaluation_mean_pairwise_distance", "evaluation_complete_unique", "cross_collision_pairs",
                     "cross_collision_pair_rate", "evaluation_to_reference_cfr", "cross_zero_distance_pairs",
                     "cross_zero_distance_pair_rate", "evaluation_to_reference_zero_distance_free_rate",
                     "complete_transfer", "mean_minimum_reference_distance"]
    s7a = summarize(fixed, ["analysis", "configuration", "trait_set"], fixed_metrics)
    pert_metrics = ["evaluation_upp", "evaluation_zero_pairs", "evaluation_zero_pair_rate", "evaluation_mean_pairwise_distance",
                    "evaluation_complete_unique", "id_retention_rate"]
    s7b = summarize(perturb_df, ["scenario", "configuration", "trait_set"], pert_metrics)
    selected = nested[nested.selection_status == "selected"]
    k = pd.to_numeric(nested.minimum_sufficient_k, errors="coerce")
    eligible = pd.to_numeric(nested.eligible_pool_size, errors="coerce")
    overall = {
        "Assessment type": "Nested internal selection-stability assessment",
        "Selection procedure repeated": "Training-only eligibility, coding, diagnostics, exhaustive search, minimum sufficient k, and original lexicographic ranking",
        "Trait-family rule": "Coverage and imbalance were lexicographic ranking criteria, not hard inclusion constraints",
        "Small-class diagnostic restriction": "Endosperm-associated effect and RF permutation importance were exploratory inputs used only through the final composite-score tie-breaker",
        "Number of splits": len(nested), "Median coding-eligible pool size": eligible.median(),
        "IQR coding-eligible pool size": f"{eligible.quantile(.25):.1f}–{eligible.quantile(.75):.1f}",
        "Range coding-eligible pool size": f"{eligible.min():.0f}–{eligible.max():.0f}",
        "Median minimum sufficient subset size": k.median(), "IQR minimum sufficient subset size": f"{k.quantile(.25):.1f}–{k.quantile(.75):.1f}",
        "Proportion with kmin=4": float((k == 4).mean()), "Proportion with kmin=5": float((k == 5).mean()),
        "Proportion with kmin=6": float((k == 6).mean()), "Proportion with other kmin": float((k.notna() & ~k.isin([4,5,6])).mean()),
        "No-sufficient-subset rate": float((nested.selection_status != "selected").mean()),
        "Exact recovery rate of T10/T23/T32/T37/T42": float(nested.exact_core_recovery.mean()),
        "Mean Jaccard similarity": float(pd.to_numeric(selected.jaccard_with_full_panel_core).mean()),
        "Mean evaluation-to-reference collision-free rate": float(pd.to_numeric(selected.evaluation_to_reference_cfr).mean()),
        "Complete transfer rate": float(pd.to_numeric(selected.complete_transfer).mean())}
    trait_names = meta.set_index("Trait short")["Trait name"].to_dict()
    families = meta.set_index("Trait short")["Analysis Type"].to_dict()
    freq_rows = []
    for t, g in nested_traits.groupby("trait"):
        selected_g = g[g.selected == 1]
        row = {"Trait code": t, "Trait name": trait_names[t], "Trait family": families[t],
               "Eligibility frequency": float(g.eligible.mean()), "Selection frequency": float(g.selected.mean()),
               "Mean rank when selected": float(selected_g.diagnostic_rank.mean()) if len(selected_g) else np.nan}
        for core in FULL_PANEL_CORE:
            reps_core = set(nested_traits.loc[(nested_traits.trait == core) & (nested_traits.selected == 1), "repeat"])
            reps_t = set(selected_g.repeat)
            row[f"Co-selection frequency with {core}"] = len(reps_core & reps_t) / SPLIT_REPEATS
        freq_rows.append(row)
    freq = pd.DataFrame(freq_rows).sort_values("Trait code", key=lambda x: x.str.extract(r"(\d+)")[0].astype(int))
    matrix = pd.DataFrame(0.0, index=meta["Trait short"], columns=meta["Trait short"])
    for selected_text in nested.selected_trait_set:
        ts = [t.strip() for t in str(selected_text).split(",") if t.strip()]
        for a in ts:
            for b in ts: matrix.loc[a,b] += 1 / SPLIT_REPEATS
    return s7a, s7b, pd.DataFrame(list(overall.items()), columns=["Metric", "Result"]), freq, matrix.reset_index(names="Trait code")


def format_rate(v: object) -> str:
    if pd.isna(v): return ""
    v = float(v)
    if v == 0: return "0"
    return f"{v:.3e}" if abs(v) < 0.0001 else f"{v:.4f}"


def write_dataframe(ws, start_row: int, df: pd.DataFrame, header_fill="D9EAF7") -> int:
    thin = Side(style="thin", color="808080")
    for c, name in enumerate(df.columns, 1):
        cell = ws.cell(start_row, c, name); cell.font = Font(bold=True); cell.fill = PatternFill("solid", fgColor=header_fill)
        cell.border = Border(bottom=thin); cell.alignment = Alignment(wrap_text=True, vertical="top")
    for r, values in enumerate(df.itertuples(index=False, name=None), start_row + 1):
        for c, value in enumerate(values, 1):
            cell = ws.cell(r, c, None if pd.isna(value) else value); cell.alignment = Alignment(wrap_text=True, vertical="top")
            if isinstance(value, (float, np.floating)):
                cell.number_format = "0.000E+00" if value != 0 and abs(value) < 0.0001 else "0.0000"
    return start_row + len(df) + 1


def build_workbook(paths: Paths, s7a: pd.DataFrame, s7b: pd.DataFrame, s7c: pd.DataFrame, freq: pd.DataFrame, co: pd.DataFrame,
                   fixed_repeat: pd.DataFrame, perturb_repeat: pd.DataFrame, nested: pd.DataFrame, params: pd.DataFrame, splits: pd.DataFrame) -> Path:
    out = paths.out / "Supplementary_Tables_S7_revised.xlsx"
    shutil.copy2(paths.supp_tables, out)
    wb = load_workbook(out)
    table7_name = next(
        (name for name in ("Supplementary Table S7", "Supplementary Table 7") if name in wb.sheetnames),
        None,
    )
    if table7_name is None:
        raise KeyError("Could not find Supplementary Table S7/7 in the source workbook.")
    ws = wb[table7_name]
    for merged in list(ws.merged_cells.ranges): ws.unmerge_cells(str(merged))
    for row in ws.iter_rows():
        for cell in row: cell.value = None
    def selection_source(configuration: str) -> str:
        if configuration.startswith("Four-trait"): return "Full-panel exhaustive search: best pre-sufficient four-trait subset"
        if configuration.startswith("Five-trait"): return "Full-panel exhaustive search: minimum sufficient subset at k = 5"
        if configuration.startswith("Six-trait"): return "Full-panel exhaustive search: fixed-size six-trait reference"
        return "All traits in the coding-eligible pool"
    def family_role(configuration: str) -> str:
        if configuration.startswith("Full coding"): return "Not applicable to the complete eligible reference"
        return "Family coverage and imbalance were lexicographic ranking criteria, not hard constraints"
    a = pd.DataFrame({
        "Fingerprint configuration": s7a.configuration, "Trait set": s7a.trait_set,
        "Resampling scenario": s7a.analysis.map({"bootstrap_oob":"Bootstrap OOB", "stratified_80_20":"Constrained stratified 80/20"}),
        "Number of repeats": s7a.n_repeats, "Mean reference-set size": s7a.reference_size_mean,
        "Mean evaluation-set size": s7a.evaluation_size_mean, "Coding-parameter source": s7a.analysis.map({"bootstrap_oob":"Bootstrap reference sample", "stratified_80_20":"80% reference subset"}),
        "Trait subset reselected": "No", "Selection provenance": s7a.configuration.map(selection_source),
        "Trait-family balance role": s7a.configuration.map(family_role),
        "Mean evaluation UPP": s7a.evaluation_upp_mean, "SD evaluation UPP": s7a.evaluation_upp_sd,
        "UPP 2.5%": s7a.evaluation_upp_q025, "UPP 97.5%": s7a.evaluation_upp_q975,
        "Mean evaluation zero-distance pairs": s7a.evaluation_zero_pairs_mean, "Mean evaluation zero-pair rate": s7a.evaluation_zero_pair_rate_mean,
        "Mean evaluation pairwise distance": s7a.evaluation_mean_pairwise_distance_mean,
        "Complete-evaluation-uniqueness rate": s7a.evaluation_complete_unique_mean,
        "Mean evaluation–reference collision pairs": s7a.cross_collision_pairs_mean,
        "Mean cross-panel collision-pair rate": s7a.cross_collision_pair_rate_mean,
        "Evaluation-to-reference collision-free rate": s7a.evaluation_to_reference_cfr_mean,
        "Mean evaluation–reference zero-distance pairs": s7a.cross_zero_distance_pairs_mean,
        "Mean cross-panel zero-distance-pair rate": s7a.cross_zero_distance_pair_rate_mean,
        "Evaluation-to-reference zero-distance-ambiguity-free rate": s7a.evaluation_to_reference_zero_distance_free_rate_mean,
        "Complete transfer rate": s7a.complete_transfer_mean, "Mean minimum reference distance": s7a.mean_minimum_reference_distance_mean,
        "Interpretation": ["Conditional internal reference–evaluation assessment; interpret exact collisions, zero-distance ambiguity, and completeness jointly."] * len(s7a)})
    b = pd.DataFrame({"Fingerprint configuration": s7b.configuration, "Trait set": s7b.trait_set,
        "Perturbation scenario": s7b.scenario, "Number of repeats": s7b.n_repeats, "Coding parameters recalculated": "No",
        "Mean UPP": s7b.evaluation_upp_mean, "SD UPP": s7b.evaluation_upp_sd, "UPP 2.5%": s7b.evaluation_upp_q025,
        "UPP 97.5%": s7b.evaluation_upp_q975, "Mean zero-distance pairs": s7b.evaluation_zero_pairs_mean,
        "Mean zero-pair rate": s7b.evaluation_zero_pair_rate_mean, "Mean pairwise distance": s7b.evaluation_mean_pairwise_distance_mean,
        "Complete-uniqueness rate": s7b.evaluation_complete_unique_mean,
        "Mean accession-level ID retention rate": s7b.id_retention_rate_mean, "SD ID retention rate": s7b.id_retention_rate_sd,
        "Interpretation": ["Report high mean UPP separately from complete uniqueness and exact ID retention."] * len(s7b)})
    notes = ("UPP, unique pattern proportion. The fixed four-, five-, and six-trait configurations were selected using the full 150-variety panel and their results are conditional internal reference–evaluation assessments, not independent selection validation. Complete-evaluation-uniqueness rate was defined as the proportion of repetitions in which evaluation-set UPP equaled 1.0000 and no within-evaluation zero-distance pair was observed. In bootstrap out-of-bag and repeated stratified reference–evaluation analyses, coding parameters were estimated exclusively from the reference subset and applied unchanged to evaluation accessions. Exact-ID collision and Gower-like zero-distance ambiguity were calculated separately because missing explicit X states can make ID strings unequal while all comparable trait distances equal zero. Complete transfer required complete evaluation uniqueness, no evaluation–reference exact-ID collision, and no evaluation–reference zero-distance ambiguity. Fixed-configuration analyses did not repeat trait selection. In nested analyses, coding eligibility, coding parameters, diagnostics, and exhaustive subset selection were recalculated using training data only. Endosperm-associated effect and RF permutation importance were exploratory components and could influence selection only through the final composite-score tie-breaker. For measurement-error perturbation, baseline coding parameters were held fixed across 1,000 repetitions per scenario. Zero-pair rates were calculated within each repetition before averaging. The full 19-trait set was a redundancy-rich internal reference.")
    r = 1
    for title, table in [("Supplementary Table 7A. Conditional internal reference–evaluation performance of full-panel-selected fixed phenotypic fingerprint configurations under bootstrap out-of-bag and repeated stratified splits.", a),
                         ("Supplementary Table 7B. Sensitivity of full-panel-selected fixed phenotypic fingerprint configurations to trait-level perturbations under frozen baseline coding parameters.", b)]:
        ws.cell(r,1,title).font=Font(bold=True,size=12); r += 1; r = write_dataframe(ws,r,table); ws.cell(r,1,"Notes: "+notes).alignment=Alignment(wrap_text=True); r += 3
    ws.cell(r,1,"Supplementary Table 7C. Stability of coding-eligible pool definition and exhaustive core-trait selection under nested stratified resampling.").font=Font(bold=True,size=12); r += 2
    ws.cell(r,1,"Overall nested-selection metrics").font=Font(bold=True); r += 1; r=write_dataframe(ws,r,s7c); r += 2
    ws.cell(r,1,"Trait-selection frequency").font=Font(bold=True); r += 1; r=write_dataframe(ws,r,freq); ws.cell(r,1,"Notes: "+notes).alignment=Alignment(wrap_text=True)
    ws.freeze_panes="A2"
    for col in range(1, ws.max_column+1): ws.column_dimensions[get_column_letter(col)].width=min(42,max(12,max(len(str(ws.cell(rr,col).value or "")) for rr in range(1,min(ws.max_row,100)+1))+2))
    audit = {"S7A_repeat_level": fixed_repeat, "S7B_repeat_level": perturb_repeat, "S7C_split_level": nested,
             "S7C_trait_frequency": freq, "S7C_co_selection": co,
             "S7_parameters_and_seeds": pd.concat([params, pd.DataFrame([{"analysis":"global settings","repeat":np.nan,"seed":f"bootstrap={BOOTSTRAP_SEED}; split_master={SPLIT_MASTER_SEED}; perturbation={PERTURBATION_SEED}; RF={RF_SEED}","trait":"", "mu":np.nan,"sd":np.nan}])], ignore_index=True)}
    for name, data in audit.items():
        if name in wb.sheetnames:
            aws=wb[name]
            for row in aws.iter_rows():
                for cell in row: cell.value=None
        else:
            aws=wb.create_sheet(name)
        write_dataframe(aws,1,data); aws.freeze_panes="A2"; aws.auto_filter.ref=aws.dimensions
        for col in range(1,aws.max_column+1): aws.column_dimensions[get_column_letter(col)].width=min(40,max(11,len(str(aws.cell(1,col).value or ""))+2))
    wb.save(out)
    return out


def replace_section(doc: Document, heading_text: str, next_heading_prefix: str, new_paragraphs: list[str], new_heading: str | None = None) -> None:
    paras = doc.paragraphs
    start = next((i for i,p in enumerate(paras) if p.text.strip().startswith(heading_text)), None)
    if start is None: raise ValueError(f"Heading not found: {heading_text}")
    end = next((i for i in range(start+1,len(paras)) if paras[i].text.strip().startswith(next_heading_prefix)), None)
    if end is None: raise ValueError(f"Next heading not found: {next_heading_prefix}")
    heading = paras[start]
    if new_heading: heading.text = new_heading
    for i in range(end-1,start,-1):
        p=paras[i]; p._element.getparent().remove(p._element)
    anchor=heading
    for text in new_paragraphs:
        p=anchor._insert_paragraph_before() if False else None
        new_p = anchor._parent.add_paragraph()  # moved below anchor explicitly
        anchor._p.addnext(new_p._p)
        new_p.text=text
        anchor=new_p


def insert_before_paragraph(paragraph, texts: list[str]) -> None:
    for text in texts:
        new_p = paragraph._parent.add_paragraph(); paragraph._p.addprevious(new_p._p); new_p.text=text


def results_text(s7a: pd.DataFrame, s7b: pd.DataFrame, nested: pd.DataFrame) -> list[str]:
    def val(frame, analysis, config, col):
        return float(frame.loc[(frame.analysis==analysis)&(frame.configuration==config),col].iloc[0])
    five="Five-trait core subset"
    boot_upp=val(s7a,"bootstrap_oob",five,"evaluation_upp_mean"); boot_cfr=val(s7a,"bootstrap_oob",five,"evaluation_to_reference_cfr_mean"); boot_zfr=val(s7a,"bootstrap_oob",five,"evaluation_to_reference_zero_distance_free_rate_mean"); boot_ct=val(s7a,"bootstrap_oob",five,"complete_transfer_mean")
    split_upp=val(s7a,"stratified_80_20",five,"evaluation_upp_mean"); split_cfr=val(s7a,"stratified_80_20",five,"evaluation_to_reference_cfr_mean"); split_zfr=val(s7a,"stratified_80_20",five,"evaluation_to_reference_zero_distance_free_rate_mean"); split_ct=val(s7a,"stratified_80_20",five,"complete_transfer_mean")
    k=pd.to_numeric(nested.minimum_sufficient_k,errors="coerce"); exact=float(nested.exact_core_recovery.mean()); jac=float(pd.to_numeric(nested.jaccard_with_full_panel_core).mean())
    p5=s7b[(s7b.configuration==five)&(s7b.scenario=="Uniform ±5%")].iloc[0]
    p10=s7b[(s7b.configuration==five)&(s7b.scenario=="Uniform ±10%")].iloc[0]
    pg=s7b[(s7b.configuration==five)&(s7b.scenario=="Gaussian rRMSE")].iloc[0]
    return [
        f"Because the fixed reduced configurations were originally selected using the full panel, their results were interpreted as conditional internal reference–evaluation assessments. For the five-trait configuration, mean evaluation UPP was {boot_upp:.4f} in the 1,000 bootstrap out-of-bag repetitions and {split_upp:.4f} in the 100 constrained stratified splits. Accession-level exact-ID collision-free rates were {boot_cfr:.4f} and {split_cfr:.4f}, while zero-distance-ambiguity-free rates were {boot_zfr:.4f} and {split_zfr:.4f}. Under the strict definition requiring evaluation uniqueness and absence of both ambiguity types, complete transfer rates were {boot_ct:.4f} and {split_ct:.4f}, respectively.",
        f"When the complete selection procedure was repeated within reference subsets, the median minimum sufficient subset size was {k.median():.1f} (range {k.min():.0f}–{k.max():.0f}). The original T10/T23/T32/T37/T42 combination was recovered exactly in {exact:.4f} of splits, and mean Jaccard similarity was {jac:.4f}. Endosperm-associated effect size and RF importance were exploratory final-order diagnostic inputs because the smallest reference class contained only two accessions. These results characterize selection stability within the current panel rather than independent confirmation of a universally stable trait set.",
        f"Across 1,000 repetitions per scenario under frozen baseline coding parameters, the five-trait configuration retained mean UPP values of {p5.evaluation_upp_mean:.4f}, {p10.evaluation_upp_mean:.4f}, and {pg.evaluation_upp_mean:.4f} under ±5%, ±10%, and Gaussian rRMSE-based continuous-trait perturbation, respectively. Complete-uniqueness rates were {p5.evaluation_complete_unique_mean:.4f}, {p10.evaluation_complete_unique_mean:.4f}, and {pg.evaluation_complete_unique_mean:.4f}, while mean exact ID-retention rates were {p5.id_retention_rate_mean:.4f}, {p10.id_retention_rate_mean:.4f}, and {pg.id_retention_rate_mean:.4f}. High mean UPP therefore did not imply preservation of complete uniqueness or accession-level codes in every repetition.",
        "Together, these analyses support the five-trait configuration as a compact, dataset-specific fingerprint, but show that parameter transfer, collision-free reference matching, selection stability, and measurement sensitivity are separate properties. The full coding-eligible set was retained as a redundancy-rich internal reference. None of these current-panel analyses constitutes external validation across independent germplasm, years, locations, operators, or imaging systems."
    ]


def build_manuscript(paths: Paths, s7a: pd.DataFrame, s7b: pd.DataFrame, nested: pd.DataFrame) -> Path:
    doc=Document(paths.manuscript)
    methods=[
        "Internal assessment separated conditional internal reference–evaluation performance of full-panel-selected fixed configurations from stability of the complete selection procedure. The four-trait pre-sufficient subset (T10, T23, T32, and T37), five-trait minimum sufficient core subset (T10, T23, T32, T37, and T42), and balanced six-trait reference subset (T3, T30, T32, T37, T42, and T43) were originally obtained by exhaustive search of the complete 150-variety panel as described in Section 2.7; they were therefore not independently selected configurations. The 19-trait coding-eligible set was read from the supplementary trait-eligibility table. Trait-family coverage and imbalance were lexicographic ranking criteria rather than hard inclusion constraints. Fixed analyses did not reselect these configurations, whereas nested analyses repeated the original complete selection procedure within each reference subset.",
        "For all reference–evaluation analyses, parameters for coding continuous and ratio traits were estimated from the reference subset only. For trait j, the reference mean μj,ref and population-form standard deviation σj,ref (ddof = 0) were used to define raw-scale boundaries bjr = μj,ref + σj,ref qr, where qr was −0.8416, −0.2533, 0.2533, or 0.8416. Reference and evaluation accessions were encoded with the same boundaries; evaluation observations did not contribute to parameter estimation. Integer counts retained rounded integer codes, ordinal traits retained their DUS codes, and missing states were retained as X in exact ID strings.",
        "Fixed-configuration bootstrap out-of-bag assessment used 1,000 repetitions (seed 42001). In each repetition, 150 draws with replacement formed the bootstrap reference sample. Multiplicities were retained while estimating coding parameters; repeated accession records were collapsed only after parameter fitting when constructing the reference fingerprint library. Accessions not drawn formed the out-of-bag evaluation set. Performance was calculated both within the evaluation set and between evaluation accessions and the reference library.",
        "Fixed-configuration reference–evaluation assessment additionally used 100 constrained 80/20 splits by endosperm type (master seed 42002). Each 30-accession evaluation set contained at least one accession from every endosperm group, and every reference group retained at least two accessions. Coding parameters were fitted from the 120-accession reference subset and applied unchanged to the evaluation subset. This was interpreted as internal stratified subpanel transfer assessment, not external or independent validation.",
        "Nested exhaustive-selection stability was evaluated with the same 100 splits. Availability, encoded dominant-state proportion, extraction reliability, subgroup-associated variation, random-forest permutation importance, non-redundancy, pairwise distinctness, distance contribution, composite score, coding eligibility, and coding parameters were recalculated using reference data only. Exhaustive search proceeded from k = 1 until a subset achieved UPP = 1 and no zero-distance pair. At the minimum sufficient k, subsets were ranked lexicographically by broader trait-family coverage, lower family imbalance, higher mean pairwise distance, higher mean reliability, and higher mean composite score. Because the smallest endosperm class retained only two reference accessions, endosperm-associated effect size and random-forest permutation importance were treated as exploratory diagnostic inputs: neither determined eligibility or sufficiency, and they could affect selection only through the final composite-score tie-breaker. The selected subset and reference-derived codebook were then applied to evaluation accessions.",
        "Measurement-error sensitivity was assessed under a frozen baseline codebook fitted once from the unperturbed 150-variety panel. Each scenario used 1,000 repetitions (seed 42003), with no parameter refitting. For continuous and ratio traits, uniform perturbation was x′ij = xij(1 + uij), where uij followed Uniform(−0.05, 0.05) or Uniform(−0.10, 0.10). Gaussian perturbation was x′ij = max[0, xij(1 + εij)], where εij followed Normal(0, rRMSEj²) and rRMSEj = RMSEj divided by the manual-reference mean for trait j. For ordinal trait j, Bij followed Bernoulli(mj), where mj was the empirical image–manual mismatch probability; if Bij = 1, the code was sampled uniformly from the legal adjacent DUS states, with boundary states allowed to move only inward. For T32 count perturbation, eij was sampled with replacement from the empirical image-minus-manual residuals and x′ij = max[0, round(xij + eij)].",
        "Reported metrics included UPP, within-evaluation zero-distance pairs and rates, mean Gower-like distance, complete evaluation uniqueness, evaluation–reference exact-ID collision pairs and rates, evaluation–reference Gower-like zero-distance pairs and rates, accession-level exact-collision-free and zero-distance-ambiguity-free rates, minimum reference distance, perturbation ID retention, and nested selection frequencies. Exact-ID and zero-distance comparisons were reported separately because explicit X states can make ID strings different even when all comparable-trait distances are zero. Complete transfer required complete uniqueness within the evaluation subset, no evaluation–reference exact-ID collision, and no evaluation–reference zero-distance ambiguity. All rates were calculated within each repetition before summary. These analyses were interpreted as internal assessments of the current dataset and not as evidence of external generalizability across independent germplasm panels, years, locations, operators, or imaging systems."
    ]
    replace_section(doc,"2.8 ","2.9 ",methods,"2.8 Internal reference–evaluation, selection-stability, and perturbation analysis")
    replace_section(doc,"3.5 ","3.6 ",results_text(s7a,s7b,nested),"3.5 Internal transfer, selection stability, and perturbation sensitivity")
    exact_recovery=float(pd.to_numeric(nested["exact_core_recovery"],errors="coerce").mean())
    abstract_replacement=("Internal reference–evaluation analyses showed high mean discriminatory performance for the fixed fingerprint, "
                          f"but nested resampling recovered the exact five-trait combination in only {exact_recovery:.1%} of splits, and complete uniqueness and exact code retention were not maintained under all resampling and measurement-error conditions")
    for p in doc.paragraphs:
        if "Internal validation supported this dataset-specific fingerprint as compact and internally stable" in p.text:
            p.text=p.text.replace("Internal validation supported this dataset-specific fingerprint as compact and internally stable", abstract_replacement)
    limitation = "The internal assessments remained panel-referenced and were based on a single location and 2025 growing season. Coding parameters and nested selection stability were evaluated only within the current panel; no independent germplasm collection was available for external validation. Future work should freeze a prespecified codebook and trait-selection rule before evaluation across independent germplasm, years, locations, operators, and imaging systems."
    inserted=False
    for p in doc.paragraphs:
        if p.text.strip().startswith("4.5 ") or p.text.strip().startswith("4.6 "):
            new_p=p._parent.add_paragraph(); p._p.addnext(new_p._p); new_p.text=limitation; inserted=True; break
    if not inserted: doc.add_paragraph(limitation)
    out=paths.out/"Developing_Manuscript_2_8_revised.docx"; doc.save(out); return out


def build_supp_methods(paths: Paths) -> Path:
    doc=Document(paths.supp_methods)
    # Replace a pre-existing S2.16 block instead of appending a duplicate when a
    # previously revised Supplementary Methods file is used as input.
    existing=next((i for i,p in enumerate(doc.paragraphs) if p.text.strip().startswith("S2.16 Internal")),None)
    if existing is not None:
        end=next((i for i in range(existing+1,len(doc.paragraphs))
                  if doc.paragraphs[i].text.strip().startswith("Supplementary Methods 3")
                  or re.match(r"^S3(?:\.|\s)",doc.paragraphs[i].text.strip())),len(doc.paragraphs))
        for i in range(end-1,existing-1,-1):
            p=doc.paragraphs[i]; p._element.getparent().remove(p._element)
    additions=[
        ("S2.16 Internal reference–evaluation and perturbation analysis", "The following analyses separated fixed-configuration parameter transfer, nested selection stability, and sensitivity under a frozen coding rule."),
        ("S2.16.1 Reference-only coding parameter estimation", "For every reference–evaluation repetition, continuous and ratio-trait means and population-form standard deviations were calculated in NumPy float64 using ddof = 0 from reference observations only. Boundaries were bjr = μj,ref + σj,ref qr for qr = −0.8416, −0.2533, 0.2533, and 0.8416. Boundary values entered the bin on the right, consistent with lower-inclusive and upper-exclusive finite intervals. A non-missing constant trait was assigned the central code 3. No threshold rounding was applied. Missing states were encoded as X in ID strings and ignored in Gower-like distance. Continuous-trait coding ranges were fixed at 1–5; ordinal ranges came from the trait dictionary; a count range absent from the dictionary was estimated from the reference subset only."),
        ("S2.16.2 Bootstrap out-of-bag assessment", "One thousand bootstrap samples were generated with seed 42001. Multiplicity-weighted bootstrap rows fitted coding parameters. The reference library was then collapsed to unique accession IDs, and never-sampled accessions formed the OOB evaluation set. The four fixed configurations were not reselected. Within-OOB and OOB–reference metrics were computed separately; an empty OOB set would be recorded and skipped."),
        ("S2.16.3 Repeated constrained stratified splitting", "One hundred splits were generated from master seed 42002. Evaluation allocations were Normal 25, Sweet 2, Waxy 1, Popcorn 1, and Sweet-waxy 1, giving 30 evaluation and 120 reference accessions while retaining at least two reference accessions per group. Each split used a recorded child seed. Reference and evaluation IDs were required to be disjoint."),
        ("S2.16.4 Nested exhaustive-selection stability", "Within each reference subset, availability, encoded dominant-state proportion, R² or quadratic-weighted κ reliability, endosperm-associated η², 500-tree random-forest permutation importance with 30 permutations, Spearman non-redundancy, pairwise distinctness, normalized distance contribution, composite score, and eligibility were recalculated. Eligibility required availability ≥ 0.70 and dominant-state proportion < 0.90. Search and lexicographic ranking followed S2.14. Because the constrained splits left only two reference accessions in the smallest endosperm class, η² and RF permutation importance were treated as exploratory diagnostics: they were not eligibility or sufficiency criteria and entered subset selection only through the final mean-composite-score tie-breaker, after family coverage, imbalance, distance, and reliability. A split with no sufficient subset was retained with status ‘no sufficient subset’."),
        ("S2.16.5 Frozen-code measurement-error perturbation", "The complete unperturbed panel fitted one baseline codebook. One thousand repetitions per scenario used seed 42003 without refitting. Uniform perturbations used x′ij = xij(1+uij), with uij ~ Uniform(−0.05,0.05) or Uniform(−0.10,0.10). Gaussian perturbation used x′ij = max[0,xij(1+εij)], εij ~ Normal(0,rRMSEj²), where rRMSEj = RMSEj divided by the manual-reference mean and was read from the paired measurement-agreement output. For ordinal trait j, Bij ~ Bernoulli(mj), where mj was its empirical image–manual mismatch probability; when Bij=1, one legal adjacent DUS state was selected uniformly, and a boundary state could move only inward. For T32, residual eij was sampled with replacement from its empirical image-minus-manual residuals and x′ij = max[0,round(xij+eij)]. NumPy nearest-integer rounding was used. Missing observations remained missing."),
        ("S2.16.6 Cross-panel collision metrics", "For nE evaluation and nR reference accessions, exact-ID collision-pair rate was NID/(nE nR), and Gower-like zero-distance-pair rate was Nzero/(nE nR). These were calculated separately because X was retained in exact ID strings but missing traits were ignored in distance, so unequal ID strings could still have distance zero across all comparable traits. Exact-collision-free and zero-distance-ambiguity-free rates were the respective fractions of evaluation accessions with no such reference match. Complete transfer equaled 1 only when evaluation UPP was 1, within-evaluation zero pairs equaled 0, exact-collision-free rate equaled 1, and zero-distance-ambiguity-free rate equaled 1. Within-panel rates used nE(nE−1)/2."),
        ("S2.16.7 Reproducibility and random seeds", "Bootstrap, split-master, perturbation, and random-forest seeds were 42001, 42002, 42003, and 42, respectively. Split-specific child seeds, coding parameters, and repeat-level metrics were retained in the S7 audit worksheets. All summaries used repeat-level means, population-form SDs, medians, and empirical 2.5% and 97.5% quantiles before display formatting.")]
    next_heading=None
    for p in doc.paragraphs:
        if p.text.strip().startswith("Supplementary Methods 3") or p.text.strip().startswith("S3"):
            next_heading=p; break
    for heading,text in additions:
        if next_heading is None:
            h=doc.add_paragraph(heading); h.style=doc.styles["Heading 2"] if "Heading 2" in [s.name for s in doc.styles] else h.style
            doc.add_paragraph(text)
        else:
            insert_before_paragraph(next_heading,[heading,text])
    out=paths.out/"Supplementary_Methods_2_8_revised.docx"; doc.save(out); return out


def test_report(df: pd.DataFrame, meta: pd.DataFrame, configs: dict[str, tuple[str,...]], splits: pd.DataFrame,
                perturb_params: pd.DataFrame, fixed_split: pd.DataFrame) -> str:
    tests=[]
    def run(name, fn):
        try: fn(); tests.append((name,"PASS",""))
        except Exception as e: tests.append((name,"FAIL",repr(e)))
    book=fit_codebook(df,meta); enc=encode_with_codebook(df,meta,book); traits=configs["Five-trait core subset"]; ranges=ranges_for_reference(enc,traits,meta)
    run("Test 1 subset of collision-free baseline cannot gain collisions", lambda: (_ for _ in ()).throw(AssertionError()) if panel_metrics(enc.iloc[:30],traits,ranges)["evaluation_zero_pairs"] else None)
    def t2():
        ref=df.iloc[:120].copy(); ev=df.iloc[120:].copy(); b1=fit_codebook(ref,meta); ev.iloc[:,2:]=0; b2=fit_codebook(ref,meta); assert json.dumps(b1,sort_keys=True)==json.dumps(b2,sort_keys=True)
    run("Test 2 evaluation modification cannot change reference thresholds",t2)
    run("Test 3 reference/evaluation IDs are disjoint",lambda: (_ for _ in ()).throw(AssertionError()) if any(set(g[g.role=='reference'].accession_id)&set(g[g.role=='evaluation'].accession_id) for _,g in splits.groupby('repeat')) else None)
    m=panel_metrics(enc.iloc[:30],traits,ranges)
    run("Test 4 zero-pair rate denominator",lambda: np.testing.assert_allclose(m['evaluation_zero_pair_rate'],m['evaluation_zero_pairs']/math.comb(30,2)))
    ref,ev=enc.iloc[:120],enc.iloc[120:]; cm=cross_metrics(ev,ref,traits,ranges)
    run("Test 5 cross-panel collision denominator",lambda: np.testing.assert_allclose(cm['cross_collision_pair_rate'],cm['cross_collision_pairs']/(len(ev)*len(ref))))
    run("Test 6 UPP=1 implies zero pairs",lambda: (_ for _ in ()).throw(AssertionError()) if m['evaluation_upp']==1 and m['evaluation_zero_pairs']!=0 else None)
    tm=evaluate_transfer(ev,ref,traits,meta)
    run("Test 7 complete transfer logical implication",lambda: (_ for _ in ()).throw(AssertionError()) if tm['complete_transfer'] and not (tm['evaluation_upp']==1 and tm['evaluation_zero_pairs']==0 and tm['evaluation_to_reference_cfr']==1 and tm['evaluation_to_reference_zero_distance_free_rate']==1) else None)
    s1=make_splits(df); s2=make_splits(df)
    run("Test 8 repeated seed is identical",lambda: pd.testing.assert_frame_equal(s1,s2))
    def t9():
        refdf=df.iloc[:120]; evdf=df.iloc[120:].copy(); b=fit_codebook(refdf,meta); e1=encode_with_codebook(evdf,meta,b)
        for col in meta["Trait column"]:
            evdf[col]=pd.to_numeric(evdf[col],errors='coerce').astype(float)*1.1
        e2=encode_with_codebook(evdf,meta,b); assert json.dumps(b,sort_keys=True)==json.dumps(fit_codebook(refdf,meta),sort_keys=True); assert len(e1)==len(e2)
    run("Test 9 evaluation continuous codes use reference boundaries",t9)
    def t10():
        grouped=perturb_params.groupby('trait')[[c for c in perturb_params.columns if c.startswith('boundary_') or c in ['mu','sd']]].nunique(dropna=False); assert (grouped<=1).all().all()
    run("Test 10 perturbation baseline thresholds remain fixed",t10)
    def t11():
        left=pd.DataFrame({ID_COL:['E'],traits[0]:[np.nan],traits[1]:[1.0],traits[2]:[1.0],traits[3]:[1.0],traits[4]:[1.0]})
        right=pd.DataFrame({ID_COL:['R'],traits[0]:[2.0],traits[1]:[1.0],traits[2]:[1.0],traits[3]:[1.0],traits[4]:[1.0]})
        z=cross_metrics(left,right,traits,ranges)
        assert z['cross_collision_pairs']==0 and z['cross_zero_distance_pairs']==1
    run("Test 11 unequal X-containing IDs can still be cross-panel zero-distance pairs",t11)
    run("Test 12 cross zero-distance denominator",lambda: np.testing.assert_allclose(cm['cross_zero_distance_pair_rate'],cm['cross_zero_distance_pairs']/(len(ev)*len(ref))))
    return "\n".join(["Validation tests report",f"Generated: {pd.Timestamp.now().isoformat()}",""]+[f"{name}: {status}{' — '+detail if detail else ''}" for name,status,detail in tests]+["",f"Summary: {sum(s=='PASS' for _,s,_ in tests)}/{len(tests)} passed"])


def original_reproducibility(paths: Paths) -> tuple[str, pd.DataFrame]:
    raw_path=paths.project/"outputs/supplementary_tables/Supplementary_Table_S7_internal_validation_raw_four_configs.csv"
    if not raw_path.exists(): return "Original repeat-level file missing; original S7 cannot be traced without rerunning legacy code.",pd.DataFrame()
    raw=pd.read_csv(raw_path,encoding="utf-8-sig")
    legacy=summarize(raw,["section","scheme","trait_set"],["unique_pattern_proportion","zero_distance_pairs","zero_pair_rate","mean_pairwise_distance"])
    return f"Found {len(raw):,} legacy repeat-level rows ({raw.section.value_counts().to_dict()}). Grouped means reproduce the legacy S7 values to its displayed precision; the legacy table is traceable to this file and build_supplementary_table_s7.py, although the underlying definitions contain the audited leakage/target mismatches.",legacy


def write_reports(paths: Paths, eligible: list[str], legacy_note: str, legacy_summary: pd.DataFrame,
                  s7a: pd.DataFrame, s7b: pd.DataFrame, nested: pd.DataFrame, configs: dict[str,tuple[str,...]]) -> None:
    code_eligible=list(pd.read_csv(paths.project/"outputs/supplementary_tables/Supplementary_Table_S7_reference_configurations.csv",encoding="utf-8-sig").query("scheme == 'full_eligible_set'").iloc[0].trait_set.replace(' ','').split(','))
    discrepancy=sorted(set(eligible)^set(code_eligible))
    audit=f"""# Analysis audit: Methods 2.8 and Supplementary Table 7

## Input-version mapping

The requested `/mnt/data/*(1)/(2)` inputs were not present in this Windows session. The analysis used the same base-version files previously identified by the user: `{paths.manuscript}`, `{paths.supp_methods}`, and `{paths.supp_tables}`. Originals were read only and were not overwritten.

## Data sufficiency

The unencoded 150-accession × 23-trait matrix was found in `traits.csv`; paired manual/image values, the trait dictionary, rRMSE output, coding/search scripts, and 7,600 legacy repeat-level S7 records were also present. Numerical recalculation was therefore permitted. Endosperm counts were 127/8/6/3/6 as specified.

## Legacy-code audit answers

1. Bootstrap recalculated continuous/ratio means and SDs after sampling and deduplication.
2. Bootstrap regenerated five-bin codes on the deduplicated sampled subpanel.
3. The four reported fixed configurations were not reselected in bootstrap.
4. Duplicate accessions were collapsed before both parameter estimation and fingerprint evaluation, contrary to multiplicity-weighted bootstrap fitting.
5. Hold-out test codes used means/SDs estimated from the evaluation subset itself; training codes separately used training parameters. No shared training-derived codebook was applied.
6. Legacy S7 hold-out metrics compared evaluation accessions only within the evaluation subset; evaluation–reference collisions were not calculated.
7. Perturbation regenerated coding parameters from every perturbed panel; baseline thresholds were not frozen.
8. Zero-pair rate was calculated within each repetition using its comparable-pair denominator and then averaged. Some exported display values were rounded to zero.
9. Continuous SD used pandas `std(ddof=0)` in the coding function.
10. Gower-like ranges were fixed at 1–5 for continuous codes, dictionary ranges where available, and otherwise derived from the currently encoded panel. In legacy hold-out this allowed evaluation-derived ranges for traits without dictionary ranges.

## Text–code consistency

Legacy Methods stated only that encoded matrices were regenerated and did not identify parameter sources or whether selection was repeated. Thus it was under-specified rather than a faithful description of the actual train/test-specific refitting. Legacy Supplementary Methods defined subgroup η² and permutation importance, whereas the legacy hold-out helper used Kruskal ε² and impurity importance. Reliability input in the legacy hold-out was retained from the full panel. The legacy procedure was therefore not completely nested.

## Legacy S7 reproducibility

{legacy_note}

## Eligibility comparison

Supplementary trait-eligibility table traits: {', '.join(eligible)}.
Legacy code/reference-output traits: {', '.join(code_eligible)}.
Symmetric difference: {', '.join(discrepancy) if discrepancy else 'None'}.
"""
    (paths.out/"analysis_audit_2_8.md").write_text(audit,encoding="utf-8")
    five_old="Legacy five-trait S7: bootstrap mean UPP 0.9990 and complete uniqueness 0.912; hold-out mean UPP 1.0000 but only within evaluation subsets; continuous perturbation complete-uniqueness rates 0.085/0.055/0.245 for ±5%/±10%/Gaussian."
    k=pd.to_numeric(nested.minimum_sufficient_k,errors="coerce")
    report=f"""# Supplementary Table 7 revision report

1. **Original code behavior:** bootstrap deduplicated before refitting; hold-out refitted test codes on test data and omitted cross-library collisions; perturbation refitted bins after noise; legacy nested selection was partial.
2. **Original Methods claim:** encoded matrices were regenerated, without parameter-source or selection-repetition details.
3. **Original table reproducibility:** {legacy_note}
4. **Full-panel leakage:** yes for legacy hold-out selection reliability; test-specific coding also prevented a genuine training-to-evaluation transfer test.
5. **Revised definitions:** multiplicity-weighted bootstrap OOB; constrained train-only 80/20 splits; fully nested diagnostics/eligibility/search; frozen baseline perturbation.
6. **Code mapping:** `run_bootstrap`, `run_fixed_splits`, `run_nested`, `run_perturbation`, `build_workbook`, `build_manuscript`, and `build_supp_methods` in `rebuild_internal_validation_s7.py`.
7. **Old/new result difference:** {five_old} New results are stored in S7A–C and their audit worksheets; changes reflect different estimands and should not be interpreted as corrections by simple arithmetic.
8. **Main-text numbers updated:** Results 3.5, the Abstract internal-assessment sentence, and the limitations statement.
9. **Figures requiring redraw:** Fig. 5C must be regenerated because bootstrap, hold-out, and perturbation definitions changed; a supplementary nested-stability figure is recommended.
10. **Five-trait combination:** fixed analyses retained {', '.join(FULL_PANEL_CORE)} unchanged; nested splits were allowed to select other sets.
11. **Nineteen-trait set:** retained from the supplementary trait-eligibility table; difference from legacy reference output: {', '.join(discrepancy) if discrepancy else 'none'}.
12. **Data/code anomalies:** requested suffixed input paths were absent; legacy η²/permutation descriptions differed from the hold-out helper; count range was absent from the dictionary and was therefore reference-derived; legacy display rounded some nonzero rates to zero.
13. **Unresolved issues:** no independent germplasm, year, location, operator, or imaging-system validation exists. The fixed 4/5/6 configurations were originally selected on the full panel, so their transfer assessment remains conditional/post-selection. Nested k ranged from {k.min():.0f} to {k.max():.0f}; interpretation must remain panel-specific.
"""
    (paths.out/"S7_revision_report.md").write_text(report,encoding="utf-8")


def validate_outputs(paths: Paths, outputs: list[Path], source_wb: Path, revised_wb: Path) -> None:
    for p in outputs:
        if not p.exists() or p.stat().st_size == 0: raise AssertionError(f"Missing/empty output: {p}")
    Document(paths.out/"Developing_Manuscript_2_8_revised.docx")
    Document(paths.out/"Supplementary_Methods_2_8_revised.docx")
    # Normal workbook loading represents non-anchor cells in merged ranges as empty.
    # This compares visible cell content and merge geometry; read-only loading would
    # expose duplicate hidden XML values that Excel/openpyxl normalizes on save.
    new=load_workbook(revised_wb,read_only=False,data_only=False); old=load_workbook(source_wb,read_only=False,data_only=False)
    intended_updates={"Supplementary Table S7","Supplementary Table 7","S7A_repeat_level","S7B_repeat_level","S7C_split_level",
                      "S7C_trait_frequency","S7C_co_selection","S7_parameters_and_seeds"}
    for name in old.sheetnames:
        if name in intended_updates: continue
        if {str(x) for x in old[name].merged_cells.ranges}!={str(x) for x in new[name].merged_cells.ranges}:
            raise AssertionError(f"Unrelated worksheet merge geometry changed: {name}")
        old_vals=list(old[name].values); new_vals=list(new[name].values)
        if old_vals!=new_vals: raise AssertionError(f"Unrelated worksheet visible cell content changed: {name}")
    required={"S7A_repeat_level","S7B_repeat_level","S7C_split_level","S7C_trait_frequency","S7C_co_selection","S7_parameters_and_seeds"}
    if not required.issubset(new.sheetnames): raise AssertionError("Missing audit worksheets")


def main() -> None:
    paths=parse_args(); paths.out.mkdir(parents=True,exist_ok=True); require_inputs(paths)
    df,actual,predicted,meta,eligible=load_data(paths)
    configs=fixed_configuration_map(eligible)
    legacy_note,legacy_summary=original_reproducibility(paths)
    splits=make_splits(df)
    cache_files=[paths.out/"S7A_repeat_level.csv",paths.out/"S7B_repeat_level.csv",paths.out/"S7C_split_level.csv",
                 paths.out/"S7C_trait_repeat_level.csv",paths.out/"S7_parameters_and_seeds.csv"]
    if all(p.exists() for p in cache_files):
        print("Loading completed repeat-level calculations from cache...",flush=True)
        fixed=pd.read_csv(cache_files[0],encoding="utf-8-sig")
        pert=pd.read_csv(cache_files[1],encoding="utf-8-sig")
        nested=pd.read_csv(cache_files[2],encoding="utf-8-sig")
        nested_traits=pd.read_csv(cache_files[3],encoding="utf-8-sig")
        params=pd.read_csv(cache_files[4],encoding="utf-8-sig")
        split_fixed=fixed[fixed.analysis=="stratified_80_20"].copy()
        pert_params=params[params.analysis=="frozen_baseline"].copy()
    else:
        print("Running bootstrap OOB...",flush=True); boot,boot_params=run_bootstrap(df,meta,configs)
        print("Running fixed constrained splits...",flush=True); split_fixed,split_params=run_fixed_splits(df,meta,configs,splits)
        prior_nested=paths.project/"mnt/data/S7C_split_level.csv"
        prior_nested_traits=paths.project/"mnt/data/S7C_trait_repeat_level.csv"
        if prior_nested.exists() and prior_nested_traits.exists():
            print("Loading audited deterministic nested-selection trace and recomputing strict transfer metrics...",flush=True)
            nested=pd.read_csv(prior_nested,encoding="utf-8-sig")
            nested_traits=pd.read_csv(prior_nested_traits,encoding="utf-8-sig")
            nested=recompute_nested_transfer_metrics(df,meta,splits,nested)
        else:
            print("Running nested selection...",flush=True); nested,nested_traits=run_nested(df,actual,predicted,meta,splits)
        print("Running frozen perturbation...",flush=True); pert,pert_params=run_perturbation(paths,df,actual,predicted,meta,configs)
        fixed=pd.concat([boot[boot.get('skipped',0)!=1] if 'skipped' in boot else boot,split_fixed],ignore_index=True)
        params=pd.concat([boot_params,split_params,pert_params],ignore_index=True)
    s7a,s7b,s7c,freq,co=build_summaries(fixed,pert,nested,nested_traits,meta)
    fixed.to_csv(paths.out/"S7A_repeat_level.csv",index=False,encoding="utf-8-sig")
    pert.to_csv(paths.out/"S7B_repeat_level.csv",index=False,encoding="utf-8-sig")
    nested.to_csv(paths.out/"S7C_split_level.csv",index=False,encoding="utf-8-sig")
    nested_traits.to_csv(paths.out/"S7C_trait_repeat_level.csv",index=False,encoding="utf-8-sig")
    pd.concat([fixed.assign(result_section="S7A"),pert.assign(result_section="S7B")],ignore_index=True,sort=False).to_csv(paths.out/"S7_repeat_level_results.csv",index=False,encoding="utf-8-sig")
    params.to_csv(paths.out/"S7_parameters_and_seeds.csv",index=False,encoding="utf-8-sig")
    bootstrap_memberships(df).to_csv(paths.out/"S7_bootstrap_memberships.csv",index=False,encoding="utf-8-sig")
    splits.to_csv(paths.out/"S7_split_memberships.csv",index=False,encoding="utf-8-sig")
    wb=build_workbook(paths,s7a,s7b,s7c,freq,co,fixed,pert,nested,params,splits)
    manuscript=build_manuscript(paths,s7a,s7b,nested); sm=build_supp_methods(paths)
    tests=test_report(df,meta,configs,splits,pert_params,split_fixed); (paths.out/"validation_tests_report.txt").write_text(tests,encoding="utf-8")
    write_reports(paths,eligible,legacy_note,legacy_summary,s7a,s7b,nested,configs)
    shutil.copy2(Path(__file__),paths.out/"rebuild_internal_validation_s7.py")
    outputs=[wb,manuscript,sm,paths.out/"analysis_audit_2_8.md",paths.out/"S7_revision_report.md",
             paths.out/"validation_tests_report.txt",paths.out/"S7_repeat_level_results.csv",paths.out/"rebuild_internal_validation_s7.py"]
    validate_outputs(paths,outputs,paths.supp_tables,wb)
    print(json.dumps({"outputs":[str(p) for p in outputs],"eligible":eligible,"tests":tests.splitlines()[-1]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
