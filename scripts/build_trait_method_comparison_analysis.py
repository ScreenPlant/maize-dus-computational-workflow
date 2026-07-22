from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
MASTER_XLSX = ROOT / "Supplementary Table1-9.xlsx"
TRAIT_DICT = ROOT / "traits dictionary.csv"
OUTPUT_DIR = ROOT / "outputs" / "method_comparison_analysis"

MEASURED_CSV = OUTPUT_DIR / "measured_traits_agreement_summary.csv"
VISUAL_CSV = OUTPUT_DIR / "visual_traits_agreement_summary.csv"
CONFUSION_CSV = OUTPUT_DIR / "visual_traits_confusion_matrices_long.csv"
DETAILED_MEASURED_CSV = OUTPUT_DIR / "measured_traits_detailed_stats.csv"
REPORT_MD = OUTPUT_DIR / "trait_method_comparison_report.md"
WORKBOOK_XLSX = OUTPUT_DIR / "trait_method_comparison_analysis.xlsx"
WORKBOOK_XLSX_FALLBACK = OUTPUT_DIR / "trait_method_comparison_analysis_latest.xlsx"

CORE_TRAIT_IDS = [1, 2, 3, 6, 7, 9, 10, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 40, 41, 42, 43]


def load_trait_meta() -> pd.DataFrame:
    meta = pd.read_excel(MASTER_XLSX, sheet_name="Supplementary Table 1", header=1)
    meta = meta.dropna(subset=["Trait No.", "Trait name"]).copy()
    meta["Trait No."] = pd.to_numeric(meta["Trait No."], errors="coerce").astype(int)
    meta = meta[meta["Trait No."].isin(CORE_TRAIT_IDS)].copy()
    meta["Trait ID"] = meta["Trait No."].map(lambda x: f"T{x}")

    dict_df = pd.read_csv(TRAIT_DICT)
    dict_df["Trait No."] = pd.to_numeric(dict_df["Trait No."], errors="coerce").astype(int)
    meta = meta.merge(
        dict_df[["Trait No.", "Data type", "Code range"]],
        on="Trait No.",
        how="left",
    )
    meta["Trait class"] = meta["Observation Method"].map({"MS": "measured", "VG": "visual"})
    return meta.sort_values("Trait No.").reset_index(drop=True)


def load_table3() -> pd.DataFrame:
    return pd.read_excel(MASTER_XLSX, sheet_name="Supplementary Table 3", header=1)


def safe_float(value: float | int | np.floating | np.integer | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def concordance_correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    mx = float(np.mean(x))
    my = float(np.mean(y))
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    denominator = vx + vy + (mx - my) ** 2
    if denominator == 0:
        return None
    return 2 * cov / denominator


def near_zero_threshold(x: np.ndarray) -> float:
    mean_abs = float(np.mean(np.abs(x))) if len(x) else 0.0
    return max(1e-8, 0.01 * mean_abs)


def measured_category(
    *,
    r2: float | None,
    ccc: float | None,
    rrmse: float | None,
    slope: float | None,
    intercept: float | None,
    bias: float | None,
    x_mean: float,
    x_sd: float,
    heteroscedastic: bool,
) -> str:
    if None in {r2, ccc, rrmse, slope, intercept, bias}:
        return "Low agreement / calibration required"

    intercept_close = abs(float(intercept)) <= max(0.10 * x_sd, 0.05 * abs(x_mean)) if (x_sd > 0 or x_mean != 0) else abs(float(intercept)) <= 1e-8
    bias_close = abs(float(bias)) <= max(0.10 * x_sd, 0.02 * abs(x_mean)) if (x_sd > 0 or x_mean != 0) else abs(float(bias)) <= 1e-8
    if (
        float(r2) >= 0.90
        and float(ccc) >= 0.90
        and float(rrmse) <= 10
        and 0.90 <= float(slope) <= 1.10
        and intercept_close
        and bias_close
        and not heteroscedastic
    ):
        return "High agreement"
    if (
        float(r2) >= 0.75
        and float(ccc) >= 0.70
        and float(rrmse) <= 20
    ):
        return "Moderate agreement"
    return "Low agreement / calibration required"


def measured_interpretation(
    *,
    trait_id: str,
    category: str,
    r2: float | None,
    slope: float | None,
    intercept: float | None,
    bias: float | None,
    rrmse: float | None,
    mape_note: str,
    x_mean: float,
    x_sd: float,
    heteroscedastic: bool,
) -> str:
    if None in {r2, slope, intercept, bias, rrmse}:
        return f"{trait_id} had insufficient paired observations for robust agreement analysis."

    additive_bias = abs(float(intercept)) > max(0.10 * x_sd, 0.05 * abs(x_mean)) if (x_sd > 0 or x_mean != 0) else False
    proportional_bias = not (0.90 <= float(slope) <= 1.10)
    bias_small = abs(float(bias)) <= max(0.10 * x_sd, 0.02 * abs(x_mean)) if (x_sd > 0 or x_mean != 0) else abs(float(bias)) <= 1e-8

    pieces: list[str] = []
    if category == "High agreement":
        pieces.append(
            "Image-derived values showed strong agreement with manual measurements and can be considered reliable for direct phenotyping use under the current conditions."
        )
    elif float(r2) >= 0.85 and (proportional_bias or additive_bias or not bias_small):
        pieces.append(
            "Image-derived values tracked relative variation well, but direct substitution would require calibration because systematic bias remained."
        )
    elif category == "Moderate agreement":
        pieces.append(
            "Image-derived values reflected biological ranking reasonably well but were better suited to assisted measurement or trend analysis than uncorrected replacement."
        )
    else:
        pieces.append(
            "Agreement was limited, indicating that the image-derived values should not be treated as direct replacements without substantial recalibration."
        )

    if proportional_bias:
        pieces.append("The slope departed from 1, indicating proportional bias.")
    if additive_bias:
        pieces.append("The intercept departed from 0, indicating additive bias.")
    if not bias_small:
        pieces.append("Mean bias was non-negligible.")
    if heteroscedastic:
        pieces.append("Residual spread increased with trait magnitude, suggesting heteroscedastic error.")
    if mape_note:
        pieces.append(mape_note)
    return " ".join(pieces)


def measured_analysis(meta: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    detailed_rows: list[dict[str, object]] = []
    measured_meta = meta[meta["Trait class"] == "measured"].copy()

    for _, trait in measured_meta.iterrows():
        trait_no = int(trait["Trait No."])
        trait_id = str(trait["Trait ID"])
        x = pd.to_numeric(df[f"Trait_{trait_no}_Actual"], errors="coerce")
        y = pd.to_numeric(df[f"Trait_{trait_no}_Predict"], errors="coerce")
        mask = x.notna() & y.notna()
        x_pair = x[mask].to_numpy(dtype=float)
        y_pair = y[mask].to_numpy(dtype=float)
        n = len(x_pair)
        missing_n = int((~mask).sum())

        slope = intercept = r2 = pearson_r = bias = mae = rmse = rrmse = mape = ccc = None
        ba_mean_diff = ba_loa_lower = ba_loa_upper = None
        proportional_bias = additive_bias = heteroscedastic = None
        mape_note = ""

        if n >= 2:
            fit = linregress(x_pair, y_pair)
            slope = float(fit.slope)
            intercept = float(fit.intercept)
            pearson_r = float(fit.rvalue)
            r2 = float(fit.rvalue ** 2)
            diff = y_pair - x_pair
            bias = float(np.mean(diff))
            mae = float(mean_absolute_error(x_pair, y_pair))
            rmse = float(np.sqrt(mean_squared_error(x_pair, y_pair)))
            manual_mean = float(np.mean(x_pair))
            if manual_mean != 0:
                rrmse = float(rmse / manual_mean * 100.0)
            threshold = near_zero_threshold(x_pair)
            safe_mask = np.abs(x_pair) > threshold
            if safe_mask.all():
                mape = float(np.mean(np.abs((y_pair - x_pair) / x_pair)) * 100.0)
            elif safe_mask.any():
                mape = float(np.mean(np.abs((y_pair[safe_mask] - x_pair[safe_mask]) / x_pair[safe_mask])) * 100.0)
                mape_note = "MAPE was computed after excluding near-zero manual values and should be interpreted cautiously."
            else:
                mape_note = "MAPE was not stable because manual values were zero or near zero throughout."
            ccc = safe_float(concordance_correlation_coefficient(x_pair, y_pair))
            if n >= 3:
                sd_diff = float(np.std(diff, ddof=1))
                ba_mean_diff = bias
                ba_loa_lower = float(bias - 1.96 * sd_diff)
                ba_loa_upper = float(bias + 1.96 * sd_diff)
            x_sd = float(np.std(x_pair, ddof=1)) if n >= 2 else 0.0
            proportional_bias = abs(float(slope) - 1.0) > 0.10
            additive_bias = abs(float(intercept)) > 0.10 * x_sd if x_sd > 0 else abs(float(intercept)) > 1e-8
            resid = y_pair - (slope * x_pair + intercept)
            if n >= 3:
                abs_resid_corr = pearsonr(np.abs(resid), x_pair).statistic
                heteroscedastic = bool(pd.notna(abs_resid_corr) and abs_resid_corr > 0.30)
            else:
                heteroscedastic = False
            x_mean = manual_mean
            category = measured_category(
                r2=r2,
                ccc=ccc,
                rrmse=rrmse,
                slope=slope,
                intercept=intercept,
                bias=bias,
                x_mean=x_mean,
                x_sd=x_sd,
                heteroscedastic=bool(heteroscedastic),
            )
            interpretation = measured_interpretation(
                trait_id=trait_id,
                category=category,
                r2=r2,
                slope=slope,
                intercept=intercept,
                bias=bias,
                rrmse=rrmse,
                mape_note=mape_note,
                x_mean=x_mean,
                x_sd=x_sd,
                heteroscedastic=bool(heteroscedastic),
            )
        else:
            category = "Low agreement / calibration required"
            interpretation = f"{trait_id} had insufficient paired observations for robust agreement analysis."
            x_sd = np.nan

        rows.append(
            {
                "Trait ID": trait_id,
                "n": n,
                "slope": safe_float(slope),
                "intercept": safe_float(intercept),
                "R2": safe_float(r2),
                "Pearson r": safe_float(pearson_r),
                "Bias": safe_float(bias),
                "MAE": safe_float(mae),
                "RMSE": safe_float(rmse),
                "rRMSE (%)": safe_float(rrmse),
                "MAPE (%)": safe_float(mape),
                "CCC": safe_float(ccc),
                "agreement category": category,
                "interpretation": interpretation,
            }
        )
        detailed_rows.append(
            {
                "Trait ID": trait_id,
                "Trait name": trait["Trait name"],
                "n": n,
                "missing_n": missing_n,
                "manual_mean": safe_float(np.mean(x_pair) if n else None),
                "manual_sd": safe_float(np.std(x_pair, ddof=1) if n >= 2 else None),
                "image_mean": safe_float(np.mean(y_pair) if n else None),
                "image_sd": safe_float(np.std(y_pair, ddof=1) if n >= 2 else None),
                "slope": safe_float(slope),
                "intercept": safe_float(intercept),
                "regression_equation": f"Image-derived = {slope:.4f} × Manual + {intercept:.4f}" if slope is not None else "",
                "R2": safe_float(r2),
                "Pearson r": safe_float(pearson_r),
                "Bias": safe_float(bias),
                "MAE": safe_float(mae),
                "RMSE": safe_float(rmse),
                "rRMSE (%)": safe_float(rrmse),
                "MAPE (%)": safe_float(mape),
                "CCC": safe_float(ccc),
                "Bland-Altman mean difference": safe_float(ba_mean_diff),
                "Bland-Altman lower LoA": safe_float(ba_loa_lower),
                "Bland-Altman upper LoA": safe_float(ba_loa_upper),
                "proportional_bias_flag": "Yes" if proportional_bias else "No" if proportional_bias is not None else "",
                "additive_bias_flag": "Yes" if additive_bias else "No" if additive_bias is not None else "",
                "error_increases_with_magnitude": "Yes" if heteroscedastic else "No" if heteroscedastic is not None else "",
                "MAPE note": mape_note,
                "agreement category": category,
                "interpretation": interpretation,
            }
        )

    summary = pd.DataFrame(rows)
    detailed = pd.DataFrame(detailed_rows)
    numeric_cols = ["slope", "intercept", "R2", "Pearson r", "Bias", "MAE", "RMSE", "rRMSE (%)", "MAPE (%)", "CCC"]
    summary[numeric_cols] = summary[numeric_cols].round(4)
    detail_numeric = [
        "manual_mean",
        "manual_sd",
        "image_mean",
        "image_sd",
        "slope",
        "intercept",
        "R2",
        "Pearson r",
        "Bias",
        "MAE",
        "RMSE",
        "rRMSE (%)",
        "MAPE (%)",
        "CCC",
        "Bland-Altman mean difference",
        "Bland-Altman lower LoA",
        "Bland-Altman upper LoA",
    ]
    detailed[detail_numeric] = detailed[detail_numeric].round(4)
    return summary, detailed


def classify_visual_agreement(
    *,
    rho: float | None,
    weighted_kappa: float | None,
    exact_pct: float | None,
    adjacent_pct: float | None,
    sparse_class: bool,
) -> str:
    if None in {rho, weighted_kappa, exact_pct, adjacent_pct}:
        return "Low agreement"
    if (
        float(weighted_kappa) >= 0.80
        and float(adjacent_pct) >= 90
        and (
            (float(rho) >= 0.75 and float(exact_pct) >= 60)
            or (sparse_class and float(exact_pct) >= 75)
        )
    ):
        return "High ordinal agreement"
    if (
        (float(weighted_kappa) >= 0.60 and float(adjacent_pct) >= 80)
        or (sparse_class and float(weighted_kappa) >= 0.70 and float(exact_pct) >= 90 and float(adjacent_pct) >= 98)
    ):
        return "Moderate ordinal agreement"
    return "Low agreement"


def main_error_pattern(
    manual_codes: list[int],
    image_codes: list[int],
    exact_pct: float,
    adjacent_pct: float,
) -> str:
    states = sorted(pd.unique(pd.Series(manual_codes + image_codes)).astype(int).tolist())
    state_to_rank = {state: idx for idx, state in enumerate(states)}
    diffs = np.array([state_to_rank[i] - state_to_rank[m] for m, i in zip(manual_codes, image_codes)], dtype=float)
    over = int((diffs > 0).sum())
    under = int((diffs < 0).sum())
    if exact_pct >= 70:
        return "mostly exact matches"
    if adjacent_pct >= 85 and over + under > 0:
        if over >= under * 1.5:
            return "adjacent over-grading"
        if under >= over * 1.5:
            return "adjacent under-grading"
        return "mostly adjacent-grade confusion"
    if over >= under * 1.5 and over > 0:
        return "systematic over-grading"
    if under >= over * 1.5 and under > 0:
        return "systematic under-grading"
    return "mixed grade confusion"


def visual_interpretation(
    *,
    category: str,
    rho: float | None,
    weighted_kappa: float | None,
    exact_pct: float | None,
    adjacent_pct: float | None,
    pattern: str,
    sparse_class: bool,
) -> str:
    if None in {rho, weighted_kappa, exact_pct, adjacent_pct}:
        return "Insufficient paired ratings prevented robust ordinal-agreement analysis."
    pieces: list[str] = []
    if category == "High ordinal agreement":
        pieces.append(
            "Image-derived grades were closely aligned with manual DUS ratings in both ordinal ranking and chance-corrected agreement."
        )
    elif sparse_class and float(weighted_kappa) >= 0.70 and float(exact_pct) >= 90:
        pieces.append(
            "Agreement at the observed dominant grade was strong, but the low-information class distribution made rank-based coefficients conservative."
        )
    elif float(rho) >= 0.70 and float(weighted_kappa) < 0.50:
        pieces.append(
            "The image-derived grades preserved ordinal ranking reasonably well, but grade-level agreement was weaker, indicating systematic category displacement."
        )
    elif category == "Moderate ordinal agreement":
        pieces.append(
            "The image-derived grades captured part of the ordinal structure, but consistency at the exact grade level remained moderate."
        )
    else:
        pieces.append(
            "Ordinal agreement was limited, suggesting that image-derived grades should be used with caution and not treated as interchangeable with manual ratings."
        )
    pieces.append(f"The dominant error pattern was {pattern}.")
    if sparse_class:
        pieces.append("Some grades were sparse, so kappa should be interpreted alongside the confusion matrix and adjacent agreement.")
    return " ".join(pieces)


def visual_analysis(meta: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    visual_meta = meta[meta["Trait class"] == "visual"].copy()

    for _, trait in visual_meta.iterrows():
        trait_no = int(trait["Trait No."])
        trait_id = str(trait["Trait ID"])
        manual = pd.to_numeric(df[f"Trait_{trait_no}_Actual"], errors="coerce")
        image = pd.to_numeric(df[f"Trait_{trait_no}_Predict"], errors="coerce")
        mask = manual.notna() & image.notna()
        manual_pair = manual[mask].astype(int)
        image_pair = image[mask].astype(int)
        n = len(manual_pair)

        manual_dist = manual_pair.value_counts().sort_index().to_dict()
        image_dist = image_pair.value_counts().sort_index().to_dict()
        states = sorted(pd.unique(pd.concat([manual_pair, image_pair], ignore_index=True)).astype(int).tolist())
        num_classes = len(states)
        sparse_class = any(count < 5 for count in manual_dist.values()) or any(count < 5 for count in image_dist.values())

        rho = rho_p = weighted_kappa = linear_kappa = exact_pct = adjacent_pct = None
        pattern = "not estimable"

        if n >= 2:
            rho_res = spearmanr(manual_pair, image_pair)
            rho = safe_float(rho_res.statistic)
            rho_p = safe_float(rho_res.pvalue)
            weighted_kappa = safe_float(cohen_kappa_score(manual_pair, image_pair, weights="quadratic"))
            linear_kappa = safe_float(cohen_kappa_score(manual_pair, image_pair, weights="linear"))
            exact_pct = safe_float((manual_pair.to_numpy() == image_pair.to_numpy()).mean() * 100.0)

            state_to_rank = {state: idx for idx, state in enumerate(states)}
            manual_rank = manual_pair.map(state_to_rank).to_numpy(dtype=int)
            image_rank = image_pair.map(state_to_rank).to_numpy(dtype=int)
            adjacent_pct = safe_float((np.abs(image_rank - manual_rank) <= 1).mean() * 100.0)
            pattern = main_error_pattern(
                manual_pair.astype(int).tolist(),
                image_pair.astype(int).tolist(),
                float(exact_pct),
                float(adjacent_pct),
            )

            conf = pd.crosstab(manual_pair, image_pair, dropna=False)
            conf = conf.reindex(index=states, columns=states, fill_value=0)
            for manual_code in conf.index:
                for image_code in conf.columns:
                    confusion_rows.append(
                        {
                            "Trait ID": trait_id,
                            "Trait name": trait["Trait name"],
                            "Manual_code": int(manual_code),
                            "Image_code": int(image_code),
                            "Count": int(conf.loc[manual_code, image_code]),
                        }
                    )

        category = classify_visual_agreement(
            rho=rho,
            weighted_kappa=weighted_kappa,
            exact_pct=exact_pct,
            adjacent_pct=adjacent_pct,
            sparse_class=sparse_class,
        )
        interpretation = visual_interpretation(
            category=category,
            rho=rho,
            weighted_kappa=weighted_kappa,
            exact_pct=exact_pct,
            adjacent_pct=adjacent_pct,
            pattern=pattern,
            sparse_class=sparse_class,
        )

        rows.append(
            {
                "Trait ID": trait_id,
                "n": n,
                "number of classes": num_classes,
                "Spearman rho": safe_float(rho),
                "p-value": safe_float(rho_p),
                "weighted kappa": safe_float(weighted_kappa),
                "exact agreement (%)": safe_float(exact_pct),
                "adjacent agreement (%)": safe_float(adjacent_pct),
                "main error pattern": pattern,
                "agreement category": category,
                "interpretation": interpretation,
                "manual grade distribution": "; ".join(f"{k}:{v}" for k, v in manual_dist.items()),
                "image grade distribution": "; ".join(f"{k}:{v}" for k, v in image_dist.items()),
                "linear weighted kappa": safe_float(linear_kappa),
            }
        )

    summary = pd.DataFrame(rows)
    confusion = pd.DataFrame(confusion_rows)
    num_cols = ["Spearman rho", "p-value", "weighted kappa", "exact agreement (%)", "adjacent agreement (%)", "linear weighted kappa"]
    summary[num_cols] = summary[num_cols].round(4)
    return summary, confusion


def build_english_sections(measured: pd.DataFrame, visual: pd.DataFrame) -> tuple[str, str]:
    def join_trait_ids(series: pd.Series) -> str:
        ids = series.tolist()
        return ", ".join(ids) if ids else "none"

    direct_replace = measured[
        (measured["agreement category"] == "High agreement")
        & measured["interpretation"].str.contains("direct phenotyping use", na=False)
    ]["Trait ID"]
    trend_only = measured[
        measured["interpretation"].str.contains("trend analysis|calibration", case=False, na=False)
    ]["Trait ID"]
    low_measured = measured[measured["agreement category"] == "Low agreement / calibration required"]["Trait ID"]

    high_visual = visual[visual["agreement category"] == "High ordinal agreement"]["Trait ID"]
    moderate_visual = visual[visual["agreement category"] == "Moderate ordinal agreement"]["Trait ID"]
    low_visual = visual[visual["agreement category"] == "Low agreement"]["Trait ID"]

    results_text = (
        "Measured traits were evaluated against manual measurements using method-comparison statistics rather than a prediction-accuracy framework. "
        f"Traits showing the strongest practical agreement and the best support for direct phenotyping use were {join_trait_ids(direct_replace)}. "
        f"Several additional measured traits ({join_trait_ids(trend_only)}) captured relative variation well but showed systematic bias or calibration-sensitive error, indicating that they are more suitable for assisted measurement or post hoc linear correction than for uncorrected substitution. "
        f"Traits with limited agreement ({join_trait_ids(low_measured)}) should not be interpreted as direct replacements for manual measurements under the current pipeline. "
        "For visually assessed traits, ordinal agreement was summarized using Spearman’s rank correlation, weighted kappa, exact agreement, adjacent agreement, and confusion matrices. "
        f"High ordinal agreement was observed for {join_trait_ids(high_visual)}, whereas {join_trait_ids(moderate_visual)} retained ordinal structure but still showed grade-level disagreement. "
        f"Low-agreement traits ({join_trait_ids(low_visual)}) displayed either weak rank consistency or systematic grade displacement and therefore should be treated as supportive rather than interchangeable with manual DUS scoring."
    )

    methods_text = (
        "For measured traits, paired manual (Actual) and image-derived (Predict) values were compared trait by trait using ordinary least-squares regression with manual values as the independent variable and image-derived values as the dependent variable. "
        "For each trait, we reported the fitted regression equation, the coefficient of determination (R2) and Pearson’s correlation coefficient to describe linear association, together with bias, MAE, RMSE, relative RMSE (rRMSE), MAPE, and Lin’s concordance correlation coefficient (CCC). "
        "Bland–Altman statistics (mean difference and 95% limits of agreement) were additionally calculated to characterize absolute agreement. "
        "Interpretation focused on agreement, calibration bias, and substitution potential rather than predictive performance. "
        "For visually assessed traits, manual and image-derived DUS grades were treated as ordered categorical responses. "
        "We calculated Spearman’s rank correlation coefficient and its p-value to describe monotonic rank consistency, quadratic weighted Cohen’s kappa as the primary chance-corrected ordinal agreement metric, linear weighted kappa as a secondary reference, exact agreement, adjacent agreement based on neighboring ordered states, and confusion matrices with manual grades as rows and image-derived grades as columns. "
        "Trait-level conclusions were based on the combined evidence from association, agreement, and error-pattern diagnostics."
    )
    return results_text, methods_text


def write_workbook(
    measured: pd.DataFrame,
    measured_detailed: pd.DataFrame,
    visual: pd.DataFrame,
    confusion: pd.DataFrame,
    results_text: str,
    methods_text: str,
    path: Path,
) -> None:
    narrative = pd.DataFrame(
        {
            "Section": ["Results wording", "Methods wording"],
            "Text": [results_text, methods_text],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        measured.to_excel(writer, index=False, sheet_name="Measured summary", startrow=1)
        visual.to_excel(writer, index=False, sheet_name="Visual summary", startrow=1)
        measured_detailed.to_excel(writer, index=False, sheet_name="Measured detailed", startrow=1)
        confusion.to_excel(writer, index=False, sheet_name="Confusion matrices", startrow=1)
        narrative.to_excel(writer, index=False, sheet_name="English wording", startrow=1)

        for sheet_name, title in [
            ("Measured summary", "Measured traits | Method-comparison agreement summary"),
            ("Visual summary", "Visual traits | Ordinal agreement summary"),
            ("Measured detailed", "Measured traits | Detailed agreement diagnostics"),
            ("Confusion matrices", "Visual traits | Confusion matrices in long format"),
            ("English wording", "Suggested English wording for Results and Methods"),
        ]:
            ws = writer.book[sheet_name]
            ws["A1"] = title
            ws["A1"].font = Font(bold=True)
            ws.freeze_panes = "A3"
            for cell in ws[2]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def write_report(
    measured: pd.DataFrame,
    visual: pd.DataFrame,
    confusion: pd.DataFrame,
    results_text: str,
    methods_text: str,
) -> None:
    lines = [
        "# Trait Method-Comparison Analysis",
        "",
        "## Data Source",
        "",
        f"- Mother table: `{MASTER_XLSX}`",
        "- Supplementary Table 3 provided paired manual/reference and image-derived values.",
        "- Supplementary Table 1 and `traits dictionary.csv` provided trait metadata.",
        "",
        "## Measured Traits Summary",
        "",
        measured.to_markdown(index=False),
        "",
        "## Visually Assessed Traits Summary",
        "",
        visual[
            [
                "Trait ID",
                "n",
                "number of classes",
                "Spearman rho",
                "p-value",
                "weighted kappa",
                "exact agreement (%)",
                "adjacent agreement (%)",
                "main error pattern",
                "agreement category",
                "interpretation",
            ]
        ].to_markdown(index=False),
        "",
        "## Suggested Results Wording",
        "",
        results_text,
        "",
        "## Suggested Methods Wording",
        "",
        methods_text,
        "",
        "## Confusion Matrix Preview",
        "",
        confusion.head(40).to_markdown(index=False),
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_trait_meta()
    df = load_table3()

    measured, measured_detailed = measured_analysis(meta, df)
    visual, confusion = visual_analysis(meta, df)

    measured.to_csv(MEASURED_CSV, index=False, encoding="utf-8-sig")
    measured_detailed.to_csv(DETAILED_MEASURED_CSV, index=False, encoding="utf-8-sig")
    visual.to_csv(VISUAL_CSV, index=False, encoding="utf-8-sig")
    confusion.to_csv(CONFUSION_CSV, index=False, encoding="utf-8-sig")

    results_text, methods_text = build_english_sections(measured, visual)

    try:
        write_workbook(measured, measured_detailed, visual, confusion, results_text, methods_text, WORKBOOK_XLSX)
        workbook_written = WORKBOOK_XLSX
    except PermissionError:
        write_workbook(measured, measured_detailed, visual, confusion, results_text, methods_text, WORKBOOK_XLSX_FALLBACK)
        workbook_written = WORKBOOK_XLSX_FALLBACK

    write_report(measured, visual, confusion, results_text, methods_text)

    print(f"Wrote measured summary: {MEASURED_CSV}")
    print(f"Wrote visual summary: {VISUAL_CSV}")
    print(f"Wrote confusion matrices: {CONFUSION_CSV}")
    print(f"Wrote workbook: {workbook_written}")
    print(f"Wrote report: {REPORT_MD}")


if __name__ == "__main__":
    main()
