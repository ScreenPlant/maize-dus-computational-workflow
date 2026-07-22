from __future__ import annotations

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
from copy import copy
from itertools import combinations
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from docx import Document
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter


GROUP_ORDER = [
    "Normal type",
    "Sweet type",
    "Waxy type",
    "Sweet-waxy hybrid type",
    "Popcorn type",
]


METHOD_PARAGRAPH = (
    "For the four measured traits displayed in Fig. 3A (T30, T31, T36, and T37), pairwise "
    "differences among the five endosperm-type groups were examined using Games–Howell comparisons "
    "(Games and Howell, 1976), because group sizes were strongly unequal and homogeneity of variance "
    "was not assumed. For groups i and j, qij = √2|x̄i − x̄j|/√(si²/ni + sj²/nj), with sample "
    "variances calculated using n − 1 degrees of freedom. The comparison degrees of freedom were "
    "estimated by the Welch–Satterthwaite expression νij = (si²/ni + sj²/nj)²/"
    "[(si²/ni)²/(ni − 1) + (sj²/nj)²/(nj − 1)]. Multiplicity-adjusted P values were "
    "obtained from the studentized-range distribution using five groups and νij degrees of freedom. "
    "At α = 0.05, a compact letter display was generated separately for each trait by constructing a "
    "non-significance graph in which groups were connected when the Games–Howell adjusted P value was "
    "at least 0.05. Maximal mutually non-significant group sets were assigned shared lowercase letters, "
    "ordered by descending group mean; consequently, groups sharing at least one letter were not "
    "significantly different, whereas groups with no common letter differed significantly. A shared "
    "letter was interpreted as failure to detect a difference rather than evidence of equivalence. "
    "The 40 pairwise comparisons and the resulting display letters are reported in Supplementary Table 5."
)


FIG3_CAPTION = (
    "Fig. 3. Representative agreement and endosperm-type variation patterns of automated maize DUS "
    "traits. (A) Endosperm-type differences in four measured traits: T30, T31, T36, and T37. Lowercase "
    "letters show compact letter displays derived from Games–Howell pairwise comparisons using "
    "Welch–Satterthwaite degrees of freedom and studentized-range–adjusted P values (α = 0.05 within "
    "each trait). Groups sharing at least one letter were not significantly different; pairs with no "
    "common letter differed significantly. (B) Agreement between image-derived and manual reference "
    "measurements for the same four measured traits. Each point represents one maize variety based on "
    "paired variety-level observations (n = 150 for each trait). (C) Endosperm-type grade distributions "
    "of four visually assessed ordinal traits: T10, T40, T41, and T43. (D) Confusion matrices between "
    "automated predictions and expert consensus grades for the same four visually assessed ordinal traits."
)


REFERENCE = (
    "Games, P.A., Howell, J.F., 1976. Pairwise multiple comparison procedures with unequal N’s and/or "
    "variances: A Monte Carlo study. Journal of Educational Statistics 1, 113–125. "
    "https://doi.org/10.3102/10769986001002113."
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Add the formal Games–Howell basis for Fig. 3A.")
    parser.add_argument(
        "--manuscript",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--supp-tables",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--posthoc",
        type=Path,
        default=root / "results" / "manuscript_source_data" / "fig3" / "Fig3_panel_A_games_howell_posthoc.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "results" / "manuscript_source_data" / "fig3" / "Fig3_panel_A_boxplot_summary.csv",
    )
    parser.add_argument(
        "--attachment",
        type=Path,
        default=root / "results" / "manuscript_source_data" / "fig3" / "Fig3_Games_Howell_independent_audit.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "fig3_games_howell_basis_260715_v3",
    )
    return parser.parse_args()


def insert_after(paragraph, text: str):
    new_paragraph = paragraph._parent.add_paragraph()
    paragraph._p.addnext(new_paragraph._p)
    new_paragraph.style = paragraph.style
    new_paragraph.add_run(text)
    return new_paragraph


def clear_and_set(paragraph, text: str, bold_prefix: str | None = None) -> None:
    paragraph.clear()
    if bold_prefix and text.startswith(bold_prefix):
        paragraph.add_run(bold_prefix).bold = True
        paragraph.add_run(text[len(bold_prefix) :])
    else:
        paragraph.add_run(text)


def validate_pairwise(project_csv: Path, attachment: Path) -> pd.DataFrame:
    project = pd.read_csv(project_csv, encoding="utf-8-sig")
    attached = pd.read_csv(attachment, encoding="utf-8-sig")
    keys = ["Trait ID", "Group 1", "Group 2"]
    if len(project) != 40 or len(attached) != 40:
        raise AssertionError(f"Expected 40 comparisons; project={len(project)}, attachment={len(attached)}")
    merged = project.merge(attached, on=keys, suffixes=("_project", "_attachment"), validate="one_to_one")
    numeric = [
        "n1", "n2", "Mean 1", "Mean 2", "Mean difference (1-2)", "Games-Howell q",
        "Degrees of freedom", "Games-Howell adjusted P",
    ]
    for column in numeric:
        left = pd.to_numeric(merged[f"{column}_project"])
        right = pd.to_numeric(merged[f"{column}_attachment"])
        if not np.allclose(left, right, rtol=1e-12, atol=1e-14, equal_nan=True):
            raise AssertionError(f"Attachment differs from project CSV in {column}")
    if not (
        merged["Significant at 0.05_project"].astype(str)
        == merged["Significant at 0.05_attachment"].astype(str)
    ).all():
        raise AssertionError("Significance flags differ between attachment and project CSV")
    return project


def compact_letters(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for trait_id, frame in pairwise.groupby("Trait ID", sort=False):
        means: dict[str, float] = {}
        for _, row in frame.iterrows():
            means[str(row["Group 1"])] = float(row["Mean 1"])
            means[str(row["Group 2"])] = float(row["Mean 2"])
        ranked = sorted(GROUP_ORDER, key=lambda group: means[group], reverse=True)
        nonsignificant = {
            frozenset([str(row["Group 1"]), str(row["Group 2"])]):
                float(row["Games-Howell adjusted P"]) >= 0.05
            for _, row in frame.iterrows()
        }
        valid: list[set[str]] = []
        for size in range(1, len(ranked) + 1):
            for subset_tuple in combinations(ranked, size):
                subset = set(subset_tuple)
                if all(
                    nonsignificant.get(frozenset([a, b]), False)
                    for a, b in combinations(subset, 2)
                ):
                    valid.append(subset)
        maximal = [clique for clique in valid if not any(clique < other for other in valid)]
        rank = {group: index for index, group in enumerate(ranked)}
        maximal.sort(key=lambda clique: (min(rank[g] for g in clique), -len(clique)))
        assigned = {group: "" for group in GROUP_ORDER}
        for index, clique in enumerate(maximal):
            for group in clique:
                assigned[group] += alphabet[index]
        trait_name = str(frame["Trait name"].iloc[0])
        for group in GROUP_ORDER:
            rows.append(
                {
                    "Trait ID": trait_id,
                    "Trait name": trait_name,
                    "Endosperm type": group,
                    "n": int(frame.loc[frame["Group 1"].eq(group), "n1"].iloc[0])
                    if frame["Group 1"].eq(group).any()
                    else int(frame.loc[frame["Group 2"].eq(group), "n2"].iloc[0]),
                    "Mean": means[group],
                    "Games-Howell letter": assigned[group],
                }
            )
    return pd.DataFrame(rows)


def validate_letters(letters: pd.DataFrame, summary_path: Path) -> None:
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    check = letters.merge(
        summary[["Trait ID", "Endosperm Type", "Games-Howell letter"]],
        left_on=["Trait ID", "Endosperm type"],
        right_on=["Trait ID", "Endosperm Type"],
        suffixes=("_calculated", "_figure"),
        validate="one_to_one",
    )
    if not (
        check["Games-Howell letter_calculated"].astype(str)
        == check["Games-Howell letter_figure"].astype(str)
    ).all():
        raise AssertionError("Recalculated compact letters do not match Fig. 3 source data")


def revise_manuscript(source: Path, output: Path) -> None:
    document = Document(source)
    paragraphs = document.paragraphs
    heading_index = next(i for i, p in enumerate(paragraphs) if p.text.strip().startswith("2.5 Auxiliary assessment"))
    next_heading_index = next(i for i, p in enumerate(paragraphs) if p.text.strip().startswith("2.6 "))
    section = paragraphs[heading_index:next_heading_index]
    if any("Games–Howell" in p.text or "Games-Howell" in p.text for p in section):
        raise RuntimeError("Methods 2.5 already contains Games–Howell text")
    target = next(p for p in section if p.text.startswith("Kruskal–Wallis tests were used"))
    insert_after(target, METHOD_PARAGRAPH)

    caption = next(p for p in document.paragraphs if p.text.strip().startswith("Fig. 3."))
    clear_and_set(caption, FIG3_CAPTION, "Fig. 3.")
    supp_entry = next(p for p in document.paragraphs if p.text.strip().startswith("Supplementary Table 5."))
    clear_and_set(
        supp_entry,
        "Supplementary Table 5. Biological discrimination statistics for extracted maize DUS traits "
        "and Games–Howell pairwise comparisons underlying the Fig. 3A compact letter displays.",
    )

    if not any("Games, P.A., Howell, J.F., 1976" in p.text for p in document.paragraphs):
        fan = next(p for p in document.paragraphs if p.text.startswith("Fan, S."))
        insert_after(fan, REFERENCE)
    document.save(output)


def copy_style(source_cell, target_cell) -> None:
    if source_cell.has_style:
        target_cell._style = copy(source_cell._style)
    target_cell.number_format = source_cell.number_format
    target_cell.alignment = copy(source_cell.alignment)
    target_cell.font = copy(source_cell.font)
    target_cell.fill = copy(source_cell.fill)
    target_cell.border = copy(source_cell.border)
    target_cell.protection = copy(source_cell.protection)


def write_section_title(ws, row: int, text: str, max_column: int = 13) -> None:
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=max_column)
    cell = ws.cell(row, 1, text)
    copy_style(ws.cell(1, 1), cell)
    ws.row_dimensions[row].height = 30


def append_supplementary_table5(source: Path, output: Path, pairwise: pd.DataFrame, letters: pd.DataFrame) -> None:
    # Edit only the Table 5 XML. The original workbook contains merged cells
    # whose covered cells retain audit values in the underlying XML; a normal
    # openpyxl save would discard those values. Direct XML appending preserves
    # all pre-existing cells and every other XLSX member exactly.
    sheet_member = "xl/worksheets/sheet5.xml"
    with ZipFile(source, "r") as original_zip:
        edited_sheet = build_table5_xml(original_zip.read(sheet_member), pairwise, letters)
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as output_zip:
            for info in original_zip.infolist():
                payload = edited_sheet if info.filename == sheet_member else original_zip.read(info.filename)
                output_zip.writestr(info, payload)


def build_table5_xml(original_xml: bytes, pairwise: pd.DataFrame, letters: pd.DataFrame) -> bytes:
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    xml_namespace = "http://www.w3.org/XML/1998/namespace"
    ET.register_namespace("", namespace)

    def q(tag: str) -> str:
        return f"{{{namespace}}}{tag}"

    root = ET.fromstring(original_xml)
    sheet_data = root.find(q("sheetData"))
    if sheet_data is None:
        raise RuntimeError("Table 5 worksheet has no sheetData element")

    def find_cell(reference: str):
        cell = root.find(f".//{q('c')}[@r='{reference}']")
        if cell is None:
            raise RuntimeError(f"Reference style cell {reference} not found")
        return cell

    title_style = find_cell("A1").get("s", "0")
    header_styles = {col: find_cell(f"{get_column_letter(col)}2").get("s", "0") for col in range(1, 14)}
    data_styles = {col: find_cell(f"{get_column_letter(col)}3").get("s", "0") for col in range(1, 14)}

    def set_inline_cell(row_element, row_number: int, column: int, value: object, style: str) -> None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        reference = f"{get_column_letter(column)}{row_number}"
        attributes = {"r": reference, "s": style}
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            cell = ET.SubElement(row_element, q("c"), attributes)
            ET.SubElement(cell, q("v")).text = repr(float(value)) if isinstance(value, (float, np.floating)) else str(int(value))
        else:
            attributes["t"] = "inlineStr"
            cell = ET.SubElement(row_element, q("c"), attributes)
            inline = ET.SubElement(cell, q("is"))
            text = ET.SubElement(inline, q("t"))
            rendered = str(value)
            if rendered[:1].isspace() or rendered[-1:].isspace():
                text.set(f"{{{xml_namespace}}}space", "preserve")
            text.text = rendered

    def append_row(row_number: int, values: list[object], styles: dict[int, str]) -> None:
        row = ET.SubElement(sheet_data, q("row"), {"r": str(row_number)})
        for column, value in enumerate(values, 1):
            set_inline_cell(row, row_number, column, value, styles.get(column, "0"))

    def append_title(row_number: int, text: str) -> None:
        append_row(row_number, [text], {1: title_style})

    # Replace only the visible title cell; all other original cells remain.
    title_cell = find_cell("A1")
    for child in list(title_cell):
        title_cell.remove(child)
    title_cell.set("t", "inlineStr")
    inline = ET.SubElement(title_cell, q("is"))
    ET.SubElement(inline, q("t")).text = (
        "Supplementary Table 5 | Biological discrimination statistics for extracted maize DUS traits "
        "and Games–Howell pairwise comparisons underlying Fig. 3A"
    )

    append_title(
        97,
        "Supplementary Table 5B. Games–Howell pairwise comparisons among endosperm types for the four measured traits displayed in Fig. 3A.",
    )
    append_row(98, list(pairwise.columns), header_styles)
    for offset, values in enumerate(pairwise.itertuples(index=False, name=None), start=99):
        append_row(offset, list(values), data_styles)
    append_title(
        139,
        "Notes: Games–Howell comparisons used sample variances with n − 1 degrees of freedom, Welch–Satterthwaite comparison degrees of freedom, and studentized-range adjusted P values for five groups. Significance was defined as adjusted P < 0.05 within each trait. These P values were not the Benjamini–Hochberg FDR values used for the trait-wise omnibus tests in Supplementary Table 5A.",
    )
    append_title(142, "Supplementary Table 5C. Compact letter displays used in Fig. 3A.")
    append_row(143, list(letters.columns), header_styles)
    for offset, values in enumerate(letters.itertuples(index=False, name=None), start=144):
        append_row(offset, list(values), data_styles)
    append_title(
        164,
        "Notes: Within each trait, groups sharing at least one lowercase letter were not significantly different by Games–Howell comparison at adjusted P ≥ 0.05; pairs with no common letter differed at adjusted P < 0.05. Shared letters indicate failure to detect a difference, not statistical equivalence. Letters were assigned to maximal mutually non-significant group sets ordered by descending group mean.",
    )

    merge_cells = root.find(q("mergeCells"))
    if merge_cells is None:
        merge_cells = ET.Element(q("mergeCells"))
        root.insert(list(root).index(sheet_data) + 1, merge_cells)
    for reference in ["A97:M97", "A139:M139", "A142:M142", "A164:M164"]:
        ET.SubElement(merge_cells, q("mergeCell"), {"ref": reference})
    merge_cells.set("count", str(len(merge_cells.findall(q("mergeCell")))))

    dimension = root.find(q("dimension"))
    if dimension is not None:
        dimension.set("ref", "A1:N164")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _append_table5_content(ws, pairwise: pd.DataFrame, letters: pd.DataFrame) -> None:
    ws.cell(1, 1).value = (
        "Supplementary Table 5 | Biological discrimination statistics for extracted maize DUS traits "
        "and Games–Howell pairwise comparisons underlying Fig. 3A"
    )
    start = ws.max_row + 3
    write_section_title(
        ws, start,
        "Supplementary Table 5B. Games–Howell pairwise comparisons among endosperm types for the four measured traits displayed in Fig. 3A.",
    )
    header_row = start + 1
    for col, header in enumerate(pairwise.columns, 1):
        target = ws.cell(header_row, col, header)
        copy_style(ws.cell(2, min(col, ws.max_column)), target)
    for r_offset, row in enumerate(pairwise.itertuples(index=False, name=None), 1):
        for col, value in enumerate(row, 1):
            target = ws.cell(header_row + r_offset, col, value)
            copy_style(ws.cell(3, min(col, ws.max_column)), target)
    note_row = header_row + len(pairwise) + 1
    write_section_title(
        ws, note_row,
        "Notes: Games–Howell comparisons used sample variances with n − 1 degrees of freedom, Welch–Satterthwaite comparison degrees of freedom, and studentized-range adjusted P values for five groups. Significance was defined as adjusted P < 0.05 within each trait. These P values were not the Benjamini–Hochberg FDR values used for the trait-wise omnibus tests in Supplementary Table 5A.",
    )

    letter_title = note_row + 3
    write_section_title(
        ws, letter_title,
        "Supplementary Table 5C. Compact letter displays used in Fig. 3A.",
    )
    letter_header = letter_title + 1
    for col, header in enumerate(letters.columns, 1):
        target = ws.cell(letter_header, col, header)
        copy_style(ws.cell(2, min(col, ws.max_column)), target)
    for r_offset, row in enumerate(letters.itertuples(index=False, name=None), 1):
        for col, value in enumerate(row, 1):
            target = ws.cell(letter_header + r_offset, col, value)
            copy_style(ws.cell(3, min(col, ws.max_column)), target)
    final_note = letter_header + len(letters) + 1
    write_section_title(
        ws, final_note,
        "Notes: Within each trait, groups sharing at least one lowercase letter were not significantly different by Games–Howell comparison at adjusted P ≥ 0.05; pairs with no common letter differed at adjusted P < 0.05. Shared letters indicate failure to detect a difference, not statistical equivalence. Letters were assigned to maximal mutually non-significant group sets ordered by descending group mean.",
    )

    widths = {1: 12, 2: 40, 3: 27, 4: 27, 5: 8, 6: 8, 7: 16, 8: 16, 9: 20, 10: 18, 11: 18, 12: 22, 13: 20}
    for col, width in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = max(
            ws.column_dimensions[get_column_letter(col)].width or 0, width
        )


def write_report(path: Path, manuscript: Path, tables: Path, pairwise: pd.DataFrame, letters: pd.DataFrame) -> None:
    significant = int(pairwise["Significant at 0.05"].astype(str).str.casefold().eq("yes").sum())
    lines = [
        "# Fig. 3 Games–Howell method and letter-display audit",
        "",
        f"- Revised manuscript: `{manuscript}`",
        f"- Revised supplementary tables: `{tables}`",
        f"- Pairwise comparisons: {len(pairwise)} (4 traits × 10 group pairs)",
        f"- Comparisons with Games–Howell adjusted P < 0.05: {significant}",
        f"- Compact-letter rows: {len(letters)} (4 traits × 5 endosperm groups)",
        "- Attachment and project post-hoc CSV agreed for every numeric field and significance flag.",
        "- Recalculated maximal-clique compact letters agreed with all letters in the Fig. 3 source summary.",
        "- The original manuscript and workbook were not overwritten.",
        "",
        "## Display letters",
        "",
        letters.to_markdown(index=False),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise = validate_pairwise(args.posthoc.resolve(), args.attachment.resolve())
    letters = compact_letters(pairwise)
    validate_letters(letters, args.summary.resolve())

    manuscript_output = output_dir / "Developing_Manuscript_260715_Fig3_Games_Howell_revised.docx"
    tables_output = output_dir / "Supplementary_Tables_260715_Fig3_Games_Howell_revised.xlsx"
    if manuscript_output.exists() or tables_output.exists():
        raise FileExistsError("Output exists; choose a new output directory to preserve prior results.")
    revise_manuscript(args.manuscript.resolve(), manuscript_output)
    append_supplementary_table5(args.supp_tables.resolve(), tables_output, pairwise, letters)
    pairwise.to_csv(output_dir / "Fig3_Games_Howell_40_pairwise_comparisons.csv", index=False, encoding="utf-8-sig")
    letters.to_csv(output_dir / "Fig3_compact_letter_display.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir / "Fig3_Games_Howell_audit.md", manuscript_output, tables_output, pairwise, letters)
    print(manuscript_output)
    print(tables_output)


if __name__ == "__main__":
    main()
