"""Generate a pandoc reference.docx that encodes the UPT thesis style.

Run once:
    .venv/bin/python master/thesis_word/upt/_make_reference_docx.py

Pandoc will then use the produced reference.docx as a style template:
    pandoc input.md -o output.docx --reference-doc reference.docx
"""
from __future__ import annotations

import os
from pathlib import Path

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.shared import Cm, Pt


def _style(doc, name, font_name="Times New Roman", size_pt=12, bold=False,
           line_spacing=1.0):
    """Configure a built-in style with UPT-friendly defaults."""
    s = doc.styles[name]
    s.font.name = font_name
    s.font.size = Pt(size_pt)
    s.font.bold = bold
    pf = s.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE if line_spacing == 1.0 else WD_LINE_SPACING.MULTIPLE
    if line_spacing != 1.0:
        pf.line_spacing = line_spacing
    pf.space_before = Pt(0)
    pf.space_after = Pt(6)


def main():
    out = Path(__file__).resolve().parent / "upt_reference.docx"
    doc = Document()

    # --- Page setup: A4, UPT margins (top/bottom/right 2.54cm, left 3.8cm) ---
    for section in doc.sections:
        section.page_height = Cm(29.7)
        section.page_width = Cm(21.0)
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(3.8)
        section.right_margin = Cm(2.54)

    # --- Body / Normal style: TNR 12pt, single spacing ---
    _style(doc, "Normal", size_pt=12, line_spacing=1.0)

    # --- Headings: KREU titles use Heading 1; section titles use Heading 2/3 ---
    _style(doc, "Heading 1", size_pt=16, bold=True, line_spacing=1.0)
    _style(doc, "Heading 2", size_pt=14, bold=True, line_spacing=1.0)
    _style(doc, "Heading 3", size_pt=12, bold=True, line_spacing=1.0)

    # --- TOC styles ---
    for tname in ["TOC 1", "TOC 2", "TOC 3"]:
        if tname in [s.name for s in doc.styles]:
            _style(doc, tname, size_pt=12, line_spacing=1.0)

    # --- Caption style for tables/figures: 11pt per UPT manual ---
    # Pandoc uses 'Image Caption' for figures and 'Caption' (or 'Table Caption')
    # for tables. We configure all three at 11pt so any of them applies.
    for cname in ["Caption", "Image Caption", "Table Caption"]:
        if cname in [s.name for s in doc.styles]:
            _style(doc, cname, size_pt=11, bold=False, line_spacing=1.0)
        else:
            # Add the style if it doesn't exist (python-docx exposes only
            # built-in styles by default; new paragraph styles can be added
            # via the styles XML root).
            from docx.enum.style import WD_STYLE_TYPE
            try:
                ns = doc.styles.add_style(cname, WD_STYLE_TYPE.PARAGRAPH)
                ns.font.name = "Times New Roman"
                ns.font.size = Pt(11)
                ns.font.bold = False
                pf = ns.paragraph_format
                pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
                pf.space_before = Pt(0)
                pf.space_after = Pt(6)
            except Exception:
                pass

    # --- Footnote text: 10pt per UPT manual ---
    for fname in ["footnote text", "Footnote Text"]:
        if fname in [s.name for s in doc.styles]:
            _style(doc, fname, size_pt=10, line_spacing=1.0)

    # Pandoc looks for these specific styles when building the output.
    # Add an empty paragraph so the doc isn't completely blank.
    doc.add_paragraph("")

    doc.save(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
