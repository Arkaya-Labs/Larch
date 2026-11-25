#!/usr/bin/env python3
"""Writes lease-alder-street.pdf, a text PDF of ../docs/lease-alder-street.md (each line of the
Markdown that has text a line of the PDF, so pdftotext and the built-in reader give the same lines),
and copies the scanned passport beside it. Needs reportlab.

    python3 bench/agent/docs-pdf/make_pdf.py
"""

import pathlib
import shutil

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

HERE = pathlib.Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"


def main():
    lines = (DOCS / "lease-alder-street.md").read_text().splitlines()
    pdf = canvas.Canvas(str(HERE / "lease-alder-street.pdf"), pagesize=letter, invariant=1)
    pdf.setFont("Helvetica", 9)
    y = 750
    for line in filter(None, lines):
        if y < 40:
            pdf.showPage()
            pdf.setFont("Helvetica", 9)
            y = 750
        pdf.drawString(40, y, line)
        y -= 12
    pdf.save()
    shutil.copy(DOCS / "scan-passport.pdf", HERE / "scan-passport.pdf")


if __name__ == "__main__":
    main()
