"""Render the compact complete-corpus plaintext/DPE utility figure."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from reportlab.lib.colors import Color, HexColor, black
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
DATASETS = ("nq", "hotpotqa", "msmarco")
DATASET_LABELS = {
    "nq": ("NQ", "2.68M"),
    "hotpotqa": ("HotpotQA", "5.23M"),
    "msmarco": ("MS MARCO", "8.84M"),
}
METHODS = ("semantic", "lexical", "dual")
METHOD_LABELS = {"semantic": "Semantic", "lexical": "Lexical", "dual": "Dual"}
COLORS = {"semantic": "#4C78A8", "lexical": "#F58518", "dual": "#54A24B"}


def hatch_segments(x0: float, y0: float, width: float, height: float, spacing: float):
    """Yield clipped bottom-left to top-right hatch segments for a rectangle."""
    offset = -height
    while offset <= width:
        sx = x0 + max(offset, 0.0)
        sy = y0 + max(-offset, 0.0)
        length = min(x0 + width - sx, y0 + height - sy)
        if length > 0:
            yield sx, sy, sx + length, sy + length
        offset += spacing


def draw_hatched_rect(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    *,
    hatch: bool,
    line_width: float = 0.45,
) -> None:
    c.setFillColor(HexColor(color))
    c.setStrokeColor(black)
    c.setLineWidth(line_width)
    c.rect(x, y, width, height, fill=1, stroke=1)
    if hatch:
        c.setStrokeColor(Color(0.12, 0.12, 0.12))
        c.setLineWidth(0.32)
        for x0, y0, x1, y1 in hatch_segments(x, y, width, height, 2.7):
            c.line(x0, y0, x1, y1)


def draw_pdf(rows: dict[str, dict], path: Path) -> None:
    # Compact 4.25 x 2.36 inch layout with dense grouped bars.
    width, height = 306.0, 170.0
    c = canvas.Canvas(str(path), pagesize=(width, height))
    c.setTitle("Complete-corpus plaintext and DPE path utility")

    # One-line legend shared by both panels.
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 6.5)
    c.drawString(18, 159.0, "Path")
    legend_x = 40.0
    c.setFont("Helvetica", 6.5)
    for method in METHODS:
        draw_hatched_rect(c, legend_x, 156.3, 8.0, 6.2, COLORS[method], hatch=False)
        c.setFillColor(black)
        c.drawString(legend_x + 10.0, 157.2, METHOD_LABELS[method])
        legend_x += 52.0 if method != "dual" else 43.0
    c.setFont("Helvetica-Bold", 6.5)
    c.drawString(190, 159.0, "Execution")
    legend_x = 232.0
    c.setFont("Helvetica", 6.5)
    for label, hatch in (("Plain", False), ("DPE", True)):
        draw_hatched_rect(c, legend_x, 156.3, 8.0, 6.2, "#BFBFBF", hatch=hatch)
        c.setFillColor(black)
        c.drawString(legend_x + 10.0, 157.2, label)
        legend_x += 40.0 if not hatch else 32.0

    # Absolute path utility without a truncated axis.
    left, right, bottom, top = 28.0, 296.0, 28.0, 140.0
    plot_width, plot_height = right - left, top - bottom
    ymax = 0.82
    centers = (70.0, 158.0, 246.0)
    method_offsets = (-18.0, 0.0, 18.0)
    bar_width = 6.8
    c.setFont("Helvetica", 6.2)
    for value in (0.0, 0.2, 0.4, 0.6):
        y = bottom + plot_height * value / ymax
        c.setStrokeColor(Color(0.78, 0.78, 0.78))
        c.setLineWidth(0.3)
        c.setDash(2, 2)
        c.line(left, y, right, y)
        c.setDash()
        c.setFillColor(black)
        c.drawRightString(left - 3.5, y - 2.0, f"{value:.1f}")
    c.setStrokeColor(black)
    c.setLineWidth(0.5)
    c.line(left, bottom, right, bottom)
    c.line(left, bottom, left, top)

    for dataset_index, dataset in enumerate(DATASETS):
        for method_index, method in enumerate(METHODS):
            center = centers[dataset_index] + method_offsets[method_index]
            values = (rows[dataset]["plaintext"][method], rows[dataset]["dpe"][method])
            value_tops = []
            for mode_index, value in enumerate(values):
                cx = center + (-bar_width / 2 if mode_index == 0 else bar_width / 2)
                y1 = bottom + plot_height * value / ymax
                value_tops.append(y1)
                draw_hatched_rect(
                    c,
                    cx - bar_width / 2,
                    bottom,
                    bar_width,
                    y1 - bottom,
                    COLORS[method],
                    hatch=mode_index == 1,
                    line_width=0.75 if method == "dual" else 0.4,
                )
            c.setFillColor(black)
            c.setFont("Helvetica-Bold" if method == "dual" else "Helvetica", 4.25)
            label_y = max(value_tops) + 2.2
            plain_label = f"{values[0]:.4f}".removeprefix("0")
            dpe_label = f"{values[1]:.4f}".removeprefix("0")
            c.drawCentredString(center, label_y, plain_label + "/" + dpe_label)
        c.setFillColor(black)
        name, _ = DATASET_LABELS[dataset]
        c.setFont("Helvetica", 6.2)
        c.drawCentredString(centers[dataset_index], bottom - 9.0, name)

    c.saveState()
    c.translate(9.5, bottom + plot_height / 2)
    c.rotate(90)
    c.setFont("Helvetica", 6.7)
    c.drawCentredString(0, 0, "nDCG@10")
    c.restoreState()
    c.showPage()
    c.save()


def render_preview(pdf_path: Path, output_base: Path) -> None:
    poppler = (
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe"
    )
    if not poppler.exists():
        raise FileNotFoundError(f"Poppler renderer not found: {poppler}")
    subprocess.run(
        [
            str(poppler),
            "-f",
            "1",
            "-singlefile",
            "-png",
            "-r",
            "300",
            str(pdf_path),
            str(output_base),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "results/plaintext_duetrank/summary.json"
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=ROOT.parent / "results/generated/figures/complete_corpus_path_utility",
    )
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    if not payload.get("all_datasets_match_expected_relationship", False):
        raise RuntimeError("expected relationship failed; refusing to render claim figure")
    rows = {row["dataset"]: row for row in payload["datasets"]}
    if any(dataset not in rows for dataset in DATASETS):
        raise ValueError("missing one or more complete-corpus path-utility datasets")
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_base.with_suffix(".pdf")
    draw_pdf(rows, pdf_path)
    render_preview(pdf_path, args.output_base)
    print(pdf_path)
    print(args.output_base.with_suffix(".png"))


if __name__ == "__main__":
    main()
