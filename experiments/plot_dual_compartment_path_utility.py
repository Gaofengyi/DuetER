"""Plot paired Plain/CCADPE utility in the paper's original visual style."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor, black
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "revised_path_utility" / "summary.json"
FIGURE_ROOT = ROOT.parent / "results" / "generated" / "figures"
AUDIT_PATH = ROOT / "results" / "dual_compartment_full_p256" / "path_utility.json"
METHODS = (
    ("Semantic", "semantic", "#4C78A8"),
    ("Lexical", "lexical", "#F58518"),
    ("Dual", "dual", "#54A24B"),
)


def hatch_pdf(pdf: canvas.Canvas, x: float, y: float, w: float, h: float) -> None:
    pdf.saveState()
    clip = pdf.beginPath()
    clip.rect(x, y, w, h)
    pdf.clipPath(clip, stroke=0, fill=0)
    pdf.setStrokeColor(black)
    pdf.setLineWidth(0.45)
    offset = -h
    while offset <= w:
        pdf.line(x + offset, y, x + offset + h, y + h)
        offset += 3.2
    pdf.restoreState()


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = Path(r"C:\Windows\Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def centered(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    text_font: ImageFont.ImageFont,
) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=text_font, fill="black")


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = payload["datasets"]
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    width, height = 540.0, 216.0
    left, right, bottom, plot_top = 52.0, 10.0, 30.0, 163.0
    plot_w, plot_h, ymax = width - left - right, plot_top - bottom, 0.75
    bar_w, pair_gap, path_gap = 10.0, 1.2, 14.0
    pair_w = 2 * bar_w + pair_gap
    method_group_w = 3 * pair_w + 2 * path_gap

    pdf = canvas.Canvas(
        str(FIGURE_ROOT / "complete_corpus_path_utility.pdf"),
        pagesize=(width, height),
    )
    for tick in (0.0, 0.2, 0.4, 0.6):
        y = bottom + tick / ymax * plot_h
        pdf.setStrokeColor(HexColor("#CFCFCF"))
        pdf.setLineWidth(0.45)
        pdf.setDash(3, 3)
        pdf.line(left, y, width - right, y)
        pdf.setDash()
        pdf.setFillColor(black)
        pdf.setFont("Helvetica", 9.5)
        pdf.drawRightString(left - 6, y - 3.2, f"{tick:.1f}")
    pdf.setStrokeColor(black)
    pdf.setLineWidth(0.8)
    pdf.line(left, bottom, width - right, bottom)
    pdf.line(left, bottom, left, plot_top)

    slot_w = plot_w / len(rows)
    for dataset_index, row in enumerate(rows):
        center = left + slot_w * (dataset_index + 0.5)
        group_left = center - method_group_w / 2
        for method_index, (_, key, color) in enumerate(METHODS):
            pair_left = group_left + method_index * (pair_w + path_gap)
            values = (float(row["plain"][key]), float(row["ccadpe"][key]))
            for execution_index, value in enumerate(values):
                x = pair_left + execution_index * (bar_w + pair_gap)
                h = value / ymax * plot_h
                pdf.setFillColor(HexColor(color))
                pdf.setStrokeColor(black)
                pdf.setLineWidth(0.55)
                pdf.rect(x, bottom, bar_w, h, fill=1, stroke=1)
                if execution_index == 1:
                    hatch_pdf(pdf, x, bottom, bar_w, h)
            pdf.setFillColor(black)
            pdf.setFont("Helvetica-Bold" if key == "dual" else "Helvetica", 7.4)
            pdf.drawCentredString(
                pair_left + pair_w / 2,
                bottom + max(values) / ymax * plot_h + 4.2,
                f"{values[0]:.4f}/{values[1]:.4f}",
            )
        pdf.setFillColor(black)
        pdf.setFont("Helvetica", 10.2)
        pdf.drawCentredString(center, 13.0, str(row["dataset"]))

    pdf.saveState()
    pdf.translate(14.0, bottom + plot_h / 2)
    pdf.rotate(90)
    pdf.setFont("Helvetica", 10.5)
    pdf.drawCentredString(0, 0, "Outer-holdout nDCG@10")
    pdf.restoreState()

    legend_y = 196.0
    pdf.setFillColor(black)
    pdf.setFont("Helvetica-Bold", 10.0)
    pdf.drawString(36, legend_y, "Path")
    legend_x = 72.0
    for name, _, color in METHODS:
        pdf.setFillColor(HexColor(color))
        pdf.setStrokeColor(black)
        pdf.rect(legend_x, legend_y - 2, 13, 8, fill=1, stroke=1)
        pdf.setFillColor(black)
        pdf.setFont("Helvetica", 9.7)
        pdf.drawString(legend_x + 18, legend_y - 1, name)
        legend_x += 82
    pdf.setFont("Helvetica-Bold", 10.0)
    pdf.drawString(320, legend_y, "Execution")
    for legend_x, name, hatched in ((384.0, "Plain", False), (465.0, "CCADPE", True)):
        pdf.setFillColor(HexColor("#BDBDBD"))
        pdf.setStrokeColor(black)
        pdf.rect(legend_x, legend_y - 2, 13, 8, fill=1, stroke=1)
        if hatched:
            hatch_pdf(pdf, legend_x, legend_y - 2, 13, 8)
        pdf.setFillColor(black)
        pdf.setFont("Helvetica", 9.7)
        pdf.drawString(legend_x + 18, legend_y - 1, name)
    pdf.save()

    scale = 3
    image = Image.new("RGB", (int(width * scale), int(height * scale)), "white")
    draw = ImageDraw.Draw(image)

    def point(x: float, y: float) -> tuple[int, int]:
        return int(round(x * scale)), int(round((height - y) * scale))

    regular = load_font(28)
    value_font = load_font(22)
    value_bold = load_font(22, bold=True)
    legend_bold = load_font(29, bold=True)
    for tick in (0.0, 0.2, 0.4, 0.6):
        y = bottom + tick / ymax * plot_h
        x0, yy = point(left, y)
        x1, _ = point(width - right, y)
        for start in range(x0, x1, 18):
            draw.line([(start, yy), (min(start + 9, x1), yy)], fill="#CFCFCF", width=1)
        centered(draw, (left - 14) * scale, yy - 14, f"{tick:.1f}", regular)
    draw.line([point(left, bottom), point(width - right, bottom)], fill="black", width=3)
    draw.line([point(left, bottom), point(left, plot_top)], fill="black", width=3)

    for dataset_index, row in enumerate(rows):
        center = left + slot_w * (dataset_index + 0.5)
        group_left = center - method_group_w / 2
        for method_index, (_, key, color) in enumerate(METHODS):
            pair_left = group_left + method_index * (pair_w + path_gap)
            values = (float(row["plain"][key]), float(row["ccadpe"][key]))
            for execution_index, value in enumerate(values):
                x = pair_left + execution_index * (bar_w + pair_gap)
                h = value / ymax * plot_h
                x0, y0 = point(x, bottom + h)
                x1, y1 = point(x + bar_w, bottom)
                draw.rectangle((x0, y0, x1, y1), fill=color, outline="black", width=2)
                if execution_index == 1:
                    for offset in range(-(y1 - y0), x1 - x0 + 10, 10):
                        sx, sy = x0 + max(offset, 0), y1 - max(-offset, 0)
                        length = min(x1 - sx, sy - y0)
                        if length >= 0:
                            draw.line((sx, sy, sx + length, sy - length), fill="black", width=2)
            label_y = point(0, bottom + max(values) / ymax * plot_h + 6)[1]
            centered(
                draw,
                (pair_left + pair_w / 2) * scale,
                label_y,
                f"{values[0]:.4f}/{values[1]:.4f}",
                value_bold if key == "dual" else value_font,
            )
        centered(draw, center * scale, (height - 16) * scale, str(row["dataset"]), regular)

    y_label = Image.new("RGBA", (420, 42), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 0), "Outer-holdout nDCG@10", font=regular, fill="black")
    y_label = y_label.rotate(90, expand=True)
    image.paste(y_label, (2, int((image.height - y_label.height) / 2)), y_label)

    legend_text_y = (height - legend_y - 6) * scale
    draw.text((36 * scale, legend_text_y), "Path", font=legend_bold, fill="black")
    legend_x = 72.0
    for name, _, color in METHODS:
        x0, y0 = point(legend_x, legend_y + 6)
        x1, y1 = point(legend_x + 13, legend_y - 2)
        draw.rectangle((x0, y0, x1, y1), fill=color, outline="black", width=2)
        draw.text(((legend_x + 18) * scale, legend_text_y), name, font=regular, fill="black")
        legend_x += 82
    draw.text((320 * scale, legend_text_y), "Execution", font=legend_bold, fill="black")
    for legend_x, name, hatched in ((384.0, "Plain", False), (465.0, "CCADPE", True)):
        x0, y0 = point(legend_x, legend_y + 6)
        x1, y1 = point(legend_x + 13, legend_y - 2)
        draw.rectangle((x0, y0, x1, y1), fill="#BDBDBD", outline="black", width=2)
        if hatched:
            for offset in range(-24, 40, 9):
                sx, sy = x0 + max(offset, 0), y1 - max(-offset, 0)
                length = min(x1 - sx, sy - y0)
                if length >= 0:
                    draw.line((sx, sy, sx + length, sy - length), fill="black", width=2)
        draw.text(((legend_x + 18) * scale, legend_text_y), name, font=regular, fill="black")

    image.save(FIGURE_ROOT / "complete_corpus_path_utility.png")
    AUDIT_PATH.write_text(
        json.dumps(
            {
                "metric": payload["metric"],
                "comparison": "Plain/CCADPE under frozen exact-BM25 DuetRank",
                "datasets": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
