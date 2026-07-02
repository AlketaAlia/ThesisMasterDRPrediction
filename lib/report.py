"""PDF report generation."""
import io
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lib.config import PDF_IMAGE_PAGE_LIMIT, PDF_SUMMARY_ROW_LIMIT


def build_pdf_report(dataframe, report_entries, primary_model, threshold, tr):
    """Render a one-page summary plus up to PDF_IMAGE_PAGE_LIMIT detail pages.

    Truncation is disclosed on the cover page so users aren't surprised when
    the PDF is shorter than their CSV.
    """
    buffer = io.BytesIO()
    total_rows = len(dataframe)
    total_entries = len(report_entries)
    summary_truncated = total_rows > PDF_SUMMARY_ROW_LIMIT
    images_truncated = total_entries > PDF_IMAGE_PAGE_LIMIT

    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = [
            tr("report_title"),
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Primary model: {primary_model}",
            f"Threshold: {threshold:.2f}",
            f"Rows: {total_rows}",
        ]
        if summary_truncated:
            lines.append(
                f"Summary preview: showing first {PDF_SUMMARY_ROW_LIMIT} of {total_rows} rows"
            )
        if images_truncated:
            lines.append(
                f"Detail pages: showing first {PDF_IMAGE_PAGE_LIMIT} of {total_entries} images"
            )
        if summary_truncated or images_truncated:
            lines.append("(use the CSV export for the full set)")
        lines.append("")
        preview = dataframe[["file", "model", "prediction", "confidence"]].head(PDF_SUMMARY_ROW_LIMIT)
        lines.extend(preview.to_string(index=False).splitlines())
        ax.text(0.03, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for entry in report_entries[:PDF_IMAGE_PAGE_LIMIT]:
            fig, axes = plt.subplots(1, 2, figsize=(11.69, 5.5))
            axes[0].imshow(entry["image"])
            axes[0].set_title(entry["file"])
            axes[0].axis("off")
            axes[1].axis("off")
            text = (
                f"Model: {entry['model']}\n"
                f"Prediction: {entry['prediction']}\n"
                f"Probability DR: {entry['probability_dr']:.4f}\n"
                f"Confidence: {entry['confidence']:.2%}\n"
                f"Threshold: {entry['threshold']:.2f}\n"
                f"Image quality: {entry['quality_rating']}\n"
                f"Inference time: {entry['elapsed_ms']} ms"
            )
            axes[1].text(0.02, 0.9, text, va="top", fontsize=11)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()
