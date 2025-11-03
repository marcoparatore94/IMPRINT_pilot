# utils/pdf_export.py
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def export_patient_report(pdf_path: str, inputs: dict, prob: float, cls: str, recommendation: str):
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        header = "IMPRINT Risk Report"
        ax.text(0, 1, header, fontsize=16, weight="bold", va="top")

        lines = ["Input del paziente:"]
        for k, v in inputs.items():
            lines.append(f" - {k}: {v}")
        lines.append("")
        lines.append(f"Probabilità di malignità: {prob:.2%}")
        lines.append(f"Classe di rischio: {cls}")
        lines.append(f"Suggerimento: {recommendation}")
        lines.append("")
        lines.append("Nota: modello pilota addestrato su coorte simulata (benigni). "
                     "Da ricalibrare con dati benigni reali.")

        ax.text(0, 0.95, "\n".join(lines), fontsize=11, va="top", wrap=True)
        pdf.savefig(fig)
        plt.close(fig)