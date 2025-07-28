from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import ollama

def generate_textual_explanation(pred_class, gradcam_info, lime_info, occlusion_info, use_gpt=True):
    # Lesbare Diagnose-Beschreibungen
    diagnosis_map = {
        "colon_aca": "colon adenocarcinoma, a type of cancer that develops in the lining of the colon",
        "colon_n": "benign colon tissue without signs of cancer",
        "lung_aca": "lung adenocarcinoma, a type of cancer that originates in the glandular cells of the lung",
        "lung_scc": "lung squamous cell carcinoma, a form of lung cancer affecting squamous cells",
        "lung_n": "benign lung tissue without signs of cancer",
    }

    diagnosis_text = diagnosis_map.get(pred_class, f"condition: {pred_class}")

    if use_gpt:
        try:
            prompt = f"""
            Explain the following diagnosis in 2-3 sentences for a doctor colleague.

            Diagnosis: {diagnosis_text}
            Grad-CAM highlights: {gradcam_info}
            LIME focus: {lime_info}
            Occlusion sensitivity: {occlusion_info}

            Make it concise, medically correct, and easy to understand.
            """

            response = ollama.chat(
                model="deepseek-r1:8b",
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]

        except Exception:
            pass  # Falls GPT nicht funktioniert, nutze Fallback

    # Dynamische Fallback-Erklärung
    return (
        f"The diagnosis is **{diagnosis_text}**. "
        f"The Grad-CAM model emphasizes {gradcam_info}, highlighting key areas for examination. "
        f"LIME focuses on {lime_info}, pointing out relevant cell patterns or abnormalities. "
        f"However, accuracy may decrease if certain areas are not fully visible, "
        f"as occlusion sensitivity shows that the model relies on {occlusion_info} for correct classification."
    )

def generate_pdf_report(original_img, gradcam_img, lime_img, occlusion_img, explanation_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "ExplainMed Diagnosebericht")
    c.setFont("Helvetica", 12)
    c.drawString(50, 780, explanation_text[:90] + "...")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer