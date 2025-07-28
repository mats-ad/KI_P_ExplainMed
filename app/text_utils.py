from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import ollama

def generate_textual_explanation(pred_class, gradcam_info, lime_info, occlusion_info):
    prompt = f"""
    Erstelle eine kurze, ärztlich verständliche Erklärung für die Diagnose.

    Diagnose: {pred_class}
    Grad-CAM Hauptregionen: {gradcam_info}
    LIME Fokus: {lime_info}
    Occlusion Sensitivity: {occlusion_info}

    Antworte in 4-5 Sätzen, wie ein Arzt es einem Kollegen erklären würde. Answer in English I only want to see the answer and not the thought process
    """

    response = ollama.chat(
        model="deepseek-r1:8b",  # Lokales Modell, das du installiert hast
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def generate_text_explanation(label, probability):
    text = f"Das Modell diagnostiziert **{label}** mit einer Wahrscheinlichkeit von {probability:.2%}. "

    if "Adenocarcinoma" in label:
        text += "Die KI erkannte typische Muster für ein Adenokarzinom in auffälligen Regionen."
    elif "Squamous" in label:
        text += "Das Modell identifizierte verdickte epitheliale Strukturen, die für Plattenepithelkarzinome typisch sind."
    else:
        text += "Die erkannten Muster entsprechen gesunden Strukturen ohne Anzeichen von Karzinomen."

    return text

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