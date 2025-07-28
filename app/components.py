import streamlit as st
from PIL import Image
from xai_utils import gradcam_explain, lime_explain, occlusion_sensitivity_analysis
from text_utils import generate_text_explanation, generate_pdf_report, generate_textual_explanation
import torch

def landing_page(navigate):
    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(to right, #EBF4FF, #FFFFFF);
            padding: 70px 40px;
            text-align: center;
            border-radius: 16px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        .hero h1 {
            color: #1F2937;
            font-size: 44px;
            margin-bottom: 12px;
            font-weight: 700;
        }
        .hero p {
            color: #4B5563;
            font-size: 20px;
            margin-bottom: 20px;
        }
        .cta-btn {
            background: #2F80ED;
            color: white;
            padding: 14px 30px;
            border-radius: 8px;
            font-size: 18px;
            text-decoration: none;
            transition: background 0.3s ease-in-out, transform 0.2s ease-in-out;
        }
        .cta-btn:hover {
            background: #2563EB;
            transform: scale(1.03);
        }
        .feature-card {
            background: #FFFFFF;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            text-align: center;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        .feature-card:hover {
            transform: scale(1.03);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .feature-card h3 {
            color: #2F80ED;
            font-weight: 600;
        }
        .demo-box {
            margin-top: 50px;
            text-align: center;
        }
        .demo-img {
            border-radius: 12px;
            cursor: pointer;
            border: 3px solid #2F80ED;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        .demo-img:hover {
            transform: scale(1.03);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        </style>
        """, 
        unsafe_allow_html=True,
    )

    # Hero Section
    st.markdown(
        """
        <div class="hero">
            <h1>ExplainMed – Erklärbare KI-Diagnostik für Ärzte</h1>
            <p>Schnellere, nachvollziehbare Diagnosen für bessere Patientenversorgung.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Warum ExplainMed?")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("<div class='feature-card'>🧠<br><b>KI-gestützte Analyse</b><br>Automatische Klassifikation mit hoher Genauigkeit.</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<div class='feature-card'>🔍<br><b>Erklärbare Ergebnisse</b><br>Grad-CAM, LIME & Occlusion für Transparenz.</div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("<div class='feature-card'>📄<br><b>Sofortige Berichte</b><br>PDF-Export für Patientenakten.</div>", unsafe_allow_html=True)

    # Demo-Box unten
    st.markdown(
    """
    <div class="demo-box">
        <h3 style="color:#D0DDFE;">💻 Schau dir das Dashboard in Aktion an:</h3>
    </div>
    """,
    unsafe_allow_html=True,
    )

    dashboard_img = "./assets/dashboard.png"
    st.image(dashboard_img, caption="ExplainMed Dashboard Vorschau", use_container_width=True)

    if st.button("👉 Zum Dashboard", key="landing-to-dashboard"):
        navigate("dashboard")
        
    st.markdown("## 💰 Preisstruktur")

    cols = st.columns(3)
    plans = [
        ("Starter", "Für kleine Praxen", "49€ / Monat", ["Bis 100 Diagnosen", "Grundlegende XAI-Erklärungen", "PDF-Berichte"]),
        ("Pro", "Für Kliniken", "199€ / Monat", ["Unbegrenzte Diagnosen", "Erweiterte XAI-Methoden", "Teamzugang", "Premium Support"]),
        ("Enterprise", "Für große Krankenhausketten", "Kontakt", ["API-Zugang", "Integration in bestehende Systeme", "Priorisierter Support", "On-Premise Option"])
    ]

    for col, (title, subtitle, price, features) in zip(cols, plans):
        with col:
            st.markdown(f"### {title}")
            st.markdown(f"_{subtitle}_")
            st.markdown(f"**{price}**")
            for feat in features:
                st.markdown(f"- ✅ {feat}")
            st.button(f"{title} wählen")

def dashboard(model, device, transform, all_classes, class_labels):
    # Sidebar für Upload
    st.sidebar.header("📤 Bild hochladen")
    uploaded_file = st.sidebar.file_uploader("Wähle ein Bild", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.sidebar.image(image, caption="Vorschau", use_container_width=True)

        # Modellvorhersage
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        class_key = all_classes[pred_class]
        readable_label = class_labels[class_key]
        probability = probs[0][pred_class].item()

        # XAI-Bilder generieren
        gradcam_img = gradcam_explain(model, img_tensor, model.layer3[-1], pred_class)
        lime_img = lime_explain(model, image, transform, pred_class, device)
        occlusion_img = occlusion_sensitivity_analysis(model, image, transform, pred_class, device)

        gradcam_info = "Modell hebt zentrale Gewebestrukturen hervor"
        lime_info = "Fokus auf auffällige Zellmuster"
        occlusion_info = "Genauigkeit sinkt bei Abdeckung bestimmter Bereiche"

        explanation_text = generate_textual_explanation(
            readable_label, gradcam_info, lime_info, occlusion_info
        )

        st.markdown(f"### 🩺 Diagnose: **{readable_label}** ({probability:.2%})")

        # 2x2 kompakteres Grid
        col1, col2 = st.columns([1.2, 1])  # Erstes Row
        col3, col4 = st.columns([1.2, 1])  # Zweites Row

        with col1:
            st.markdown("#### 📄 Erklärung")
            st.info(explanation_text)

        with col2:
            st.image(gradcam_img, caption="Grad-CAM", use_container_width=True)

        with col3:
            st.image(lime_img, caption="LIME", use_container_width=True)

        with col4:
            st.image(occlusion_img, caption="Occlusion", use_container_width=True)

        # PDF Download Button
        st.markdown("---")
        if st.button("📄 Diagnosebericht als PDF herunterladen"):
            pdf_bytes = generate_pdf_report(image, gradcam_img, lime_img, occlusion_img, explanation_text)
            st.download_button("⬇️ PDF speichern", pdf_bytes, "ExplainMed_Report.pdf", "application/pdf")

    else:
        st.info("📥 Lade ein Bild in der Sidebar hoch, um eine Diagnose zu erhalten.")