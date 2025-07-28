import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model_utils import ResNet_Definition
from components import landing_page, dashboard
from text_utils import generate_textual_explanation

# App Config
st.set_page_config(page_title="ExplainMed", page_icon="🩺", layout="wide")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Klassenlabels
class_labels = {
    "colon_aca": "Colon Adenocarcinoma",
    "colon_n": "Colon Benign Tissue",
    "lung_aca": "Lung Adenocarcinoma",
    "lung_n": "Lung Benign Tissue",
    "lung_scc": "Lung Squamous Cell Carcinoma"
}
all_classes = list(class_labels.keys())

# Modell laden
model = ResNet_Definition(num_classes=len(all_classes))
model.load_state_dict(torch.load("models/resnet_model.pth", map_location=device))
model.to(device)
model.eval()

# Bildtransformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Session-State für Navigation
if "page" not in st.session_state:
    st.session_state.page = "landing"

def navigate(page):
    st.session_state.page = page

# ---------- Navbar ----------
st.markdown(
    """
    <style>
    .navbar {
        background-color: #FFFFFF;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 40px;
        font-family: Arial, sans-serif;
        border-bottom: 1px solid #E5E7EB;
    }
    .navbar-title {
        color: #2F80ED;
        font-size: 24px;
        font-weight: bold;
    }
    .nav-btn button {
        background: none;
        border: none;
        font-size: 16px;
        cursor: pointer;
        color: #1F2937;
    }
    .nav-btn button:hover {
        color: #2F80ED;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("<div class='navbar-title'>🩺 ExplainMed</div>", unsafe_allow_html=True)
with col2:
    c1, c2 = st.columns(2)
    if c1.button("🏠 Startseite", key="home-btn"):
        st.session_state.page = "landing"
    if c2.button("📊 Dashboard", key="dash-btn"):
        st.session_state.page = "dashboard"

st.markdown("---")

# ---------- Seiten-Logik ----------
if st.session_state.page == "landing":
    landing_page(navigate)
elif st.session_state.page == "dashboard":
    dashboard(model, device, transform, all_classes, class_labels)