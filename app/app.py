# app/app.py
import os
import sys
import streamlit as st
import numpy as np
from joblib import load

# Fix per importare moduli dalla root del progetto
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.info_panel import get_info_html  # solo info_panel, niente pdf_export

# --- LOGIN SEMPLICE ---
PASSWORD = "Imprintfpg2025"
st.sidebar.subheader("🔒 Accesso")
password = st.sidebar.text_input("Inserisci la password", type="password")
if password != PASSWORD:
    st.error("Accesso negato. Inserisci la password corretta.")
    st.stop()

# --- PATH MODELLO (relativo, funziona anche su Cloud) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "imprint_risk_model.joblib")
MODEL_PATH = os.path.abspath(MODEL_PATH)

# --- Configurazione pagina ---
st.set_page_config(page_title="IMPRINT Risk Calculator", page_icon="🧮", layout="centered")
st.title("Calcolatore rischio malignità (IMPRINT)")

# --- Pannello informativo ---
with st.expander("Informazioni e cut-off"):
    st.markdown(get_info_html(), unsafe_allow_html=True)

# --- Form di input ---
with st.form("risk_form"):
    col1, col2 = st.columns(2)
    age = col1.number_input("Età (anni)", min_value=15, max_value=95, value=50)
    diameter = col2.number_input("Diametro target (mm)", min_value=10, max_value=250, value=75)

    st.subheader("Ecografia (MUSA/MYLUNAR)")
    c1, c2, c3 = st.columns(3)
    irregular_margins = c1.selectbox("Margini irregolari?", [0,1], format_func=lambda x: "No" if x==0 else "Sì")
    color4 = c2.selectbox("Color score 4?", [0,1], format_func=lambda x: "No" if x==0 else "Sì")
    shadows_absent = c3.selectbox("Assenza di ombre acustiche?", [0,1], format_func=lambda x: "No" if x==0 else "Sì")

    st.subheader("Emocromo (valori assoluti, x10^9/L)")
    neutrofili = st.number_input("Neutrofili", min_value=0.0, max_value=30.0, value=4.0)
    linfociti = st.number_input("Linfociti", min_value=0.0, max_value=10.0, value=2.0)
    monociti = st.number_input("Monociti", min_value=0.0, max_value=5.0, value=0.5)
    eosinofili = st.number_input("Eosinofili", min_value=0.0, max_value=5.0, value=0.2)
    piastrine = st.number_input("Piastrine", min_value=10.0, max_value=1000.0, value=250.0)

    submitted = st.form_submit_button("Calcola rischio")

# --- Funzioni di classificazione ---
def classify_risk(p: float) -> str:
    if p < 0.004:
        return "Basso"
    elif p < 0.023:
        return "Intermedio"
    else:
        return "Alto"

def suggest_action(cls: str) -> str:
    if cls == "Alto":
        return "Imaging avanzato/MDT; considerare biopsia/chirurgia con approccio oncologico."
    elif cls == "Intermedio":
        return "Rivalutazione ecografica, ripetere markers; considerare MRI."
    return "Gestione conservativa come leiomioma, follow-up."

# --- Calcolo e output ---
if submitted:
    if linfociti > 0:
        NLR = neutrofili / linfociti
        PLR = piastrine / linfociti
        SII = (neutrofili * piastrine) / linfociti
        SIRI = (neutrofili * monociti) / linfociti
        PIV = (neutrofili * piastrine * monociti) / linfociti
    else:
        NLR = PLR = SII = SIRI = PIV = 0

    diameter_gt8 = 1 if diameter > 80 else 0
    x = np.array([[NLR, SII, SIRI, PIV, age, diameter_gt8,
                   irregular_margins, color4, shadows_absent]])

    model = load(MODEL_PATH)
    p = float(model.predict_proba(x)[0,1])
    cls = classify_risk(p)
    recommendation = suggest_action(cls)

    st.markdown(f"### Probabilità di malignità: {p:.3%}")

    if cls == "Basso":
        color = "green"
    elif cls == "Intermedio":
        color = "orange"
    else:
        color = "red"

    st.markdown(
        f"### Classe di rischio: <span style='color:{color}'>{cls}</span>",
        unsafe_allow_html=True
    )

    st.write(recommendation)
    st.caption("Nota: modello pilota addestrato su coorte simulata (benigni). Da ricalibrare con dati reali del centro.")
