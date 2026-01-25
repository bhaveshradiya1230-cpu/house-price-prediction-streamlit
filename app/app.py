import streamlit as st
import numpy as np
import pickle

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# 2. PROFESSIONAL DARK UI + BORDERS CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

.stApp {
    background-color: #0b0f19 !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Hide Streamlit default UI */
header, footer, .stDeployButton {display: none !important;}

/* MAIN APP BORDER */
.main-container {
    border: 2px solid #22d3ee;
    border-radius: 18px;
    padding: 35px 30px;
    box-shadow: 0 0 30px rgba(34, 211, 238, 0.3);
    background: linear-gradient(180deg, #0b1220, #020617);
    max-width: 900px;
    margin: auto;
}

/* INPUT CARD */
.input-card {
    border: 2px solid #22d3ee;
    background-color: #111827;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.25);
    margin-top: 20px;
}

/* HEADINGS */
h1, h2, p {
    color: white !important;
    text-align: center;
}

/* LABELS */
label p {
    color: #22d3ee !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}

/* BUTTON */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #0891b2, #0e7490) !important;
    color: white !important;
    border: 1px solid #22d3ee !important;
    height: 50px;
    font-size: 18px;
    font-weight: 700;
    border-radius: 10px;
    margin-top: 15px;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: #22d3ee !important;
    color: #0b0f19 !important;
    box-shadow: 0 0 15px #22d3ee;
}

/* RESULT BOX */
.result-box {
    background: #064e3b;
    border: 2px solid #10b981;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.5);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL & ENCODER
# -----------------------------
@st.cache_resource
def load_files():
    model = pickle.load(open("Model/house_model.pkl", "rb"))
    encoder = pickle.load(open("Model/location_encoder.pkl", "rb"))
    return model, encoder

model, encoder = load_files()
locations = list(encoder.classes_)

# -----------------------------
# START MAIN BORDER
# -----------------------------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1>🏠 House Price Prediction Using Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9ca3af !important;'>AI-Powered Real Estate Valuation</p>", unsafe_allow_html=True)

# -----------------------------
# INPUT CARD
# -----------------------------
st.markdown("<div class='input-card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 SELECT CITY", locations)
    area = st.number_input("📐 AREA (SQFT)", min_value=100, value=1200)

with col2:
    bhk = st.selectbox("🛏️ BHK STYLE", [1, 2, 3, 4, 5], index=1)
    bath = st.selectbox("🚿 BATHROOMS", [1, 2, 3, 4, 5], index=1)

predict = st.button("CALCULATE MARKET VALUE")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# RESULT
# -----------------------------
if predict:
    try:
        loc_idx = encoder.transform([location])[0]
        price = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]

        st.markdown(f"""
        <div class='result-box'>
            <p style='font-size:14px; opacity:0.8;'>ESTIMATED MARKET PRICE</p>
            <h2 style='font-size:38px;'>₹ {price:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()
    except:
        st.error("Model files not found or corrupted.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    "<p style='text-align:center; color:#4b5563 !important; font-size:12px; margin-top:30px;'>Optimized for Mobile & Desktop View</p>",
    unsafe_allow_html=True
)

# -----------------------------
# END MAIN BORDER
# -----------------------------
st.markdown("</div>", unsafe_allow_html=True)
