import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Function
# -----------------------------
def format_inr(amount):
    amount = round(amount, 2)
    s = f"{amount:.2f}"
    integer, decimal = s.split(".")
    if len(integer) > 3:
        integer = integer[:-3][::-1]
        groups = [integer[i:i+2] for i in range(0, len(integer), 2)]
        integer = ",".join(groups)[::-1] + "," + s[-6:-3]
    return f"₹ {integer}.{decimal}"

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# REAL WEBSITE CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0f172a;
}

/* Hide default UI */
header, footer, .stDeployButton { display: none !important; }

/* ================= MAIN WRAPPER ================= */
.main-wrapper {
    max-width: 900px;
    margin: auto;
    padding: 40px 20px;
}

/* ================= CARD ================= */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 35px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.15);
}

/* ================= HEADINGS ================= */
h1 {
    text-align: center;
    color: #0f172a;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 30px;
    font-size: 14px;
}

/* ================= FORM BORDER ================= */
.form-box {
    border: 1.5px solid #e5e7eb;
    border-radius: 14px;
    padding: 25px;
    background: #f8fafc;
}

/* ================= LABELS ================= */
label p {
    font-size: 13px;
    font-weight: 600;
    color: #334155;
}

/* ================= INPUTS ================= */
input, select {
    background-color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
}

/* Selectbox wrapper */
div[data-baseweb="select"] {
    border-radius: 10px;
    border: 1px solid #cbd5e1;
}

/* ================= BUTTON ================= */
div.stButton > button {
    width: 100%;
    height: 48px;
    margin-top: 20px;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 600;
    background: #2563eb;
    color: white;
    border: none;
    transition: 0.25s;
}

div.stButton > button:hover {
    background: #1d4ed8;
}

/* ================= RESULT ================= */
.result-box {
    margin-top: 25px;
    padding: 22px;
    border-radius: 14px;
    background: #ecfeff;
    border: 1.5px solid #67e8f9;
    text-align: center;
}

.result-box h2 {
    color: #0f172a;
    font-size: 34px;
    margin-top: 5px;
}

/* ================= FOOTER ================= */
.footer {
    text-align: center;
    font-size: 12px;
    color: #94a3b8;
    margin-top: 20px;
}

/* ================= MOBILE ================= */
@media (max-width: 600px) {
    .card {
        padding: 25px;
    }
    h1 {
        font-size: 22px;
    }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_files():
    model = pickle.load(open("Model/house_model.pkl", "rb"))
    encoder = pickle.load(open("Model/location_encoder.pkl", "rb"))
    return model, encoder

model, encoder = load_files()
locations = list(encoder.classes_)

# ================= UI =================
st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("<h1>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-world AI based property price estimation</div>", unsafe_allow_html=True)

# -------- FORM BOX (SINGLE BORDER) --------
st.markdown("<div class='form-box'>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    location = st.selectbox("📍 Location", locations)
    area = st.number_input("📐 Area (sqft)", min_value=100, value=1200)

with c2:
    bhk = st.selectbox("🛏️ BHK", [1,2,3,4,5], index=1)
    bath = st.selectbox("🚿 Bathrooms", [1,2,3,4,5], index=1)

predict = st.button("Calculate Price")

st.markdown("</div>", unsafe_allow_html=True)

# -------- RESULT --------
if predict:
    loc_idx = encoder.transform([location])[0]
    price = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]

    st.markdown(f"""
    <div class='result-box'>
        <p style="font-size:13px;color:#475569;">Estimated Market Price</p>
        <h2>{format_inr(price)}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer'>Designed like a real-world web application</div>", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)
