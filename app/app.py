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
# ADVANCED CSS (FIXED INPUT + MOBILE)
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

* { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
}

/* Hide Streamlit default UI */
header, footer, .stDeployButton { display: none !important; }

/* ================= MAIN CARD ================= */
.main-container {
    max-width: 900px;
    margin: auto;
    padding: 35px 30px;
    border-radius: 22px;
    background: linear-gradient(180deg, #020617, #020617);
    border: 2px solid rgba(34,211,238,0.5);
    box-shadow: 0 0 30px rgba(34,211,238,0.25);
}

/* ================= INPUT CARD ================= */
.input-card {
    background: #020617;
    border-radius: 18px;
    padding: 28px;
    margin-top: 25px;
    border: 1px solid rgba(34,211,238,0.4);
    box-shadow: inset 0 0 20px rgba(34,211,238,0.05);
}

/* ================= HEADINGS ================= */
h1 {
    font-weight: 800;
    text-align: center;
    color: #e5e7eb;
}

p {
    text-align: center;
    color: #94a3b8;
}

/* ================= LABELS ================= */
label p {
    color: #22d3ee !important;
    font-weight: 600;
    font-size: 13px;
}

/* ================= INPUTS ================= */
input, select, textarea {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 12px !important;
    border: 1px solid rgba(34,211,238,0.4) !important;
}

/* Selectbox arrow + container */
div[data-baseweb="select"] {
    background-color: #020617;
    border-radius: 12px;
    border: 1px solid rgba(34,211,238,0.4);
}

/* ================= BUTTON ================= */
div.stButton > button {
    width: 100%;
    height: 52px;
    margin-top: 15px;
    border-radius: 14px;
    font-size: 17px;
    font-weight: 700;
    background: linear-gradient(90deg, #0891b2, #0e7490);
    border: none;
    color: white;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    background: #22d3ee;
    color: #020617;
    box-shadow: 0 0 25px #22d3ee;
    transform: scale(1.02);
}

/* ================= RESULT ================= */
.result-box {
    margin-top: 30px;
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(180deg, #022c22, #064e3b);
    border: 2px solid #10b981;
    text-align: center;
    box-shadow: 0 0 30px rgba(16,185,129,0.6);
}

.result-box h2 {
    color: #ecfdf5;
    font-size: 38px;
}

/* ================= FOOTER ================= */
.footer-text {
    margin-top: 25px;
    font-size: 12px;
    text-align: center;
    color: #64748b;
}

/* ================= MOBILE ================= */
@media (max-width: 600px) {
    .main-container {
        padding: 25px 18px;
    }
    h1 {
        font-size: 22px;
    }
    .result-box h2 {
        font-size: 30px;
    }
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

# ================= UI =================
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("<h1>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p>AI-Powered Real Estate Valuation</p>", unsafe_allow_html=True)

st.markdown("<div class='input-card'>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    location = st.selectbox("📍 LOCATION", locations)
    area = st.number_input("📐 AREA (SQFT)", min_value=100, value=1200)

with c2:
    bhk = st.selectbox("🛏️ BHK", [1,2,3,4,5], index=1)
    bath = st.selectbox("🚿 BATH", [1,2,3,4,5], index=1)

predict = st.button("💰 CALCULATE PRICE")

st.markdown("</div>", unsafe_allow_html=True)

if predict:
    loc_idx = encoder.transform([location])[0]
    price = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]

    st.markdown(f"""
    <div class='result-box'>
        <p>ESTIMATED PRICE</p>
        <h2>{format_inr(price)}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.balloons()

st.markdown("<p class='footer-text'>Optimized for Mobile & Desktop</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
