import streamlit as st
import numpy as np
import pickle

# -----------------------------
# INR FORMAT (INT ONLY)
# -----------------------------
def format_inr(amount):
    amount = int(amount)
    s = str(amount)

    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3][::-1]
        groups = [rest[i:i+2] for i in range(0, len(rest), 2)]
        s = ",".join(groups)[::-1] + "," + last3

    return f"₹ {s}"

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# FINAL POLISHED CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
}

header, footer, .stDeployButton { display: none !important; }

/* MAIN CARD */
.main-container {
    max-width: 900px;
    margin: auto;
    padding: 40px 30px;
    background: #020617;
    border-radius: 18px;
    border: 2px solid #1e293b;
    box-shadow: 0 30px 80px rgba(0,0,0,0.6);
}

/* FORM */
.form-container {
    margin-top: 30px;
    padding: 30px;
    border-radius: 14px;
    background: #020617;
    border: 2px solid #38bdf8;
}

/* TITLES */
h1 {
    color: #e5e7eb;
    text-align: center;
    font-weight: 700;
}

.subtitle {
    color: #94a3b8;
    text-align: center;
    font-size: 14px;
}

/* LABELS */
label p {
    color: #cbd5f5 !important;
    font-size: 13px;
    font-weight: 600;
}

/* ===== INPUTS (IMPORTANT FIX) ===== */
input, textarea {
    background: #020617 !important;
    color: #f8fafc !important;
    border-radius: 10px !important;
    border: 2px solid #38bdf8 !important;
    font-size: 14px !important;
}

input::placeholder {
    color: #64748b !important;
}

/* INPUT FOCUS */
input:focus {
    outline: none !important;
    border: 2px solid #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.35) !important;
}

/* SELECT BOX */
div[data-baseweb="select"] {
    background: #020617 !important;
    border-radius: 10px;
    border: 2px solid #38bdf8;
}

div[data-baseweb="select"] span {
    color: #f8fafc !important;
}

/* BUTTON */
div.stButton > button {
    width: 100%;
    height: 52px;
    margin-top: 22px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    border: none;
    color: white;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
}

/* RESULT */
.result-box {
    margin-top: 35px;
    padding: 28px;
    border-radius: 16px;
    background: #020617;
    border: 2px solid #22c55e;
    text-align: center;
}

.result-box h2 {
    color: #dcfce7;
    font-size: 38px;
}

/* FOOTER */
.footer-text {
    text-align: center;
    font-size: 12px;
    color: #64748b;
    margin-top: 25px;
}

/* MOBILE */
@media (max-width:600px){
    .main-container { padding: 25px 20px; }
    h1 { font-size: 22px; }
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

# -----------------------------
# UI
# -----------------------------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("<h1>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Real Estate Valuation System</div>", unsafe_allow_html=True)

st.markdown("<div class='form-container'>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    location = st.selectbox("📍 Location", locations)
    area = st.number_input("📐 Area (sqft)", min_value=100, value=1500)

with c2:
    bhk = st.selectbox("🛏️ BHK", [1,2,3,4,5], index=4)
    bath = st.selectbox("🚿 Bathrooms", [1,2,3,4,5], index=1)

predict = st.button("Calculate Property Price")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# RESULT
# -----------------------------
if predict:
    loc_idx = encoder.transform([location])[0]
    raw_price = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]
    final_price = int(round(raw_price))

    st.markdown(f"""
    <div class="result-box">
        <p style="font-size:13px;color:#94a3b8;">Estimated Market Value</p>
        <h2>{format_inr(final_price)}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer-text'>Client Demo • Internship • Placement Ready 💼</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
