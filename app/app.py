import streamlit as st
import numpy as np
import pickle

# -----------------------------
# INR FORMAT
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
# STRONG BORDER CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #0f172a;
}

header, footer, .stDeployButton { display: none !important; }

/* MAIN PAGE */
.page {
    max-width: 900px;
    margin: auto;
    padding: 50px 20px;
}

/* MAIN CARD */
.card {
    background: #ffffff;
    border-radius: 18px;
    padding: 40px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.25);
}

/* TITLE */
h1 {
    text-align: center;
    color: #0f172a;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #475569;
    margin-bottom: 30px;
    font-size: 14px;
}

/* ================== FORM BORDER (VERY CLEAR) ================== */
.form-border {
    border: 2.5px solid #1e293b;   /* <<< STRONG BORDER */
    border-radius: 14px;
    padding: 30px;
    background: #f9fafb;
    box-shadow: inset 0 0 0 1px #e5e7eb;
}

/* LABELS */
label p {
    font-size: 13px;
    font-weight: 600;
    color: #1e293b;
}

/* INPUTS */
input, select {
    background: #ffffff !important;
    border: 1.5px solid #94a3b8 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
}

/* SELECTBOX FIX */
div[data-baseweb="select"] {
    border-radius: 10px;
    border: 1.5px solid #94a3b8;
}

/* BUTTON */
div.stButton > button {
    width: 100%;
    height: 50px;
    margin-top: 25px;
    border-radius: 10px;
    background: #2563eb;
    color: white;
    font-size: 16px;
    font-weight: 600;
    border: none;
}

div.stButton > button:hover {
    background: #1d4ed8;
}

/* RESULT */
.result-box {
    margin-top: 30px;
    padding: 24px;
    border-radius: 12px;
    border: 2px solid #2563eb;
    background: #eff6ff;
    text-align: center;
}

.result-box h2 {
    font-size: 34px;
    color: #0f172a;
}

/* FOOTER */
.footer {
    margin-top: 25px;
    font-size: 12px;
    text-align: center;
    color: #64748b;
}

/* MOBILE */
@media (max-width:600px){
    .card { padding: 25px; }
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
st.markdown("<div class='page'>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("<h1>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Corporate real-world property valuation system</div>", unsafe_allow_html=True)

# -------- FORM WITH CLEAR BORDER --------
st.markdown("<div class='form-border'>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    location = st.selectbox("📍 Location", locations)
    area = st.number_input("📐 Area (sqft)", min_value=100, value=1200)

with c2:
    bhk = st.selectbox("🛏️ BHK", [1,2,3,4,5], index=1)
    bath = st.selectbox("🚿 Bathrooms", [1,2,3,4,5], index=1)

predict = st.button("Calculate Property Price")

st.markdown("</div>", unsafe_allow_html=True)

# RESULT
if predict:
    loc_idx = encoder.transform([location])[0]
    price = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]

    st.markdown(f"""
    <div class="result-box">
        <p style="font-size:13px;color:#475569;">Estimated Market Value</p>
        <h2>{format_inr(price)}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer'>Client-Demo • Internship • Placement Ready</div>", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)
