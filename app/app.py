import streamlit as st
import numpy as np
import pickle

# -----------------------------
# INR FORMAT (INT ONLY – REAL WORLD)
# -----------------------------
def format_inr(amount):
    amount = int(amount)   # 🔥 force integer
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
# CORPORATE CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background: #0b0f19; }

header, footer, .stDeployButton { display: none !important; }

.main-container {
    max-width: 900px;
    margin: auto;
    padding: 40px 30px;
    background: #020617;
    border-radius: 18px;
    border: 2.5px solid #1e293b;
    box-shadow: 0 25px 60px rgba(0,0,0,0.4);
}

.form-container {
    margin-top: 30px;
    padding: 30px;
    border-radius: 14px;
    background: #020617;
    border: 2.5px solid #38bdf8;
}

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

label p {
    color: #e5e7eb !important;
    font-size: 13px;
    font-weight: 600;
}

input, select {
    background: #020617 !important;
    border-radius: 10px !important;
    border: 1.5px solid #475569 !important;
    color: #e5e7eb !important;
}

div[data-baseweb="select"] {
    border-radius: 10px;
    border: 1.5px solid #475569;
}

div.stButton > button {
    width: 100%;
    height: 52px;
    margin-top: 22px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    background: #2563eb;
    border: none;
    color: white;
}

div.stButton > button:hover {
    background: #1d4ed8;
}

.result-box {
    margin-top: 30px;
    padding: 25px;
    border-radius: 14px;
    background: #020617;
    border: 2.5px solid #22c55e;
    text-align: center;
}

.result-box h2 {
    color: #dcfce7;
    font-size: 36px;
}

.footer-text {
    text-align: center;
    font-size: 12px;
    color: #64748b;
    margin-top: 25px;
}

@media (max-width:600px){
    .main-container { padding: 25px 20px; }
    h1 { font-size: 22px; }
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
# UI
# -----------------------------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("<h1>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Real Estate Valuation System</div>", unsafe_allow_html=True)

st.markdown("<div class='form-container'>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    location = st.selectbox("📍 Location", locations)
    area = st.number_input("📐 Area (sqft)", min_value=100, value=1200)

with c2:
    bhk = st.selectbox("🛏️ BHK", [1,2,3,4,5], index=1)
    bath = st.selectbox("🚿 Bathrooms", [1,2,3,4,5], index=1)

predict = st.button("Calculate Property Price")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# RESULT (REAL-WORLD CLEAN PRICE)
# -----------------------------
if predict:
    loc_idx = encoder.transform([location])[0]

    raw_price = model.predict(
        np.array([[area, bhk, bath, loc_idx]])
    )[0]

    # 🔥 REAL-WORLD CLEAN PRICE
    final_price = int(round(raw_price))

    st.markdown(f"""
    <div class="result-box">
        <p style="font-size:13px;color:#94a3b8;">Estimated Market Value</p>
        <h2>{format_inr(final_price)}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer-text'>Client Demo • Internship • Placement Ready 💼</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
