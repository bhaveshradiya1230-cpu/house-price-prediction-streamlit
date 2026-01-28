import streamlit as st
import numpy as np
import pickle

# -----------------------------
# INR FORMAT (NO CHANGE)
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
# RECOMMENDATION ENGINE (NO CHANGE)
# -----------------------------
def get_recommendation(price, area, bhk):
    if price < 3000000:
        category = "Budget Property"
        advice = "Good for first-time buyers and rental income."
    elif price < 7000000:
        category = "Mid-Range Property"
        advice = "Balanced pricing, suitable for families."
    elif price < 15000000:
        category = "Premium Property"
        advice = "High demand segment, good appreciation potential."
    else:
        category = "Luxury Property"
        advice = "Ideal for premium buyers and long-term investment."

    space_tip = (
        "Compact layout, space utilization is average."
        if area / bhk < 350
        else "Spacious layout with good livability."
    )

    low = int(price * 0.9)
    high = int(price * 1.1)

    return category, advice, space_tip, low, high

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# 🔥 REAL-WORLD MODERN CSS (ONLY NEW PART)
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top, #020617, #000000);
}

header, footer {display:none;}

.main-container {
    max-width: 900px;
    margin: auto;
    padding: 40px;
    border-radius: 22px;
    background: linear-gradient(180deg, #020617, #020617);
    border: 2px solid #22d3ee;
    box-shadow: 0 0 35px rgba(34,211,238,0.25);
    text-align: center;
}

h1 {
    font-size: 38px;
    font-weight: 800;
    color: #e5e7eb;
}

.subtitle {
    color: #94a3b8;
    margin-top: -10px;
}

label p {
    color: #67e8f9 !important;
    font-weight: 600 !important;
}

div.stButton > button {
    width: 100%;
    height: 54px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(90deg, #06b6d4, #0ea5e9);
    border: none;
    color: #020617;
    transition: 0.3s;
}

div.stButton > button:hover {
    box-shadow: 0 0 25px #22d3ee;
    transform: scale(1.02);
}

.result-box {
    margin-top: 30px;
    padding: 26px;
    border-radius: 18px;
    background: linear-gradient(180deg, #022c22, #064e3b);
    border: 2px solid #10b981;
    color: #ecfdf5;
    box-shadow: 0 0 30px rgba(16,185,129,0.6);
}

.reco-box {
    margin-top: 20px;
    padding: 22px;
    border-radius: 18px;
    background: #020617;
    border: 2px solid #38bdf8;
    color: #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL (NO CHANGE)
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
st.markdown("""
<div class="main-container">
    <h1>🏠 House Price Prediction</h1>
    <p class="subtitle">AI Powered Valuation & Smart Recommendation</p>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    location = st.selectbox("Location", locations)
    area = st.number_input("Area (sqft)", min_value=100, value=1500)

with c2:
    bhk = st.selectbox("BHK", [1,2,3,4,5], index=2)
    bath = st.selectbox("Bathrooms", [1,2,3,4,5], index=1)

predict = st.button("Calculate Property Price")

# -----------------------------
# RESULT
# -----------------------------
if predict:
    loc_idx = encoder.transform([location])[0]
    raw_price = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]
    final_price = int(round(raw_price))

    category, advice, space_tip, low, high = get_recommendation(
        final_price, area, bhk
    )

    st.markdown(f"""
    <div class="result-box">
        <h2>{format_inr(final_price)}</h2>
        <p>{category}</p>
    </div>

    <div class="reco-box">
        <p>{advice}</p>
        <p>{space_tip}</p>
        <p><b>Expected Range:</b> {format_inr(low)} - {format_inr(high)}</p>
    </div>
    """, unsafe_allow_html=True)
