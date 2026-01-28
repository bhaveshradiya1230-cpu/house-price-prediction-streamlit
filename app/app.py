import streamlit as st
import numpy as np
import pickle

# -----------------------------
# INR FORMAT
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
# RECOMMENDATION + CONFIDENCE
# -----------------------------
def get_recommendation(price, area, bhk):
    if price < 3000000:
        category = "Budget Property"
        badge = "🟢 Affordable Zone"
        advice = "Good for first-time buyers & rental income."
        confidence = 70
    elif price < 7000000:
        category = "Mid-Range Property"
        badge = "🔵 High Demand Area"
        advice = "Balanced pricing. Family-friendly choice."
        confidence = 82
    elif price < 15000000:
        category = "Premium Property"
        badge = "🟣 Premium Growth Zone"
        advice = "Strong appreciation potential."
        confidence = 88
    else:
        category = "Luxury Property"
        badge = "🔴 Elite Locality"
        advice = "Ideal for wealth parking."
        confidence = 92

    space_tip = (
        "⚠ Compact layout."
        if area / bhk < 350
        else "✅ Spacious & well-planned."
    )

    low = int(price * 0.9)
    high = int(price * 1.1)

    return category, badge, advice, space_tip, low, high, confidence

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# PREMIUM UI CSS + FONT
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Slab:wght@400;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
    transition: all 0.3s ease;
}

.stApp {
    background: radial-gradient(circle at top, #020617, #000000);
    color: #e5e7eb;
}

header, footer {
    display: none;
}

.main {
    max-width: 920px;
    margin: auto;
    padding: 40px;
    border-radius: 22px;
    background: #020617;
    border: 2px solid #22d3ee;
    box-shadow: 0 0 45px rgba(34,211,238,.35);
    font-family: 'Roboto Slab', serif;
}

h1 {
    color: #e5e7eb;
    font-weight: 800;
    text-align: center;
    font-size: 36px;
    letter-spacing: 1px;
}

.subtitle {
    color: #94a3b8;
    text-align: center;
    margin-top: -8px;
    font-size: 16px;
    font-weight: 500;
}

.card {
    margin-top: 25px;
    padding: 26px;
    border-radius: 18px;
    background: #020617;
    border: 2px solid #38bdf8;
    box-shadow: 0 0 25px rgba(56,189,248,.25);
}

.result {
    background: linear-gradient(180deg, #022c22, #064e3b);
    border: 2px solid #10b981;
    box-shadow: 0 0 35px rgba(16,185,129,.6);
    color: #ecfdf5;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    background: #0f172a;
    border: 1px solid #38bdf8;
    color: #7dd3fc;
    font-size: 12px;
    margin-top: 10px;
    font-weight: 600;
}

.progress {
    height: 12px;
    background: #020617;
    border-radius: 999px;
    border: 1px solid #38bdf8;
    overflow: hidden;
    margin-top: 8px;
}

.progress span {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, #22d3ee, #10b981);
}

button {
    border-radius: 14px !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg,#22d3ee,#10b981) !important;
    color: #020617 !important;
    padding: 10px 20px !important;
}

button:hover {
    transform: scale(1.05);
}

@media(max-width:600px){
    .main {
        padding: 25px;
    }
    h1 {
        font-size: 24px;
    }
    .subtitle {
        font-size: 14px;
    }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_files():
    model = pickle.load(open("Model/house_model.pkl","rb"))
    encoder = pickle.load(open("Model/location_encoder.pkl","rb"))
    return model, encoder

model, encoder = load_files()
locations = list(encoder.classes_)

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<div class="main">
<h1>🏠 House Price Prediction</h1>
<div class="subtitle">AI-Powered Valuation & Smart Insights</div>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    location = st.selectbox("📍 Location", locations)
    area = st.number_input("📐 Area (sqft)", min_value=100, value=1500)
with c2:
    bhk = st.selectbox("🛏 BHK", [1,2,3,4,5], index=2)
    bath = st.selectbox("🚿 Bathrooms", [1,2,3,4,5], index=1)

predict = st.button("🚀 Calculate Property Price")

# -----------------------------
# RESULT
# -----------------------------
if predict:
    loc_idx = encoder.transform([location])[0]
    price = int(model.predict(np.array([[area,bhk,bath,loc_idx]]))[0])

    category,badge,advice,space_tip,low,high,confidence = get_recommendation(
        price, area, bhk
    )

    st.markdown(f"""
    <div class="card result">
        <h2>{format_inr(price)}</h2>
        <div class="badge">{badge}</div>
        <p style="margin-top:10px;">{category}</p>
    </div>

    <div class="card">
        <h4>📊 Price Confidence</h4>
        <div class="progress"><span style="width:{confidence}%"></span></div>
        <p style="font-size:13px;margin-top:6px;">Confidence Score: {confidence}%</p>

        <p>💡 {advice}</p>
        <p>{space_tip}</p>
        <p><b>Expected Range:</b> {format_inr(low)} - {format_inr(high)}</p>
    </div>
    """, unsafe_allow_html=True)
