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
# RECOMMENDATION ENGINE
# -----------------------------
def get_recommendation(price, area, bhk):
    # Category
    if price < 30_00_000:
        category = "🟢 Budget Property"
        advice = "Good for first-time buyers & rental income."
    elif price < 70_00_000:
        category = "🔵 Mid-Range Property"
        advice = "Balanced price. Suitable for end-users."
    elif price < 1_50_00_000:
        category = "🟣 Premium Property"
        advice = "High-demand segment. Long-term appreciation expected."
    else:
        category = "🔴 Luxury Property"
        advice = "Premium buyers zone. Ideal for wealth parking."

    # Area + BHK insight
    if area / bhk < 350:
        space_tip = "⚠️ Compact layout. Space efficiency is average."
    else:
        space_tip = "✅ Spacious layout. Good livability score."

    # Confidence range
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
# CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
}

header, footer, .stDeployButton { display: none !important; }

.main-container {
    max-width: 900px;
    margin: auto;
    padding: 40px 30px;
    background: #020617;
    border-radius: 18px;
    border: 2px solid #1e293b;
}

.form-container {
    margin-top: 30px;
    padding: 30px;
    border-radius: 14px;
    background: #020617;
    border: 2px solid #38bdf8;
}

h1 {
    color: #e5e7eb;
    text-align: center;
}

.subtitle {
    color: #94a3b8;
    text-align: center;
    font-size: 14px;
}

label p {
    color: #cbd5f5 !important;
    font-size: 13px;
    font-weight: 600;
}

input {
    background: #020617 !important;
    color: #f8fafc !important;
    border-radius: 10px !important;
    border: 2px solid #38bdf8 !important;
}

div[data-baseweb="select"] {
    background: #020617 !important;
    border-radius: 10px;
    border: 2px solid #38bdf8;
}

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

.result-box {
    margin-top: 35px;
    padding: 28px;
    border-radius: 16px;
    background: #020617;
    border: 2px solid #22c55e;
    text-align: center;
}

.reco-box {
    margin-top: 25px;
    padding: 22px;
    border-radius: 14px;
    background: #020617;
    border: 2px dashed #38bdf8;
    color: #e5e7eb;
}

.footer-text {
    text-align: center;
    font-size: 12px;
    color: #64748b;
    margin-top: 25px;
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


st.markdown("<h1>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Powered Valuation & Recommendation System</div>", unsafe_allow_html=True)

st.markdown("<div class='form-container'>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    location = st.selectbox("📍 Location", locations)
    area = st.number_input("📐 Area (sqft)", min_value=100, value=1500)

with c2:
    bhk = st.selectbox("🛏️ BHK", [1,2,3,4,5], index=2)
    bath = st.selectbox("🚿 Bathrooms", [1,2,3,4,5], index=1)

predict = st.button("Calculate Property Price")


# -----------------------------
# RESULT + RECOMMENDATION
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
        <p style="color:#94a3b8;">Estimated Market Value</p>
        <h2>{format_inr(final_price)}</h2>
    </div>

    <div class="reco-box">
        <h4>{category}</h4>
        <p>💡 {advice}</p>
        <p>{space_tip}</p>
        <p>📊 Expected Price Range: <b>{format_inr(low)} – {format_inr(high)}</b></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer-text'>Client Demo • Internship • Placement Ready 💼</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

