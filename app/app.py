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
# RECOMMENDATION ENGINE
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
# CSS
# -----------------------------
st.markdown("""
<style>
* { font-family: Inter, sans-serif; }
.stApp { background: radial-gradient(circle at top, #0f172a, #020617); }
header, footer { display: none; }

.main-container {
    max-width: 900px;
    margin: auto;
    padding: 40px;
    background: #020617;
    border-radius: 18px;
    border: 2px solid #1e293b;
}

.result-box, .reco-box {
    margin-top: 25px;
    padding: 22px;
    border-radius: 14px;
    border: 2px solid #38bdf8;
    color: #e5e7eb;
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
st.markdown("""
<div class="main-container">
<h1>House Price Prediction</h1>
<p>AI Powered Valuation & Recommendation System</p>
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
