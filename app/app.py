import streamlit as st
import numpy as np
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="ProphetHouse AI",
    page_icon="🏠",
    layout="centered"
)

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
# CUSTOM CSS (DARK MODERN UI)
# -----------------------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f2027, #000);
    color: white;
}
.main {
    background: transparent;
}
.block-container {
    max-width: 420px;
    padding-top: 30px;
}
input, select {
    border-radius: 8px !important;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-size: 16px;
}
.result-box {
    margin-top: 20px;
    padding: 20px;
    background: #064e3b;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
}
small {
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<h2 style='text-align:center;'>🏡 ProphetHouse AI</h2>"
    "<p style='text-align:center;opacity:0.7;'>AI Powered Real Estate Valuation</p>",
    unsafe_allow_html=True
)

# -----------------------------
# INPUTS
# -----------------------------
location = st.selectbox("📍 Select City", sorted(locations))
area = st.number_input("📐 Area (sq.ft)", min_value=300, max_value=10000, value=1200)
bhk = st.selectbox("🛏 BHK", [1, 2, 3, 4, 5])
bath = st.selectbox("🛁 Bathrooms", [1, 2, 3, 4])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("💰 Calculate Market Value"):

    if location not in encoder.classes_:
        st.error("❌ Selected city not supported by model")
        st.stop()

    loc_encoded = encoder.transform([location])[0]

    features = np.array([[area, bhk, bath, loc_encoded]])
    price = model.predict(features)[0]

    st.markdown(
        f"""
        <div class="result-box">
            <small>Estimated Property Price</small><br><br>
            <b>₹ {price:,.2f}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("⚡ Prediction powered by Machine Learning model")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    "<hr><p style='text-align:center;opacity:0.5;'>Built for AI-ML Internship Project 🚀</p>",
    unsafe_allow_html=True
)
