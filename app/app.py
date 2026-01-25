import streamlit as st
import numpy as np
import pickle

# Function to format INR
def format_inr(amount):
    amount = round(amount, 2)
    s = f"{amount:.2f}"
    integer, decimal = s.split(".")
    if len(integer) > 3:
        integer = integer[:-3][::-1]
        groups = [integer[i:i+2] for i in range(0, len(integer), 2)]
        integer = ",".join(groups)[::-1] + "," + s[-6:-3]
    return f"₹ {integer}.{decimal}"

# PAGE CONFIG
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ADVANCED PROFESSIONAL CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

.stApp {
    background-color: #0b0f19;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Hide Streamlit UI */
header, footer, .stDeployButton {display: none !important;}

/* MAIN CONTAINER */
.main-container {
    padding: 20px;
    max-width: 900px;
    margin: auto;
}

/* ===========================
   INPUT CARD WITH GLOW BORDER
   (Isse aapka pura input section border me aayega)
=========================== */
.input-card {
    border: 2px solid #22d3ee;
    background-color: #0f172a;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.2);
    margin-bottom: 20px;
}

/* HEADINGS */
h1 { color: #ffffff !important; text-align: center; font-weight: 800; margin-bottom: 5px; }
.sub-text { color: #9ca3af !important; text-align: center; margin-bottom: 30px; }

/* LABELS STYLE */
div[data-testid="stMarkdownContainer"] p {
    color: #22d3ee !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 13px !important;
    letter-spacing: 1px;
}

/* BUTTON */
div.stButton > button {
    width: 100%;
    height: 55px;
    margin-top: 20px;
    font-size: 18px;
    font-weight: 700;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #0891b2, #0e7490);
    color: white;
    transition: 0.3s;
    text-transform: uppercase;
}

div.stButton > button:hover {
    background: #22d3ee;
    color: #020617;
    box-shadow: 0 0 25px rgba(34,211,238,0.5);
}

/* RESULT BOX */
.result-box {
    background: rgba(16, 185, 129, 0.1);
    border: 2px solid #10b981;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL & ENCODER (Dummy list if files not found for testing)
@st.cache_resource
def load_files():
    try:
        model = pickle.load(open("Model/house_model.pkl", "rb"))
        encoder = pickle.load(open("Model/location_encoder.pkl", "rb"))
        return model, encoder
    except:
        return None, None

model, encoder = load_files()
locations = list(encoder.classes_) if encoder else ["Sample City"]

# =============================
# MAIN UI
# =============================
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("<h1>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>AI-Powered Real Estate Valuation</p>", unsafe_allow_html=True)

# Sab kuch is "input-card" div ke andar border me dikhega
st.markdown("<div class='input-card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Select City", locations)
    area = st.number_input("📐 Area (SQFT)", min_value=100, value=1200)

with col2:
    bhk = st.selectbox("🛏️ BHK Style", [1, 2, 3, 4, 5], index=1)
    bath = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5], index=1)

predict = st.button("Calculate Market Value")

st.markdown("</div>", unsafe_allow_html=True) # Input card ends here

# RESULT
if predict:
    if model and encoder:
        try:
            loc_idx = encoder.transform([location])[0]
            price = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]

            st.markdown(f"""
            <div class='result-box'>
                <p style='color:#10b981 !important; margin-bottom:0;'>ESTIMATED MARKET PRICE</p>
                <h2 style='font-size:45px; color:white; margin-top:10px;'>{format_inr(price)}</h2>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.warning("⚠️ Model files not found. Please check paths.")

st.markdown("</div>", unsafe_allow_html=True) # Main container ends
