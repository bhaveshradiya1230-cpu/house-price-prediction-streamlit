import streamlit as st
import numpy as np
import pickle

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="ProphetHouse AI",
    page_icon="🏠",
    layout="centered"import streamlit as st
import numpy as np
import pickle
import os

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="ProphetHouse AI",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# 2. High-Contrast Professional CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

.stApp {
    background-color: #0b0f19 !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

header, footer, .stDeployButton {display: none !important;}

.input-table-card {
    border: 2px solid #22d3ee;
    background-color: #111827;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.2);
    margin: 20px 0;
}

h1, h2, p {
    color: #ffffff !important;
    text-align: center;
}

label p {
    color: #22d3ee !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #0891b2, #0e7490) !important;
    color: white !important;
    border: 1px solid #22d3ee !important;
    height: 50px;
    font-size: 18px;
    font-weight: 700;
    border-radius: 10px;
    margin-top: 15px;
}

div.stButton > button:hover {
    background: #22d3ee !important;
    color: #0b0f19 !important;
    box-shadow: 0 0 15px #22d3ee;
}

.result-display {
    background: #064e3b;
    border: 2px solid #10b981;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. Load Model (CLOUD SAFE)
# -----------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "house_model.pkl")
    ENCODER_PATH = os.path.join(BASE_DIR, "..", "Model", "location_encoder.pkl")

    model = pickle.load(open(MODEL_PATH, "rb"))
    encoder = pickle.load(open(ENCODER_PATH, "rb"))

    locations = encoder.classes_

except Exception as e:
    st.error(f"Model loading failed: {e}")
    locations = ["Ahmedabad", "Mumbai", "Rajkot", "Jamnagar", "Gandhinagar"]

# -----------------------------
# 4. Header Section
# -----------------------------
st.markdown("<h1 style='font-size:40px;'>🏠 ProphetHouse AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9ca3af !important;'>AI-Powered Real Estate Valuation</p>", unsafe_allow_html=True)

# -----------------------------
# 5. Input Section
# -----------------------------
st.markdown('<div class="input-table-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 SELECT CITY", locations)
    area = st.number_input("📐 AREA (SQFT)", min_value=100, value=1200)

with col2:
    bhk = st.selectbox("🛏️ BHK STYLE", [1, 2, 3, 4, 5], index=1)
    bath = st.selectbox("🚿 BATHROOMS", [1, 2, 3, 4, 5], index=1)

predict_click = st.button("CALCULATE MARKET VALUE")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 6. Prediction Result
# -----------------------------
if predict_click:
    try:
        loc_idx = encoder.transform([location])[0]
        price = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]

        st.markdown(f"""
        <div class='result-display'>
            <p style='font-size:14px; opacity:0.8;'>ESTIMATED MARKET PRICE</p>
            <h2 style='font-size:38px;'>₹ {price:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "<p style='text-align:center; color:#4b5563 !important; font-size:12px; margin-top:30px;'>Optimized for Mobile & Desktop View</p>",
    unsafe_allow_html=True
)

)

# -----------------------------
# 2. High-Contrast Professional CSS
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    /* Pure Dark Professional Background */
    .stApp {
        background-color: #0b0f19 !important; 
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hide Streamlit Default UI Elements */
    header, footer, .stDeployButton {display: none !important;}
    [data-testid="stVerticalBlock"] > div:empty {display: none !important;}

    /* VISIBLE BORDERED TABLE CARD */
    .input-table-card {
        border: 2px solid #22d3ee; /* Bright Cyan Border */
        background-color: #111827; /* Darker Card Color */
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.2);
        margin: 20px 0;
    }

    /* TEXT VISIBILITY FIX */
    h1, h2, p {
        color: #ffffff !important;
        text-align: center;
    }

    /* INPUT LABELS (VERY IMPORTANT) */
    label p {
        color: #22d3ee !important; /* Cyan color for labels */
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 0.5px;
        margin-bottom: 8px !important;
    }

    /* BUTTON DESIGN */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #0891b2, #0e7490) !important;
        color: white !important;
        border: 1px solid #22d3ee !important;
        height: 50px;
        font-size: 18px;
        font-weight: 700;
        border-radius: 10px;
        margin-top: 15px;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background: #22d3ee !important;
        color: #0b0f19 !important;
        box-shadow: 0 0 15px #22d3ee;
    }

    /* PREDICTION BOX (Solid Visibility) */
    .result-display {
        background: #064e3b;
        border: 2px solid #10b981;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. Load Model
# -----------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "house_model.pkl")
    ENCODER_PATH = os.path.join(BASE_DIR, "..", "Model", "location_encoder.pkl")

    model = pickle.load(open(MODEL_PATH, "rb"))
    encoder = pickle.load(open(ENCODER_PATH, "rb"))

    locations = encoder.classes_

except Exception as e:
    st.error(f"Model loading error: {e}")
    locations = ["Ahmedabad", "Mumbai", "Rajkot", "Jamnagar", "Gandhinagar"]

# -----------------------------
# 4. Header Section
# -----------------------------
st.markdown("<h1 style='font-size: 40px; margin-bottom: 0;'>🏠 ProphetHouse AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #9ca3af !important;'>AI-Powered Real Estate Valuation</p>", unsafe_allow_html=True)

# -----------------------------
# 5. The Input Table (Visible Box)
# -----------------------------
st.markdown('<div class="input-table-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("📍 SELECT CITY", locations)
    area = st.number_input("📐 AREA (SQFT)", min_value=100, value=1200)

with col2:
    bhk = st.selectbox("🛏️ BHK STYLE", [1, 2, 3, 4, 5], index=1)
    bath = st.selectbox("🚿 BATHROOMS", [1, 2, 3, 4, 5], index=1)

# Button is inside the bordered table
predict_click = st.button("CALCULATE MARKET VALUE")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 6. Result Display
# -----------------------------
if predict_click:
    try:
        loc_idx = encoder.transform([location])[0]
        price = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]

        st.markdown(f"""
            <div class='result-display'>
                <p style='font-size:14px; margin:0; opacity:0.8;'>ESTIMATED MARKET PRICE</p>
                <h2 style='font-size:38px; margin:5px 0 0 0;'>₹ {price:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
    except:
        st.error("Model files not found or corrupted.")

st.markdown(
    "<p style='text-align:center; color:#4b5563 !important; font-size:12px; margin-top:30px;'>Optimized for Mobile & Desktop View</p>",

    unsafe_allow_html=True)

