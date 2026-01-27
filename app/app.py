import streamlit as st
import numpy as np
import pickle

# Function for INR Formatting
def format_inr(amount):
    amount = round(amount, 2)
    s = f"{amount:.2f}"
    integer, decimal = s.split(".")
    if len(integer) > 3:
        integer_part = integer[:-3][::-1]
        groups = [integer_part[i:i+2] for i in range(0, len(integer_part), 2)]
        integer = ",".join(groups)[::-1] + "," + integer[-3:]
    return f"₹ {integer}.{decimal}"

st.set_page_config(page_title="PropAI Predictor", page_icon="🏠", layout="centered")

# -----------------------------
# FIXED BORDER CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com');

.stApp {
    background: #0f172a;
    font-family: 'Inter', sans-serif;
}

header, footer, .stDeployButton {display: none !important;}

/* Fixed Outside Border Container */
.main-card {
    border: 2px solid #22d3ee !important; /* Always Visible */
    background: rgba(30, 41, 59, 0.7);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    margin: 10px auto;
}

/* Internal Fixed Section for Inputs */
.input-section {
    border: 1px solid rgba(34, 211, 238, 0.3);
    background: rgba(15, 23, 42, 0.5);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}

/* Ensuring Inputs stay inside borders */
.stSelectbox, .stNumberInput {
    margin-bottom: 10px;
}

/* Button & Result */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border: none;
    height: 50px;
    font-weight: 700;
    border-radius: 10px;
}

.result-box {
    border: 2px solid #10b981;
    background: rgba(16, 185, 129, 0.1);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    text-align: center;
}

h1 { color: white !important; text-align: center; font-size: 2rem !important; }
label p { color: #22d3ee !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_files():
    try:
        model = pickle.load(open("Model/house_model.pkl", "rb"))
        encoder = pickle.load(open("Model/location_encoder.pkl", "rb"))
        return model, encoder
    except: return None, None

model, encoder = load_files()

# -----------------------------
# UI STRUCTURE
# -----------------------------
# Wrapper Start
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("<h1>PropAI Predictor 🏠</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8;'>Enterprise-Grade Real Estate Analysis</p>", unsafe_allow_html=True)

if model and encoder:
    locations = list(encoder.classes_)
    
    # Input Section Start
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("📍 Location", locations)
        area = st.number_input("📐 Area (Sqft)", min_value=100, value=1200)
    with col2:
        bhk = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5], index=1)
        bath = st.selectbox("🚿 Bath", [1, 2, 3, 4, 5], index=1)
    
    st.markdown("</div>", unsafe_allow_html=True)
    # Input Section End

    st.write("") # Spacer

    if st.button("CALCULATE PRICE"):
        try:
            loc_idx = encoder.transform([location])[0]
            prediction = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]
            
            st.markdown(f"""
                <div class='result-box'>
                    <p style='color: #a7f3d0; margin:0;'>ESTIMATED VALUE</p>
                    <h2 style='color: #10b981; margin:5px 0;'>{format_inr(prediction)}</h2>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        except:
            st.error("Prediction Error")
else:
    st.error("Model Files Missing")

# Wrapper End
st.markdown("</div>", unsafe_allow_html=True)
