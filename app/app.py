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

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="PropAI | House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# CORPORATE PREMIUM CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com');

.stApp {
    background: radial-gradient(circle at top, #1e293b, #0f172a);
    font-family: 'Inter', sans-serif;
}

header, footer, .stDeployButton {display: none !important;}

/* Border for the main container */
.main-card {
    border: 2px solid #22d3ee;
    background: rgba(15, 23, 42, 0.9);
    padding: 40px;
    border-radius: 24px;
    box-shadow: 0 0 30px rgba(34, 211, 238, 0.2);
    margin-bottom: 20px;
}

/* Border for the input area */
.input-section {
    background: rgba(255, 255, 255, 0.03);
    border: 1.5px solid #22d3ee;
    padding: 25px;
    border-radius: 16px;
    margin-top: 20px;
}

/* ADDING BORDER TO INPUT FIELDS */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    border: 1px solid #22d3ee !important;
    border-radius: 8px !important;
    background-color: #0f172a !important;
    color: white !important;
}

/* FOCUS EFFECT */
.stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
    border: 2px solid #22d3ee !important;
    box-shadow: 0 0 10px rgba(34, 211, 238, 0.5) !important;
}

h1 { color: #f8fafc !important; font-weight: 800 !important; text-align: center; }
p.subtitle { color: #94a3b8 !important; text-align: center; margin-top: -15px; }

label p {
    color: #22d3ee !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
    color: white;
    border: none;
    padding: 15px;
    font-weight: 700;
    border-radius: 12px;
}

.result-box {
    background: rgba(16, 185, 129, 0.1);
    border: 2px solid #10b981;
    padding: 30px;
    border-radius: 18px;
    margin-top: 30px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD ASSETS
# -----------------------------
@st.cache_resource
def load_files():
    try:
        model = pickle.load(open("Model/house_model.pkl", "rb"))
        encoder = pickle.load(open("Model/location_encoder.pkl", "rb"))
        return model, encoder
    except:
        return None, None

model, encoder = load_files()

# -----------------------------
# MAIN UI
# -----------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("<h1>PropAI Predictor 🏠</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enterprise-grade Real Estate Analysis</p>", unsafe_allow_html=True)

if model and encoder:
    locations = list(encoder.classes_)
    
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("📍 Location / Area", locations)
        area = st.number_input("📐 Carpet Area (Sqft)", min_value=100, step=50, value=1200)
    
    with col2:
        bhk = st.selectbox("🛏️ Apartment Type", [1, 2, 3, 4, 5], index=1, format_func=lambda x: f"{x} BHK")
        bath = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5], index=1)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("") # Spacer
    
    if st.button("GET VALUATION REPORT"):
        try:
            loc_idx = encoder.transform([location])[0]
            prediction = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]
            
            st.markdown(f"""
                <div class='result-box'>
                    <p style='color: #a7f3d0; margin:0; font-size: 0.9rem; font-weight:600;'>ESTIMATED VALUATION</p>
                    <h1 style='color: #10b981 !important; margin:10px 0; font-size: 2.5rem;'>{format_inr(prediction)}</h1>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.error("Model files missing in 'Model/' folder.")

st.markdown("</div>", unsafe_allow_html=True)
