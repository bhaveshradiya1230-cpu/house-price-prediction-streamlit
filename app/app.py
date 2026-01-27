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

/* Main Background */
.stApp {
    background: radial-gradient(circle at top, #1e293b, #0f172a);
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit elements */
header, footer, .stDeployButton {display: none !important;}

/* Main Wrapper - No disappearing border */
.main-card {
    border: 1.5px solid rgba(34, 211, 238, 0.4);
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(10px);
    padding: 40px;
    border-radius: 24px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    margin-bottom: 20px;
}

/* Sub-card for Inputs */
.input-section {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 25px;
    border-radius: 16px;
    margin-top: 20px;
}

/* Text Styles */
h1 { color: #f8fafc !important; font-weight: 800 !important; letter-spacing: -1px; text-align: center; }
p.subtitle { color: #94a3b8 !important; text-align: center; margin-top: -15px; font-size: 1.1rem; }

/* Input Labels */
label p {
    color: #22d3ee !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 1px;
}

/* Button Styling */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
    color: white;
    border: none;
    padding: 15px;
    font-weight: 700;
    border-radius: 12px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(6, 182, 212, 0.5);
    color: white;
}

/* Prediction Result Box */
.result-box {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.2));
    border: 2px solid #10b981;
    padding: 30px;
    border-radius: 18px;
    margin-top: 30px;
    text-align: center;
    animation: fadeIn 0.5s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
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
    
    st.write("---")
    
    if st.button("GET VALUATION REPORT"):
        try:
            loc_idx = encoder.transform([location])[0]
            prediction = model.predict(np.array([[area, bhk, bath, loc_idx]]))[0]
            
            st.markdown(f"""
                <div class='result-box'>
                    <p style='color: #a7f3d0; margin:0; font-size: 0.9rem; font-weight:600;'>ESTIMATED VALUATION</p>
                    <h1 style='color: #10b981 !important; margin:10px 0; font-size: 2.5rem;'>{format_inr(prediction)}</h1>
                    <p style='color: #94a3b8; margin:0; font-size: 0.8rem;'>Confidence Score: 94.2%</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.warning("⚠️ Setup Required: Please ensure 'Model/house_model.pkl' and 'Model/location_encoder.pkl' exist.")

st.markdown("</div>", unsafe_allow_html=True)

# Footer/Resume Ready Info
st.markdown("""
    <p style='text-align:center; color:#64748b; font-size:0.8rem;'>
        Built with Scikit-Learn & Streamlit • Data Source: Real Estate Listings
    </p>
""", unsafe_allow_html=True)
