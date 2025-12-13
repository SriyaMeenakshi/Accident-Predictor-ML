import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIG & BACKGROUND
# ---------------------------------------------------------
st.set_page_config(page_title="Road Accident Risk Intelligence", page_icon="🛡️", layout="wide")

# High-quality "Night City/Traffic" Background
BACKGROUND_URL = "https://images.unsplash.com/photo-1545153243-7f212239c8e2?q=80&w=2071&auto=format&fit=crop"

st.markdown(f"""
    <style>
    /* Background Image */
    .stApp {{
        background-image: url("{BACKGROUND_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Overlay to make text readable */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(15, 23, 42, 0.85); /* Dark Blue overlay */
        z-index: -1;
    }}
    
    /* Global Text */
    .main {{ color: #E0E0E0; }}
    h1, h2, h3, p, label {{ color: #f1f2f6 !important; }}
    
    /* Metrics containers */
    div[data-testid="stMetric"] {{
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 10px;
        border-radius: 8px;
        backdrop-filter: blur(5px);
    }}
    
    /* Custom Header */
    .header-style {{
        font-size: 24px;
        font-weight: bold;
        color: #f1f2f6;
        margin-bottom: 10px;
        border-bottom: 2px solid #2e86de;
        padding-bottom: 5px;
    }}
    
    /* Risk Levels for Main Card */
    .risk-high {{ color: #ff4757; font-weight: 800; font-size: 48px; text-shadow: 0px 0px 15px rgba(255, 71, 87, 0.6); }}
    .risk-medium {{ color: #ffa502; font-weight: 800; font-size: 48px; text-shadow: 0px 0px 15px rgba(255, 165, 2, 0.6); }}
    .risk-low {{ color: #2ed573; font-weight: 800; font-size: 48px; text-shadow: 0px 0px 15px rgba(46, 213, 115, 0.6); }}
    
    /* Comparison Table Styling */
    .dataframe {{ font-size: 14px !important; background-color: rgba(0,0,0,0.3) !important; }}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD RESOURCES
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    assets = {}
    try:
        assets["label_encoder"] = joblib.load("label_encoder.pkl")
        assets["scaler"] = joblib.load("scaler.pkl")
        assets["models"] = {
            "Random Forest": joblib.load("rf_model.pkl"),
            "XGBoost": joblib.load("xgb_model.pkl"),
            "Logistic Regression": joblib.load("lr_model.pkl"),
            "Voting Ensemble": joblib.load("voting_model.pkl")
        }
        with open("metrics.json", "r") as f:
            assets["metrics"] = pd.DataFrame(json.load(f))
    except Exception as e:
        st.error(f"Error loading files: {e}. Run train_models.py first.")
        st.stop()
    return assets

data = load_assets()
le = data["label_encoder"]
scaler = data["scaler"]
models = data["models"]
metrics_df = data["metrics"]

@st.cache_data
def load_raw_data():
    return pd.read_csv("processed_accidents.csv")
df = load_raw_data()

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.title("🛡️ Live Input Panel")
st.sidebar.info("Adjust conditions to see how different ML models react.")

def get_median(col, default):
    return float(df[col].median()) if col in df.columns else default

# User Inputs
st.sidebar.markdown("### 🕒 Temporal Inputs")
hour = st.sidebar.slider("Hour of Day", 0, 23, 18)
day_week = st.sidebar.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], index=0)
month = st.sidebar.selectbox("Month", range(1, 13), index=10)

st.sidebar.markdown("### 🛣️ Road Infrastructure")
c1, c2 = st.sidebar.columns(2)
with c1:
    junction = st.checkbox("Junction", value=False)
    signal = st.checkbox("Traffic Signal", value=False)
    crossing = st.checkbox("Crossing", value=False)
with c2:
    station = st.checkbox("Station", value=False)
    stop = st.checkbox("Stop Sign", value=False)
    bump = st.checkbox("Speed Bump", value=False)

st.sidebar.markdown("### 🌤️ Weather Conditions")
temp = st.sidebar.slider("Temperature (°F)", -10, 110, int(get_median("Temperature(F)", 70)))
humid = st.sidebar.slider("Humidity (%)", 0, 100, int(get_median("Humidity(%)", 60)))
wind = st.sidebar.slider("Wind Speed (mph)", 0, 100, int(get_median("Wind_Speed(mph)", 10)))
vis = st.sidebar.slider("Visibility (mi)", 0.0, 10.0, 10.0)
pressure = st.sidebar.number_input("Pressure (in)", 20.0, 35.0, 29.0)

# ---------------------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------------------
day_map = {k:v for v,k in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])}
is_rush = 1 if hour in [7,8,9,16,17,18,19] else 0

input_data = pd.DataFrame({
    "Hour": [hour],
    "Month": [month],
    "DayOfWeek": [day_map[day_week] if df["DayOfWeek"].dtype != object else day_week],
    "Temperature(F)": [temp],
    "Humidity(%)": [humid],
    "Pressure(in)": [pressure],
    "Visibility(mi)": [vis],
    "Wind_Speed(mph)": [wind],
    "Junction": [1 if junction else 0],
    "Traffic_Signal": [1 if signal else 0],
    "Crossing": [1 if crossing else 0],
    "Station": [1 if station else 0],
    "Stop": [1 if stop else 0],
    "Bump": [1 if bump else 0],
    "Is_Rush_Hour": [is_rush]
})

results = []

# Loop through all models
for name, model in models.items():
    if name == "Logistic Regression":
        final_input = scaler.transform(input_data)
    else:
        final_input = input_data
    
    try:
        pred_idx = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0]
        pred_label = le.inverse_transform([int(pred_idx)])[0]
        confidence = np.max(prob)
        
        results.append({
            "Model": name,
            "Prediction": pred_label,
            "Confidence": f"{confidence:.2%}",
            "Raw_Prob": prob 
        })
    except:
        results.append({"Model": name, "Prediction": "Error", "Confidence": "0%", "Raw_Prob": [0,0,0]})

res_df = pd.DataFrame(results)
voting_result = res_df[res_df["Model"] == "Voting Ensemble"].iloc[0]
final_decision = voting_result["Prediction"]

# ---------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------
st.title("🚦 AI-Driven Road Safety System")
st.markdown("### 3-1 Semester Term Project | Machine Learning Classification")

tab1, tab2, tab3 = st.tabs(["🖥️ Main Dashboard", "📊 Model Performance Analytics", "ℹ️ About"])

with tab1:
    # --- SECTION 1: THE BIG RESULT ---
    st.markdown("<div class='header-style'>System Prediction (Consensus)</div>", unsafe_allow_html=True)
    
    col_main, col_chart = st.columns([1.5, 2.5])
    
    with col_main:
        st.write("Based on the **Voting Ensemble** (RF + XGB + LR):")
        
        if "High" in final_decision:
            st.markdown(f"<div class='risk-high'>HIGH RISK 🚨</div>", unsafe_allow_html=True)
            st.error("⚠️ Advisory: Dangerous conditions. Emergency teams on standby.")
        elif "Medium" in final_decision:
            st.markdown(f"<div class='risk-medium'>MEDIUM RISK ⚠️</div>", unsafe_allow_html=True)
            st.warning("⚠️ Advisory: Exercise caution. Increased traffic likely.")
        else:
            st.markdown(f"<div class='risk-low'>LOW RISK ✅</div>", unsafe_allow_html=True)
            st.success("✅ Status: Standard driving conditions.")

    with col_chart:
        st.write("**Probability Distribution (Voting Model)**")
        prob_data = pd.DataFrame({
            "Risk Level": le.classes_,
            "Probability": voting_result["Raw_Prob"]
        })
        fig, ax = plt.subplots(figsize=(6, 2), facecolor='none')
        sns.barplot(x="Probability", y="Risk Level", data=prob_data, palette="viridis", ax=ax)
        ax.set_facecolor("none")
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.set_xlim(0, 1)
        st.pyplot(fig)

    # --- SECTION 2: THE "PROOF" (COMPARISON TABLE) ---
    st.markdown("---")
    st.markdown("<div class='header-style'>Real-Time Model Consensus</div>", unsafe_allow_html=True)
    
    def color_risk(val):
        color = 'white'
        if 'High' in val: color = '#ff4757'
        elif 'Medium' in val: color = '#ffa502'
        elif 'Low' in val: color = '#2ed573'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        res_df[["Model", "Prediction", "Confidence"]].style.applymap(color_risk, subset=['Prediction']),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.header("Training Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### F1-Score Comparison")
        st.bar_chart(metrics_df.set_index("Model")["F1_Score"])
    
    with c2:
        st.markdown("#### Detailed Metrics")
        st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Feature Importance (XGBoost)")
    
    # Feature Importance Plot with dark theme support
    xgb_model = models["XGBoost"]
    imp_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": xgb_model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(7)
    
    fig2, ax2 = plt.subplots(figsize=(10, 3), facecolor='none')
    sns.barplot(x="Importance", y="Feature", data=imp_df, palette="magma", ax=ax2)
    ax2.set_facecolor("none")
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('white')
    st.pyplot(fig2)

with tab3:
    st.markdown("### About this ML Project")
    st.markdown("Predictive modeling of accident severity using balanced ensemble learning techniques.")