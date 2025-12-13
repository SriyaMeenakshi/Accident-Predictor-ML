import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIG & PROFESSIONAL DARK THEME
# ---------------------------------------------------------
st.set_page_config(page_title="Road Accident Risk Intelligence", page_icon="🚗", layout="wide")

# Professional Dark Theme CSS
st.markdown("""
    <style>
    /* Main App Background - Dark Slate/Black */
    .stApp {
        background-color: #0f172a; /* Slate 900 */
        color: #f1f5f9;
    }
    
    /* Metrics/Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1e293b; /* Slate 800 */
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Custom Header Underline */
    .header-style {
        font-size: 20px;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 15px;
        border-bottom: 2px solid #3b82f6; /* Blue border */
        padding-bottom: 8px;
    }
    
    /* Risk Levels Typography */
    .risk-high { 
        color: #ef4444; 
        font-weight: 800; 
        font-size: 42px; 
        letter-spacing: 1px;
    }
    .risk-medium { 
        color: #f59e0b; 
        font-weight: 800; 
        font-size: 42px; 
        letter-spacing: 1px;
    }
    .risk-low { 
        color: #22c55e; 
        font-weight: 800; 
        font-size: 42px; 
        letter-spacing: 1px;
    }
    
    /* Table Styling to match Dark Theme */
    .dataframe {
        font-size: 14px !important;
        background-color: #1e293b !important;
        color: white !important;
    }
    
    /* Force image to fit container nicely */
    img {
        border-radius: 12px;
    }
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
st.sidebar.title("🛡️ Simulation Panel")
st.sidebar.markdown("Adjust parameters below to see how the AI predicts risk.")

def get_median(col, default):
    return float(df[col].median()) if col in df.columns else default

st.sidebar.markdown("### 🕒 Time & Date")
hour = st.sidebar.slider("Hour (24h)", 0, 23, 18)
day_week = st.sidebar.selectbox("Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], index=0)
month = st.sidebar.selectbox("Month", range(1, 13), index=10)

st.sidebar.markdown("### 🛣️ Infrastructure")
c1, c2 = st.sidebar.columns(2)
with c1:
    junction = st.checkbox("Junction", value=False)
    signal = st.checkbox("Traffic Signal", value=False)
    crossing = st.checkbox("Crossing", value=False)
with c2:
    station = st.checkbox("Station", value=False)
    stop = st.checkbox("Stop Sign", value=False)
    bump = st.checkbox("Speed Bump", value=False)

st.sidebar.markdown("### 🌤️ Environment")
temp = st.sidebar.slider("Temp (°F)", -10, 110, int(get_median("Temperature(F)", 70)))
humid = st.sidebar.slider("Humidity (%)", 0, 100, int(get_median("Humidity(%)", 60)))
wind = st.sidebar.slider("Wind (mph)", 0, 100, int(get_median("Wind_Speed(mph)", 10)))
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
for name, model in models.items():
    final_input = scaler.transform(input_data) if name == "Logistic Regression" else input_data
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
# MAIN UI: HEADER LAYOUT (Left Text, Right Image)
# ---------------------------------------------------------
col_head_text, col_head_img = st.columns([2, 1])

with col_head_text:
    st.title("🚦AI-Driven Road Safety System ")
    st.markdown("Machine Learning Project - Predicting Accident Risk Levels")
    st.markdown("""
    This intelligent system uses **Machine Learning** to predict the risk of road accidents in real-time. 
    By analyzing weather, time, and road conditions, it helps authorities deploy resources efficiently.
    """)

with col_head_img:
    # High-quality tech/road image, displayed cleanly on the right
    st.image("https://c1.wallpaperflare.com/preview/412/860/56/traffic-autos-vehicles-road.jpg", 
             use_container_width=True)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Prediction Dashboard", "📈 Model Analytics", "ℹ️ About Project"])

with tab1:
    st.info("💡 **Dashboard View:** Shows the live prediction based on the inputs you selected in the sidebar.")
    
    # --- RESULT SECTION ---
    st.markdown("<div class='header-style'>System Prediction (Consensus)</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.5, 2.5])
    with c1:
        st.write("Voting Ensemble Output:")
        if "High" in final_decision:
            st.markdown(f"<div class='risk-high'>HIGH RISK 🚨</div>", unsafe_allow_html=True)
            st.error("Advisory: Deploy emergency teams immediately.")
        elif "Medium" in final_decision:
            st.markdown(f"<div class='risk-medium'>MEDIUM RISK ⚠️</div>", unsafe_allow_html=True)
            st.warning("Advisory: Exercise caution. Standard patrols.")
        else:
            st.markdown(f"<div class='risk-low'>LOW RISK ✅</div>", unsafe_allow_html=True)
            st.success("Status: Safe driving conditions.")

    with c2:
        # Probability Chart - CLEAN DARK THEME
        st.write("Probability Distribution (Voting Model)")
        prob_data = pd.DataFrame({"Risk": le.classes_, "Prob": voting_result["Raw_Prob"]})
        
        # Transparent background for the plot
        fig, ax = plt.subplots(figsize=(6, 2.2), facecolor='#0f172a') 
        sns.barplot(x="Prob", y="Risk", data=prob_data, palette="viridis", ax=ax)
        
        # Dark theme adjustments
        ax.set_facecolor("#0f172a") 
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white') 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 1)
        st.pyplot(fig)

    # --- TABLE SECTION ---
    st.markdown("---")
    st.markdown("<div class='header-style'>Model Consensus Table</div>", unsafe_allow_html=True)
    st.write("Comparison of different AI models to ensure the final decision is accurate.")
    
    def color_risk(val):
        color = '#22c55e' if 'Low' in val else '#f59e0b' if 'Medium' in val else '#ef4444'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        res_df[["Model", "Prediction", "Confidence"]].style.applymap(color_risk, subset=['Prediction']),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.info("💡 **Analytics View:** Shows how well the models performed during training and what data they find most important.")
    
    st.header("Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**F1-Score Comparison**")
        st.write("Higher score means better accuracy.")
        st.bar_chart(metrics_df.set_index("Model")["F1_Score"])
    with c2:
        st.markdown("**Detailed Metrics**")
        st.write("Precision, Recall, and Accuracy data.")
        st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)

    st.markdown("---")
    st.markdown("**Feature Importance (XGBoost)**")
    st.write("These are the factors that most strongly influence the risk calculation.")
    xgb_model = models["XGBoost"]
    imp_df = pd.DataFrame({"Feature": input_data.columns, "Importance": xgb_model.feature_importances_}).sort_values(by="Importance", ascending=False).head(8)
    
    # Feature Importance Plot
    fig2, ax2 = plt.subplots(figsize=(10, 3), facecolor='#0f172a')
    sns.barplot(x="Importance", y="Feature", data=imp_df, palette="magma", ax=ax2)
    
    ax2.set_facecolor("#0f172a")
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    st.pyplot(fig2)

with tab3:
    st.header("📖 About the Project")
    
    st.markdown("""
    ### 1. What is this project?
    This is a **Predictive Analytics System** for Road Safety. Instead of reacting to accidents after they happen, this system uses Artificial Intelligence to predict **where and when** accidents are likely to occur based on live conditions.
    
    ### 2. How does it work?
    The system follows a simple 3-step process:
    * **Step 1 (Input):** It takes data about the **Environment** (Rain, Fog, Wind), **Time** (Rush hour, Night/Day), and **Road** (Junctions, Signals).
    * **Step 2 (Processing):** It feeds this data into 4 different Machine Learning models (**Random Forest, XGBoost, Logistic Regression, and a Voting Ensemble**).
    * **Step 3 (Output):** The models vote to decide if the current situation is **High Risk, Medium Risk, or Low Risk**.
    
    ### 3. Understanding the Sections
    * **📊 Prediction Dashboard:** This is the main control center. You can simulate different weather/road scenarios using the sidebar and see the AI's risk prediction instantly.
    * **📈 Model Analytics:** This section is for the technical deep-dive. It proves that the models are accurate by showing their test scores (F1-Score, Accuracy) and graphs showing which factors (like Speed or Rain) are most dangerous.
    
    ### 4. Why is this useful?
    * **For Traffic Police:** They can place patrol cars at "High Risk" locations before accidents happen.
    * **For Hospitals:** Ambulances can be put on standby during high-risk weather.
    * **For Drivers:** Navigation apps could warn drivers to be careful in specific zones.
    """)
    
    st.success("Developed by **Ch.Sriya Meenakshi, G. Divya Samhitha Varshini, I. Spandana, D. Satya Harshitha, D. Kavya Malini** | Department of CSE(AI & ML) | 3 CSM1 | GVPCEW")