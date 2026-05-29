# ==============================================================================
# DIABETES PREDICTION INTERACTIVE STREAMLIT APPLICATION (DARK THEMED)
# ==============================================================================
# This Streamlit application serves as a dark-themed user interface for our
# trained machine learning model. It allows users to input health indicators,
# scales the data, and displays predicted risk via a progress gauge, prediction cards,
# and dynamically rendered dark-style plots.
# ==============================================================================

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------------------------
# SECTION 1: Page Configuration & Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Classifier (Dark Mode)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for forcing a high-end Dark Mode UI across the app
st.markdown("""
<style>
    /* Root application style */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Text colors and titles */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #f8fafc !important;
    }
    
    /* Slider widget styles */
    .stSlider > label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    
    /* Dashboard card containers */
    .card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Prediction success/failure banner cards */
    .result-negative {
        background-color: #065f46;
        border: 1px solid #10b981;
        color: #a7f3d0 !important;
        padding: 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .result-positive {
        background-color: #7f1d1d;
        border: 1px solid #ef4444;
        color: #fca5a5 !important;
        padding: 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Actionable tips box */
    .tip-box {
        background-color: #1e3a8a;
        border: 1px solid #3b82f6;
        color: #bfdbfe !important;
        padding: 1.2rem;
        border-radius: 8px;
        font-size: 1rem;
    }
    
    /* Centered big title */
    .center-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3b82f6 !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# SECTION 2: Data Loading & Imputation Utilities
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    """
    Loads raw Pima Indians dataset and pre-processes invalid zero entries
    with median values to ensure statistical accuracy in visual graphs.
    """
    df = pd.read_csv('diabetes.csv')
    # Replace invalid 0 values with column medians in Glucose, BMI, BloodPressure
    for col in ['Glucose', 'BMI', 'BloodPressure']:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df

@st.cache_resource
def load_model_assets():
    """
    Loads saved pickle files: Trained ML model and fitted StandardScaler.
    """
    model_path = 'model.pkl'
    scaler_path = 'scaler.pkl'
    
    model = None
    scaler = None
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
    return model, scaler

@st.cache_resource
def train_rf_importances():
    """
    Trains a Random Forest classifier on preprocessed dataset to calculate
    stable feature importance scores for Visual Analytics.
    """
    df_rf = pd.read_csv('diabetes.csv')
    for col in ['Glucose', 'BMI', 'BloodPressure']:
        df_rf[col] = df_rf[col].replace(0, np.nan)
        df_rf[col] = df_rf[col].fillna(df_rf[col].median())
    X_rf = df_rf.drop(columns=['Outcome'])
    y_rf = df_rf['Outcome']
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_rf, y_rf)
    return rf.feature_importances_, X_rf.columns

# Load assets
df = load_and_preprocess_data()
model, scaler = load_model_assets()


# ------------------------------------------------------------------------------
# SECTION 3: Left Sidebar (Patient Diagnostics Input)
# ------------------------------------------------------------------------------
st.sidebar.markdown("<h2 style='margin-bottom:0;'>Patient Diagnostics Input</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:0.9rem; color:#94a3b8 !important; margin-bottom:1.5rem;'>Adjust the values below based on physical clinical measurements</p>", unsafe_allow_html=True)

# 8 features sliders with info hover icons showing normal medical range
pregnancies = st.sidebar.slider(
    "Pregnancies",
    min_value=0, max_value=20, value=2, step=1,
    help="Pregnancies: Normal Range: 0 - 4 is typical, but varies widely per patient."
)

glucose = st.sidebar.slider(
    "Glucose Level (mg/dL)",
    min_value=40, max_value=200, value=117, step=1,
    help="Glucose: Normal Range: < 140 mg/dL (2-hour oral glucose tolerance test)."
)

bp = st.sidebar.slider(
    "Blood Pressure (mm Hg)",
    min_value=40, max_value=130, value=72, step=1,
    help="Blood Pressure (Diastolic): Normal Range: 60 - 80 mm Hg."
)

skin = st.sidebar.slider(
    "Skin Thickness (mm)",
    min_value=0, max_value=100, value=23, step=1,
    help="Skin Thickness (Triceps skin fold): Normal Range: 10 - 30 mm."
)

insulin = st.sidebar.slider(
    "Insulin Level (mu U/ml)",
    min_value=0, max_value=900, value=80, step=5,
    help="Insulin: Normal Range: 16 - 166 mu U/ml (2-hour post glucose load)."
)

bmi = st.sidebar.slider(
    "BMI (Body Mass Index)",
    min_value=10.0, max_value=70.0, value=32.0, step=0.1,
    help="BMI: Normal Range: 18.5 - 24.9 kg/m²."
)

dpf = st.sidebar.slider(
    "Diabetes Pedigree Function",
    min_value=0.05, max_value=3.0, value=0.37, step=0.01,
    help="Diabetes Pedigree Function (Hereditary score): Normal Range: < 0.5 (lower represents lower genetic risk)."
)

age = st.sidebar.slider(
    "Age (Years)",
    min_value=21, max_value=120, value=29, step=1,
    help="Age: Age of patient in years (Dataset ranges: 21 - 81 years)."
)


# ------------------------------------------------------------------------------
# SECTION 4: Main Layout Tabs
# ------------------------------------------------------------------------------
st.markdown("<div class='center-title'>🩺 Diabetes Health Risk Predictor</div>", unsafe_allow_html=True)

# Initialize Session State to track predictions across tab actions
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
    st.session_state.prob = 0.0
    st.session_state.pred_class = 0

tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📊 Visual Analytics", "📁 Dataset Overview"])


# ==============================================================================
# TAB 1: Make Prediction
# ==============================================================================
with tab1:
    # 1. Gauge/Progress Bar at the top showing risk percentage
    if st.session_state.predicted:
        risk_percentage = st.session_state.prob * 100
        color = "#ef4444" if risk_percentage >= 50 else "#10b981"
        
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <h4 style="margin: 0 0 0.5rem 0; color: #f8fafc !important;">Estimated Diabetes Risk Probability: <strong style="color:{color};">{risk_percentage:.1f}%</strong></h4>
            <div style="background-color: #334155; border-radius: 10px; height: 24px; width: 100%; overflow: hidden;">
                <div style="background-color: {color}; width: {risk_percentage}%; height: 100%; transition: width 0.5s ease-in-out;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align: center; border-style: dashed;">
            <h4 style="margin: 0; color: #94a3b8 !important;">💡 Ready for Analysis</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.95rem; color: #94a3b8 !important;">Click the <strong>'Calculate Diabetes Risk'</strong> button to evaluate indicators and check risk probability gauge.</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Split Panel Layout
    col1, col2 = st.columns([1, 1])
    
    # Left Panel: Summary Table
    with col1:
        st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
        st.subheader("📋 Captured Clinical Indicators")
        st.write("Summary of inputs adjusted in the left sidebar:")
        
        summary_data = {
            "Feature Indicator": [
                "Pregnancies", "Glucose Level (mg/dL)", "Blood Pressure (mm Hg)", 
                "Skin Thickness (mm)", "Insulin (mu U/ml)", "BMI", 
                "Diabetes Pedigree Function", "Age"
            ],
            "Value Entered": [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        }
        st.table(pd.DataFrame(summary_data))
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Right Panel: ML Prediction Control & Banners
    with col2:
        st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
        st.subheader("🤖 ML Prediction Model")
        st.write("Process indicators using the scaled Decision Tree classifier.")
        
        # Red prediction action button
        predict_btn = st.button("Calculate Diabetes Risk", type="primary", use_container_width=True)
        
        if predict_btn:
            with st.spinner("Analyzing patient metrics..."):
                # Simulating calculation delay for UI spinner effect
                time.sleep(0.6)
                
                # Check for model and scaler availability
                if model is None or scaler is None:
                    st.error("Model files missing! Run diabetes_prediction.py first.")
                else:
                    # Arrange features as trained DataFrame
                    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                    input_df = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], columns=feature_names)
                    
                    # Transform inputs
                    input_scaled = scaler.transform(input_df)
                    
                    # Run ML prediction
                    pred_class = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0][1]
                    
                    # Store in state
                    st.session_state.predicted = True
                    st.session_state.prob = prob
                    st.session_state.pred_class = pred_class
                    
                    # Force rerun to show the gauge at the top
                    st.rerun()

        # Display result banners if prediction was executed
        if st.session_state.predicted:
            if st.session_state.pred_class == 1:
                st.markdown("""
                <div class="result-positive">
                    ⚠️ DIABETIC RISK DETECTED
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.95rem; font-weight: normal; color: #fca5a5 !important;">
                        The classifier predicts a HIGH risk of Diabetes. Clinical evaluation is highly recommended.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="tip-box">
                    <strong>💡 Actionable Medical Advice:</strong><br>
                    - Schedule a formal blood diagnostic panel (A1C, Fasting Blood Glucose).<br>
                    - Monitor carbohydrate and refined sugar intake.<br>
                    - Incorporate consistent cardiovascular and aerobic exercise (30+ minutes daily).<br>
                    - Speak to an endocrinologist or primary care physician.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-negative">
                    ✅ NO DIABETES DETECTED
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.95rem; font-weight: normal; color: #a7f3d0 !important;">
                        The classifier predicts a LOW risk of Diabetes. Patient characteristics appear healthy.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="tip-box" style="background-color: #064e3b; border-color: #059669; color: #d1fae5 !important;">
                    <strong>💡 Maintenance Advice:</strong><br>
                    - Maintain a nutrient-dense diet focusing on whole grains, lean proteins, and healthy fats.<br>
                    - Stay active and maintain your current healthy BMI range.<br>
                    - Keep up with annual routine clinical screening evaluations.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Input patient parameters and click 'Calculate Diabetes Risk' above to view predictions.")
            
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 2: Visual Analytics (Dark Theme)
# ==============================================================================
with tab2:
    st.subheader("📊 Dark-Themed Diagnostic Visualizations")
    st.write("Visual inspection of underlying distributions, attributes correlations, and feature importances.")
    
    # Configure Matplotlib dark style
    plt.style.use('dark_background')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>🔗 Features Correlation Heatmap</h4>", unsafe_allow_html=True)
        # 1. Heatmap
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax1, cbar=True)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>⚖️ Outcome Cases Distribution</h4>", unsafe_allow_html=True)
        # 3. Outcome Pie Chart
        fig3, ax3 = plt.subplots(figsize=(7, 7))
        counts = df['Outcome'].value_counts()
        ax3.pie(counts, labels=['Non-Diabetic', 'Diabetic'], autopct='%1.1f%%', 
                colors=['#10b981', '#ef4444'], startangle=90, explode=[0, 0.08],
                textprops={'fontsize': 12, 'color': '#f8fafc'})
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>👑 Random Forest Feature Importance</h4>", unsafe_allow_html=True)
        # 2. RF Importances
        try:
            importances, col_names = train_rf_importances()
            indices = np.argsort(importances)[::-1]
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            sns.barplot(x=importances[indices], y=col_names[indices], palette='viridis', ax=ax2)
            ax2.set_xlabel('Relative Importance Value')
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close()
        except Exception as e:
            st.error(f"Failed to calculate importances: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>💧 Glucose Distribution by Diabetic Class</h4>", unsafe_allow_html=True)
        # 4. Glucose Histplot
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        sns.histplot(data=df, x='Glucose', hue='Outcome', kde=True, multiple='stack', 
                     palette=['#10b981', '#ef4444'], ax=ax4, alpha=0.7)
        ax4.set_xlabel('Glucose (mg/dL)')
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 3: Dataset Overview
# ==============================================================================
with tab3:
    st.subheader("📁 Clinical Dataset Explorer")
    st.write("Browse descriptive values, shape sizes, and raw data entries.")
    
    # 1. Basic Stats metrics
    total_records = len(df)
    diabetic_count = (df['Outcome'] == 1).sum()
    non_diabetic_count = (df['Outcome'] == 0).sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", total_records)
    with col2:
        st.metric("Diabetic Count", diabetic_count, delta=None)
    with col3:
        st.metric("Non-Diabetic Count", non_diabetic_count)
        
    st.markdown("---")
    
    # 2. Describe Stats
    st.subheader("📋 Descriptive Statistics summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # 3. Full Dataframe view
    st.subheader("🔍 Raw Data Records")
    st.dataframe(df, use_container_width=True)
