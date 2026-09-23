import streamlit as st
import joblib
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

st.set_page_config(page_title="PredictX Dashboard", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f172a; color: #f8fafc; }
    h1, h2, h3 { color: #e2e8f0; }
    .metric-card {
        background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px;
    }
    .metric-value { font-size: 2.5rem; font-weight: 600; color: #38bdf8; }
    .metric-label { font-size: 1rem; color: #94a3b8; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def load_data():
    try:
        reg_models = joblib.load(os.path.join(MODEL_DIR, 'all_reg_models.pkl'))
        reg_scaler = joblib.load(os.path.join(MODEL_DIR, 'reg_scaler.pkl'))
        reg_metrics = pd.read_csv(os.path.join(MODEL_DIR, 'reg_metrics.csv'))
        
        clf_models = joblib.load(os.path.join(MODEL_DIR, 'all_clf_models.pkl'))
        clf_scaler = joblib.load(os.path.join(MODEL_DIR, 'clf_scaler.pkl'))
        clf_metrics = pd.read_csv(os.path.join(MODEL_DIR, 'clf_metrics.csv'))
        clf_features = joblib.load(os.path.join(MODEL_DIR, 'clf_features.pkl'))
        clf_sample = pd.read_csv(os.path.join(MODEL_DIR, 'clf_sample_data.csv'))
        
        return reg_models, reg_scaler, reg_metrics, clf_models, clf_scaler, clf_metrics, clf_features, clf_sample
    except Exception as e:
        return None, None, None, None, None, None, None, None

reg_models, reg_scaler, reg_metrics, clf_models, clf_scaler, clf_metrics, clf_features, clf_sample = load_data()

if reg_models is None:
    st.error("Failed to load models. Please ensure the background training script has finished.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("⚡ PredictX Navigation")
track = st.sidebar.radio("Select Track", ["Regression (Power Plant)", "Classification (Steel Faults)"])
page = st.sidebar.radio("Select Page", ["Interactive Prediction", "Model Analysis & Leaderboard"])

if track == "Regression (Power Plant)":
    st.title("🏭 Combined Cycle Power Plant")
    if page == "Interactive Prediction":
        st.subheader("Interactive Prediction")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Inputs")
            with st.form("reg_form"):
                at = st.number_input("Ambient Temperature (AT)", value=20.0)
                v = st.number_input("Exhaust Vacuum (V)", value=50.0)
                ap = st.number_input("Ambient Pressure (AP)", value=1013.0)
                rh = st.number_input("Relative Humidity (RH)", value=75.0)
                selected_model = st.selectbox("Select Model", list(reg_models.keys()), index=6)
                submit = st.form_submit_button("Predict & Analyze")
                
        with col2:
            st.markdown("### Output")
            if submit:
                with st.status("Predicting...", expanded=True) as status:
                    st.write("Extracting environmental conditions...")
                    time.sleep(0.3)
                    st.write("Applying feature engineering (Temp-Humidity Index, AT-V Interaction)...")
                    time.sleep(0.4)
                    
                    input_data = pd.DataFrame({'AT': [at], 'V': [v], 'AP': [ap], 'RH': [rh]})
                    input_data['Temp_Humidity_Index'] = input_data['AT'] * input_data['RH'] / 100.0
                    input_data['AT_V_interaction'] = input_data['AT'] * input_data['V']
                    features = ['AT', 'V', 'AP', 'RH', 'Temp_Humidity_Index', 'AT_V_interaction']
                    
                    st.write("Scaling data with standard scaler...")
                    time.sleep(0.3)
                    scaled_input = reg_scaler.transform(input_data[features])
                    
                    st.write(f"Running {selected_model}...")
                    time.sleep(0.4)
                    pred = reg_models[selected_model].predict(scaled_input)[0]
                    status.update(label="Prediction Complete!", state="complete", expanded=False)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{selected_model} Prediction</div>
                    <div class="metric-value">{pred:.2f} MW</div>
                </div>
                """, unsafe_allow_html=True)
                
                all_preds = {name: model.predict(scaled_input)[0] for name, model in reg_models.items()}
                pred_df = pd.DataFrame(list(all_preds.items()), columns=['Model', 'Prediction']).sort_values('Prediction')
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=pred_df, x='Prediction', y='Model', ax=ax, palette='viridis')
                ax.set_title("Predictions Across All Models")
                ax.set_xlim(pred_df['Prediction'].min() - 5, pred_df['Prediction'].max() + 5)
                st.pyplot(fig)
    
    else:
        st.subheader("Model Analysis & Leaderboard")
        st.dataframe(reg_metrics.sort_values('R2 Score', ascending=False).style.background_gradient(cmap='Blues'), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.barplot(data=reg_metrics.sort_values('R2 Score', ascending=False), y='Model', x='R2 Score', palette='mako', ax=ax)
            ax.set_title("R2 Score Comparison")
            st.pyplot(fig)
            
        with col2:
            rf = reg_models['Random Forest']
            features = ['AT', 'V', 'AP', 'RH', 'Temp_Index', 'AT_V']
            importances = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
            fig, ax = plt.subplots()
            sns.barplot(data=importances.sort_values('Importance', ascending=False), y='Feature', x='Importance', palette='rocket', ax=ax)
            ax.set_title("Random Forest Feature Importance")
            st.pyplot(fig)

elif track == "Classification (Steel Faults)":
    st.title("🛡️ Steel Plates Faults Classification")
    if page == "Interactive Prediction":
        st.subheader("Interactive Prediction with XAI")
        
        st.markdown("Click the button below to load a random sample, then **double-click any cell in the table to edit** the 27 features directly before predicting!")
        
        if 'sample_idx' not in st.session_state:
            st.session_state.sample_idx = 0
            
        if st.button("Load Random Sample"):
            st.session_state.sample_idx = np.random.randint(0, len(clf_sample))
            
        sample = clf_sample.iloc[st.session_state.sample_idx]
        actual_target = sample['target']
        input_features = sample.drop('target').to_frame().T
        
        # EDITABLE DATA EDITOR (No 27 input boxes!)
        edited_features = st.data_editor(input_features, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_model = st.selectbox("Select Model", list(clf_models.keys()), index=3) # Decision Tree default for XAI speed
            st.markdown(f"**Actual Fault Type (From Sample):** `{actual_target}`")
            submit = st.button("Predict & Explain", type="primary", use_container_width=True)
                
        with col2:
            if submit:
                with st.status("Analyzing and Predicting...", expanded=True) as status:
                    st.write("Loading editable features...")
                    time.sleep(0.3)
                    st.write("Scaling input data using trained Standard Scaler...")
                    scaled_input = clf_scaler.transform(edited_features)
                    time.sleep(0.4)
                    
                    st.write(f"Executing {selected_model}...")
                    model = clf_models[selected_model]
                    pred = model.predict(scaled_input)[0]
                    time.sleep(0.4)
                    
                    st.write("Generating Explainable AI (XAI) insights...")
                    # Generate a simple feature impact plot for XAI
                    feature_impact = None
                    if hasattr(model, "feature_importances_"):
                        # Tree based
                        feature_impact = edited_features.iloc[0] * model.feature_importances_
                    elif hasattr(model, "coef_"):
                        # Linear based
                        # Take the mean of absolute coefficients across all classes for simplicity
                        coefs = np.mean(np.abs(model.coef_), axis=0)
                        feature_impact = edited_features.iloc[0] * coefs
                    time.sleep(0.5)
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                # Result
                color = "#22c55e" if pred == actual_target else "#ef4444"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{selected_model} Prediction</div>
                    <div class="metric-value" style="color: {color};">{pred}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Class Probs
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(scaled_input)[0]
                    classes = model.classes_
                    prob_df = pd.DataFrame({'Fault Type': classes, 'Probability': probs})
                    fig, ax = plt.subplots(figsize=(10, 3))
                    sns.barplot(data=prob_df, x='Probability', y='Fault Type', ax=ax, palette='magma')
                    ax.set_title("Class Probabilities")
                    st.pyplot(fig)
                
                # XAI Plot
                if feature_impact is not None:
                    impact_df = pd.DataFrame({'Feature': feature_impact.index, 'Impact Contribution': feature_impact.values})
                    # Top 10 features
                    impact_df = impact_df.reindex(impact_df['Impact Contribution'].abs().sort_values(ascending=False).index).head(10)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=impact_df, x='Impact Contribution', y='Feature', ax=ax2, palette='coolwarm')
                    ax2.set_title("XAI: Top 10 Feature Contributions to this Prediction")
                    st.pyplot(fig2)
                else:
                    st.info("XAI Feature Contribution is not natively supported for this specific model architecture in this dashboard.")
    
    else:
        st.subheader("Model Analysis & Leaderboard")
        st.dataframe(clf_metrics.sort_values('Accuracy', ascending=False).style.background_gradient(cmap='Greens'), use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=clf_metrics.sort_values('Accuracy', ascending=False), y='Model', x='Accuracy', palette='viridis', ax=ax)
        ax.set_title("Accuracy Comparison")
        st.pyplot(fig)
