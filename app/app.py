import streamlit as st
import joblib
import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="PredictX | Multi-Domain Predictive Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        font-size: 0.98rem;
        color: #94a3b8;
        margin-bottom: 0px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    
    .metric-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Pill Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-primary { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-success { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    
    /* Info Box */
    .info-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 18px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #080c14;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'clustering')

# Dark theme for matplotlib/seaborn
def apply_dark_plot_style(ax, fig=None):
    if fig:
        fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.xaxis.label.set_color('#cbd5e1')
    ax.yaxis.label.set_color('#cbd5e1')
    ax.title.set_color('#f8fafc')
    for spine in ax.spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.1))
    ax.grid(True, linestyle='--', alpha=0.15, color='#ffffff')

@st.cache_resource
def load_all_artifacts():
    try:
        # 1. Regression Models & Artifacts
        reg_models = joblib.load(os.path.join(MODEL_DIR, 'all_reg_models.pkl'))
        reg_scaler = joblib.load(os.path.join(MODEL_DIR, 'reg_scaler.pkl'))
        reg_metrics = pd.read_csv(os.path.join(MODEL_DIR, 'reg_metrics.csv'))
        
        # 2. Classification Models & Artifacts (Review 1 + Review 2)
        clf_models = joblib.load(os.path.join(MODEL_DIR, 'all_clf_models.pkl'))
        clf_scaler = joblib.load(os.path.join(MODEL_DIR, 'clf_scaler.pkl'))
        clf_metrics = pd.read_csv(os.path.join(MODEL_DIR, 'clf_metrics.csv'))
        clf_features = joblib.load(os.path.join(MODEL_DIR, 'clf_features.pkl'))
        clf_sample = pd.read_csv(os.path.join(MODEL_DIR, 'clf_sample_data.csv'))
        
        # 3. Clustering Artifacts (Review 2 Newly Added Domain)
        clustering_metrics_path = os.path.join(MODEL_DIR, 'clustering_metrics.csv')
        clustering_metrics = pd.read_csv(clustering_metrics_path) if os.path.exists(clustering_metrics_path) else None
        
        clustering_summary_path = os.path.join(MODEL_DIR, 'clustering_summary.json')
        clustering_summary = None
        if os.path.exists(clustering_summary_path):
            with open(clustering_summary_path, 'r', encoding='utf-8') as f:
                clustering_summary = json.load(f)
                
        return {
            'reg_models': reg_models, 'reg_scaler': reg_scaler, 'reg_metrics': reg_metrics,
            'clf_models': clf_models, 'clf_scaler': clf_scaler, 'clf_metrics': clf_metrics,
            'clf_features': clf_features, 'clf_sample': clf_sample,
            'clustering_metrics': clustering_metrics, 'clustering_summary': clustering_summary
        }
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None

artifacts = load_all_artifacts()

if artifacts is None or artifacts['reg_models'] is None:
    st.error("Failed to load models. Please ensure the background training artifacts exist in `models/`.")
    st.stop()

reg_models = artifacts['reg_models']
reg_scaler = artifacts['reg_scaler']
reg_metrics = artifacts['reg_metrics']

clf_models = artifacts['clf_models']
clf_scaler = artifacts['clf_scaler']
clf_metrics = artifacts['clf_metrics']
clf_features = artifacts['clf_features']
clf_sample = artifacts['clf_sample']

clustering_metrics = artifacts['clustering_metrics']
clustering_summary = artifacts['clustering_summary']

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("""
<div style="padding: 10px 0 15px 0;">
    <h2 style="margin: 0; color: #38bdf8; font-size: 1.5rem; font-weight: 700;">⚡ PredictX</h2>
    <p style="margin: 2px 0 0 0; color: #64748b; font-size: 0.8rem; font-weight: 500;">Multi-Domain Predictive Intelligence</p>
</div>
""", unsafe_allow_html=True)

track = st.sidebar.radio(
    "Select Track Domain",
    [
        "Regression (Power Plant)",
        "Classification (Steel Faults)",
        "Clustering (Human Activity)"
    ],
    index=0
)

st.sidebar.markdown("---")

if track == "Clustering (Human Activity)":
    page = st.sidebar.radio(
        "Select Module Page",
        [
            "Cluster Analysis & Projections",
            "Clustering Leaderboard & Evaluation"
        ]
    )
else:
    page = st.sidebar.radio(
        "Select Module Page",
        [
            "Interactive Prediction",
            "Model Analysis & Leaderboard"
        ]
    )

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.78rem; color: #64748b; line-height: 1.5;">
    <p style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Review 2 Newly Added Models:</strong></p>
    <div><span class="badge badge-success">Clustering Track</span> K-Means, Agglomerative Hierarchical (Ward, Complete, Avg)</div>
    <div style="margin-top: 4px;"><span class="badge badge-purple">Classifiers</span> Random Forest, Bagging, Gradient Boosting, MLP, AdaBoost</div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# TRACK 1: REGRESSION (POWER PLANT)
# ==============================================================================
if track == "Regression (Power Plant)":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🏭 Combined Cycle Power Plant Output Prediction</div>
        <div class="hero-subtitle">UCI ID: 294 | Regression Track | Predicting Net Hourly Electrical Energy Output (PE in MW)</div>
    </div>
    """, unsafe_allow_html=True)
    
    if page == "Interactive Prediction":
        st.subheader("⚡ Real-time Operating Prediction")
        
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.markdown("### Plant Sensor Inputs")
            with st.form("reg_form"):
                at = st.number_input("Ambient Temperature (AT) [°C]", value=20.0, step=0.5, format="%.2f")
                v = st.number_input("Exhaust Vacuum (V) [cm Hg]", value=50.0, step=0.5, format="%.2f")
                ap = st.number_input("Ambient Pressure (AP) [mbar]", value=1013.0, step=1.0, format="%.2f")
                rh = st.number_input("Relative Humidity (RH) [%]", value=75.0, step=1.0, format="%.2f")
                
                model_names = list(reg_models.keys())
                rf_idx = model_names.index("Random Forest") if "Random Forest" in model_names else 0
                selected_model = st.selectbox("Select Regression Model", model_names, index=rf_idx)
                submit = st.form_submit_button("Predict Plant Output", use_container_width=True)
                
        with col2:
            st.markdown("### Energy Output & Cross-Model Comparison")
            if submit:
                with st.status("Executing prediction pipeline...", expanded=True) as status:
                    st.write("Extracting environmental conditions...")
                    time.sleep(0.15)
                    st.write("Computing engineered features (Temp-Humidity Index, AT-V Interaction)...")
                    
                    input_data = pd.DataFrame({'AT': [at], 'V': [v], 'AP': [ap], 'RH': [rh]})
                    input_data['Temp_Humidity_Index'] = input_data['AT'] * input_data['RH'] / 100.0
                    input_data['AT_V_interaction'] = input_data['AT'] * input_data['V']
                    features = ['AT', 'V', 'AP', 'RH', 'Temp_Humidity_Index', 'AT_V_interaction']
                    
                    st.write("Scaling input vector via Standard Scaler...")
                    scaled_input = reg_scaler.transform(input_data[features])
                    
                    st.write(f"Executing {selected_model} regressor...")
                    pred = reg_models[selected_model].predict(scaled_input)[0]
                    time.sleep(0.15)
                    status.update(label="Prediction Completed Successfully!", state="complete", expanded=False)
                
                col_res1, col_res2 = st.columns([1, 1])
                with col_res1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{selected_model} Output</div>
                        <div class="metric-value">{pred:.2f} <span style="font-size: 1.2rem; color: #94a3b8;">MW</span></div>
                        <div class="metric-subtitle">Net hourly electrical energy</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_res2:
                    rf_pred = reg_models['Random Forest'].predict(scaled_input)[0] if 'Random Forest' in reg_models else pred
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Benchmark (Random Forest)</div>
                        <div class="metric-value">{rf_pred:.2f} <span style="font-size: 1.2rem; color: #94a3b8;">MW</span></div>
                        <div class="metric-subtitle">R² = 0.995 Top Model</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                all_preds = {name: m.predict(scaled_input)[0] for name, m in reg_models.items()}
                pred_df = pd.DataFrame(list(all_preds.items()), columns=['Model', 'Prediction']).sort_values('Prediction')
                
                fig, ax = plt.subplots(figsize=(10, 4.5))
                apply_dark_plot_style(ax, fig)
                bars = ax.barh(pred_df['Model'], pred_df['Prediction'], color='#38bdf8', alpha=0.85, edgecolor=(1.0, 1.0, 1.0, 0.1))
                for bar in bars:
                    ax.text(bar.get_width() - 8, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f} MW",
                            va='center', ha='right', color='#ffffff', fontsize=8, fontweight='bold')
                ax.set_title("Cross-Model Predicted Energy Output (MW)", fontsize=11, fontweight='bold')
                ax.set_xlim(pred_df['Prediction'].min() - 15, pred_df['Prediction'].max() + 10)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("👈 Enter operating environmental conditions on the left and click **Predict Plant Output**.")
    
    else:
        st.subheader("📊 Regression Performance & Model Comparison")
        st.dataframe(
            reg_metrics.sort_values('R2 Score', ascending=False).style.format({
                'R2 Score': '{:.4f}',
                'RMSE': '{:.4f}',
                'MAE': '{:.4f}'
            }).background_gradient(cmap='Blues', subset=['R2 Score']),
            use_container_width=True
        )
        
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            apply_dark_plot_style(ax, fig)
            sorted_reg = reg_metrics.sort_values('R2 Score', ascending=True)
            ax.barh(sorted_reg['Model'], sorted_reg['R2 Score'], color='#38bdf8', alpha=0.85)
            ax.set_title("R² Score Comparison (Higher is Better)", fontsize=11, fontweight='bold')
            ax.set_xlim(0.85, 1.02)
            plt.tight_layout()
            st.pyplot(fig)
            
        with col2:
            rf = reg_models.get('Random Forest', None)
            if rf and hasattr(rf, 'feature_importances_'):
                features = ['AT', 'V', 'AP', 'RH', 'Temp_Index', 'AT_V']
                importances = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=True)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                apply_dark_plot_style(ax, fig)
                ax.barh(importances['Feature'], importances['Importance'], color='#818cf8', alpha=0.85)
                ax.set_title("Random Forest: Feature Importance", fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)


# ==============================================================================
# TRACK 2: CLASSIFICATION (STEEL PLATES FAULTS) - INCLUDES NEW REVIEW 2 MODELS
# ==============================================================================
elif track == "Classification (Steel Faults)":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🛡️ Steel Plates Faults Classification</div>
        <div class="hero-subtitle">UCI ID: 198 | Classification Track | 10 Models (Review 1 Baselines + Review 2 Ensembles & Neural Nets)</div>
    </div>
    """, unsafe_allow_html=True)
    
    if page == "Interactive Prediction":
        st.subheader("🔬 Fault Prediction & Explainable AI (XAI)")
        
        st.markdown("""
        <div class="info-card">
            <span class="badge badge-purple">Review 2 Upgraded</span>
            Now supporting all <strong>10 classification models</strong> including <strong>Random Forest, Bagging, Gradient Boosting, MLP Neural Network, and AdaBoost</strong> alongside original Review 1 baselines.
            Click <strong>Load Random Sample</strong> or double-click to edit any feature before predicting.
        </div>
        """, unsafe_allow_html=True)
        
        if 'sample_idx' not in st.session_state:
            st.session_state.sample_idx = 0
            
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🎲 Load Random Sample", use_container_width=True):
                st.session_state.sample_idx = int(np.random.randint(0, len(clf_sample)))
                st.rerun()
                
        sample = clf_sample.iloc[st.session_state.sample_idx]
        actual_target = sample['target']
        input_features = sample.drop('target').to_frame().T
        
        # Interactive Feature Editor
        edited_features = st.data_editor(input_features, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.markdown("### Model Selection")
            model_list = list(clf_models.keys())
            
            # Categorize models for nice UI
            st.caption("⚡ Review 2 Ensemble & Neural Models:")
            review_2_models = [m for m in model_list if m in ['Random Forest', 'Bagging Classifier', 'Gradient Boosting', 'MLP Classifier', 'AdaBoost']]
            st.caption("🛡️ Review 1 Baseline Models:")
            review_1_models = [m for m in model_list if m not in review_2_models]
            
            # Default to top performer: Random Forest
            default_idx = model_list.index('Random Forest') if 'Random Forest' in model_list else 0
            selected_model = st.selectbox("Select Classifier", model_list, index=default_idx)
            
            is_r2 = selected_model in review_2_models
            badge_html = '<span class="badge badge-purple">Review 2 Model</span>' if is_r2 else '<span class="badge badge-primary">Review 1 Baseline</span>'
            st.markdown(f"**Model Type:** {badge_html}", unsafe_allow_html=True)
            st.markdown(f"**Ground Truth Fault:** `{actual_target}`")
            
            submit = st.button("⚡ Predict & Explain", type="primary", use_container_width=True)
                
        with col2:
            st.markdown("### Prediction Insights")
            if submit:
                with st.status("Analyzing plate features...", expanded=True) as status:
                    st.write("Loading feature vector...")
                    time.sleep(0.15)
                    st.write("Applying trained Standard Scaler...")
                    scaled_input = clf_scaler.transform(edited_features)
                    
                    st.write(f"Executing {selected_model} inference...")
                    model = clf_models[selected_model]
                    pred = model.predict(scaled_input)[0]
                    time.sleep(0.15)
                    
                    st.write("Generating Explainable AI (XAI) feature importance...")
                    feature_impact = None
                    if hasattr(model, "feature_importances_"):
                        # Tree-based (Random Forest, Gradient Boosting, AdaBoost, Decision Tree)
                        feature_impact = edited_features.iloc[0] * model.feature_importances_
                    elif hasattr(model, "estimators_") and len(model.estimators_) > 0 and hasattr(model.estimators_[0], "feature_importances_"):
                        # Bagging Classifier
                        imp = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
                        feature_impact = edited_features.iloc[0] * imp
                    elif hasattr(model, "coef_"):
                        # Linear (Logistic Regression)
                        coefs = np.mean(np.abs(model.coef_), axis=0)
                        feature_impact = edited_features.iloc[0] * coefs
                    elif hasattr(model, "coefs_"):
                        # MLP Neural Net (Input layer weight magnitude)
                        w_input = np.mean(np.abs(model.coefs_[0]), axis=1)
                        feature_impact = edited_features.iloc[0] * w_input
                        
                    time.sleep(0.2)
                    status.update(label="Inference & XAI Complete!", state="complete", expanded=False)
                
                # Result Card
                is_correct = (pred == actual_target)
                color = "#22c55e" if is_correct else "#ef4444"
                status_text = "MATCHES GROUND TRUTH" if is_correct else "MISMATCH / SAMPLE OUTLIER"
                
                st.markdown(f"""
                <div class="metric-card" style="border-color: {color}44;">
                    <div class="metric-label">{selected_model} Classified Result</div>
                    <div class="metric-value" style="color: {color};">{pred}</div>
                    <div class="metric-subtitle" style="color: {color}; font-weight: 600;">● {status_text} (Actual: {actual_target})</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Class Probabilities (if supported)
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(scaled_input)[0]
                        classes = model.classes_
                        prob_df = pd.DataFrame({'Fault Type': classes, 'Probability': probs}).sort_values('Probability', ascending=True)
                        fig, ax = plt.subplots(figsize=(10, 3.2))
                        apply_dark_plot_style(ax, fig)
                        bars = ax.barh(prob_df['Fault Type'], prob_df['Probability'], color='#818cf8', alpha=0.85)
                        for bar in bars:
                            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width()*100:.1f}%",
                                    va='center', ha='left', color='#e2e8f0', fontsize=8)
                        ax.set_title(f"Class Probability Distribution ({selected_model})", fontsize=10, fontweight='bold')
                        ax.set_xlim(0, 1.08)
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception:
                        pass
                
                # XAI Plot
                if feature_impact is not None:
                    impact_df = pd.DataFrame({'Feature': feature_impact.index, 'Impact Contribution': feature_impact.values})
                    impact_df = impact_df.reindex(impact_df['Impact Contribution'].abs().sort_values(ascending=False).index).head(10).sort_values('Impact Contribution', ascending=True)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 3.8))
                    apply_dark_plot_style(ax2, fig2)
                    colors = ['#22c55e' if v >= 0 else '#ef4444' for v in impact_df['Impact Contribution']]
                    ax2.barh(impact_df['Feature'], impact_df['Impact Contribution'], color=colors, alpha=0.85)
                    ax2.set_title("XAI: Top 10 Feature Contributions for this Plate Specimen", fontsize=10, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig2)
                else:
                    st.info("Feature importance explanation is not supported for this specific model architecture.")
            else:
                st.info("👈 Select a model and click **Predict & Explain** to run inference.")
    
    else:
        st.subheader("📊 Complete 10-Model Classification Leaderboard")
        
        # Add Generation Column
        df_display = clf_metrics.copy()
        r2_names = ['Random Forest', 'Bagging Classifier', 'Gradient Boosting', 'MLP Classifier', 'AdaBoost']
        df_display['Generation'] = df_display['Model'].apply(lambda x: '⚡ Review 2 (Ensemble/Deep)' if x in r2_names else 'Review 1 (Baseline)')
        
        st.dataframe(
            df_display.sort_values('Accuracy', ascending=False).style.format({
                'Accuracy': '{:.4f}',
                'Weighted F1': '{:.4f}'
            }).background_gradient(cmap='Greens', subset=['Accuracy', 'Weighted F1']),
            use_container_width=True
        )
        
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5.2))
            apply_dark_plot_style(ax, fig)
            sorted_clf = df_display.sort_values('Accuracy', ascending=True)
            colors = ['#c084fc' if 'Review 2' in gen else '#38bdf8' for gen in sorted_clf['Generation']]
            bars = ax.barh(sorted_clf['Model'], sorted_clf['Accuracy'], color=colors, alpha=0.85)
            for bar in bars:
                ax.text(bar.get_width() - 0.08, bar.get_y() + bar.get_height()/2, f"{bar.get_width()*100:.1f}%",
                        va='center', ha='right', color='#ffffff', fontsize=8, fontweight='bold')
            ax.set_title("Model Accuracy Across All 10 Classifiers", fontsize=11, fontweight='bold')
            ax.set_xlim(0.4, 0.9)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("🟣 Purple = Review 2 Newly Added Models | 🔵 Blue = Review 1 Baseline Models")
            
        with col2:
            st.markdown("### 🚀 Review 2 Uplift Analysis")
            best_r1 = df_display[df_display['Generation'].str.contains('Review 1')]['Accuracy'].max()
            best_r2 = df_display[df_display['Generation'].str.contains('Review 2')]['Accuracy'].max()
            uplift = (best_r2 - best_r1) * 100
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Review 2 Accuracy Boost</div>
                <div class="metric-value" style="color: #22c55e;">+{uplift:.2f}%</div>
                <div class="metric-subtitle">Best R1: {best_r1*100:.1f}% ➔ Best R2: {best_r2*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card" style="font-size: 0.85rem; line-height: 1.6;">
                <strong>Key Observations:</strong><br>
                • <strong>Random Forest & Bagging</strong> achieved top-tier performance (>80% accuracy, >0.80 F1-score), substantially outperforming single decision trees.<br>
                • <strong>Gradient Boosting & MLP</strong> show strong generalization across complex minority fault categories (e.g., Stains, Dirtiness).<br>
                • Ensembles effectively mitigated high variance in high-dimensional tabular plate defect features.
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TRACK 3: CLUSTERING (HUMAN ACTIVITY RECOGNITION) - NEW REVIEW 2 TRACK
# ==============================================================================
elif track == "Clustering (Human Activity)":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🏃 Human Activity Kinematic Profiling</div>
        <div class="hero-subtitle">UCI ID: 240 | Clustering Track | Unsupervised Analysis on Smartphone Sensor Signals (10,299 instances, 561 features)</div>
    </div>
    """, unsafe_allow_html=True)
    
    if page == "Cluster Analysis & Projections":
        st.subheader("🌐 Unsupervised Cluster Explorer & 2D Manifolds")
        
        # Summary Overview Cards
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Dataset Instances</div>
                <div class="metric-value">10,299</div>
                <div class="metric-subtitle">Train + Test combined</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Kinematic Features</div>
                <div class="metric-value">561</div>
                <div class="metric-subtitle">Sensor signals & FFT</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Target Clusters (K)</div>
                <div class="metric-value">6</div>
                <div class="metric-subtitle">Identified via Elbow Method</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">PCA Variance Explained</div>
                <div class="metric-value" style="color: #a855f7;">56.98%</div>
                <div class="metric-subtitle">First 2 principal components</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        tabs = st.tabs([
            "📍 2D Manifold Projections (PCA & t-SNE)",
            "📊 Cluster Balance & Sizes",
            "📉 Elbow Curve & Optimal K",
            "🌲 Hierarchical Dendrograms",
            "🏃 Ground Truth Activity Profile"
        ])
        
        # TAB 1: 2D PROJECTIONS
        with tabs[0]:
            st.markdown("### 2D Dimensionality Reduction & Cluster Manifolds")
            st.markdown("""
            High-dimensional kinematic sensor spaces (561 dimensions) are projected into 2D manifolds using **PCA (Linear Orthogonal Projection)** 
            and **t-SNE (Non-linear Probabilistic Manifold Embedding)** to inspect cluster boundaries.
            """)
            
            proj_choice = st.radio("Select 2D Projection Method", ["PCA Projections Across All 4 Models", "t-SNE Manifold (K-Means Clusters)"], horizontal=True)
            
            if proj_choice == "PCA Projections Across All 4 Models":
                pca_img = os.path.join(ASSET_DIR, 'pca_clusters.png')
                if os.path.exists(pca_img):
                    st.image(pca_img, caption="Figure: 2D Principal Component Analysis (PCA) across K-Means and Agglomerative Hierarchical (Ward, Complete, Average) Linkages (56.98% variance explained)", use_container_width=True)
                else:
                    st.info("PCA image generated in clustering notebook.")
            else:
                tsne_img = os.path.join(ASSET_DIR, 'tsne_clusters.png')
                if os.path.exists(tsne_img):
                    st.image(tsne_img, caption="Figure: 2D t-SNE Embedding of K-Means Clusters revealing distinct kinematic sub-clusters", use_container_width=True)
                else:
                    st.info("t-SNE image generated in clustering notebook.")
                    
        # TAB 2: CLUSTER BALANCE & SIZES
        with tabs[1]:
            st.markdown("### Cluster Size Distributions Across Algorithms")
            
            cluster_sizes = clustering_summary.get('cluster_sizes', {}) if clustering_summary else {
                'K-Means': {0: 2638, 1: 307, 2: 1920, 3: 2449, 4: 1338, 5: 1647},
                'Agglomerative - Ward': {0: 3208, 1: 1313, 2: 1206, 3: 1834, 4: 151, 5: 2587},
                'Agglomerative - Complete': {0: 5626, 1: 218, 2: 2, 3: 12, 4: 4439, 5: 2},
                'Agglomerative - Average': {0: 138, 1: 10141, 2: 15, 3: 2, 4: 2, 5: 1}
            }
            
            sel_algo = st.selectbox("Select Clustering Algorithm for Size Breakdown", list(cluster_sizes.keys()))
            sizes = cluster_sizes[sel_algo]
            sizes_df = pd.DataFrame(list(sizes.items()), columns=['Cluster ID', 'Observation Count'])
            
            col_t1, col_t2 = st.columns([1, 1], gap="medium")
            with col_t1:
                st.dataframe(sizes_df, use_container_width=True, hide_index=True)
                
            with col_t2:
                fig, ax = plt.subplots(figsize=(8, 4))
                apply_dark_plot_style(ax, fig)
                ax.bar(sizes_df['Cluster ID'].astype(str), sizes_df['Observation Count'], color='#38bdf8', alpha=0.85)
                ax.set_title(f"Cluster Observation Count: {sel_algo}", fontsize=11, fontweight='bold')
                ax.set_xlabel("Cluster Index")
                ax.set_ylabel("Number of Observations")
                plt.tight_layout()
                st.pyplot(fig)
                
            st.markdown("""
            <div class="info-card">
                <strong>Clustering Behavior Insight:</strong><br>
                • <strong>K-Means & Agglomerative Ward:</strong> Produce well-balanced clusters (150 – 3,200 points per cluster), aligning closely with physical activity distributions.<br>
                • <strong>Complete & Average Linkages:</strong> Group the vast majority of observations into one or two large clusters, treating rare movement transitions as tiny micro-clusters.
            </div>
            """, unsafe_allow_html=True)

        # TAB 3: ELBOW METHOD
        with tabs[2]:
            st.markdown("### Optimal K Determination via Elbow Method")
            st.markdown("""
            The Elbow Method computes the Within-Cluster Sum of Squares (Inertia) for $K \\in [2, 10]$.
            An inflection point ("elbow") is observed, validating **K = 6** as the optimal configuration corresponding to the 6 physical activities.
            """)
            elbow_img = os.path.join(ASSET_DIR, 'elbow_method.png')
            if os.path.exists(elbow_img):
                st.image(elbow_img, caption="Figure: K-Means Elbow Curve (Inertia vs Number of Clusters K)", use_container_width=True)
                
        # TAB 4: DENDROGRAMS
        with tabs[3]:
            st.markdown("### Hierarchical Clustering Dendrograms")
            st.markdown("""
            Dendrograms visualize the hierarchical tree synthesis under **Ward**, **Complete**, and **Average** linkage strategies.
            """)
            dendro_img = os.path.join(ASSET_DIR, 'dendrograms.png')
            if os.path.exists(dendro_img):
                st.image(dendro_img, caption="Figure: Hierarchical Clustering Dendrograms (Ward Linkage, Complete Linkage, Average Linkage)", use_container_width=True)
                
        # TAB 5: GROUND TRUTH
        with tabs[4]:
            st.markdown("### Ground Truth Activity Reference Profile")
            st.markdown("""
            In unsupervised clustering, actual activity labels are withheld during training and utilized strictly for **post-hoc kinematic validation**.
            """)
            act_img = os.path.join(ASSET_DIR, 'activity_distribution.png')
            if os.path.exists(act_img):
                st.image(act_img, caption="Figure: Ground Truth Activity Distribution across 10,299 observations in UCI HAR Dataset", use_container_width=True)
                
    else:
        # CLUSTERING LEADERBOARD & EVALUATION
        st.subheader("🏆 Clustering Performance Comparison Leaderboard")
        
        # Best Performer Metric Cards
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Best Silhouette Score</div>
                <div class="metric-value" style="color: #22c55e;">0.4287</div>
                <div class="metric-subtitle">Agglomerative - Average (Higher is better)</div>
            </div>
            """, unsafe_allow_html=True)
        with bcol2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Best Davies-Bouldin Index</div>
                <div class="metric-value" style="color: #38bdf8;">1.1020</div>
                <div class="metric-subtitle">Agglomerative - Average (Lower is better)</div>
            </div>
            """, unsafe_allow_html=True)
        with bcol3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Best Calinski-Harabasz Index</div>
                <div class="metric-value" style="color: #f59e0b;">2556.54</div>
                <div class="metric-subtitle">K-Means (Higher is better)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Performance Leaderboard Table
        st.markdown("### Unsupervised Evaluation Metrics")
        if clustering_metrics is not None:
            st.dataframe(
                clustering_metrics.style.format({
                    'Silhouette Score': '{:.4f}',
                    'Davies-Bouldin Index': '{:.4f}',
                    'Calinski-Harabasz Index': '{:.2f}'
                }).highlight_max(subset=['Silhouette Score', 'Calinski-Harabasz Index'], color='rgba(34, 197, 94, 0.3)')
                  .highlight_min(subset=['Davies-Bouldin Index'], color='rgba(56, 189, 248, 0.3)'),
                use_container_width=True
            )
            
        # Metric Charts
        col_c1, col_c2, col_c3 = st.columns(3, gap="medium")
        
        with col_c1:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            apply_dark_plot_style(ax, fig)
            ax.barh(clustering_metrics['Algorithm'], clustering_metrics['Silhouette Score'], color='#22c55e', alpha=0.85)
            ax.set_title("Silhouette Score (Higher is Better)", fontsize=9, fontweight='bold')
            ax.set_xlim(0, 0.5)
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_c2:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            apply_dark_plot_style(ax, fig)
            ax.barh(clustering_metrics['Algorithm'], clustering_metrics['Davies-Bouldin Index'], color='#38bdf8', alpha=0.85)
            ax.set_title("Davies-Bouldin Index (Lower is Better)", fontsize=9, fontweight='bold')
            ax.set_xlim(0, 3.0)
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_c3:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            apply_dark_plot_style(ax, fig)
            ax.barh(clustering_metrics['Algorithm'], clustering_metrics['Calinski-Harabasz Index'], color='#f59e0b', alpha=0.85)
            ax.set_title("Calinski-Harabasz Index (Higher is Better)", fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
        # Deep Evaluation Insights
        st.markdown("""
        <div class="info-card" style="margin-top: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #f8fafc; font-size: 1rem;">🔬 Comparative Clustering Insights & Mathematical Trade-Offs</h4>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                1. <strong>Spherical vs Manifold Clusters:</strong> 
                <strong>K-Means</strong> optimizes the Calinski-Harabasz Index (2556.54) because it explicitly maximizes between-cluster variance relative to within-cluster variance under Euclidean assumptions. It creates well-partitioned clusters across high-dimensional sensor data.<br><br>
                2. <strong>Average Linkage Phenomenon:</strong> 
                <strong>Agglomerative - Average</strong> achieves the highest Silhouette Score (0.4287) and lowest Davies-Bouldin Index (1.1020). However, this is largely driven by forming one massive cluster alongside isolated peripheral points, creating artificially high pairwise silhouette separation.<br><br>
                3. <strong>Ward Linkage Practical Balance:</strong> 
                <strong>Agglomerative - Ward</strong> minimizes total within-cluster variance at each merge step, producing the most balanced cluster sizes that accurately mirror physical activity divisions between dynamic movements (Walking, Walking Upstairs, Walking Downstairs) and static postures (Sitting, Standing, Laying).
            </div>
        </div>
        """, unsafe_allow_html=True)
