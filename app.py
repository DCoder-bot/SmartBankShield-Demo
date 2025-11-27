
import streamlit as st
import pandas as pd
import pickle
import os

# ========== SETUP ==========
st.set_page_config(page_title="SmartBankShield Dashboard", layout="wide")
st.title("💳 SmartBankShield: ML-Based Fraud Detection & Insights")

# Base paths (relative for GitHub/Streamlit Cloud)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = BASE_PATH
ROC_PATH = os.path.join(BASE_PATH, "roc")
SMOTE_PATH = os.path.join(BASE_PATH, "methdology 1a and 1b")

# ========== LOAD MODELS ==========
def load_model(model_name):
    import joblib
    file_path = os.path.join(MODEL_PATH, model_name)
    try:
        return pickle.load(open(file_path, "rb"))
    except:
        try:
            return joblib.load(file_path)
        except Exception as e:
            st.error(f"❌ Failed to load {model_name}: {e}")
            return None

models = {
    "Logistic Regression": load_model("lr_model_v2_recalculated.pkl"),
    "Random Forest": load_model("rf_model_v2_recalculated.pkl"),
    "XGBoost": load_model("xgb_model_v2_recalculated.pkl"),
}

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Navigation")
option = st.sidebar.selectbox(
    "Select Section",
    (
        "🏠 Dashboard Overview",
        "📊 Model Performance Comparison",
        "📈 SMOTE Impact Analysis",
        "🔍 Feature Importance (XGBoost)",
        "🧠 Explainability (SHAP Summary)",
        "🧮 Try Custom Predictions",
    ),
)

# ========== 1. DASHBOARD OVERVIEW ==========
if option == "🏠 Dashboard Overview":
    st.subheader("📊 Project Overview")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        ### SmartBankShield
        An ML-based framework designed to detect fraudulent banking transactions with **explainability**.

        **Key Features:**
        - ✅ Three classification models (Logistic Regression, Random Forest, XGBoost)
        - ✅ SMOTE-based class imbalance handling
        - ✅ SHAP-based explainable AI (XAI)
        - ✅ Interactive dashboard for real-time predictions
        - ✅ 99.95% accuracy, 82.7% recall, 0.974 AUC-ROC

        **Use Case:** Financial institutions detecting fraudulent credit card transactions in real-time.
        """
        )

    with col2:
        st.metric("Model Accuracy", "99.95%", "XGBoost")
        st.metric("AUC-ROC Score", "0.974", "Excellent")
        st.metric("Detection Recall", "82.7%", "High")

# ========== 2. MODEL PERFORMANCE ==========
elif option == "📊 Model Performance Comparison":
    st.subheader("📈 Model Performance Comparison")

    st.markdown("### Performance Metrics Table")
    table_path = os.path.join(BASE_PATH, "model_performance_table.png")
    if os.path.exists(table_path):
        st.image(
            table_path,
            caption="Table 1: Comparative Performance Metrics",
            use_container_width=True,
        )
    else:
        st.warning("⚠️ Performance table image not found. Please generate it first.")

    st.markdown("### ROC Curves Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        roc_lr_path = os.path.join(ROC_PATH, "roc_logistic.png")
        if os.path.exists(roc_lr_path):
            st.image(
                roc_lr_path,
                caption="ROC - Logistic Regression (AUC: 0.972)",
                use_container_width=True,
            )

    with col2:
        roc_rf_path = os.path.join(ROC_PATH, "roc_randomforest.png")
        if os.path.exists(roc_rf_path):
            st.image(
                roc_rf_path,
                caption="ROC - Random Forest (AUC: 0.986)",
                use_container_width=True,
            )

    with col3:
        roc_xgb_path = os.path.join(ROC_PATH, "roc_xgboost.png")
        if os.path.exists(roc_xgb_path):
            st.image(
                roc_xgb_path,
                caption="ROC - XGBoost (AUC: 0.974)",
                use_container_width=True,
            )

# ========== 3. SMOTE IMPACT ==========
elif option == "📈 SMOTE Impact Analysis":
    st.subheader("📊 Class Imbalance: Before vs After SMOTE")
    st.markdown(
        """
    SMOTE (Synthetic Minority Oversampling Technique) addresses severe class imbalance
    by generating synthetic fraudulent transaction samples.
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        fig1a_path = os.path.join(SMOTE_PATH, "figure_1a_before_smote.png")
        if os.path.exists(fig1a_path):
            st.image(
                fig1a_path,
                caption="Figure 1A: Before SMOTE (Imbalanced)",
                use_container_width=True,
            )
        else:
            st.warning("⚠️ Figure 1A not found at: " + fig1a_path)

    with col2:
        fig1b_path = os.path.join(SMOTE_PATH, "figure_1b_after_smote.png")
        if os.path.exists(fig1b_path):
            st.image(
                fig1b_path,
                caption="Figure 1B: After SMOTE (Balanced)",
                use_container_width=True,
            )
        else:
            st.warning("⚠️ Figure 1B not found at: " + fig1b_path)

    st.info(
        "✅ SMOTE transformed imbalanced data (1:578 ratio) to balanced (1:1), enabling better model training."
    )

# ========== 4. FEATURE IMPORTANCE ==========
elif option == "🔍 Feature Importance (XGBoost)":
    st.subheader("🔍 XGBoost Feature Importance")
    st.markdown(
        """
    These are the most influential features for fraud detection. Variables V14 and V17
    are the top predictors, followed by transaction Amount.
    """
    )

    fig_path = os.path.join(BASE_PATH, "xgb_feature_importance.png")
    if os.path.exists(fig_path):
        st.image(
            fig_path,
            caption="Figure 3: XGBoost Feature Importance Ranking",
            use_container_width=True,
        )
    else:
        st.warning("⚠️ Feature importance image not found")

# ========== 5. EXPLAINABILITY ==========
elif option == "🧠 Explainability (SHAP Summary)":
    st.subheader("🧠 Explainable AI - SHAP Summary Plot")
    st.markdown(
        """
    SHAP (SHapley Additive exPlanations) provides interpretable explanations for each prediction.
    - **Red colors**: Features pushing prediction towards fraud
    - **Blue colors**: Features pushing prediction towards legitimate
    """
    )

    shap_path = os.path.join(BASE_PATH, "shap_summary_plot.png")
    if os.path.exists(shap_path):
        st.image(
            shap_path, caption="Figure 4: SHAP Summary Plot", use_container_width=True
        )
    else:
        st.warning("⚠️ SHAP summary plot not found")

# ========== 6. CUSTOM PREDICTION ==========
elif option == "🧮 Try Custom Predictions":
    st.subheader("🧮 Real-Time Fraud Prediction")
    st.markdown(
        "Upload a transaction dataset or enter transaction details to predict fraud probability."
    )

    model_choice = st.selectbox("Select Model", list(models.keys()))
    model = models[model_choice]
    st.divider()

    uploaded = st.file_uploader("Upload transaction dataset (CSV format)", type=["csv"])

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            st.write("**Preview of uploaded **")
            st.dataframe(data.head())

            X_pred = data.drop("Class", axis=1, errors="ignore")

            if model is None:
                st.error("❌ Model not loaded. Please check your model files.")
            else:
                if hasattr(model, "predict"):
                    if st.button("🔍 Predict Fraudulent Transactions"):
                        try:
                            preds = model.predict(X_pred)
                            if hasattr(model, "predict_proba"):
                                probs = model.predict_proba(X_pred)[:, 1]
                                data["Fraud_Probability"] = probs
                            data["Predicted_Fraud"] = preds

                            st.success("✅ Prediction complete!")
                            st.dataframe(data.head(10))

                            fraud_count = (preds == 1).sum()
                            fraud_pct = (fraud_count / len(preds)) * 100

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Transactions", len(preds))
                            with col2:
                                st.metric("Flagged as Fraud", fraud_count)
                            with col3:
                                st.metric("Fraud %", f"{fraud_pct:.2f}%")

                            csv = data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv,
                                file_name="fraud_predictions.csv",
                                mime="text/csv",
                            )
                        except Exception as e:
                            st.error(f"❌ Prediction error: {e}")
                else:
                    st.error(
                        "❌ The loaded object is not a model. Please check the file and load a valid model (.pkl)."
                    )
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    else:
        st.info("📤 Please upload a CSV file to start predictions.")

# ========== FOOTER ==========
st.markdown("---")
st.markdown(
    """
**SmartBankShield v1.0** | Built with Streamlit, XGBoost, and SHAP  
For research and educational purposes.
"""
)
