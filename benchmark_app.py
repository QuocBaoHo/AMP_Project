import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import joblib
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_recall_fscore_support
import warnings
import sys
import plotly.graph_objects as go

warnings.filterwarnings('ignore') # Suppress warnings
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# Cấu hình giao diện Streamlit
# ==========================================
st.set_page_config(page_title="Benchmark Dashboard", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .stApp {
        background-color: #0F1A15;
    }
    h1 { font-size: 46px !important; color: #E6F4F1 !important; }
    h2 { font-size: 36px !important; color: #E6F4F1 !important; }
    h3 { font-size: 28px !important; color: #E6F4F1 !important; }
    h4 { font-size: 22px !important; color: #E6F4F1 !important; }
    p, div, span, label {
        font-size: 18px !important;
        color: #B8D8CB !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 AMP & Toxicity Benchmark Dashboard")
st.markdown("Run the evaluation on the full **Test set** and visualize the performance metrics for both AMP and Toxicity prediction.")

@st.cache_resource(show_spinner=False)
def load_models():
    mcnn_model = tf.keras.models.load_model('mcnn_model.keras')
    cnn_model = tf.keras.models.load_model('cnn_model.keras')
    rf_model = joblib.load('rf_model.pkl')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Thử load vectorizer cho RF (nếu có)
    try:
        vectorizer = joblib.load('vectorizer.pkl')
    except FileNotFoundError:
        vectorizer = None
        
    esm2_tokenizer = AutoTokenizer.from_pretrained('./esm2_multitask_final')
    esm2_model = AutoModelForSequenceClassification.from_pretrained('./esm2_multitask_final')
    esm2_model.eval()
        
    return mcnn_model, cnn_model, rf_model, tokenizer, vectorizer, esm2_tokenizer, esm2_model

@st.cache_data(show_spinner=False)
def get_dataset():
    df = pd.read_csv('test_multitask.csv')
    return df['sequence'].tolist(), df['is_amp'].values, df['is_toxic'].values

if st.button("🔥 Run Benchmark", type="primary"):
    with st.status("Running Benchmark Pipeline...", expanded=True) as status:
        st.write("1. Initializing models (including ESM-2 Multitask)...")
        mcnn_model, cnn_model, rf_model, tokenizer, vectorizer, esm2_tokenizer, esm2_model = load_models()
        
        st.write("2. Fetching test set (test_multitask.csv)...")
        sequences, y_true_amp, y_true_tox = get_dataset()
        st.write(f"👉 Total samples in test set: {len(sequences)}")
        
        st.write("3. Preprocessing data...")
        seq_seq = tokenizer.texts_to_sequences(sequences)
        X_padded = keras.utils.pad_sequences(seq_seq, maxlen=50, padding='post')
        
        st.write("4. Evaluating models...")
        st.write(" -> Evaluating mCNN (Proposed)...")
        prob_mcnn = mcnn_model.predict(X_padded, batch_size=256, verbose=0)
        y_pred_mcnn = (prob_mcnn > 0.5).astype(int).flatten()
        
        st.write(" -> Evaluating 1D CNN...")
        prob_cnn = cnn_model.predict(X_padded, batch_size=256, verbose=0)
        y_pred_cnn = (prob_cnn > 0.5).astype(int).flatten()
        
        st.write(" -> Evaluating Random Forest...")
        if vectorizer is not None:
            try:
                X_rf = vectorizer.transform(sequences)
                y_pred_rf = rf_model.predict(X_rf)
                prob_rf = rf_model.predict_proba(X_rf)[:, 1]
                rf_error = False
            except Exception as e:
                st.error(f"   [ERROR] RF Prediction failed: {e}")
                rf_error = True
                y_pred_rf = np.zeros_like(y_true_amp)
                prob_rf = np.zeros_like(y_true_amp, dtype=float)
        else:
            st.error("   [WARNING] Missing 'vectorizer.pkl'! Random Forest will blindly guess Non-AMP (0).")
            rf_error = True
            y_pred_rf = np.zeros_like(y_true_amp)
            prob_rf = np.zeros_like(y_true_amp, dtype=float)

        st.write(" -> Evaluating ESM-2 Multitask (AMP & Toxicity)...")
        batch_size = 128
        all_amp_probs = []
        all_tox_probs = []
        
        progress_bar = st.progress(0, text="ESM-2 Inference: 0%")
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            inputs = esm2_tokenizer(batch_seqs, return_tensors='pt', padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                logits = esm2_model(**inputs).logits
                probs = torch.sigmoid(logits)
            all_amp_probs.extend(probs[:, 0].numpy())
            all_tox_probs.extend(probs[:, 1].numpy())
            progress_bar.progress(min((i + batch_size) / len(sequences), 1.0), text=f"ESM-2 Inference: {min(i+batch_size, len(sequences))}/{len(sequences)}")
            
        progress_bar.empty()
        prob_esm2_amp = np.array(all_amp_probs)
        y_pred_esm2_amp = (prob_esm2_amp > 0.5).astype(int)
        
        prob_esm2_tox = np.array(all_tox_probs)
        
        status.update(label="Benchmark Completed!", state="complete", expanded=False)

    y_true = y_true_amp
    
    # Toxicity evaluation subset
    valid_tox_indices = y_true_tox != -1
    y_true_tox_valid = y_true_tox[valid_tox_indices]
    prob_esm2_tox_valid = prob_esm2_tox[valid_tox_indices]
    y_pred_esm2_tox_valid = (prob_esm2_tox_valid > 0.5).astype(int)

    # Calculate accuracy AMP
    acc_rf = accuracy_score(y_true, y_pred_rf) * 100
    acc_cnn = accuracy_score(y_true, y_pred_cnn) * 100
    acc_mcnn = accuracy_score(y_true, y_pred_mcnn) * 100
    acc_esm2_amp = accuracy_score(y_true, y_pred_esm2_amp) * 100

    # Calculate ROC AUC AMP
    fpr_mcnn, tpr_mcnn, _ = roc_curve(y_true, prob_mcnn)
    roc_auc_mcnn = auc(fpr_mcnn, tpr_mcnn)
    
    fpr_cnn, tpr_cnn, _ = roc_curve(y_true, prob_cnn)
    roc_auc_cnn = auc(fpr_cnn, tpr_cnn)

    fpr_rf, tpr_rf, _ = roc_curve(y_true, prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    fpr_esm2, tpr_esm2, _ = roc_curve(y_true, prob_esm2_amp)
    roc_auc_esm2 = auc(fpr_esm2, tpr_esm2)

    metrics_labels = ['Precision', 'Recall', 'F1-Score']

    st.markdown("---")
    
    # ==========================
    # 1. ACCURACY COMPARISON
    # ==========================
    st.subheader("🏆 Accuracy Comparison (AMP)")
    models = ['Random Forest', '1D CNN', 'mCNN (Proposed)', 'ESM-2 (Multitask)']
    accs = [acc_rf, acc_cnn, acc_mcnn, acc_esm2_amp]
    colors = ['#FF6B6B', '#4A90E2', '#4CAF50', '#9C27B0']
    
    fig_acc = go.Figure(data=[
        go.Bar(x=models, y=accs, marker_color=colors, text=[f"{val:.2f}%" for val in accs], textposition='auto', textfont=dict(size=24, color='white', family='Arial Black'))
    ])
    fig_acc.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E6F4F1',
        margin=dict(t=20, b=20, l=0, r=0), height=400,
        xaxis=dict(tickfont=dict(size=16, color='white')),
        yaxis=dict(title=dict(text='Accuracy (%)', font=dict(size=18, color='white')), range=[0, 100], gridcolor='#2D5A27', tickfont=dict(size=14, color='white'))
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")

    # ==========================
    # 2. mCNN CLASS-WISE METRICS
    # ==========================
    st.subheader("👑 Proposed Model Performance: mCNN")
    p_mcnn, r_mcnn, f_mcnn, _ = precision_recall_fscore_support(y_true, y_pred_mcnn)
    
    fig_mcnn = go.Figure(data=[
        go.Bar(name='Non-AMP (0)', x=metrics_labels, y=[p_mcnn[0], r_mcnn[0], f_mcnn[0]], marker_color='#A0B0C0', text=[f"{val:.2f}" for val in [p_mcnn[0], r_mcnn[0], f_mcnn[0]]], textposition='auto', textfont=dict(size=24, color='white', family='Arial Black')),
        go.Bar(name='AMP (1)', x=metrics_labels, y=[p_mcnn[1], r_mcnn[1], f_mcnn[1]], marker_color='#4CAF50', text=[f"{val:.2f}" for val in [p_mcnn[1], r_mcnn[1], f_mcnn[1]]], textposition='auto', textfont=dict(size=24, color='white', family='Arial Black'))
    ])
    fig_mcnn.update_layout(
        barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E6F4F1',
        margin=dict(t=20, b=20, l=0, r=0), height=450,
        xaxis=dict(tickfont=dict(size=18, color='white')),
        yaxis=dict(title=dict(text='Score', font=dict(size=18, color='white')), range=[0, 1.1], gridcolor='#2D5A27', tickfont=dict(size=14, color='white')),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=20, color="white"))
    )
    st.plotly_chart(fig_mcnn, use_container_width=True)

    st.markdown("---")

    # ==========================
    # 3. BASELINES: 1D CNN, RF, ESM-2
    # ==========================
    st.subheader("🛠️ Baseline Models Performance (AMP)")
    col_base1, col_base2, col_base3 = st.columns(3)
    
    with col_base1:
        st.markdown("**1D CNN Metrics**")
        p_cnn, r_cnn, f_cnn, _ = precision_recall_fscore_support(y_true, y_pred_cnn)
        fig_cnn = go.Figure(data=[
            go.Bar(name='Non-AMP (0)', x=metrics_labels, y=[p_cnn[0], r_cnn[0], f_cnn[0]], marker_color='#A0B0C0', text=[f"{val:.2f}" for val in [p_cnn[0], r_cnn[0], f_cnn[0]]], textposition='auto', textfont=dict(size=16, color='white', family='Arial Black')),
            go.Bar(name='AMP (1)', x=metrics_labels, y=[p_cnn[1], r_cnn[1], f_cnn[1]], marker_color='#4A90E2', text=[f"{val:.2f}" for val in [p_cnn[1], r_cnn[1], f_cnn[1]]], textposition='auto', textfont=dict(size=16, color='white', family='Arial Black'))
        ])
        fig_cnn.update_layout(
            barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E6F4F1',
            margin=dict(t=20, b=20, l=0, r=0), height=350,
            xaxis=dict(tickfont=dict(size=12, color='white')),
            yaxis=dict(title=dict(text='Score', font=dict(size=14, color='white')), range=[0, 1.1], gridcolor='#2D5A27', tickfont=dict(size=10, color='white')),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color="white"))
        )
        st.plotly_chart(fig_cnn, use_container_width=True)

    with col_base2:
        st.markdown("**Random Forest Metrics**")
        p_rf, r_rf, f_rf, _ = precision_recall_fscore_support(y_true, y_pred_rf)
        fig_rf = go.Figure(data=[
            go.Bar(name='Non-AMP (0)', x=metrics_labels, y=[p_rf[0], r_rf[0], f_rf[0]], marker_color='#A0B0C0', text=[f"{val:.2f}" for val in [p_rf[0], r_rf[0], f_rf[0]]], textposition='auto', textfont=dict(size=16, color='white', family='Arial Black')),
            go.Bar(name='AMP (1)', x=metrics_labels, y=[p_rf[1], r_rf[1], f_rf[1]], marker_color='#FF6B6B', text=[f"{val:.2f}" for val in [p_rf[1], r_rf[1], f_rf[1]]], textposition='auto', textfont=dict(size=16, color='white', family='Arial Black'))
        ])
        fig_rf.update_layout(
            barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E6F4F1',
            margin=dict(t=20, b=20, l=0, r=0), height=350,
            xaxis=dict(tickfont=dict(size=12, color='white')),
            yaxis=dict(title=dict(text='Score', font=dict(size=14, color='white')), range=[0, 1.1], gridcolor='#2D5A27', tickfont=dict(size=10, color='white')),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color="white"))
        )
        st.plotly_chart(fig_rf, use_container_width=True)

    with col_base3:
        st.markdown("**ESM-2 Metrics**")
        p_esm2, r_esm2, f_esm2, _ = precision_recall_fscore_support(y_true, y_pred_esm2_amp)
        fig_esm2_bar = go.Figure(data=[
            go.Bar(name='Non-AMP (0)', x=metrics_labels, y=[p_esm2[0], r_esm2[0], f_esm2[0]], marker_color='#A0B0C0', text=[f"{val:.2f}" for val in [p_esm2[0], r_esm2[0], f_esm2[0]]], textposition='auto', textfont=dict(size=16, color='white', family='Arial Black')),
            go.Bar(name='AMP (1)', x=metrics_labels, y=[p_esm2[1], r_esm2[1], f_esm2[1]], marker_color='#9C27B0', text=[f"{val:.2f}" for val in [p_esm2[1], r_esm2[1], f_esm2[1]]], textposition='auto', textfont=dict(size=16, color='white', family='Arial Black'))
        ])
        fig_esm2_bar.update_layout(
            barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E6F4F1',
            margin=dict(t=20, b=20, l=0, r=0), height=350,
            xaxis=dict(tickfont=dict(size=12, color='white')),
            yaxis=dict(title=dict(text='Score', font=dict(size=14, color='white')), range=[0, 1.1], gridcolor='#2D5A27', tickfont=dict(size=10, color='white')),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color="white"))
        )
        st.plotly_chart(fig_esm2_bar, use_container_width=True)

    st.markdown("---")

    # ==========================
    # 4. ROC CURVE
    # ==========================
    st.subheader("📈 ROC Curve Analysis (AMP)")
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_mcnn, y=tpr_mcnn, mode='lines', name=f'mCNN (AUC = {roc_auc_mcnn:.4f})', line=dict(color='#4CAF50', width=4)))
    fig_roc.add_trace(go.Scatter(x=fpr_cnn, y=tpr_cnn, mode='lines', name=f'1D CNN (AUC = {roc_auc_cnn:.4f})', line=dict(color='#4A90E2', width=3)))
    fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'Random Forest (AUC = {roc_auc_rf:.4f})', line=dict(color='#FF6B6B', width=2)))
    fig_roc.add_trace(go.Scatter(x=fpr_esm2, y=tpr_esm2, mode='lines', name=f'ESM-2 (AUC = {roc_auc_esm2:.4f})', line=dict(color='#9C27B0', width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='gray', dash='dash')))
    
    fig_roc.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E6F4F1',
        margin=dict(t=20, b=20, l=0, r=0),
        xaxis=dict(title=dict(text='False Positive Rate', font=dict(size=18, color='white')), gridcolor='#2D5A27', range=[0, 1], tickfont=dict(size=14, color='white')),
        yaxis=dict(title=dict(text='True Positive Rate', font=dict(size=18, color='white')), gridcolor='#2D5A27', range=[0, 1.05], tickfont=dict(size=14, color='white')),
        legend=dict(x=0.75, y=0.1, bgcolor='rgba(0,0,0,0.8)', font=dict(size=20, color="white"))
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ==========================
    # 5. DETAILED REPORTS (AMP)
    # ==========================
    st.subheader("### 📋 Detailed Classification Reports (AMP)")
    
    col_rep1, col_rep2, col_rep3, col_rep4 = st.columns(4)
    
    def format_report(rep, color, bg_color):
        html = rep.replace(' ', '&nbsp;').replace('\n', '<br>')
        return f"""
        <div style='background-color: {bg_color}; padding: 15px; border-radius: 10px; border: 1px solid {color}; margin-top: 10px;'>
            <div style='color: #E6F4F1; font-size: 13px; font-family: monospace; line-height: 1.6; white-space: nowrap; overflow-x: auto;'>{html}</div>
        </div>
        """

    with col_rep1:
        st.markdown("#### 🌲 Random Forest")
        st.markdown(format_report(classification_report(y_true, y_pred_rf, target_names=["Non-AMP", "AMP"]), '#444', '#1E1E1E'), unsafe_allow_html=True)

    with col_rep2:
        st.markdown("#### 🛠️ 1D CNN")
        st.markdown(format_report(classification_report(y_true, y_pred_cnn, target_names=["Non-AMP", "AMP"]), '#2B5E8C', '#121B24'), unsafe_allow_html=True)

    with col_rep3:
        st.markdown("#### 👑 mCNN (Proposed)")
        st.markdown(format_report(classification_report(y_true, y_pred_mcnn, target_names=["Non-AMP", "AMP"]), '#4CAF50', '#0B2516'), unsafe_allow_html=True)

    with col_rep4:
        st.markdown("#### 🧬 ESM-2")
        st.markdown(format_report(classification_report(y_true, y_pred_esm2_amp, target_names=["Non-AMP", "AMP"]), '#9C27B0', '#250B25'), unsafe_allow_html=True)
        
    st.markdown("---")

    # ==========================
    # 6. TOXICITY EVALUATION
    # ==========================
    st.subheader("☠️ Toxicity Evaluation (ESM-2 Multitask)")
    st.markdown(f"Evaluating Toxicity prediction on the subset of data with valid toxicity labels ({len(y_true_tox_valid)} samples).")
    
    if len(y_true_tox_valid) > 0:
        acc_tox = accuracy_score(y_true_tox_valid, y_pred_esm2_tox_valid) * 100
        fpr_tox, tpr_tox, _ = roc_curve(y_true_tox_valid, prob_esm2_tox_valid)
        roc_auc_tox = auc(fpr_tox, tpr_tox)
        
        col_tox1, col_tox2 = st.columns(2)
        
        with col_tox1:
            st.markdown(f"**Accuracy:** {acc_tox:.2f}%")
            st.markdown("**Classification Report**")
            st.markdown(format_report(classification_report(y_true_tox_valid, y_pred_esm2_tox_valid, target_names=["Non-Toxic", "Toxic"]), '#E24A4A', '#2B1212'), unsafe_allow_html=True)
            
        with col_tox2:
            fig_roc_tox = go.Figure()
            fig_roc_tox.add_trace(go.Scatter(x=fpr_tox, y=tpr_tox, mode='lines', name=f'ESM-2 Toxicity (AUC = {roc_auc_tox:.4f})', line=dict(color='#E24A4A', width=4)))
            fig_roc_tox.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='gray', dash='dash')))
            fig_roc_tox.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E6F4F1',
                margin=dict(t=20, b=20, l=0, r=0), height=350,
                xaxis=dict(title=dict(text='False Positive Rate', font=dict(size=14, color='white')), gridcolor='#2D5A27', range=[0, 1], tickfont=dict(size=12, color='white')),
                yaxis=dict(title=dict(text='True Positive Rate', font=dict(size=14, color='white')), gridcolor='#2D5A27', range=[0, 1.05], tickfont=dict(size=12, color='white')),
                legend=dict(x=0.5, y=0.1, bgcolor='rgba(0,0,0,0.8)', font=dict(size=14, color="white"))
            )
            st.plotly_chart(fig_roc_tox, use_container_width=True)
    else:
        st.warning("No valid toxicity labels found in the dataset.")