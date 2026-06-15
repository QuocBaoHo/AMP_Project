import streamlit as st
import numpy as np
import tensorflow as tf
import keras
import joblib
import pickle
from datasets import load_dataset
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

st.title("🚀 AMP Models Benchmark Dashboard")
st.markdown("Run the evaluation on the full **VIRI test set (28.5k samples)** and visualize the performance metrics.")

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
        
    return mcnn_model, cnn_model, rf_model, tokenizer, vectorizer

@st.cache_data(show_spinner=False)
def get_dataset():
    dataset = load_dataset("ps29/amp_dataset_viri", split="viri_test")
    return dataset['seq'], dataset['label']

if st.button("🔥 Run Benchmark (28.5k samples)", type="primary"):
    with st.status("Running Benchmark Pipeline...", expanded=True) as status:
        st.write("1. Initializing 3 baseline models...")
        mcnn_model, cnn_model, rf_model, tokenizer, vectorizer = load_models()
        
        st.write("2. Fetching test set (viri_test) from Hugging Face...")
        sequences, y_true = get_dataset()
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
                # Chuyển chuỗi chữ thành ma trận TF-IDF/Count cho RF
                X_rf = vectorizer.transform(sequences)
                y_pred_rf = rf_model.predict(X_rf)
                prob_rf = rf_model.predict_proba(X_rf)[:, 1]
                rf_error = False
            except Exception as e:
                st.error(f"   [ERROR] RF Prediction failed: {e}")
                rf_error = True
                y_pred_rf = np.zeros_like(y_true)
                prob_rf = np.zeros_like(y_true, dtype=float)
        else:
            st.error("   [WARNING] Missing 'vectorizer.pkl'! Random Forest will blindly guess Non-AMP (0).")
            rf_error = True
            y_pred_rf = np.zeros_like(y_true)
            prob_rf = np.zeros_like(y_true, dtype=float)

        status.update(label="Benchmark Completed!", state="complete", expanded=False)

    # Calculate accuracy
    acc_rf = accuracy_score(y_true, y_pred_rf) * 100
    acc_cnn = accuracy_score(y_true, y_pred_cnn) * 100
    acc_mcnn = accuracy_score(y_true, y_pred_mcnn) * 100

    # Calculate ROC AUC
    fpr_mcnn, tpr_mcnn, _ = roc_curve(y_true, prob_mcnn)
    roc_auc_mcnn = auc(fpr_mcnn, tpr_mcnn)
    
    fpr_cnn, tpr_cnn, _ = roc_curve(y_true, prob_cnn)
    roc_auc_cnn = auc(fpr_cnn, tpr_cnn)

    fpr_rf, tpr_rf, _ = roc_curve(y_true, prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    metrics_labels = ['Precision', 'Recall', 'F1-Score']

    st.markdown("---")
    
    # ==========================
    # 1. ACCURACY COMPARISON
    # ==========================
    st.subheader("🏆 Accuracy Comparison")
    models = ['Random Forest', '1D CNN', 'mCNN (Proposed)']
    accs = [acc_rf, acc_cnn, acc_mcnn]
    colors = ['#FF6B6B', '#4A90E2', '#4CAF50']
    
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
    # 3. BASELINES: 1D CNN & RF
    # ==========================
    st.subheader("🛠️ Baseline Models Performance")
    col_base1, col_base2 = st.columns(2)
    
    with col_base1:
        st.markdown("**1D CNN Metrics**")
        p_cnn, r_cnn, f_cnn, _ = precision_recall_fscore_support(y_true, y_pred_cnn)
        fig_cnn = go.Figure(data=[
            go.Bar(name='Non-AMP (0)', x=metrics_labels, y=[p_cnn[0], r_cnn[0], f_cnn[0]], marker_color='#A0B0C0', text=[f"{val:.2f}" for val in [p_cnn[0], r_cnn[0], f_cnn[0]]], textposition='auto', textfont=dict(size=20, color='white', family='Arial Black')),
            go.Bar(name='AMP (1)', x=metrics_labels, y=[p_cnn[1], r_cnn[1], f_cnn[1]], marker_color='#4A90E2', text=[f"{val:.2f}" for val in [p_cnn[1], r_cnn[1], f_cnn[1]]], textposition='auto', textfont=dict(size=20, color='white', family='Arial Black'))
        ])
        fig_cnn.update_layout(
            barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E6F4F1',
            margin=dict(t=20, b=20, l=0, r=0), height=350,
            xaxis=dict(tickfont=dict(size=14, color='white')),
            yaxis=dict(title=dict(text='Score', font=dict(size=16, color='white')), range=[0, 1.1], gridcolor='#2D5A27', tickfont=dict(size=12, color='white')),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=16, color="white"))
        )
        st.plotly_chart(fig_cnn, use_container_width=True)

    with col_base2:
        st.markdown("**Random Forest Metrics**")
        p_rf, r_rf, f_rf, _ = precision_recall_fscore_support(y_true, y_pred_rf)
        fig_rf = go.Figure(data=[
            go.Bar(name='Non-AMP (0)', x=metrics_labels, y=[p_rf[0], r_rf[0], f_rf[0]], marker_color='#A0B0C0', text=[f"{val:.2f}" for val in [p_rf[0], r_rf[0], f_rf[0]]], textposition='auto', textfont=dict(size=20, color='white', family='Arial Black')),
            go.Bar(name='AMP (1)', x=metrics_labels, y=[p_rf[1], r_rf[1], f_rf[1]], marker_color='#FF6B6B', text=[f"{val:.2f}" for val in [p_rf[1], r_rf[1], f_rf[1]]], textposition='auto', textfont=dict(size=20, color='white', family='Arial Black'))
        ])
        fig_rf.update_layout(
            barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E6F4F1',
            margin=dict(t=20, b=20, l=0, r=0), height=350,
            xaxis=dict(tickfont=dict(size=14, color='white')),
            yaxis=dict(title=dict(text='Score', font=dict(size=16, color='white')), range=[0, 1.1], gridcolor='#2D5A27', tickfont=dict(size=12, color='white')),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=16, color="white"))
        )
        st.plotly_chart(fig_rf, use_container_width=True)

    st.markdown("---")

    # ==========================
    # 4. ROC CURVE
    # ==========================
    st.subheader("📈 ROC Curve Analysis")
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_mcnn, y=tpr_mcnn, mode='lines', name=f'mCNN (AUC = {roc_auc_mcnn:.4f})', line=dict(color='#4CAF50', width=4)))
    fig_roc.add_trace(go.Scatter(x=fpr_cnn, y=tpr_cnn, mode='lines', name=f'1D CNN (AUC = {roc_auc_cnn:.4f})', line=dict(color='#4A90E2', width=3)))
    fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'Random Forest (AUC = {roc_auc_rf:.4f})', line=dict(color='#FF6B6B', width=2)))
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
    # 5. DETAILED REPORTS
    # ==========================
    st.subheader("### 📋 Detailed Classification Reports Comparison:")
    
    col_rep1, col_rep2, col_rep3 = st.columns(3)
    
    with col_rep1:
        st.markdown("#### 🌲 Random Forest")
        rep_rf = classification_report(y_true, y_pred_rf, target_names=["Non-AMP", "AMP"])
        html_rf = rep_rf.replace(' ', '&nbsp;').replace('\n', '<br>')
        st.markdown(f"""
        <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #444; margin-top: 10px;'>
            <div style='color: #999; font-size: 15px; font-family: monospace; line-height: 1.6; white-space: nowrap; overflow-x: auto;'>{html_rf}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_rep2:
        st.markdown("#### 🛠️ 1D CNN")
        rep_cnn = classification_report(y_true, y_pred_cnn, target_names=["Non-AMP", "AMP"])
        html_cnn = rep_cnn.replace(' ', '&nbsp;').replace('\n', '<br>')
        st.markdown(f"""
        <div style='background-color: #121B24; padding: 15px; border-radius: 10px; border: 1px solid #2B5E8C; margin-top: 10px;'>
            <div style='color: #A0C0DE; font-size: 15px; font-family: monospace; line-height: 1.6; white-space: nowrap; overflow-x: auto;'>{html_cnn}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_rep3:
        st.markdown("#### 👑 mCNN (Proposed)")
        rep_mcnn = classification_report(y_true, y_pred_mcnn, target_names=["Non-AMP", "AMP"])
        html_mcnn = rep_mcnn.replace(' ', '&nbsp;').replace('\n', '<br>')
        st.markdown(f"""
        <div style='background-color: #0B2516; padding: 15px; border-radius: 10px; border: 1px solid #4CAF50; margin-top: 10px;'>
            <div style='color: #E6F4F1; font-size: 15px; font-family: monospace; line-height: 1.6; white-space: nowrap; overflow-x: auto;'>{html_mcnn}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Print to console
    print("\n" + "="*50)
    print("🏆 BENCHMARK RESULTS ON TEST SET (28.5k samples) 🏆")
    print("="*50)
    print(f"🌲 Random Forest Accuracy : {acc_rf:.2f}%")
    print(f"🛠️ CNN 1D Accuracy       : {acc_cnn:.2f}%")
    print(f"👑 mCNN Accuracy         : {acc_mcnn:.2f}%")