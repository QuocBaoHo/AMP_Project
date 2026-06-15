import streamlit as st
import requests
import pandas as pd

# Cấu hình giao diện (Wide để bảng so sánh nhìn rộng rãi, icon chiếc lá cho nghệ)
st.set_page_config(page_title="AMP Multi-Model Platform", layout="wide", page_icon="🌿")

# ==========================================
# CSS ĐỘ ĐỘNG CƠ THẨM MỸ (BOTANIC THEME)
# ==========================================
st.markdown("""
<style>
    /* Nền toàn bộ trang là Xanh Rêu Đậm */
    .stApp {
        background-color: #0F1A15;
    }
    /* Các khối box (tabs, table) là Xanh Rêu Sáng hơn chút */
    [data-testid="stVerticalBlock"] {
        background-color: #17261F;
        padding: 20px;
        border-radius: 15px;
    }
    /* Chữ tiêu đề dùng màu Mint Cream (đẩy sáng một chút) */
    h1, h2, h3, h4 {
        color: #E6F4F1 !important;
    }
    /* Phóng to chữ ở các Tab */
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"] p {
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    /* Chữ thường dùng màu xanh lá nhạt (đẩy sáng và rõ hơn bản gốc) */
    p, div, span {
        color: #B8D8CB !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌿 Peptide Classification & Analysis System")
st.caption("Backend: FastAPI | Core AI: Multi-branch CNN & ESM-2 | UI: Botanic Green Aesthetic")
st.markdown("---")

# ==========================================
# HÀM VẼ HEATMAP (Theme Xanh Rêu Sang Trọng, Chống Mù Mắt)
# ==========================================
def render_xai_html(seq, scores):
    # Nền đổi sang Đen Tuyền (Black) để màu sắc của chữ nổi bần bật nhất có thể
    html = "<div style='font-family: monospace; font-size: 26px; letter-spacing: 2px; padding: 30px; border-radius: 12px; background: #000000; color: white; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.15); border: 1px solid #2D5A27; line-height: 1.8; word-break: break-all;'>"
    for char, score in zip(seq, scores):
        sharp_score = score ** 3 
        alpha = sharp_score * 0.95 
        
        # BỘ LỌC MÀU PASTEL: Dịu mắt, hòa quyện với nền xanh rêu
        if char in ['K', 'R', 'H']:
            bg_color = f"rgba(74, 144, 226, {alpha})"  # Xanh lam dịu (Nam châm)
        elif char in ['L', 'I', 'V', 'F', 'W', 'M', 'A']:
            bg_color = f"rgba(231, 76, 60, {alpha})"   # Đỏ san hô (Mũi khoan)
        elif char == 'C':
            bg_color = f"rgba(241, 196, 15, {alpha})"  # Vàng nghệ (Khung thép)
        else:
            bg_color = f"rgba(255, 255, 255, 0)"       # Trong suốt hoàn toàn
            
        # Chữ không quan trọng sẽ có màu Sage Green sáng hơn nữa để dễ đọc mà vẫn mờ hơn chữ chính
        text_color = "#FFFFFF" if sharp_score > 0.1 else "#95B5A6"
        
        html += f"<span style='display: inline-block; background-color: {bg_color}; color: {text_color}; padding: 4px 2px; border-radius: 5px; font-weight: 600; margin: 4px 1px;'>{char}</span>"
    html += "</div>"
    return html

# ==========================================
# KHỞI TẠO 2 TAB CHIẾN LƯỢC
# ==========================================
tab1, tab2 = st.tabs([
    "🎯 1. Antimicrobial Prediction (mCNN + XAI)", 
    "☠️ 2. Toxicity Analysis (ESM-2)"
])

# ==========================================
# TAB 1: MẶT TIỀN CHÍNH (mCNN + XAI)
# ==========================================
with tab1:
    st.subheader("Antimicrobial Property Checker (Proposed mCNN)")
    
    demo_sequence = "MSTKDFNLDLVSVSKKDSGASPRITSISLCTPGCKTGALMGCNMKTATCHCSIHVSK" 
    sequence_t1 = st.text_input("Enter Amino Acid Sequence (sequences > 50 AAs will be truncated to the last 50 AAs):", value=demo_sequence, key="input_tab1")
    
    if len(sequence_t1) > 50:
        st.warning(f"⚠️ The current sequence is {len(sequence_t1)} characters long. The AI system will extract the LAST 50 characters for analysis.")

    if st.button("Run Deep Analysis", type="primary", key="btn_tab1"):
        with st.spinner("Extracting biological features..."):
            try:
                res = requests.post("http://127.0.0.1:8000/predict/all", json={"sequence": sequence_t1})
                
                if res.status_code == 200:
                    data = res.json()
                    mcnn_data = data["mcnn"]
                    
                    col1, col2 = st.columns([1, 2.5]) # Chỉnh tỷ lệ cột cho heatmap rộng ra
                    
                    with col1:
                        if mcnn_data["is_amp"]:
                            st.markdown(f"""
                            <div style='background-color: #17261F; padding: 20px; border-radius: 10px; border: 2px solid #4CAF50; text-align: center;'>
                                <h2 style='color: #4CAF50; margin-bottom: 5px; font-weight: 800;'>🔥 AMP DETECTED!</h2>
                                <h4 style='color: #E6F4F1;'>Confidence: {mcnn_data['prob']*100:.2f}%</h4>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style='background-color: #17261F; padding: 20px; border-radius: 10px; border: 2px solid #FF6B6B; text-align: center;'>
                                <h2 style='color: #FF6B6B; margin-bottom: 5px; font-weight: 800;'>❄️ NON-AMP</h2>
                                <h4 style='color: #E6F4F1;'>Confidence: {(1 - mcnn_data['prob'])*100:.2f}%</h4>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**🔍 Structural Space (Explainable AI)**")
                        
                        # Vẽ HTML
                        heatmap_html = render_xai_html(data["sequence"], data["xai_scores"])
                        st.markdown(heatmap_html, unsafe_allow_html=True)
                        
                        # Bảng chú thích màu Mint Green siêu nghệ
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        if mcnn_data["is_amp"]:
                            legend_html = """
                            <div style='background-color: #1A2A22; padding: 20px; border-radius: 10px; border-left: 6px solid #4A7A62;'>
                                <h4 style='margin-top: 0; color: #D1E8E2;'>💡 Why does this sequence kill bacteria?</h4>
                                <p>The AI system identified the following <b>'Biological Weapons'</b> configured in the sequence:</p>
                                <ul style='margin-bottom: 0; line-height: 1.8;'>
                                    <li>🧲 <b style='color: #6CB4EE;'>Cationic Group (Blue - K, R, H):</b> Acts as a magnet, anchoring the peptide to the bacterial membrane.</li>
                                    <li>🗡️ <b style='color: #FF6B6B;'>Hydrophobic Group (Red - L, I, V, F...):</b> Acts as a 'drill' penetrating the lipid bilayer, causing cell lysis.</li>
                                    <li>🏗️ <b style='color: #F8C471;'>Structural Group (Yellow - C):</b> Cysteine forms Disulfide bridges, maintaining a rigid 3D conformation.</li>
                                </ul>
                            </div>
                            """
                        else:
                            legend_html = """
                            <div style='background-color: #2A1A1A; padding: 20px; border-radius: 10px; border-left: 6px solid #8B3A3A;'>
                                <h4 style='margin-top: 0; color: #E8D1D1;'>💡 Why is this sequence NOT antimicrobial?</h4>
                                <p>AI analysis reveals that this sequence <b>lacks the core biological features</b> required for an AMP:</p>
                                <ul style='margin-bottom: 0; line-height: 1.8;'>
                                    <li>🧲 <b style='color: #A0B0C0;'>Lack of Cationic Residues:</b> Insufficient electrostatic attraction to anchor to the bacterial membrane.</li>
                                    <li>🗡️ <b style='color: #C0A0A0;'>Lack of Hydrophobicity:</b> Fails to form the necessary 'drill' to penetrate the lipid bilayer.</li>
                                    <li>🛡️ <b style='color: #D1E8E2;'>Conclusion:</b> This is highly likely a standard protein or peptide lacking biocidal properties.</li>
                                </ul>
                            </div>
                            """
                        st.markdown(legend_html, unsafe_allow_html=True)
                        
                else:
                    st.warning("API returned a local error!")
            except requests.exceptions.ConnectionError:
                st.error("🚨 Failed to connect to Backend! Please ensure uvicorn is running.")

# ==========================================
# TAB 2: TRÙM CUỐI SOTA (Dự án Phase 2)
# ==========================================
with tab2:
    st.subheader("Toxicity Analysis via Transfer Learning")
    st.markdown("This module is currently being designed using Meta AI's **ESM-2 (Evolutionary Scale Modeling)** language model to extract deep biological semantics.")
    st.info("🚧 Currently in the Fine-tuning phase. The AI reactor is being initialized...")