import streamlit as st
import requests
import time

# Cấu hình giao diện
st.set_page_config(page_title="AMP Pipeline Platform", layout="wide", page_icon="🌿")

# ==========================================
# CSS ĐỘ ĐỘNG CƠ THẨM MỸ (BOTANIC THEME)
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #0F1A15; }
    [data-testid="stVerticalBlock"] { background-color: #17261F; padding: 20px; border-radius: 15px; }
    h1, h2, h3, h4 { color: #E6F4F1 !important; }
    p, div, span { color: #B8D8CB !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌿 One-Click Peptide AI Pipeline")
st.caption("Backend: FastAPI | Core AI: Multi-branch CNN & ESM-2 | Flow: Waterfall Architecture")
st.markdown("---")

# ==========================================
# HÀM VẼ HEATMAP (CHUẨN UX + LIÊM CHÍNH KHOA HỌC)
# ==========================================
def render_xai_html(seq, scores):
    html = ""
    is_saturated = sum(scores) == 0

    # 1. Nếu bão hòa, hiện một cái cảnh báo nhỏ gọn, tinh tế phía trên
    if is_saturated:
        html += """
        <div style='background-color: #2A251A; padding: 12px 20px; border-radius: 8px; border-left: 4px solid #F1C40F; margin-bottom: 15px;'>
            <span style='color: #F1C40F; font-weight: bold; font-size: 14px;'>⚠️ XAI Saturated:</span>
            <span style='color: #D1E8E2; font-size: 14px; margin-left: 5px;'>The model is 100% confident. The heatmap below displays default chemical properties (Flat).</span>
        </div>
        """

    # 2. Bắt đầu vẽ khung đen Heatmap
    html += "<div style='font-family: monospace; font-size: 26px; letter-spacing: 2px; padding: 30px; border-radius: 12px; background: #000000; color: white; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.15); border: 1px solid #2D5A27; line-height: 1.8; word-break: break-all;'>"
    
    for char, score in zip(seq, scores):
        # Nếu bão hòa, gán alpha cứng là 0.4 cho tất cả để nhìn dịu mắt. Nếu không, tính theo toán XAI.
        if is_saturated:
            alpha = 0.4
            text_color = "#FFFFFF"
        else:
            sharp_score = score ** 3 
            alpha = sharp_score * 0.95 
            text_color = "#FFFFFF" if sharp_score > 0.1 else "#95B5A6"

        # Đổ màu theo nhóm sinh học
        if char in ['K', 'R', 'H']:
            bg_color = f"rgba(74, 144, 226, {alpha})"
        elif char in ['L', 'I', 'V', 'F', 'W', 'M', 'A']:
            bg_color = f"rgba(231, 76, 60, {alpha})"
        elif char == 'C':
            bg_color = f"rgba(241, 196, 15, {alpha})"
        else:
            bg_color = f"rgba(255, 255, 255, 0.05)" if is_saturated else f"rgba(255, 255, 255, 0)"
            
        html += f"<span style='display: inline-block; background-color: {bg_color}; color: {text_color}; padding: 4px 2px; border-radius: 5px; font-weight: 600; margin: 4px 1px;'>{char}</span>"
    
    html += "</div>"
    return html
# ==========================================
# GIAO DIỆN WATERFALL (THẢ DỌC TỪ TRÊN XUỐNG)
# ==========================================

demo_sequence = "MSTKDFNLDLVSVSKKDSGASPRITSISLCTPGCKTGALMGCNMKTATCHCSIHVSK" 
sequence_input = st.text_input("🧬 Enter Amino Acid Sequence (Auto-truncates to last 50 AAs):", value=demo_sequence)

if st.button("🚀 Run Full Pipeline", type="primary"):
    
    st.markdown("### 🛡️ STAGE 1: Antimicrobial Property Gatekeeper (mCNN)")
    
    with st.spinner("Extracting biological features..."):
        try:
            res = requests.post("http://127.0.0.1:8000/predict/all", json={"sequence": sequence_input})
            
            if res.status_code == 200:
                data = res.json()
                
                # Bóc tách data từ JSON đời mới
                mcnn_data = data["mcnn"]
                is_long = data["is_long_sequence"]
                routing = data["routing_status"]
                final_status = data["final_amp_status"]
                esm2_amp = data["esm2_amp_data"]
                esm2_tox = data["esm2_toxicity_data"]
                
                col1, col2 = st.columns([1, 2.5]) 
                
                with col1:
                    # CỘT 1: BIỂN BÁO STAGE 1
                    if mcnn_data["verdict"] == "AMP":
                        st.markdown(f"""
                        <div style='background-color: #17261F; padding: 20px; border-radius: 10px; border: 2px solid #4CAF50; text-align: center;'>
                            <h2 style='color: #4CAF50; margin-bottom: 5px; font-weight: 800;'>🔥 AMP DETECTED</h2>
                            <h4 style='color: #E6F4F1;'>Confidence: {mcnn_data['prob']*100:.2f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        if is_long:
                            # Ca nghi ngờ: Chuỗi dài bị cắt đầu, mCNN phán rác -> Kích hoạt biển Vàng
                            st.markdown(f"""
                            <div style='background-color: #2A251A; padding: 20px; border-radius: 10px; border: 2px solid #F1C40F; text-align: center;'>
                                <h2 style='color: #F1C40F; margin-bottom: 5px; font-weight: 800;'>⚠️ INCONCLUSIVE</h2>
                                <h4 style='color: #E6F4F1;'>mCNN Confidence: {(1 - mcnn_data['prob'])*100:.2f}%</h4>
                                <p style='color: #D1E8E2; font-size: 13px; margin-bottom: 0;'>Sequence truncated. Needs ESM-2 verification!</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Rác xịn: Chuỗi ngắn, mCNN phán rác -> Biển Đỏ
                            st.markdown(f"""
                            <div style='background-color: #17261F; padding: 20px; border-radius: 10px; border: 2px solid #FF6B6B; text-align: center;'>
                                <h2 style='color: #FF6B6B; margin-bottom: 5px; font-weight: 800;'>❄️ NON-AMP (TRASH)</h2>
                                <h4 style='color: #E6F4F1;'>Confidence: {(1 - mcnn_data['prob'])*100:.2f}%</h4>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    # CỘT 2: HEATMAP VÀ BẢNG GIẢI THÍCH
                    st.markdown(f"**🔍 Structural Space (mCNN XAI - Last {len(data['processed_sequence'])} AAs Slice)**")
                    heatmap_html = render_xai_html(data["processed_sequence"], data["xai_scores"])
                    st.markdown(heatmap_html, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if mcnn_data["verdict"] == "AMP":
                        st.markdown("""
                        <div style='background-color: #1A2A22; padding: 20px; border-radius: 10px; border-left: 6px solid #4A7A62;'>
                            <h4 style='margin-top: 0; color: #D1E8E2;'>💡 Why does this kill bacteria?</h4>
                            <p style='color: #95B5A6; font-size: 15px;'>The AI model detected the following "biological weapons" in the sequence:</p>
                            <ul style='margin-bottom: 0; line-height: 1.8; color: #B8D8CB;'>
                                <li>🧲 <b style='color: #6CB4EE;'>Cationic (Blue):</b> Positively charged, acts as a <i>"Magnet"</i> attracting bacterial membranes.</li>
                                <li>🗡️ <b style='color: #FF6B6B;'>Hydrophobic (Red):</b> Hydrophobic group, acts as a <i>"Drill"</i> penetrating lipid membranes.</li>
                                <li>🏗️ <b style='color: #F1C40F;'>Structural (Yellow):</b> Cysteine (C) acts as a <i>"Steel frame"</i> maintaining structure.</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        if is_long:
                            # Giải thích tại sao lại hiện biển Vàng
                            st.markdown("""
                            <div style='background-color: #2A251A; padding: 20px; border-radius: 10px; border-left: 6px solid #F1C40F;'>
                                <h4 style='margin-top: 0; color: #F1C40F;'>💡 Why is this Inconclusive?</h4>
                                <p style='color: #D1E8E2; font-size: 15px; margin-bottom: 8px;'>The mCNN model found no clear antimicrobial signs in the last 50 characters. However, <b>the system detected that the sequence prefix was truncated</b> due to the Maxlen limit.</p>
                                <p style='color: #D1E8E2; font-size: 15px; margin-bottom: 0;'>To prevent false negatives, the system automatically triggers a Bypass, routing the full original sequence to <b>STAGE 2 (ESM-2)</b> for re-verification using the Attention mechanism.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style='background-color: #2A1A1A; padding: 20px; border-radius: 10px; border-left: 6px solid #8B3A3A;'>
                                <h4 style='margin-top: 0; color: #E8D1D1;'>💡 Why is this sequence rejected?</h4>
                                <p style='color: #B59595; font-size: 15px;'>AI analysis shows this sequence has a <b>severe lack</b> of biological properties (Cationic/Hydrophobic) to act as a weapon. Pipeline auto-terminated to save resources.</p>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("---")
                
                # ==========================================
                # STAGE 2: ESM-2 REACTOR (THẨM ĐỊNH & ĐỘC TÍNH)
                # ==========================================
                st.markdown("### ☠️ STAGE 2: Toxicity & Verification (ESM-2 Reactor)")
                
                if final_status == "NON-AMP" and routing == "MCNN_ONLY":
                    st.error("🛑 NON-AMP DETECTED. Pipeline Terminated. ESM-2 Status: Offline (Resource saving).")
                
                elif routing == "ESM_VERIFIED":
                    st.warning("🔄 Processing Bypass: ESM-2 is reading the full original sequence...")
                    import time
                    time.sleep(1) # Giả lập chờ ESM-2 load tí cho nghệ
                    
                    if esm2_amp["status"] == "AMP":
                        st.success(f"✅ **ESM-2 Overridden Successfully:** {esm2_amp['msg']}")
                        if esm2_tox:
                            if not esm2_tox["is_toxic"]:
                                st.info(f"🧬 **Toxicity Analysis (ESM-2):** {esm2_tox['msg']}")
                            else:
                                st.error(f"☠️ **Toxicity Analysis (ESM-2):** Warning! Toxic to human cells!")
                    else:
                        st.error(f"❌ **ESM-2 Final Verdict:** {esm2_amp['msg']}")

                elif final_status == "AMP" and routing == "MCNN_ONLY":
                    st.success("✅ mCNN confirmed AMP! ESM-2 accepted for toxicity measurement...")
                    if esm2_tox:
                        if not esm2_tox["is_toxic"]:
                            st.info(f"🧬 **Toxicity Analysis (ESM-2):** {esm2_tox['msg']}")
                        else:
                            st.error(f"☠️ **Toxicity Analysis (ESM-2):** Warning! Toxic to human cells!")
                        
            else:
                st.warning("API returned a local error!")
        except requests.exceptions.ConnectionError:
            st.error("🚨 Failed to connect to Backend! Please ensure uvicorn is running.")