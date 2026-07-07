import streamlit as st
import requests
import time

# ==========================================
# PAGE CONFIGURATION & THEMING
# ==========================================
st.set_page_config(page_title="AMP Pipeline Platform", layout="wide", page_icon="🌿")

st.markdown("""
<style>
    .stApp { background-color: #0F1A15; }
    [data-testid="stVerticalBlock"] { background-color: #17261F; padding: 20px; border-radius: 15px; }
    h1, h2, h3, h4 { color: #E6F4F1 !important; }
    p, div, span { color: #B8D8CB !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌿 Peptida: Advanced Antimicrobial Peptide AI Pipeline")
st.caption("Architecture: Dynamic Routing Cascade | Core Models: Multi-branch CNN (Local) & ESM-2 8M (Attention)")
st.markdown("---")

# ==========================================
# EXPLAINABLE AI (XAI) HEATMAP RENDERER
# ==========================================
def render_xai_html(seq, scores, theme="amp", model="mcnn"):
    html = ""
    is_saturated = sum(scores) == 0

    if is_saturated:
        html += """
        <div style='background-color: #2A251A; padding: 12px 20px; border-radius: 8px; border-left: 4px solid #F1C40F; margin-bottom: 15px;'>
            <span style='color: #F1C40F; font-weight: bold; font-size: 14px;'>⚠️ XAI Saturated:</span>
            <span style='color: #D1E8E2; font-size: 14px; margin-left: 5px;'>Maximum confidence reached. Displaying default biochemical properties.</span>
        </div>
        """

    border_color = "#2D5A27" if theme == "amp" else "#5A2727"
    html += f"<div style='font-family: monospace; font-size: 26px; letter-spacing: 2px; padding: 30px; border-radius: 12px; background: #000000; color: white; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.15); border: 1px solid {border_color}; line-height: 1.8; word-break: break-all;'>"
    
    for char, score in zip(seq, scores):
        if is_saturated:
            alpha = 0.4
            text_color = "#FFFFFF"
        else:
            if model == "esm2":
                # Transformer occlusion scores are subtle, use gentle sharpening
                sharp_score = score ** 1.2 
            else:
                # CNN occlusion scores are sharp, use aggressive sharpening
                sharp_score = score ** 3 if theme == "amp" else score ** 2
                
            alpha = sharp_score * 0.95 
            text_color = "#FFFFFF" if sharp_score > 0.1 else "#95B5A6"

        # Tùy chỉnh màu theo theme
        if theme == "toxicity":
            # Theme Toxicity: đỏ chót những chỗ gây độc
            if char in ['L', 'I', 'V', 'F', 'W', 'M', 'A']: # Hydrophobic (thường gây độc rbc)
                bg_color = f"rgba(255, 60, 60, {alpha})"
            elif char in ['K', 'R', 'H']:
                bg_color = f"rgba(255, 150, 0, {alpha})"
            else:
                bg_color = f"rgba(150, 50, 50, {alpha})"
        else:
            # Theme AMP mặc định
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
# MAIN WORKFLOW (WATERFALL ARCHITECTURE)
# ==========================================
demo_sequence = "MSTKDFNLDLVSVSKKDSGASPRITSISLCTPGCKTGALMGCNMKTATCHCSIHVSK" 
sequence_input = st.text_input("🧬 Enter Amino Acid Sequence (Auto-truncates to last 50 AAs for Stage 1):", value=demo_sequence)

if st.button("🚀 Execute Pipeline", type="primary"):
    
    st.markdown("### 🛡️ STAGE 1: Antimicrobial Gatekeeper (mCNN Inference)")
    
    with st.spinner("Extracting physiochemical features..."):
        try:
            res = requests.post("http://127.0.0.1:8000/predict/all", json={"sequence": sequence_input})
            
            if res.status_code == 200:
                data = res.json()
                
                # Parsing Backend Response
                mcnn_data = data["mcnn"]
                mcnn_verdict = mcnn_data["verdict"]
                mcnn_raw_prob = mcnn_data["prob"]
                confidence_pct = (mcnn_raw_prob if mcnn_verdict == "AMP" else (1 - mcnn_raw_prob)) * 100
                
                is_long = data["is_long_sequence"]
                routing = data["routing_status"]
                final_status = data["final_amp_status"]
                esm2_amp = data["esm2_amp_data"]
                esm2_tox = data["esm2_toxicity_data"]
                
                col1, col2 = st.columns([1, 2.5]) 
                
                with col1:
                    # CLASSIFICATION PANEL
                    if routing == "ESM_VERIFIED":
                        reason_label = "Sequence Truncated" if is_long else "Suboptimal Confidence"
                        st.markdown(f"""
                        <div style='background-color: #2A251A; padding: 20px; border-radius: 10px; border: 2px solid #F1C40F; text-align: center;'>
                            <h2 style='color: #F1C40F; margin-bottom: 5px; font-weight: 800;'>⚠️ INCONCLUSIVE</h2>
                            <h4 style='color: #E6F4F1;'>mCNN Confidence: {confidence_pct:.2f}%</h4>
                            <p style='color: #D1E8E2; font-size: 13px; margin-bottom: 0;'>{reason_label}. Re-routing to ESM-2.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif mcnn_verdict == "AMP":
                        st.markdown(f"""
                        <div style='background-color: #17261F; padding: 20px; border-radius: 10px; border: 2px solid #4CAF50; text-align: center;'>
                            <h2 style='color: #4CAF50; margin-bottom: 5px; font-weight: 800;'>🔥 AMP DETECTED</h2>
                            <h4 style='color: #E6F4F1;'>Confidence: {confidence_pct:.2f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.markdown(f"""
                        <div style='background-color: #17261F; padding: 20px; border-radius: 10px; border: 2px solid #FF6B6B; text-align: center;'>
                            <h2 style='color: #FF6B6B; margin-bottom: 5px; font-weight: 800;'>❄️ NON-AMP (INACTIVE)</h2>
                            <h4 style='color: #E6F4F1;'>Confidence: {confidence_pct:.2f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # EXPLAINABLE AI PANEL
                    st.markdown(f"**🔍 Structural Space (mCNN XAI - Processed Slice: {len(data['processed_sequence'])} AAs)**")
                    heatmap_html = render_xai_html(data["processed_sequence"], data["xai_scores"])
                    st.markdown(heatmap_html, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if routing == "ESM_VERIFIED":
                        st.markdown(f"""
                        <div style='background-color: #2A251A; padding: 20px; border-radius: 10px; border-left: 6px solid #F1C40F;'>
                            <h4 style='margin-top: 0; color: #F1C40F;'>💡 System Notification: Failsafe Protocol Activated</h4>
                            <p style='color: #D1E8E2; font-size: 15px; margin-bottom: 8px;'>The Local CNN classification did not meet the rigorous confidence threshold (≥ 85%) or structural integrity was compromised due to truncation constraints.</p>
                            <p style='color: #D1E8E2; font-size: 15px; margin-bottom: 0;'>To ensure absolute accuracy and strictly mitigate prediction errors, the pipeline has triggered the <b>Failsafe Protocol</b> and forwarded the complete sequence to <b>STAGE 2 (ESM-2 pLLM)</b> for global contextual verification.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif mcnn_verdict == "AMP":
                        st.markdown("""
                        <div style='background-color: #1A2A22; padding: 20px; border-radius: 10px; border-left: 6px solid #4A7A62;'>
                            <h4 style='margin-top: 0; color: #D1E8E2;'>💡 Biological Mechanism Identified</h4>
                            <p style='color: #95B5A6; font-size: 15px;'>The spatial architecture exhibits critical motifs necessary for membrane disruption:</p>
                            <ul style='margin-bottom: 0; line-height: 1.8; color: #B8D8CB;'>
                                <li>🧲 <b style='color: #6CB4EE;'>Cationic (Blue):</b> Positively charged residues facilitating electrostatic attraction to bacterial bilayers.</li>
                                <li>🗡️ <b style='color: #FF6B6B;'>Hydrophobic (Red):</b> Non-polar residues inserting into and destabilizing the lipid core.</li>
                                <li>🏗️ <b style='color: #F1C40F;'>Structural (Yellow):</b> Disulfide bridge anchors (Cysteine) maintaining conformational stability.</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.markdown("""
                        <div style='background-color: #2A1A1A; padding: 20px; border-radius: 10px; border-left: 6px solid #8B3A3A;'>
                            <h4 style='margin-top: 0; color: #E8D1D1;'>💡 Sequence Classification Rationale</h4>
                            <p style='color: #B59595; font-size: 15px;'>Topological analysis reveals an absence of critical amphipathic structures required for antimicrobial activity. The sequence lacks sufficient cationic affinity and hydrophobic insertion capabilities. Pipeline execution terminated to conserve computational resources.</p>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")
                
                # ==========================================
                # STAGE 2: ESM-2 ATTENTION REACTOR
                # ==========================================
                st.markdown("### ☠️ STAGE 2: Global Verification & Toxicity (ESM-2 Reactor)")
                
                if final_status == "NON-AMP" and routing == "MCNN_ONLY":
                    st.error("🛑 **NON-AMP VERIFIED.** Pipeline Terminated. ESM-2 Reactor: Standby (Resource Optimization Mode).")
                
                elif routing == "ESM_VERIFIED":
                    st.warning("🔄 **Bypass Initiated:** ESM-2 Transformer is processing the global sequence context...")
                    time.sleep(0.8) # UI aesthetic delay
                    
                    if esm2_amp["status"] == "AMP":
                        st.success(f"✅ **ESM-2 Override Successful:** Sequence structurally validated as Antimicrobial Peptide (Confidence: {esm2_amp['prob']*100:.2f}%).")
                        
                        # Render ESM-2 AMP XAI Heatmap
                        if "xai_scores" in esm2_amp and "processed_sequence" in esm2_amp:
                            st.markdown("---")
                            st.markdown("#### 🔬 AMP Verification Attention Map (ESM-2 Global Context)")
                            
                            amp_col1, amp_col2 = st.columns([1, 2.5])
                            with amp_col1:
                                st.markdown(f"""
                                <div style='background-color: #17261F; padding: 20px; border-radius: 10px; border: 2px solid #4CAF50; text-align: center; height: 100%;'>
                                    <h3 style='color: #4CAF50; margin-bottom: 5px; font-weight: 800;'>🔥 AMP VERIFIED</h3>
                                    <h5 style='color: #E6F4F1;'>Confidence: {esm2_amp['prob']*100:.2f}%</h5>
                                    <p style='color: #D1E8E2; font-size: 13px;'>Global sequence functionally verified as Antimicrobial.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            with amp_col2:
                                st.markdown(f"**🔍 Antimicrobial Motif Space (ESM-2 Attention - Processed Slice: {len(esm2_amp['processed_sequence'])} AAs)**")
                                amp_heatmap_html = render_xai_html(esm2_amp["processed_sequence"], esm2_amp["xai_scores"], theme="amp", model="esm2")
                                st.markdown(amp_heatmap_html, unsafe_allow_html=True)
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                st.markdown("""
                                <div style='background-color: #1A2A22; padding: 20px; border-radius: 10px; border-left: 6px solid #4A7A62;'>
                                    <h4 style='margin-top: 0; color: #D1E8E2;'>💡 Global Biological Mechanism Verified</h4>
                                    <p style='color: #95B5A6; font-size: 15px;'>ESM-2 Transformer attention confirms the presence of essential antimicrobial properties across the entire sequence context:</p>
                                    <ul style='margin-bottom: 0; line-height: 1.8; color: #B8D8CB;'>
                                        <li>🧲 <b style='color: #6CB4EE;'>Cationic (Blue):</b> Verified electrostatic anchors.</li>
                                        <li>🗡️ <b style='color: #FF6B6B;'>Hydrophobic (Red):</b> Verified membrane-destabilizing core.</li>
                                        <li>🏗️ <b style='color: #F1C40F;'>Structural (Yellow):</b> Verified topological stability points.</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            
                    else:
                        st.error(f"❌ **ESM-2 Final Verdict:** Sequence confirmed as inactive. No further processing required.")

                elif final_status == "AMP" and routing == "MCNN_ONLY":
                    st.success("✅ **Gatekeeper Approval:** Sequence accepted. Proceeding to secondary safety assessment...")
                
                # Render Toxicity XAI
                if final_status == "AMP" and esm2_tox:
                    st.markdown("---")
                    st.markdown("#### 🔬 Toxicity Attention Map (ESM-2 Global Context)")
                    
                    tox_col1, tox_col2 = st.columns([1, 2.5])
                    with tox_col1:
                        if not esm2_tox["is_toxic"]:
                            st.markdown(f"""
                            <div style='background-color: #17261F; padding: 20px; border-radius: 10px; border: 2px solid #4CAF50; text-align: center; height: 100%;'>
                                <h3 style='color: #4CAF50; margin-bottom: 5px; font-weight: 800;'>🧬 SAFE</h3>
                                <h5 style='color: #E6F4F1;'>Toxicity Prob: {esm2_tox['prob']*100:.2f}%</h5>
                                <p style='color: #D1E8E2; font-size: 13px;'>Negligible hemolytic activity predicted.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style='background-color: #2A1A1A; padding: 20px; border-radius: 10px; border: 2px solid #FF4B4B; text-align: center; height: 100%;'>
                                <h3 style='color: #FF4B4B; margin-bottom: 5px; font-weight: 800;'>☠️ TOXIC</h3>
                                <h5 style='color: #E6F4F1;'>Toxicity Prob: {esm2_tox['prob']*100:.2f}%</h5>
                                <p style='color: #D1E8E2; font-size: 13px;'>High probability of hemolyzing human erythrocytes.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with tox_col2:
                        if "xai_scores" in esm2_tox and "processed_sequence" in esm2_tox:
                            st.markdown(f"**🔍 Cytotoxic Motif Space (ESM-2 Attention - Processed Slice: {len(esm2_tox['processed_sequence'])} AAs)**")
                            tox_heatmap_html = render_xai_html(esm2_tox["processed_sequence"], esm2_tox["xai_scores"], theme="toxicity", model="esm2")
                            st.markdown(tox_heatmap_html, unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            if esm2_tox["is_toxic"]:
                                st.markdown("""
                                <div style='background-color: #2A1A1A; padding: 20px; border-radius: 10px; border-left: 6px solid #FF4B4B;'>
                                    <h4 style='margin-top: 0; color: #E8D1D1;'>💡 Cytotoxicity Mechanism Identified</h4>
                                    <p style='color: #B59595; font-size: 15px;'>ESM-2 Attention highlights regions with excessive hydrophobic density or highly charged clusters that lead to non-specific pore formation in mammalian cell membranes:</p>
                                    <ul style='margin-bottom: 0; line-height: 1.8; color: #E8D1D1;'>
                                        <li>🔴 <b style='color: #FF6B6B;'>Deep Red Zones:</b> Primary cytotoxic motifs. Often continuous hydrophobic domains that penetrate red blood cells.</li>
                                        <li>🟠 <b style='color: #FFA500;'>Orange Zones:</b> Secondary structural elements enhancing peptide aggregation and membrane lysis.</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style='background-color: #1A2A22; padding: 20px; border-radius: 10px; border-left: 6px solid #4CAF50;'>
                                    <h4 style='margin-top: 0; color: #D1E8E2;'>💡 Biocompatibility Rationale</h4>
                                    <p style='color: #95B5A6; font-size: 15px;'>The global attention maps do not detect consecutive sequences capable of lysing mammalian membranes. The hydrophobic-hydrophilic balance is optimal for prokaryotic targeting without harming host cells.</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
            else:
                st.warning("HTTP Error: Unexpected response from backend.")
        except requests.exceptions.ConnectionError:
            st.error("🚨 System Failure: Unable to establish connection. Verify Uvicorn backend process.")