import streamlit as st
import requests

st.set_page_config(page_title="Đồ án AMP Demo", layout="centered")

st.title("🔬 Hệ thống Phân loại Peptide (AMP)")
st.caption("Backend: FastAPI | Core AI: Multi-branch CNN | Tính năng: Explainable AI")
st.markdown("---")

st.subheader("Kiểm tra đặc tính Kháng khuẩn")

# Chuỗi mặc định
demo_sequence = "GLFDIVKKVVGALG" 
sequence = st.text_input("Nhập chuỗi Amino Acid (Tối đa 50 ký tự):", value=demo_sequence)

# Hàm vẽ HTML Heatmap
def render_xai_html(seq, scores):
    html = "<div style='font-family: monospace; font-size: 24px; letter-spacing: 2px; padding: 20px; border-radius: 10px; background: #262730; color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>"
    for char, score in zip(seq, scores):
        # Điểm càng cao, nền màu đỏ càng đậm (alpha từ 0 đến 1)
        alpha = score * 0.9 # Giới hạn 0.9 để chữ vẫn đọc được
        bg_color = f"rgba(255, 75, 75, {alpha})"
        html += f"<span style='background-color: {bg_color}; padding: 4px 2px; border-radius: 4px; font-weight: bold;'>{char}</span>"
    html += "</div>"
    return html

if st.button("Chạy Phân Tích & XAI", type="primary"):
    with st.spinner("Đang móc ruột AI để phân tích..."):
        try:
            res = requests.post("http://127.0.0.1:8000/predict/mcnn", json={"sequence": sequence})
            
            if res.status_code == 200:
                data = res.json()
                
                # Cột trái hiện kết quả, Cột phải hiện XAI
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if data["is_amp"]:
                        st.success(f"🔥 PHÁT HIỆN AMP!\n\nTự tin: {data['confidence']}")
                    else:
                        st.error(f"❄️ KHÔNG PHẢI AMP.\n\nTự tin: {data['confidence']}")
                
                with col2:
                    st.markdown("**🔍 Phân tích AI (Explainable AI - Feature Ablation)**")
                    st.caption("Các vùng màu đỏ đậm là các Amino Acid quyết định tính kháng khuẩn.")
                    
                    # Gọi hàm vẽ Heatmap ra
                    heatmap_html = render_xai_html(data["sequence"], data["xai_scores"])
                    st.markdown(heatmap_html, unsafe_allow_html=True)
                
                # Show dữ liệu ẩn bên dưới cho Sigma
                with st.expander("Bấm vào để xem Raw JSON API"):
                    st.json(data)
                    
            else:
                st.warning("API trả về lỗi cục bộ!")
        except requests.exceptions.ConnectionError:
            st.error("🚨 Không kết nối được Backend!")