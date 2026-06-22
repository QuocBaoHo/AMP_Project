from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib
import uvicorn
import tensorflow as tf
import keras
import numpy as np

# TODO: Vài bữa ráp hàng thật thì bỏ comment 2 dòng này
# import torch
# from transformers import AutoTokenizer, EsmForSequenceClassification

app = FastAPI(title="AMP & Toxicity Cascade Pipeline")

# --- NẠP BẢO VẬT GIAI ĐOẠN 1 (Gatekeeper) ---
print("Đang khởi động STAGE 1: mCNN Gatekeeper...")
mcnn_model = tf.keras.models.load_model('mcnn_model.keras')
cnn_model = tf.keras.models.load_model('cnn_model.keras')
rf_model = joblib.load('rf_model.pkl')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# --- NẠP BẢO VẬT GIAI ĐOẠN 2 (Trùm cuối ESM-2) ---
print("Đang khởi động STAGE 2: Lò phản ứng ESM-2 (Chế độ chờ)...")
# Tương lai:
# esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
# esm_model = EsmForSequenceClassification.from_pretrained("du_an_CTU_cua_ong/esm2_finetuned")

def run_esm2_reactor(full_sequence: str, task: str):
    """
    Hàm mô phỏng ESM-2. 
    task: 'verify_amp' (check lại oan sai) hoặc 'toxicity' (đo độc tính)
    """
    # Logic giả lập ESM-2 để test luồng trước khi lắp não thật:
    if task == 'verify_amp':
        # Giả vờ ESM-2 đọc từ đầu đến đuôi và thấy tổ hợp kỵ nước mạnh (Cysteine hoặc Phenylalanine/Leucine)
        if 'CC' in full_sequence or 'FFF' in full_sequence or 'LLL' in full_sequence:
            return {"status": "AMP", "prob": 0.88, "msg": "ESM-2 confirmed: The sequence was wrongfully truncated, this is a true AMP!"}
        return {"status": "NON-AMP", "prob": 0.12, "msg": "ESM-2 confirmed: Read the entire sequence, it is indeed NON-AMP."}
    
    elif task == 'toxicity':
        # Cứ là AMP thì giả vờ đo độc tính
        return {"is_toxic": False, "prob": 0.05, "msg": "Safe! Does not destroy human red blood cells (Non-toxic)."}

class PeptideData(BaseModel):
    sequence: str

@app.post("/predict/all")
def predict_all(data: PeptideData):
    original_seq = data.sequence.upper()
    is_long = len(original_seq) > 50
    
    # 1. mCNN chỉ nhai được 50 ký tự cuối
    seq_for_mcnn = original_seq[-50:] if is_long else original_seq
        
    seq_seq = tokenizer.texts_to_sequences([seq_for_mcnn])
    seq_padded = keras.utils.pad_sequences(seq_seq, maxlen=50, padding='post', truncating='pre')
    
    # 2. Gatekeeper phán xử
    prob_mcnn = float(mcnn_model.predict(seq_padded, verbose=0)[0][0])
    mcnn_verdict = "AMP" if prob_mcnn > 0.5 else "NON-AMP"
    
    # 3. ĐỊNH TUYẾN ĐỘNG (CASCADE ROUTING) - Ăn tiền là khúc này!
    final_status = mcnn_verdict
    routing_status = "MCNN_ONLY"
    esm2_amp_result = None
    esm2_tox_result = None

    if mcnn_verdict == "NON-AMP" and is_long:
        # Bị cắt đầu mà bị phán là rác -> Kích hoạt ESM-2 vớt hàng!
        routing_status = "ESM_VERIFIED"
        esm2_amp_result = run_esm2_reactor(original_seq, task='verify_amp')
        final_status = esm2_amp_result["status"] # Ghi đè phán quyết cuối cùng
    
    # Nếu kết quả cuối cùng nó là AMP (do mCNN phán chuẩn, hoặc do ESM-2 vớt lại thành công)
    if final_status == "AMP":
        # Ném sang đo độc tính
        esm2_tox_result = run_esm2_reactor(original_seq, task='toxicity')

    # 4. Tính toán XAI cho mCNN (Chỉ áp dụng trên 50 ký tự cuối)
    importances = []
    actual_len = min(len(seq_for_mcnn), 50)
    for i in range(actual_len):
        masked_seq = seq_padded.copy()
        masked_seq[0][i] = 0
        masked_prob = float(mcnn_model.predict(masked_seq, verbose=0)[0][0])
        importances.append(abs(prob_mcnn - masked_prob))
    
    max_imp = max(importances) if importances and max(importances) > 0 else 1.0
    xai_scores = [imp / max_imp for imp in importances]

    # Trả về nguyên 1 cục data đầy đủ logic rẽ nhánh cho Frontend tha hồ xào nấu
    return {
        "original_sequence": original_seq,
        "processed_sequence": seq_for_mcnn,
        "is_long_sequence": is_long,
        "mcnn": {
            "prob": prob_mcnn, 
            "verdict": mcnn_verdict
        },
        "routing_status": routing_status,
        "final_amp_status": final_status,
        "esm2_amp_data": esm2_amp_result,
        "esm2_toxicity_data": esm2_tox_result,
        "xai_scores": xai_scores
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)