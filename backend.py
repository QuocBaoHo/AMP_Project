from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib
import uvicorn
import tensorflow as tf
import keras
import numpy as np

# --- MỞ KHÓA VŨ KHÍ HẠNG NẶNG ---
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="AMP & Toxicity Cascade Pipeline")

# --- NẠP BẢO VẬT GIAI ĐOẠN 1 (Gatekeeper) ---
print("Đang khởi động STAGE 1: mCNN Gatekeeper...")
mcnn_model = tf.keras.models.load_model('mcnn_model.keras')
cnn_model = tf.keras.models.load_model('cnn_model.keras')
rf_model = joblib.load('rf_model.pkl')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# --- NẠP BẢO VẬT GIAI ĐOẠN 2 (Trùm cuối ESM-2 THẬT) ---
print("🚀 Đang khởi động Lò phản ứng ESM-2 (Hàng Real 99.6%)...")
esm2_path = "./esm2_multitask_final" # Đổi sang model multitask có đo độc tính
esm2_tokenizer = AutoTokenizer.from_pretrained(esm2_path)
esm2_model = AutoModelForSequenceClassification.from_pretrained(esm2_path, attn_implementation="eager")
esm2_model.eval() # Bật chế độ chạy thực tế, không train nữa

def run_esm2_reactor(full_sequence: str, task: str):
    """
    Hàm vận hành não ESM-2 thật. 
    task: 'verify_amp' (check lại oan sai) hoặc 'toxicity' (đo độc tính)
    """
    # Ép chuỗi vào khuôn của ESM-2
    inputs = esm2_tokenizer(full_sequence, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        # Bật output_attentions để lấy XAI
        outputs = esm2_model(**inputs, output_attentions=True)
        logits = outputs.logits
        probs = torch.sigmoid(logits)[0] # Multitask dùng sigmoid
        attentions = outputs.attentions
    
    # Lấy xác suất: vị trí 0 là AMP, vị trí 1 là Toxicity
    amp_prob = probs[0].item()
    tox_prob = probs[1].item()
    
    result = {}
    if task == 'verify_amp':
        if amp_prob > 0.5:
            result = {"status": "AMP", "prob": round(amp_prob, 3), "msg": "ESM-2 xác nhận: Chuỗi bị cắt oan, đây là AMP xịn!"}
        else:
            result = {"status": "NON-AMP", "prob": round(1-amp_prob, 3), "msg": "ESM-2 xác nhận: Đã đọc toàn bộ chuỗi gốc, đúng là rác!"}
    
    elif task == 'toxicity':
        # Logic đo độc: Tận dụng xác suất Toxicity từ label thứ 2
        is_toxic = tox_prob > 0.5 
        result = {
            "is_toxic": is_toxic, 
            "prob": round(tox_prob, 3), 
            "msg": "Cảnh báo: Có khả năng gây độc!" if is_toxic else "An toàn! Không phá hủy hồng cầu người (Non-toxic)."
        }
        
    # XAI Độc lập bằng phương pháp Occlusion (Che lấp) để phân biệt rõ AMP và Toxicity
    input_ids = inputs['input_ids'][0]
    seq_len = len(input_ids)
    actual_len = seq_len - 2 # Bỏ <s> và </s>
    
    importances = []
    mask_token_id = esm2_tokenizer.mask_token_id if esm2_tokenizer.mask_token_id is not None else esm2_tokenizer.pad_token_id
    
    for i in range(1, seq_len - 1):
        masked_input_ids = input_ids.clone()
        masked_input_ids[i] = mask_token_id
        
        masked_inputs = {'input_ids': masked_input_ids.unsqueeze(0), 'attention_mask': inputs['attention_mask']}
        with torch.no_grad():
            masked_logits = esm2_model(**masked_inputs).logits
            masked_probs = torch.sigmoid(masked_logits)[0]
            
        if task == 'verify_amp':
            # Importance = Khả năng AMP giảm bao nhiêu khi mất axit amin này
            drop = amp_prob - masked_probs[0].item()
            importances.append(max(0, drop))
        else:
            # Importance = Khả năng Độc tính giảm bao nhiêu khi mất axit amin này
            drop = tox_prob - masked_probs[1].item()
            importances.append(max(0, drop))
            
    max_imp = max(importances) if importances and max(importances) > 0 else 1.0
    xai_scores = [imp / max_imp for imp in importances]
    
    result["xai_scores"] = xai_scores
    result["processed_sequence"] = full_sequence[:actual_len]
    
    return result

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
    
    # 3. ĐỊNH TUYẾN ĐỘNG (CASCADE ROUTING) - Bản nâng cấp Threshold!
    final_status = mcnn_verdict
    routing_status = "MCNN_ONLY"
    esm2_amp_result = None
    esm2_tox_result = None

    # Tính độ tự tin (Confidence) của mCNN
    # Nếu phán AMP thì lấy prob, nếu phán NON-AMP thì lấy 1 - prob
    mcnn_confidence = prob_mcnn if mcnn_verdict == "AMP" else (1 - prob_mcnn)

    # ĐIỀU KIỆN KÍCH HOẠT ESM-2 (FAILSAFE):
    # Nếu mCNN dự đoán là AMP với độ tự tin >= 0.85, bỏ qua failsafe (kể cả chuỗi dài/cắt gọt vì đã chắc kèo).
    # Còn lại: đẩy qua ESM-2 nếu độ tự tin thấp (< 0.85) hoặc chuỗi bị cắt gọt (is_long).
    is_confident_amp = (mcnn_verdict == "AMP" and mcnn_confidence >= 0.85)
    
    if (is_long or mcnn_confidence < 0.85) and not is_confident_amp:
        
        routing_status = "ESM_VERIFIED"
        esm2_amp_result = run_esm2_reactor(original_seq, task='verify_amp')
        
        # Sửa lại câu thông báo
        if not is_long:
            if mcnn_verdict == "NON-AMP":
                esm2_amp_result["msg"] = "ESM-2 xác nhận: mCNN dự đoán sai do độ tự tin thấp, đây là AMP xịn!"
            else:
                esm2_amp_result["msg"] = "ESM-2 xác nhận: mCNN đoán đúng, nhưng tự tin thấp quá. Đã kiểm chứng lại bằng ESM-2!"
            
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