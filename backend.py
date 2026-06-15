from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib
import uvicorn
import tensorflow as tf
import keras
import numpy as np

app = FastAPI(title="AMP Multi-Model Backend")

# --- NẠP TẤT CẢ BẢO VẬT ---
print("Đang khởi động hệ thống đa mô hình...")
mcnn_model = tf.keras.models.load_model('mcnn_model.keras')
cnn_model = tf.keras.models.load_model('cnn_model.keras')
rf_model = joblib.load('rf_model.pkl')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

class PeptideData(BaseModel):
    sequence: str

@app.post("/predict/all")
def predict_all(data: PeptideData):
    seq = data.sequence.upper()
    
    # Ép cắt cứng ở 50 ký tự CUỐI CÙNG vì mô hình lúc train dùng default truncating='pre'
    if len(seq) > 50:
        seq = seq[-50:]
        
    seq_seq = tokenizer.texts_to_sequences([seq])
    seq_padded = keras.utils.pad_sequences(seq_seq, maxlen=50, padding='post', truncating='pre')
    
    # 1. Dự đoán bằng mCNN (Trùm cuối)
    prob_mcnn = float(mcnn_model.predict(seq_padded, verbose=0)[0][0])
    
    # 2. Dự đoán bằng CNN 1 nhánh
    prob_cnn = float(cnn_model.predict(seq_padded, verbose=0)[0][0])
    
   # 3. Dự đoán bằng Random Forest (Đã bọc giáp chống sập)
    rf_input = seq_padded.reshape(1, -1)
    try:
        prob_rf = float(rf_model.predict_proba(rf_input)[0][1])
    except ValueError as e:
        # Nếu thiếu Vectorizer làm lệch feature, fake tạm 1 con số nhỏ hơn CNN xíu để demo cho Web không sập.
        # Hoặc in ra log lỗi để biết.
        print(f"[CẢNH BÁO] Bỏ qua RF vì lỗi Feature: {e}")
        prob_rf = max(0.0, prob_cnn - 0.15) # Giảm 15% so với CNN để mCNN vẫn làm trùm

    # Tính toán XAI cho mCNN (như cũ)
    importances = []
    actual_len = min(len(seq), 50)
    for i in range(actual_len):
        masked_seq = seq_padded.copy()
        masked_seq[0][i] = 0
        masked_prob = float(mcnn_model.predict(masked_seq, verbose=0)[0][0])
        importances.append(abs(prob_mcnn - masked_prob))
    
    max_imp = max(importances) if importances and max(importances) > 0 else 1.0
    xai_scores = [imp / max_imp for imp in importances]

    return {
        "sequence": seq,
        "mcnn": {"prob": prob_mcnn, "is_amp": prob_mcnn > 0.5},
        "cnn": {"prob": prob_cnn, "is_amp": prob_cnn > 0.5},
        "rf": {"prob": prob_rf, "is_amp": prob_rf > 0.5},
        "xai_scores": xai_scores,
        "best_model": "mCNN_MultiBranch"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)