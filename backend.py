from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn
import tensorflow as tf
import keras

# Khởi tạo API
app = FastAPI(title="AMP Backend API")

# Nạp "Bảo vật"
print("Đang khởi động lò phản ứng mCNN...")
mcnn_model = tf.keras.models.load_model('mcnn_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

class PeptideData(BaseModel):
    sequence: str

@app.post("/predict/mcnn")
def predict_amp(data: PeptideData):
    seq = data.sequence.upper()
    seq_seq = tokenizer.texts_to_sequences([seq])
    seq_padded = keras.utils.pad_sequences(seq_seq, maxlen=50, padding='post')
    
    # 1. Dự đoán bản gốc
    original_prob = float(mcnn_model.predict(seq_padded, verbose=0)[0][0])
    is_amp = bool(original_prob > 0.5)
    confidence = original_prob if is_amp else (1 - original_prob)
    
    # ==========================================
    # 2. XAI MODULE: OCCLUSION SENSITIVITY
    # ==========================================
    importances = []
    # Chỉ xét những ký tự có thật trong chuỗi (không tính padding)
    actual_len = min(len(seq), 50) 
    
    for i in range(actual_len):
        # Tạo 1 bản sao bị "che mắt" (Biến chữ thứ i thành số 0)
        masked_seq = seq_padded.copy()
        masked_seq[0][i] = 0 
        
        # Bắt model đoán lại
        masked_prob = float(mcnn_model.predict(masked_seq, verbose=0)[0][0])
        
        # Nếu đang là AMP (original_prob cao), che đi mà prob tụt -> Chữ quan trọng!
        # Trị tuyệt đối để lấy mức độ ảnh hưởng
        drop = abs(original_prob - masked_prob)
        importances.append(drop)
    
    # Chuẩn hóa điểm quan trọng về thang 0.0 -> 1.0 để làm màu HTML
    max_imp = max(importances) if importances and max(importances) > 0 else 1.0
    normalized_importances = [imp / max_imp for imp in importances]

    return {
        "sequence": seq,
        "is_amp": is_amp,
        "probability": original_prob,
        "confidence": f"{confidence * 100:.2f}%",
        "model_used": "mCNN_MultiBranch",
        "xai_scores": normalized_importances # Gửi mảng điểm lên mặt tiền
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    