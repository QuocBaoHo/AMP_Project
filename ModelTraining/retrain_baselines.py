import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, concatenate, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle

print("🔥 BẮT ĐẦU CHIẾN DỊCH THANH TRỪNG DATA LEAKAGE...")

# 1. Nạp data siêu chuẩn (Chỉ lấy phần có nhãn AMP)
print("1. Đang nạp dữ liệu Train/Val...")
train_df = pd.read_csv("train_multitask.csv")
val_df = pd.read_csv("val_multitask.csv")

train_df = train_df[train_df['is_amp'] != -1]
val_df = val_df[val_df['is_amp'] != -1]

X_train_text = train_df['sequence'].tolist()
y_train = train_df['is_amp'].values
X_val_text = val_df['sequence'].tolist()
y_val = val_df['is_amp'].values

# --- TIỀN XỬ LÝ CHO DEEP LEARNING (Keras) ---
print("2. Đang Tokenize dữ liệu cho Keras (Giới hạn 50 ký tự cuối)...")
# Cắt 50 ký tự cuối đúng chuẩn luật Gatekeeper
X_train_cut = [seq[-50:] if len(seq) > 50 else seq for seq in X_train_text]
X_val_cut = [seq[-50:] if len(seq) > 50 else seq for seq in X_val_text]

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train_cut)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

X_train_seq = keras.utils.pad_sequences(tokenizer.texts_to_sequences(X_train_cut), maxlen=50, padding='post', truncating='pre')
X_val_seq = keras.utils.pad_sequences(tokenizer.texts_to_sequences(X_val_cut), maxlen=50, padding='post', truncating='pre')

vocab_size = len(tokenizer.word_index) + 1

# =========================================
# LÒ ĐÚC 1: STANDARD CNN
# =========================================
print("\n⚙️ Đang đúc lại Standard CNN...")
cnn_input = Input(shape=(50,))
x = Embedding(vocab_size, 64)(cnn_input)
x = Conv1D(64, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
cnn_out = Dense(1, activation='sigmoid')(x)

cnn_model = keras.Model(cnn_input, cnn_out)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), epochs=10, batch_size=128, verbose=1)
cnn_model.save('cnn_model.keras')

# =========================================
# LÒ ĐÚC 2: mCNN (MULTI-BRANCH GATEKEEPER)
# =========================================
print("\n🛡️ Đang đúc lại mCNN Gatekeeper...")
mcnn_input = Input(shape=(50,))
emb = Embedding(vocab_size, 64)(mcnn_input)

# 3 nhánh chập kích thước khác nhau (đọc motif sinh học)
branch_1 = Conv1D(64, 3, activation='relu', padding='same')(emb)
branch_1 = GlobalMaxPooling1D()(branch_1)

branch_2 = Conv1D(64, 5, activation='relu', padding='same')(emb)
branch_2 = GlobalMaxPooling1D()(branch_2)

branch_3 = Conv1D(64, 7, activation='relu', padding='same')(emb)
branch_3 = GlobalMaxPooling1D()(branch_3)

merged = concatenate([branch_1, branch_2, branch_3])
x = Dense(128, activation='relu')(merged)
x = Dropout(0.3)(x)
mcnn_out = Dense(1, activation='sigmoid')(x)

mcnn_model = keras.Model(mcnn_input, mcnn_out)
mcnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mcnn_model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), epochs=10, batch_size=128, verbose=1)
mcnn_model.save('mcnn_model.keras')

# =========================================
# LÒ ĐÚC 3: RANDOM FOREST (Machine Learning)
# =========================================
print("\n🌲 Đang trồng lại Random Forest...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=3000)
X_train_rf = vectorizer.fit_transform(X_train_text)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

rf_model = RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=42)
rf_model.fit(X_train_rf, y_train)
joblib.dump(rf_model, 'rf_model.pkl')

print("\n🎉 XONG! Tất cả vũ khí đã được rèn lại bằng Dataset chuẩn.")
print("Bây giờ ông có thể tự tin chạy file run_benchmark.py để xem kết quả đối đầu sòng phẳng!")