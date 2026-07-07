import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

print("🚀 KÍCH HOẠT QUY TRÌNH DATA FUSION (BẢN CHUẨN SOTA 2024)...")

# --- HÀM HỖ TRỢ ĐỌC VÀ LÀM SẠCH ---
def load_and_clean(file_path):
    df = pd.read_csv(file_path)
    # Đổi tên cột cho đồng nhất (bao thầu mọi chuẩn của lão Salem)
    if 'text' in df.columns:
        df = df.rename(columns={'text': 'sequence'})
    elif 'Sequence' in df.columns:
        df = df.rename(columns={'Sequence': 'sequence'})
        
    if 'labels' in df.columns:
        df = df.rename(columns={'labels': 'label'})
    elif 'y_model_2cl' in df.columns:
        df = df.rename(columns={'y_model_2cl': 'label'})
        
    # Tiêu diệt tiền tố b'...' do pandas đọc nhầm chuỗi byte
    df['sequence'] = df['sequence'].astype(str).str.replace(r"^b'|'$", "", regex=True)
    
    # Tiêu diệt khoảng trắng trong sequence
    df['sequence'] = df['sequence'].str.replace(" ", "")
    return df[['sequence', 'label']]

# --- BƯỚC 1: XỬ LÝ 5 TẬP DỮ LIỆU CỐT LÕI (SẠCH & NGUYÊN CHẤT) ---
print("1. Đang nạp 5 tập dataset Local (HemoPI 1-2-3, RNN-Hem, HLPpred-Fuse)...")

df_hemo1 = load_and_clean('HemoPI1.csv')
df_hemo2 = load_and_clean('HemoPI2.csv')
df_hemo3 = load_and_clean('HemoPI3.csv')
df_rnn = load_and_clean('rnnamp.csv')
df_hlp = load_and_clean('hlppredfuse.csv')

# --- BƯỚC 2: GẮN NHÃN "PHONG THẦN" (THE GOLDEN LABELING) ---
print("2. Đang thực thi Mapping Nhãn Tuyệt Mật...")
master_records = []

def apply_golden_labels(df, is_amp_neg, is_toxic_neg):
    for _, row in df.iterrows():
        seq = row['sequence']
        label = row['label']
        if label == 1:
            # Toàn bộ Positive của 5 file đều là AMP Kịch Độc
            master_records.append({'sequence': seq, 'is_amp': 1, 'is_toxic': 1})
        else:
            # Phân loại Negative cực gắt theo bản chất Dataset
            master_records.append({'sequence': seq, 'is_amp': is_amp_neg, 'is_toxic': is_toxic_neg})

# 2.1 - HemoPI-2 & HemoPI-3 (Negative là AMP An toàn -> Chén Thánh)
apply_golden_labels(df_hemo2, is_amp_neg=1, is_toxic_neg=0)
apply_golden_labels(df_hemo3, is_amp_neg=1, is_toxic_neg=0)

# 2.2 - HemoPI-1, rnnamp, hlppredfuse (Negative là Protein dân đen vô hại)
apply_golden_labels(df_hemo1, is_amp_neg=0, is_toxic_neg=0)
apply_golden_labels(df_rnn, is_amp_neg=0, is_toxic_neg=0)
apply_golden_labels(df_hlp, is_amp_neg=0, is_toxic_neg=0)

df_core = pd.DataFrame(master_records)

# --- BƯỚC 3: DUNG HỢP VỚI ĐẠI QUÂN VIRI ---
print("3. Đang triệu hồi dataset ps29/amp_dataset_viri từ Hugging Face...")
try:
    viri_ds = load_dataset("ps29/amp_dataset_viri", split="viri_train")
    df_viri = viri_ds.to_pandas()
    df_viri = df_viri[['seq', 'label']].rename(columns={'seq': 'sequence', 'label': 'is_amp'})
    
    # Bịt mắt ESM-2 ở phần Độc tính với tập Viri
    df_viri['is_toxic'] = -1
except Exception as e:
    print(f"Lỗi tải Viri: {e}")
    df_viri = pd.DataFrame(columns=['sequence', 'is_amp', 'is_toxic'])

df_master = pd.concat([df_core, df_viri], ignore_index=True)

# --- BƯỚC 4: THUẬT TOÁN ƯU TIÊN GIỮ NHÃN XỊN (DEDUPLICATION) ---
print("4. Đang dọn dẹp trùng lặp bằng Haki Bá Vương (Priority Deduplication)...")

# Xác định cấp độ VIP của từng loại nhãn
def get_priority(row):
    if row['is_amp'] == 1 and row['is_toxic'] == 0: return 1 # VIP 1: Chén Thánh (Cấm xóa)
    if row['is_amp'] == 1 and row['is_toxic'] == 1: return 2 # VIP 2: Hàng Kịch Độc
    if row['is_amp'] == 0 and row['is_toxic'] == 0: return 3 # VIP 3: Dân Đen
    return 4                                                 # Hàng chót: Mask -1

df_master['priority'] = df_master.apply(get_priority, axis=1)
# Xếp hạng theo độ VIP và chém sạch các chuỗi trùng lặp, chỉ giữ lại thằng VIP cao nhất
df_master = df_master.sort_values('priority', ascending=True)
df_master = df_master.drop_duplicates(subset=['sequence'], keep='first')

# Dọn dẹp cột priority
df_master = df_master[['sequence', 'is_amp', 'is_toxic']]

# --- BƯỚC 5: CHIA LÔ (TRAIN/VAL/TEST) ---
print("5. Đang băm nhỏ dữ liệu (80-10-10)...")
train_df, temp_df = train_test_split(df_master, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_csv("train_multitask.csv", index=False)
val_df.to_csv("val_multitask.csv", index=False)
test_df.to_csv("test_multitask.csv", index=False)

print(f"✅ HOÀN TẤT! Tổng dữ liệu thực tế (Không trùng lặp): {len(df_master)} chuỗi.")
print(f"📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("💻 Mời chủ tịch quăng vào RTX 3070 và đi ngủ!")