import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

print("🔥 KHỞI ĐỘNG LÒ PHẢN ỨNG ESM-2 (MULTI-TASK)...")

# 1. Load Model và Tokenizer
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# num_labels = 2 vì mình có 2 vòi ra: AMP và Toxicity
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, problem_type="multi_label_classification")

# 2. Đọc Dataset
train_df = pd.read_csv("train_multitask.csv")
val_df = pd.read_csv("val_multitask.csv")
test_df = pd.read_csv("test_multitask.csv")

# Chuyển đổi 2 cột nhãn thành 1 list [is_amp, is_toxic]
train_df["labels"] = train_df.apply(lambda row: [float(row["is_amp"]), float(row["is_toxic"])], axis=1)
val_df["labels"] = val_df.apply(lambda row: [float(row["is_amp"]), float(row["is_toxic"])], axis=1)
test_df["labels"] = test_df.apply(lambda row: [float(row["is_amp"]), float(row["is_toxic"])], axis=1)

train_ds = Dataset.from_pandas(train_df[['sequence', 'labels']])
val_ds = Dataset.from_pandas(val_df[['sequence', 'labels']])
test_ds = Dataset.from_pandas(test_df[['sequence', 'labels']])

# 3. Hàm Tokenize
def tokenize_function(examples):
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=128)

train_tokenized = train_ds.map(tokenize_function, batched=True)
val_tokenized = val_ds.map(tokenize_function, batched=True)
test_tokenized = test_ds.map(tokenize_function, batched=True)

# 4. KIẾN TRÚC SƯ: CUSTOM TRAINER (MASKED LOSS)
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Dùng BCE cho Multi-label
        loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        
        # Mask: Nếu nhãn là -1 thì mask = 0 (bỏ qua), ngược lại mask = 1 (học)
        active_loss = (labels != -1.0).float()
        
        # Tính Loss thô
        raw_loss = loss_fct(logits, labels.float())
        
        # Nhân với mask để triệt tiêu loss ở những chỗ -1
        masked_loss = raw_loss * active_loss
        
        # Tính trung bình loss dựa trên số lượng nhãn hợp lệ
        final_loss = masked_loss.sum() / active_loss.sum().clamp(min=1e-5)
        
        return (final_loss, outputs) if return_outputs else final_loss

# 5. Cấu hình Training (Ép xung RTX 3070)
training_args = TrainingArguments(
    output_dir="./esm2_multitask_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16, # Vừa vặn cho 8GB VRAM
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    fp16=True, # Bật mixed precision để chạy nhanh gấp đôi
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True
)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
)

# KÍCH HỎA!
print("🚀 BẮT ĐẦU HUẤN LUYỆN")
trainer.train()

# Đánh giá trên tập test
print("🧪 ĐÁNH GIÁ TRÊN TẬP TEST...")
test_results = trainer.evaluate(eval_dataset=test_tokenized)
print(f"Kết quả Test: {test_results}")

# Lưu thành quả
trainer.save_model("./esm2_multitask_final")
tokenizer.save_pretrained("./esm2_multitask_final")
print("🎉 ĐÃ ĐÚC XONG SIÊU NHÂN! Lưu tại ./esm2_multitask_final")