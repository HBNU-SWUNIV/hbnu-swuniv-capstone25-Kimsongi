import os
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


MODEL_NAME = "paust/pko-t5-base"
DATASET_PATH = "./drive/MyDrive/2025캡스톤프로젝트/train_dataset.json"

SAVE_DIR = "./drive/MyDrive/2025캡스톤프로젝트"
SAVE_PATH = os.path.join(SAVE_DIR, "my_finetuned_t5_model")
LOG_DIR = "./logs"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("▶ 모델/토크나이저 로드")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

model.config.dropout_rate = 0.1
model.config.attention_dropout_rate = 0.1


print("▶ 데이터 로드:", DATASET_PATH)
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)


np.random.shuffle(raw_data)
split_point = int(len(raw_data) * 0.9)
train_data = raw_data[:split_point]
eval_data = raw_data[split_point:]

print(f"훈련 데이터: {len(train_data)}개, 검증 데이터: {len(eval_data)}개")

# --- Train/Eval 입력 중복 점검 ---
train_inputs = set(ex["input"] for ex in train_data)
eval_inputs = set(ex["input"] for ex in eval_data)
overlap = sorted(list(train_inputs & eval_inputs))
print(f"[CHECK] Train/Eval 중복 입력 개수: {len(overlap)}")
if len(overlap) > 0:
    print("  예시 중복 입력(최대 10개):", overlap[:10])

# 필요 시: 중복 제거
eval_data = [ex for ex in eval_data if ex["input"] not in train_inputs]
print(f"[FIX] 중복 제거 후 검증 데이터: {len(eval_data)}개")

train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)


def preprocess_function(examples):
    # 입력
    model_inputs = tokenizer(
        examples["input"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    # 타깃 (새 API: text_target=)
    labels = tokenizer(
        text_target=examples["output"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    # pad 토큰을 -100으로 마스킹 (loss 제외)
    pad_id = tokenizer.pad_token_id
    masked_labels = []
    for seq in labels["input_ids"]:
        masked_labels.append([tid if tid != pad_id else -100 for tid in seq])
    model_inputs["labels"] = masked_labels
    return model_inputs

print("▶ 전처리 시작")
tokenized_train_dataset = train_dataset.map(
    preprocess_function, batched=True, remove_columns=train_dataset.column_names
)
tokenized_eval_dataset = eval_dataset.map(
    preprocess_function, batched=True, remove_columns=eval_dataset.column_names
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


class LossRecorder(TrainerCallback):
    def __init__(self):
        # (epoch_float, loss)
        self.train_logs = []
        self.eval_logs = []

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        if "loss" in logs and state.epoch is not None:
            # step-단위 train loss
            self.train_logs.append((float(state.epoch), float(logs["loss"])))

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics and state.epoch is not None:
            # epoch-단위 eval loss
            self.eval_logs.append((float(state.epoch), float(metrics["eval_loss"])))

    def get_epochwise_losses(self, agg="mean"):
        # 에폭별 train loss 집계 & eval loss 매핑
        bucket = defaultdict(list)
        for ep, l in self.train_logs:
            bucket[math.floor(ep)].append(l)

        train_epoch_loss = {}
        for ep in sorted(bucket.keys()):
            vals = bucket[ep]
            if agg == "last":
                train_epoch_loss[ep] = vals[-1]
            else:
                train_epoch_loss[ep] = sum(vals) / len(vals)

        eval_epoch_loss = {}
        for ep, l in self.eval_logs:
            eval_epoch_loss[math.floor(ep)] = l

        return train_epoch_loss, eval_epoch_loss

loss_recorder = LossRecorder()

@torch.no_grad()
def compute_exact_match(trainer: Trainer, eval_ds: Dataset, limit_samples: int = None, num_beams: int = 4):
    model.eval()
    dataloader = trainer.get_eval_dataloader(eval_ds)
    total, exact = 0, 0
    preds_all, refs_all = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=num_beams
        )
        pred_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # 라벨 복원(-100 -> pad_id) 후 디코드
        labels = batch["labels"].clone()
        labels = labels.cpu().numpy()
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        ref_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for p, r in zip(pred_text, ref_text):
            preds_all.append(p.strip())
            refs_all.append(r.strip())
            exact += int(p.strip() == r.strip())
            total += 1

        if limit_samples is not None and total >= limit_samples:
            break

    em = exact / max(1, total)
    print(f"[GEN-EVAL] Exact Match: {em:.3f}  ({exact}/{total})")
    return em, preds_all, refs_all

MAX_EPOCHS = 10
early_stopping_patience = 3
best_eval_loss = float("inf")
patience_counter = 0


training_args = TrainingArguments(
    output_dir=os.path.join(SAVE_DIR, "t5_finetune_results"),
    num_train_epochs=1,                     
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,           
    learning_rate=3e-5,                     
    weight_decay=0.01,                      
    logging_dir='./logs',
    logging_steps=50,                       
    save_total_limit=2,                     
    save_steps=500,                         
    prediction_loss_only=False           
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    callbacks=[loss_recorder],
)


print("▶ 수동 조기 종료 기능으로 모델 미세 조정을 시작합니다...")

for epoch in range(MAX_EPOCHS):
    print(f"--- Epoch {epoch + 1}/{MAX_EPOCHS} ---")

    trainer.train()

    eval_results = trainer.evaluate()
    current_eval_loss = float(eval_results["eval_loss"])
    print(f"Epoch {epoch + 1} - 검증 손실(Validation Loss): {current_eval_loss:.6f}")

    _ = compute_exact_match(trainer, tokenized_eval_dataset, limit_samples=None, num_beams=4)
    
    if current_eval_loss < best_eval_loss:
        best_eval_loss = current_eval_loss
        patience_counter = 0
        print(f"✅ 새로운 최고 성능 모델 발견! (eval_loss: {best_eval_loss:.6f})")
        print(f"   모델을 '{SAVE_PATH}'에 저장합니다...")
        trainer.save_model(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
    else:
        patience_counter += 1
        print(f"   성능 향상 없음. (Patience: {patience_counter}/{early_stopping_patience})")

    # 조기 종료
    if patience_counter >= early_stopping_patience:
        print(f"🔴 {early_stopping_patience}번 연속으로 성능이 향상되지 않아 학습을 조기 종료합니다.")
        break

print("✅ 모델 학습이 완료되었습니다!")
print(f"✅ 최종적으로 가장 성능이 좋았던 모델이 '{SAVE_PATH}' 폴더에 저장되었습니다.")