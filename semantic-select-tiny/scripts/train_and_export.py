# ============================================
# このスクリプトは以下を実行します：
# 1. TinyBERTのTriplet Lossファインチューニング
# 2. SBERT構造でONNXエクスポート（FP32）
# 3. INT8量子化を実行
# ============================================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import numpy as np
import random
from tqdm import tqdm

# ==========================
# 設定
# ==========================
MODEL_NAME = "prajjwal1/bert-tiny"
MODEL_DIR = "models/bert-tiny"
ONNX_EXPORT_PATH = "models/tinybert_sbert.onnx"
ONNX_INT8_EXPORT_PATH = "models/tinybert_sbert_int8.onnx"
TRAIN_DATA_PATH = "data/train_triplets.txt"
MAX_LENGTH = 32
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# SBERT構造（mean pooling）
# ==========================
class SBERTEncoder(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1)
        mean_pooled = summed / counts
        return mean_pooled

# ==========================
# Tripletデータセット
# ==========================
class TripletDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                anchor, pos, neg = line.strip().split("\t")
                self.samples.append((anchor, pos, neg))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a, p, n = self.samples[idx]
        return self.tokenizer([a, p, n], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)

# ==========================
# Triplet Loss
# ==========================
def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = (anchor - positive).pow(2).sum(1)
    d_an = (anchor - negative).pow(2).sum(1)
    return torch.relu(d_ap - d_an + margin).mean()

# ==========================
# ファインチューニング
# ==========================
def train():
    print("\n[1] モデルとトークナイザーの読み込み...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    model = SBERTEncoder(base_model).to(DEVICE)

    dataset = TripletDataset(TRAIN_DATA_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("[2] ファインチューニング開始...")
    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].squeeze(1).to(DEVICE)  # [batch, 3, len]
            attention_mask = batch["attention_mask"].squeeze(1).to(DEVICE)

            a, p, n = input_ids[:,0,:], input_ids[:,1,:], input_ids[:,2,:]
            am, pm, nm = attention_mask[:,0,:], attention_mask[:,1,:], attention_mask[:,2,:]

            va = model(a, am)
            vp = model(p, pm)
            vn = model(n, nm)

            loss = triplet_loss(va, vp, vn)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print(f"  Epoch {epoch+1} / {EPOCHS}  Loss: {np.mean(losses):.4f}")

    print(f"\n✅ ファインチューニング完了 → {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    base_model.save_pretrained(MODEL_DIR)
    return tokenizer, model

# ==========================
# ONNXエクスポート
# ==========================
def export_onnx(model, tokenizer):
    print("\n[3] ONNXエクスポート...")
    model.eval()
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, MAX_LENGTH), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, MAX_LENGTH), dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        ONNX_EXPORT_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["pooled_output"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
        opset_version=13
    )
    print(f"✅ ONNXファイル出力 → {ONNX_EXPORT_PATH}")

# ==========================
# キャリブレーション用ダミーデータ
# ==========================
class DummyCalibReader(CalibrationDataReader):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.datas = [
            tokenizer("30代男性。朝。オフィスにいる。", return_tensors="np", padding="max_length", truncation=True, max_length=MAX_LENGTH)
        ]
        self.index = 0

    def get_next(self):
        if self.index < len(self.datas):
            data = self.datas[self.index]
            self.index += 1
            return {
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"]
            }
        return None

# ==========================
# 量子化処理
# ==========================
def quantize_model(tokenizer):
    print("\n[4] INT8量子化を実行中...")
    quantize_static(
        model_input=ONNX_EXPORT_PATH,
        model_output=ONNX_INT8_EXPORT_PATH,
        calibration_data_reader=DummyCalibReader(tokenizer),
        quant_format=QuantType.QUInt8
    )
    print(f"✅ INT8量子化完了 → {ONNX_INT8_EXPORT_PATH}\n")

# ==========================
# 実行
# ==========================
if __name__ == "__main__":
    tokenizer, model = train()
    export_onnx(model, tokenizer)
    quantize_model(tokenizer)
