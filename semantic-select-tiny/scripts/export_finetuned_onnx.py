# ============================================
# このスクリプトは以下を実行します：
# 1. TinyBERTベースのSBERTエンコーダをONNX形式でエクスポート
# 2. 上記ONNXモデルをINT8形式で量子化（サイズ・速度最適化）
# ============================================

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import numpy as np

# ==========================
# 各種パスと定数設定
# ==========================
MODEL_PATH = "models/bert-tiny"                            # ファインチューニング済みモデルの保存先
ONNX_EXPORT_PATH = "models/tinybert_sbert.onnx"            # エクスポート先ONNXパス（FP32）
ONNX_INT8_EXPORT_PATH = "models/tinybert_sbert_int8.onnx"  # INT8量子化済みONNXモデル出力先
MAX_LENGTH = 32                                            # トークン長の上限

# ==========================
# SBERT構造（mean pooling）
# BERT出力を文ベクトルに集約するカスタムモジュール
# ==========================
class SBERTEncoder(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        masked = last_hidden * mask
        summed = masked.sum(1)
        counts = mask.sum(1)
        mean_pooled = summed / counts
        return mean_pooled

# ==========================
# INT8量子化用のキャリブレーションデータリーダー
# ダミー入力1件で代表値を提供
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
# ONNXエクスポート＋量子化の本体処理
# ==========================
def export_to_onnx():
    print("\n[1] モデルとトークナイザーを読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModel.from_pretrained(MODEL_PATH)
    model = SBERTEncoder(base_model).eval()

    print("[2] ダミー入力を用意してONNX形式でエクスポート...")
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
    print(f"\n✅ ONNXファイルを出力しました → {ONNX_EXPORT_PATH}")

    print("\n[3] INT8量子化を実行中...")
    quantize_static(
        model_input=ONNX_EXPORT_PATH,
        model_output=ONNX_INT8_EXPORT_PATH,
        calibration_data_reader=DummyCalibReader(tokenizer),
        quant_format=QuantType.QUInt8  # 安定動作する推奨設定
    )
    print(f"✅ INT8量子化完了 → {ONNX_INT8_EXPORT_PATH}\n")

# ==========================
# スクリプトエントリーポイント
# ==========================
if __name__ == "__main__":
    export_to_onnx()
