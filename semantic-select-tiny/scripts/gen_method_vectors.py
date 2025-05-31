# ============================================
# このスクリプトは以下を実行します：
# 1. methods.txt に書かれた各抽象メソッドを読み込む
# 2. ONNXモデルを使って各メソッド文をベクトル化
# 3. ベクトルを .npy 形式で保存（高速検索用）
# 4. 元テキストを .json 形式で保存（表示・検索用）
# ============================================

import os
import json
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

# ==========================
# パスおよび設定定数
# ==========================
MODEL_PATH = "models/tinybert_sbert_int8.onnx"         # INT8量子化済みONNXモデルのパス
TOKENIZER_PATH = "models/bert-tiny"                    # トークナイザーのローカル保存パス
METHODS_TEXT_PATH = "data/methods.txt"                 # 入力メソッドテキストファイル
OUTPUT_VEC_PATH = "data/method_vectors.npy"            # 出力ベクトル（NumPy形式）
OUTPUT_TEXT_JSON_PATH = "data/method_texts.json"       # メソッド原文保存ファイル
MAX_LENGTH = 32                                        # 最大トークン長

# ==========================
# モデル・トークナイザー読み込み
# ==========================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
session = ort.InferenceSession(MODEL_PATH)

# ==========================
# 単一文をONNXモデルでベクトル化
# ==========================
def encode(text):
    inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=MAX_LENGTH)
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    return session.run(["pooled_output"], ort_inputs)[0][0]

# ==========================
# メソッド一覧をベクトル化し保存
# ==========================
def main():
    os.makedirs("data", exist_ok=True)

    # メソッド文の読み込み
    with open(METHODS_TEXT_PATH, encoding="utf-8") as f:
        method_lines = [line.strip() for line in f if line.strip()]

    print(f"[1] {len(method_lines)}件の抽象メソッドをベクトル化中...")
    vectors = np.array([encode(line) for line in method_lines])

    # ベクトルを保存
    np.save(OUTPUT_VEC_PATH, vectors)
    print(f"[2] ベクトルを保存しました → {OUTPUT_VEC_PATH}")

    # 元テキストをJSONで保存（選択肢表示などに使う）
    with open(OUTPUT_TEXT_JSON_PATH, "w", encoding="utf-8") as jf:
        json.dump(method_lines, jf, ensure_ascii=False, indent=2)
    print(f"[3] メソッド原文を保存しました → {OUTPUT_TEXT_JSON_PATH}")

# ==========================
# 実行エントリーポイント
# ==========================
if __name__ == "__main__":
    main()
