# ================================================
# このスクリプトは以下の処理を行います：
# 1. ユーザーの入力文を意味ベクトル化（ONNX推論）
# 2. 事前にベクトル化された抽象メソッド群とコサイン類似度を計算
# 3. 最も近いメソッドを1つ選択して返す
# ================================================

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import json
import os

# ==========================
# 設定（パス定義）
# ==========================
MODEL_PATH = "models/tinybert_sbert_int8.onnx"            # 量子化済みのONNXモデル
TOKENIZER_PATH = "models/bert-tiny"                        # トークナイザー保存先
METHODS_VECTORS_PATH = "data/method_vectors.npy"           # メソッドベクトル（NumPy形式）
METHODS_TEXTS_PATH = "data/method_texts.json"              # メソッド原文（表示用）

# ==========================
# 入力文 → 意味ベクトル（ONNX推論）
# ==========================
def get_embedding(text, tokenizer, session, max_length=32):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    embedding = session.run(["pooled_output"], ort_inputs)[0]
    return embedding[0]  # shape: [hidden_dim]

# ==========================
# コサイン類似度の計算
# ==========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ==========================
# 入力文と最も近いメソッドをすべて表示（類似度順）
# ==========================
def select_best_method(user_text):
    # [1] モデル・トークナイザー読み込み
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    session = ort.InferenceSession(MODEL_PATH)

    # [2] 入力文をベクトル化
    user_vec = get_embedding(user_text, tokenizer, session)

    # [3] 事前保存されたメソッド群のベクトル・テキストを読み込み
    method_vecs = np.load(METHODS_VECTORS_PATH)
    with open(METHODS_TEXTS_PATH, "r", encoding="utf-8") as f:
        method_texts = json.load(f)

    # [4] 各メソッドとの類似度を計算
    sims = [cosine_similarity(user_vec, v) for v in method_vecs]
    
    # [5] すべてのインデックスを類似度順に取得
    sorted_indices = np.argsort(sims)[::-1]
    
    # [6] 結果を整形
    results = []
    for idx in sorted_indices:
        results.append({
            'method': method_texts[idx],
            'score': float(sims[idx])  # NumPyのfloat32をPythonのfloatに変換
        })
    
    return results

# ==========================
# 実行テスト用（CLIなど）
# ==========================
if __name__ == "__main__":
    user_input = "地震が発生した。"
    results = select_best_method(user_input)

    print("\n[ユーザー入力]", user_input)
    print("\n[類似度スコア順メソッド一覧]")
    for i, result in enumerate(results, 1):
        print(f"{i}. スコア: {round(result['score'], 4)}")
        print(f"   メソッド: {result['method']}")
        print()
