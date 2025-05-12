# OpenAI APIと数値計算、環境変数の読み込み用ライブラリをインポート
import openai
import numpy as np
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# .envに定義されたOpenAIのAPIキーを取得
api_key = os.getenv("OPENAI_API_KEY")

# OpenAIクライアントを初期化（APIキーを指定）
client = openai.OpenAI(api_key=api_key)

# 2つのベクトル間のコサイン類似度を計算する関数
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 2つのテキストの意味的な類似度を計算する関数
def word_similarity(text1, text2, model="text-embedding-ada-002"):
    # OpenAI APIで2つのテキストを埋め込みベクトル（1536次元）に変換
    response = client.embeddings.create(
        input=[text1, text2],
        model=model
    )
    # 1つ目のテキストのベクトルを取得
    vec1 = response.data[0].embedding
    # 2つ目のテキストのベクトルを取得
    vec2 = response.data[1].embedding

    # 埋め込みベクトルの中身を表示（全文）
    print(f"\nEmbedding for \"{text1}\" ({len(vec1)} dims):\n{vec1}")
    print(f"\nEmbedding for \"{text2}\" ({len(vec2)} dims):\n{vec2}")

    # コサイン類似度を返す
    return cosine_similarity(vec1, vec2)

# 比較したい2つの文を定義
sentence1 = "He showed me the way to the station."
sentence2 = "She forgot to buy milk."

# 文同士の意味的な類似度を計算
similarity = word_similarity(sentence1, sentence2)

# 類似度スコアを小数点以下4桁で表示
print(f"\nSimilarity: {similarity:.4f}")
