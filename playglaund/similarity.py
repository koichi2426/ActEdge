import openai
import numpy as np
import os
from dotenv import load_dotenv

# .envからAPIキーを読み込み
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# コサイン類似度を計算
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 入力された2つの文/単語の意味的類似度を算出
def word_similarity(text1, text2, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=[text1, text2],
        model=model
    )
    vec1 = response.data[0].embedding
    vec2 = response.data[1].embedding

    print(f"\nEmbedding for \"{text1}\" ({len(vec1)} dims):\n{vec1}")
    print(f"\nEmbedding for \"{text2}\" ({len(vec2)} dims):\n{vec2}")

    return cosine_similarity(vec1, vec2)

# --- コマンドライン入力 ---
sentence1 = input("1つ目の文または単語を入力してください： ")
sentence2 = input("2つ目の文または単語を入力してください： ")

# 類似度を計算して表示
similarity = word_similarity(sentence1, sentence2)
print(f"\nSimilarity: {similarity:.4f}")
