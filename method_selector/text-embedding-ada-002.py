import openai
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

user_sentence = (
    "35歳の女性医師。家族と暮らしており、朝型の生活スタイルでリモートワーク中。"
    "現在は秋の金曜日の朝6時、静かなオフィスにいる。"
    "曇りの日で、気温22度、湿度58%、騒音レベル55dB。"
    "カフェが好きで、瞑想をリラックス手段とし、タイ料理と映画が好み。"
)

methods = [
    "ルート最適化を行う",
    "状況に応じたスポット提案を行う",
    "食・休憩・買い物の提案を行う",
    "イベント・体験・撮影に関する提案を行う",
    "安全・緊急対策を行う",
    "何もしない"
]

user_vec = get_embedding(user_sentence)
method_vecs = [get_embedding(m) for m in methods]

scores = [cosine_similarity(user_vec, vec) for vec in method_vecs]
best_method = methods[np.argmax(scores)]

print("最も適した抽象メソッド:", best_method)
for m, s in sorted(zip(methods, scores), key=lambda x: x[1], reverse=True):
    print(f"{m}：{s:.4f}")
