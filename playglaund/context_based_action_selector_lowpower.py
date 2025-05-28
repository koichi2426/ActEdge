from sentence_transformers import SentenceTransformer, util
import numpy as np

# Step 1: ユーザー状態 → 自然文に変換
user_sentence = (
    "35歳の女性医師。家族と暮らしており、朝型の生活スタイルでリモートワーク中。"
    "現在は秋の金曜日の朝6時、静かなオフィスにいる。"
    "曇りの日で、気温22度、湿度58%、騒音レベル55dB。"
    "カフェが好きで、瞑想をリラックス手段とし、タイ料理と映画が好み。"
)

# Step 2: 候補メソッド（事前登録）
methods = [
    "ルート最適化を行う",
    "状況に応じたスポット提案を行う",
    "食・休憩・買い物の提案を行う",
    "イベント・体験・撮影に関する提案を行う",
    "安全・緊急対策を行う",
    "何もしない"
]

# Step 3: 軽量モデル読み込み
model = SentenceTransformer("all-MiniLM-L6-v2")  # 実運用ではTinyBERTのONNXに置き換え

# Step 4: ベクトル化＆類似度評価
user_vec = model.encode(user_sentence)
method_vecs = model.encode(methods)
scores = [util.cos_sim(user_vec, vec).item() for vec in method_vecs]

# Step 5: 最もスコアの高いメソッドを出力
best_method = methods[np.argmax(scores)]

print("最も適した抽象メソッド:", best_method)
for m, s in sorted(zip(methods, scores), key=lambda x: x[1], reverse=True):
    print(f"{m}：{s:.4f}")
