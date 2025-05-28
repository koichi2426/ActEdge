from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer("all-mpnet-base-v2")

user_sentence = (
    "A 35-year-old female doctor. Lives with her family and has a morning lifestyle working remotely. "
    "It's a quiet office on an autumn Friday morning at 6 a.m. "
    "Cloudy weather, 22°C, 58% humidity, 55dB noise level. "
    "She enjoys cafes, meditates to relax, and likes Thai food and movies."
)

methods = [
    "Optimize route planning",
    "Suggest spots based on the situation",
    "Recommend food, breaks, and shopping",
    "Propose events, experiences, and photo ops",
    "Provide safety and emergency measures",
    "Do nothing"
]

user_vec = model.encode(user_sentence, convert_to_tensor=True)
method_vecs = model.encode(methods, convert_to_tensor=True)

scores = util.cos_sim(user_vec, method_vecs)[0].tolist()
best_method = methods[np.argmax(scores)]

print("Best Abstract Method:", best_method)
for m, s in sorted(zip(methods, scores), key=lambda x: x[1], reverse=True):
    print(f"{m}: {s:.4f}")
