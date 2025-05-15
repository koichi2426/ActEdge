from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# モデルとトークナイザーのロード
model_id = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# プロンプト
prompt = "Explain the theory of relativity in simple terms."

# トークン化
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 推論時間の計測開始
start_time = time.time()

# 推論
outputs = model.generate(**inputs, max_new_tokens=100)

# 推論時間の計測終了
end_time = time.time()
elapsed_time = end_time - start_time  # 秒

# トークン数の計算
generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]  # 新しく生成された部分
num_generated_tokens = len(generated_ids)

# 出力表示
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"\n⏱ 推論時間: {elapsed_time:.3f} 秒 ({elapsed_time * 1000:.1f} ms)")
print(f"🔢 生成トークン数: {num_generated_tokens}")
if num_generated_tokens > 0:
    print(f"⚙️ トークンあたり平均: {elapsed_time * 1000 / num_generated_tokens:.2f} ms/token")
