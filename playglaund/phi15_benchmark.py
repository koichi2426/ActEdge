from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import subprocess

# モデルとトークナイザーのロード
model_id = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# プロンプト
prompt = "Explain the theory of relativity in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# powermetricsコマンドで測定開始前のリファレンス電力を取得
def get_power_snapshot():
    try:
        output = subprocess.check_output(
            ["sudo", "powermetrics", "-n", "1", "-i", "1000", "--samplers", "all"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8")
        for line in output.splitlines():
            if "CPU Power:" in line:
                power_watts = float(line.split(":")[1].strip().split()[0])
                return power_watts
    except Exception as e:
        print("⚠️ powermetrics 実行失敗:", e)
        return None

print("🔧 電力測定には sudo 権限が必要です。最初に sudo パスワードが求められる場合があります。")

power_before = get_power_snapshot()
start_time = time.time()
outputs = model.generate(**inputs, max_new_tokens=100)
end_time = time.time()
power_after = get_power_snapshot()

# 推論時間とトークン数
elapsed_time = end_time - start_time
generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
num_generated_tokens = len(generated_ids)

# 平均電力と推定消費エネルギー（ジュール）
if power_before is not None and power_after is not None:
    avg_power = (power_before + power_after) / 2
    estimated_energy_j = avg_power * elapsed_time
else:
    avg_power = estimated_energy_j = None

# 出力テキスト
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)

# ファイル保存
with open("phi_output.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

# メタ情報出力
print(f"\n⏱ 推論時間: {elapsed_time:.3f} 秒")
print(f"🔢 生成トークン数: {num_generated_tokens}")
if num_generated_tokens > 0:
    print(f"⚙️ 平均: {elapsed_time * 1000 / num_generated_tokens:.2f} ms/token")
if avg_power is not None:
    print(f"⚡️ 平均CPU電力: {avg_power:.2f} W")
    print(f"🔋 推定消費エネルギー: {estimated_energy_j:.2f} J")
else:
    print("⚠️ 電力測定に失敗しました（powermetrics未許可または失敗）")
