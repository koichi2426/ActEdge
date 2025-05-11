import pandas as pd

# CSVファイルの読み込み
df = pd.read_csv("1.csv")  # ファイルパスは環境に合わせて調整

# 各行を「カラム名は値-カラム名は値...」形式に変換
formatted_rows = df.apply(lambda row: "-".join([f"{col}は{row[col]}" for col in df.columns]), axis=1)

# ヘッダーなしでCSV出力
formatted_rows.to_csv("converted_output.csv", index=False, header=False, encoding="utf-8-sig")
