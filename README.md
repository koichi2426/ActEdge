# ActEdge – 現状把握リアルタイム推論エンジン

ユーザーの環境・ペルソナ、**今この瞬間に最も適した行動（または介入しないという選択）** をリアルタイムに判断する、モバイル・組み込み対応の軽量推論エンジン。

---

## ■ 目的

本エンジンは、ユーザーの **環境情報**、**ペルソナ情報** をもとに、与えられた候補メソッド群から、**「今この瞬間に最も適した行動、または何もしない（NoAction）」** を選択します。

推論結果が「何もしない（NoAction）」であっても、それは状況が安定しており、ユーザーに最適な介入が不要であると判断された結果です。
モバイル・組み込み環境での即時動作と、**過干渉を避ける設計**を重視しています。

---

## ■ 入出力仕様

| 種類  | 内容例                                                                                |
| --- | ---------------------------------------------------------------------------------- |
| 入力① | 環境情報：時間帯、場所、周囲音、照明、温湿度、使用中のデバイスなど                                                  |
| 入力② | ペルソナ情報：年齢、性格、生活習慣、疲労度、目的、嗜好など                                                                   |
| 入力③ | 候補メソッド群：RelaxMusic, SuggestBreak, NotifyHydration, LaunchFocusApp, **NoAction** など(未知メソッド) |
| 出力  | 最適なメソッド（1つ、例：`SuggestBreak` または `NoAction`）とその理由（スコア・根拠付き）                         |

### 入力仕様

```json
{
  "environment": {
    "time_of_day": "午後",
    "location": "オフィス",
    "ambient_sound": "静穏",
    "lighting": "暖色",
    "temperature": "25°C",
    "humidity": "40%",
    "device_in_use": "スマートフォン"
  },
  "persona": {
    "age": 30,
    "personality": "外向的",
    "lifestyle": "忙しい",
    "fatigue_level": "中程度",
    "purpose": "集中作業",
    "preferences": ["音楽", "静かな空間"]
  }
  "candidate_methods": [
    "RelaxMusic",
    "SuggestBreak",
    "NotifyHydration",
    "LaunchFocusApp",
    "NoAction"
  ]
}
```

### 出力仕様
```json
{
  "selected_method": "SuggestBreak",
  "reason": {
    "score": 85,
    "explanation": "ユーザーが一息つきたいという発言と疲労度から、休憩を提案するのが最適と判断された。"
  }
}
```
---

## ■ コア処理構造（Decision Engine）
![image](https://github.com/user-attachments/assets/79ede4d0-741e-4c32-bc33-c5e72cc76c17)

## ■ 特徴と強み

* ✅ **過干渉抑制**：状況が安定していれば介入せずNoActionを推奨
* ✅ **常時推論型**：発言がなくても状態変化に応じて常に判断を実行
* ✅ **説明可能性**：選択／非選択の理由をスコア・ルールで提示
* ✅ **軽量実装**：SVMや木構造により省メモリ・高速推論
* ✅ **高拡張性**：新メソッド／新ルール／スコア軸の追加が容易

---

## ■ ユースケース例

| 状況         | 推奨メソッド                     |
| ---------- | -------------------------- |
| 午後＋静穏＋集中状態 | `NoAction`（干渉せず維持）         |
| 深夜＋疲労＋静寂   | `SuggestBreak`             |
| 夏＋高温＋屋内    |　`NotifyHydration`（環境リスク検知） |
| 通勤中＋雑音環境   | `PlayCalmMusic`            |

---

## ■ データ仕様と学習方針

* **履歴件数**：3,000〜10,000件（状況・発言・メソッド含む）
* **特徴量**：

  * 環境（10項目）、ペルソナ（10項目）、発言エンベディング（任意）
* **手法**：

  * SVM、RandomForest、LightNN、小規模NN＋ルール合成

---

## ■ ライセンス

MIT License（[LICENSE](https://opensource.org/license/mit)）

---

## ■ 開発者

* 作成者: 佐藤 幸一
* GitHub: [github.com/koichi2426](https://github.com/koichi2426)
* 連絡先: [satoyskkh200426@gmail.com](mailto:satoyskkh200426@gmail.com)
