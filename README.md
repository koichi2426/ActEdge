# 最終目標

**AR空間上の汎用コンシェルジュを実現する。**
私の最終目標はユーザーの「環境・状態・意図」をリアルタイムに理解し、AR空間上で最適な行動を提示・支援するエージェントを実現することである。
参考イメージ：[YouTube動画](https://www.youtube.com/watch?v=RsXael79U5Y)

---

# 研究について

## 研究テーマ

**現状把握リアルタイム推論エンジンの研究**

## キーワード
機械学習

## 研究背景・課題
近年、AIエージェントは自然言語処理や意思決定能力の向上により、人間の指示に応じて対話や行動が可能なレベルへと急速に進化している。

このエージェントの振る舞いには「受動的」と「能動的」の2種類がある。
受動的振る舞いでは、ユーザーの発話に応じてLLMを起動することで、環境や状態を踏まえた推論が可能であり、コスト面でも現実的である。
一方、能動的振る舞いには、ユーザーと環境を常時考慮した推論が求められるが、LLMで毎回数百トークン規模のプロンプトを処理し続けるのはコストパフォーマンスが極めて悪い。

また、LLM以外の手法は、習慣的な行動の予測には有効だが、観光地での関心対象の提示や混雑回避のルート変更など、文脈依存の一時的な行動推論には対応しきれない。

このように、**常時・低コスト・高精度**をすべて満たす推論エンジンは現時点で存在せず、汎用コンシェルジュの**能動的振る舞いの実現には根本的な技術課題**がある。



## 研究目的

個人の状態や環境に応じた行動を **超低消費電力(1mW以下)** でリアルタイムに推論するエンジンを開発し、**GPT-4o相当の推論精度**を実現すること。

## 提案手法

「環境」「ユーザ」の情報から行動を予測する学習モデルを作成し、API化してエンジンを開発する。

## 実装

実装は以下の2フェーズで構成されている：

* **モデル構築：**
  ユーザー／環境情報に基づく行動予測モデルを作成。CSVデータを前処理し、ラベル付き訓練データセットを生成。学習済みモデルをONNX形式に変換。

* **推論エンジン実装：**
  Amazon EC2上にC++製推論APIサーバーを構築し、ONNX形式のモデルを使用してリアルタイム推論を実行。S3バケットを利用してモデルやログを管理。

## 使用言語や使用ツール
C++（推論APIサーバー実装）

Python（モデル学習・前処理）

Amazon EC2（APIサーバー運用）

Amazon S3（モデル・ログ管理）

Linux環境（サーバーインフラ）

## ポスター

![image](https://github.com/user-attachments/assets/a5f8540b-5013-4243-bf44-e45266c67164)

---

必要であれば、英語版への翻訳やさらに短く要約したバージョンも可能です。
