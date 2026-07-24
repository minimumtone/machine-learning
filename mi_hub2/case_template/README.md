# 解析事例: <事例名>

<!-- 事例管理テンプレート。1 事例 = 1 ディレクトリ(or リポジトリ)。実行者(人間/runcell/任意エージェント)非依存。 -->

## 目的
(1〜3 行。何を知るための解析か)

## 結論
(完了後に必ず記入。図表は MLflow artifact を参照)

## 再現
```bash
conda env create -f environment.yml -n case_<name>
conda run -n case_<name> python run.py
```

## 関連
- MLflow experiment: `<experiment名>` / mi_hub.run_id: `<uuid>`
- 入力データ kind: `<datastore kind>`
- 先行事例: `../<case>/`

## エージェント(runcell 等)への典型指示
- 「この事例を再実行し、MLflow の run_id を README の関連欄に追記せよ」
- 「事例 <X> の手法を本事例のデータに適用し、差分を報告せよ」
