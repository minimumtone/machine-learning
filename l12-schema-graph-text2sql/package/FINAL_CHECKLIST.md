# 完成版チェックリスト

パス表記は ZIP 展開先を `<EXTRACT_DIR>` とし、`<EXTRACT_DIR>/03_mdr_scripts_data` を `<MDR>` と書く。

## 現在完了済み

- 日本語版 `01_manuscript/stam-m_ja.tex`（＝`<MDR>/paper/stam-m_ja.tex`）は SSOT `<MDR>/paper/paper_data.json` と同期済み（v24: Abstract短縮・DFT由来データの早期明示・平易化。数値・図・SSOT は v23e と同一）。
- `paper_data.json` は DB-full 環境で `paper_scripts/compute_all_figures.py` により再生成済みで、図6点も `generate_figures.py` で再生成済み（ログは `<MDR>/verification_logs/`）。
- 旧英語版 `stam-m.tex` / `stam-m.pdf` は v23d で削除（v23 より前の p 値等が残存していたため）。英語版は凍結版 `stam-m_ja.tex` を複製して新規に訳出する（未着手）。
- DB なし静的検証（`verify_all.py --static-only`）は `PASS=8 WARN=0 FAIL=0`。
- DB-full 正式検証（`FULL_DB_TEST=1 verify_all.py --warnings-as-errors`）は `PASS=18 WARN=0 FAIL=0 SKIP=0`、pytest 166 passed（クリーン展開＋新規 venv で再確認済み）。
- 日本語本文の数値照合（`verify_paper_numbers.py`、引数なし＝`paper/*.tex` グロブ）は `TeX files audited: 1 (stam-m_ja.tex)`・gating 0・exit 0、`verify_ssot.py` は 5/5 PASS。「gating 0」は日本語版の結果であり、英語版は未作成である。
- 日本語 PDF は LuaLaTeX＋BibTeX でビルド済み（30ページ：本文＋参考文献 p.1–16、補足資料 S1–S14＝物理ページ 17–30。未定義参照・未定義引用 0、overfull hbox 0）。日本語版は git タグ `ja-final` で凍結済み（`<MDR>/paper/SHA256SUMS_ja.txt`、13ファイル）。
- パッケージは `scripts/build_submission_package.py` で3区分（原稿本文 / Supplementary / MDR登録用）に機械的に組み立て、全ファイルの SHA-256 を `SHA256SUMS.txt` に列挙する。区分間で重複するファイル（原稿ソース・図・per_query 表）はすべてバイト同一。

## PDF 確認

PDF コンパイルだけなら Docker、PostgreSQL、ホスト側 `psql` は不要である。`01_manuscript/stam-m_ja.tex` を LuaLaTeX でコンパイルし、表のはみ出し、参照切れ、図の更新漏れを確認する。
本文のみ／補足のみの PDF（`stam-m_ja_main.pdf` / `02_supplementary/stam-m_ja_supplementary.pdf`）は完全版からのページ抽出であり、内容は完全版と同一である。

## DB 込み完全再現

Docker が使える場合は、`<MDR>/sql_package/docker/docker-compose.yml` で PostgreSQL 15 を起動する。ホスト側 `psql` は必須ではなく、接続確認はコンテナ内 `psql` で代替できる。
Linux/macOS のシェル手順はルート `README.md` §2.3 にある。PowerShell での最小手順:

```powershell
cd <MDR>\sql_package
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-repro.txt

cd docker
$env:POSTGRES_PASSWORD = "l12_password"
$env:GIT_COMMIT = (Get-Content ..\GIT_COMMIT -Raw).Trim()
docker compose up -d
docker compose exec postgres psql -U l12_user -d l12_materials -c "SELECT version, git_commit FROM schema_initialization_status;"

cd ..
$env:POSTGRES_HOST = "127.0.0.1"
$env:POSTGRES_PASSWORD = "l12_password"
.\.venv\Scripts\python scripts\build_transfer_db.py
.\.venv\Scripts\python scripts\build_obfuscated_transfer_db.py
.\.venv\Scripts\python scripts\build_mp_transfer_db.py

$env:L12_DSN = "postgresql://l12_user:l12_password@127.0.0.1:5432/l12_materials"
$env:TRANSFER_DSN = "postgresql://l12_user:l12_password@127.0.0.1:5432/oqmd_transfer"
$env:OBF_TRANSFER_DSN = "postgresql://l12_user:l12_password@127.0.0.1:5432/oqmd_transfer_obfuscated"
$env:MP_DSN = "postgresql://l12_user:l12_password@127.0.0.1:5432/mp_transfer"
.\.venv\Scripts\python scripts\verify_all.py --warnings-as-errors
```

展開先ドライブが Docker Desktop の bind mount 対象として使えない場合は、Docker が共有できるローカル NTFS パスまたは WSL/Linux 側の作業ディレクトリで DB 検証だけを実施する。ソース内容は同一でよい。

## SSOT 再生成（実施済み）

SSOT と図は DB-full 環境で再生成済みである。別環境で再確認する場合は、DB 込み検証が通った環境で以下を実行する。

```powershell
cd <MDR>
$env:PYTHONPATH = "<MDR>\sql_package"
$env:POSTGRES_HOST = "127.0.0.1"
$env:POSTGRES_PASSWORD = "l12_password"
$env:FULL_DB_TEST = "1"
.\sql_package\.venv\Scripts\python paper_scripts\compute_all_figures.py
.\sql_package\.venv\Scripts\python -m pip install -r paper_scripts\requirements-paper.txt
.\sql_package\.venv\Scripts\python paper_scripts\generate_figures.py
.\sql_package\.venv\Scripts\python paper_scripts\verify_paper_numbers.py
.\sql_package\.venv\Scripts\python paper_scripts\verify_ssot.py
```

再生成後の `paper_data.json` は `_meta.generated_at` / `_meta.git_commit` 以外が同梱版と一致し、図6点はバイト同一になる。

## 最終化（英語版）

英語版は凍結版 `stam-m_ja.tex`（`ja-final`）を `paper/stam-m.tex` へ複製してセクション単位で訳出する（旧英語版は削除済みであり、差分修正はしない）。
数値は `paper/paper_data.json`（SSOT）由来のみとし、英語版だけで独自に数値を直さない。

訳出後の合格条件（両方を満たすこと）:

1. `verify_paper_numbers.py` を引数なしで実行して exit 0（gating 0）。
2. 同じ実行の出力に `TeX files audited: 2 (stam-m.tex, stam-m_ja.tex)` が含まれること。
   旧英語版の削除以降、引数なし実行は英語版が無くても `TeX files audited: 1 (stam-m_ja.tex)` で exit 0 になるため、exit 0 だけでは英語版が監査対象に入ったことを保証しない。

英語版を `paper/` に置いたら `SHA256SUMS_ja.txt` と同様に英語版の凍結リストを作り、`scripts/build_submission_package.py` の `MANUSCRIPT_SOURCES` に追加してパッケージを再構築する。
現状の本パッケージは英語版未作成のため submission-ready ではない。
