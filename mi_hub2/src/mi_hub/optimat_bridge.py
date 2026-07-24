"""mi_hub.optimat_bridge — OptiMat Alloys living DB の読み取りブリッジ。

前提:
  - OptiMat Alloys コンテナの DB ディレクトリをホスト側にボリュームマウントし、
    そのパスを env MI_HUB_OPTIMAT_DB で指す(setup/phase2_optimat 参照)。
  - living DB は UUID インデックス + provenance 付きで、実装上は
    SQLite / JSON ファイル群のいずれかが想定される。公開イメージの
    実スキーマはバージョン依存のため、本モジュールは
      (1) SQLite があれば全テーブルを吸い上げ
      (2) なければ JSON/JSONL を再帰走査してフラット化
    する汎用スナップショット取得に徹する。列名の正規化は取得後に
    ノートブック側(runcell に任せると速い)で行う設計。

使用例:
    from mi_hub import optimat_bridge as ob, datastore as ds
    snap = ob.snapshot()                      # {name: DataFrame}
    for name, df in snap.items():
        ds.save(df, f"optimat_{name}", source="optimat")
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd


def db_root() -> Path:
    p = os.environ.get("MI_HUB_OPTIMAT_DB")
    if not p:
        raise EnvironmentError(
            "MI_HUB_OPTIMAT_DB が未設定です。OptiMat コンテナの DB ボリューム"
            "のホスト側パスを指定してください(例: /data/optimat_db)。")
    return Path(p)


def _sqlite_tables(path: Path) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(path)
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        return {n: pd.read_sql_query(f"SELECT * FROM '{n}'", con) for n in names}
    finally:
        con.close()


def _json_records(root: Path) -> pd.DataFrame:
    rows = []
    for f in root.rglob("*.json*"):
        try:
            text = f.read_text(encoding="utf-8")
            if f.suffix == ".jsonl":
                objs = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                obj = json.loads(text)
                objs = obj if isinstance(obj, list) else [obj]
            for o in objs:
                if isinstance(o, dict):
                    o["_file"] = str(f.relative_to(root))
                    rows.append(o)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return pd.json_normalize(rows) if rows else pd.DataFrame()


def snapshot() -> dict[str, pd.DataFrame]:
    """living DB の現在状態を {name: DataFrame} で返す。"""
    root = db_root()
    out: dict[str, pd.DataFrame] = {}
    for db in root.rglob("*.db"):
        for tname, df in _sqlite_tables(db).items():
            out[f"{db.stem}__{tname}"] = df
    for db in root.rglob("*.sqlite*"):
        for tname, df in _sqlite_tables(db).items():
            out[f"{db.stem}__{tname}"] = df
    if not out:
        df = _json_records(root)
        if not df.empty:
            out["json_records"] = df
    if not out:
        raise FileNotFoundError(f"{root} に SQLite / JSON が見つかりません。")
    return out
