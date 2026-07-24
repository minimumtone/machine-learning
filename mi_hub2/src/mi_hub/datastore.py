"""mi_hub.datastore — 統合環境の共通データ層。

規約:
  - 全ツール間の受け渡しは parquet に統一(pandas + pyarrow)。
  - 全レコードに provenance 列を強制付与:
      run_id      : UUID4。MLflow run と 1:1 対応させる
      created_at  : UTC ISO8601
      source      : 生成元 ("tc_python" | "optimat" | "dft" | "manual" | ...)
      code_ver    : スクリプト側が渡すバージョン文字列(git hash 推奨)
  - 保存先は MI_HUB_DATA (env) 直下の kind 別ディレクトリ:
      data/<kind>/<run_id>.parquet
    kind 例: "ternary_sections", "hea_features", "optimat_snapshot", "predictions"
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROVENANCE_COLS = ["run_id", "created_at", "source", "code_ver"]


def data_root() -> Path:
    root = Path(os.environ.get("MI_HUB_DATA", Path.home() / "mi_hub_data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_run_id() -> str:
    return str(uuid.uuid4())


def stamp(df: pd.DataFrame, *, run_id: str, source: str, code_ver: str = "unknown") -> pd.DataFrame:
    """provenance 列を付与した新しい DataFrame を返す。"""
    out = df.copy()
    out["run_id"] = run_id
    out["created_at"] = datetime.now(timezone.utc).isoformat()
    out["source"] = source
    out["code_ver"] = code_ver
    return out


def save(df: pd.DataFrame, kind: str, *, run_id: str | None = None,
         source: str = "manual", code_ver: str = "unknown") -> Path:
    """provenance を付与して data/<kind>/<run_id>.parquet に保存し、パスを返す。"""
    run_id = run_id or new_run_id()
    if not set(PROVENANCE_COLS).issubset(df.columns):
        df = stamp(df, run_id=run_id, source=source, code_ver=code_ver)
    d = data_root() / kind
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{run_id}.parquet"
    df.to_parquet(path, index=False)
    return path


def load(kind: str, *, run_id: str | None = None) -> pd.DataFrame:
    """kind 配下を全結合して返す。run_id 指定時は単一ファイル。"""
    d = data_root() / kind
    if run_id:
        return pd.read_parquet(d / f"{run_id}.parquet")
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {d}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def catalog() -> pd.DataFrame:
    """全 kind の在庫一覧(kind, ファイル数, 最終更新)。"""
    rows = []
    for d in sorted(p for p in data_root().iterdir() if p.is_dir()):
        files = list(d.glob("*.parquet"))
        if files:
            rows.append({
                "kind": d.name,
                "n_files": len(files),
                "last_modified": datetime.fromtimestamp(
                    max(f.stat().st_mtime for f in files), tz=timezone.utc
                ).isoformat(),
            })
    return pd.DataFrame(rows)
