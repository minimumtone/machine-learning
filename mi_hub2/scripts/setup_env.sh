#!/bin/bash
# MI-HUB2 環境構築スクリプト（NIMS proxy 対応）
#
# 使い方:
#   bash mi_hub2/scripts/setup_env.sh            # proxy 自動判別
#   MI_HUB_PROXY=on  bash mi_hub2/scripts/setup_env.sh   # NIMS proxy を強制使用
#   MI_HUB_PROXY=off bash mi_hub2/scripts/setup_env.sh   # proxy を使わない
#
# NIMS 内ネットワークでは wwwout.nims.go.jp:8888 経由で外部へ出る。
# proxy の要否を自動判別し、apt / pip / curl の proxy 設定を切り替える。

set -euo pipefail

NIMS_PROXY="http://wwwout.nims.go.jp:8888"
MI_HUB_PROXY="${MI_HUB_PROXY:-auto}"

log() { echo "[setup_env] $*"; }

# ---------- 1. proxy 判別 ----------
use_proxy=""
case "$MI_HUB_PROXY" in
  on)  use_proxy="yes" ;;
  off) use_proxy="no" ;;
  auto)
    # 既に環境変数で proxy が設定されていればそれを尊重
    if [ -n "${https_proxy:-}${HTTPS_PROXY:-}" ]; then
      use_proxy="preset"
    # 直接外へ出られるか（5秒）
    elif curl -s --max-time 5 -o /dev/null https://pypi.org; then
      use_proxy="no"
    # NIMS proxy 経由で外へ出られるか
    elif curl -s --max-time 5 -x "$NIMS_PROXY" -o /dev/null https://pypi.org; then
      use_proxy="yes"
    else
      log "エラー: 直接接続・NIMS proxy 経由のいずれでも外部へ接続できません。"
      log "ネットワーク設定を確認するか、MI_HUB_PROXY=on/off を明示してください。"
      exit 1
    fi
    ;;
  *) log "エラー: MI_HUB_PROXY は auto / on / off のいずれかを指定してください。"; exit 1 ;;
esac

PIP_ARGS=()
APT_ARGS=()
if [ "$use_proxy" = "yes" ]; then
  log "NIMS proxy（$NIMS_PROXY）を使用します。"
  export http_proxy="$NIMS_PROXY" https_proxy="$NIMS_PROXY"
  export HTTP_PROXY="$NIMS_PROXY" HTTPS_PROXY="$NIMS_PROXY"
  export no_proxy="localhost,127.0.0.1,.nims.go.jp"
  PIP_ARGS=(--proxy "$NIMS_PROXY")
  APT_ARGS=(-o "Acquire::http::Proxy=$NIMS_PROXY" -o "Acquire::https::Proxy=$NIMS_PROXY")
elif [ "$use_proxy" = "preset" ]; then
  log "既存の proxy 設定（${https_proxy:-$HTTPS_PROXY}）を使用します。"
else
  log "proxy なし（直接接続）で進めます。"
fi

# ---------- 2. OS パッケージ ----------
if command -v apt-get >/dev/null 2>&1; then
  log "apt: 日本語フォント等をインストールします。"
  sudo apt-get "${APT_ARGS[@]}" update -q
  sudo apt-get "${APT_ARGS[@]}" install -y -q fonts-noto-cjk zstd
else
  log "警告: apt-get が無いため OS パッケージ（fonts-noto-cjk）はスキップします。"
fi

# ---------- 3. Python パッケージ ----------
log "pip: エージェント基盤・GraphRAG・ML・MLIP を導入します。"
pip install "${PIP_ARGS[@]}" -q --upgrade pip
pip install "${PIP_ARGS[@]}" -q \
  pydantic fastapi uvicorn streamlit httpx pytest ruff openai
pip install "${PIP_ARGS[@]}" -q fugashi unidic-lite
pip install "${PIP_ARGS[@]}" -q scikit-learn pycaret xgboost lightgbm
pip install "${PIP_ARGS[@]}" -q chgnet
# PyCaret(scikit-plot) が scipy>=1.12 で動かないためピン留め
# （CHGNet / pymatgen は 1.11.4 で動作確認済み）
pip install "${PIP_ARGS[@]}" -q "scipy==1.11.4"

# ---------- 4. 動作確認 ----------
log "動作確認: 主要モジュールの import を検証します。"
python3 - <<'PY'
import fastapi, streamlit, httpx, pydantic  # noqa: F401
import fugashi  # noqa: F401
import sklearn, xgboost, lightgbm  # noqa: F401
import scipy
from pycaret.classification import setup  # noqa: F401
from chgnet.model import CHGNet  # noqa: F401
print(f"OK: scipy {scipy.__version__} / 全モジュール import 成功")
PY

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
(cd "$REPO_ROOT/mi_hub2" && PYTHONPATH=src python3 -m pytest tests -q)

log "完了。UI 起動:"
log "  (cd mi_hub2 && PYTHONPATH=src streamlit run src/mi_hub/agent/ui_streamlit.py --server.port 8501 --server.headless true)"
log "Slurm を使う場合:"
log "  export MI_HUB_SCHEDULER=slurm"
log "  export MI_HUB_SLURM_SSH_HOST=<HPCログインノード>   # SSH経由の場合のみ"
