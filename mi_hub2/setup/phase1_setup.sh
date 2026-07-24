#!/usr/bin/env bash
# Phase 1: 操縦席(JupyterLab + runcell + pygwalker) + 記録層(MLflow) + mi_hub
set -euo pipefail

ENV_NAME="${1:-hub}"
conda create -y -n "$ENV_NAME" python=3.11
conda run -n "$ENV_NAME" pip install \
    "jupyterlab>=4.4" runcell==0.2.0 pygwalker==0.5.0.1 \
    "mlflow>=2.14" "pandas>=2.0" "pyarrow>=14" scikit-learn
conda run -n "$ENV_NAME" pip install -e "$(dirname "$0")/.."   # mi_hub 本体

echo
echo "~/.bashrc に追記推奨:"
echo "  export MI_HUB_DATA=\$HOME/mi_hub_data"
echo "  export MI_HUB_MLFLOW=sqlite:///\$HOME/mi_hub_data/mlflow.db"
echo
echo "起動:"
echo "  conda activate $ENV_NAME && jupyter lab                # runcell 拡張が現れる"
echo "  mlflow ui --backend-store-uri \$MI_HUB_MLFLOW -p 5000  # 別ターミナル"
