"""
Integrations Sub-package
外部ツール統合モジュール

Provides adapter layers for:
  mlflow_tracker   MLflow experiment tracking (metrics, params, artifacts)
  feast_store      Feast feature store (feature set registration & retrieval)
  mint_adapter     MInt workflow connection (external workflow execution)

All integrations are optional — each module gracefully falls back to
built-in implementations when the external dependency is not installed.
"""
