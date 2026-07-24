"""mi_hub — runcell/OptiMat/MLflow/Feast/AutoML 統合環境 + 研究エージェントの共通ライブラリ。"""
import importlib

__version__ = "0.2.0"

_LAZY_MODULES = ("datastore", "tracking", "automl", "optimat_bridge", "agent")


def __getattr__(name: str):
    # mlflow 等の重い依存を持つモジュールは初回アクセス時に読み込む
    if name in _LAZY_MODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
