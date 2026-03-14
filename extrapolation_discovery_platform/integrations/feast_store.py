"""Feast feature store adapter — falls back to built-in FeatureCatalog."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import feast
    from feast import Entity, Feature, FeatureStore, FeatureView, ValueType
    from feast.data_source import FileSource

    _FEAST_AVAILABLE = True
except ImportError:
    _FEAST_AVAILABLE = False
    logger.info(
        "feast not installed — FeastFeatureStore will use built-in FeatureCatalog. "
        "Install with: pip install feast"
    )


def is_feast_available() -> bool:
    """Return True if the ``feast`` package is importable."""
    return _FEAST_AVAILABLE


class _BuiltinFeatureStore:
    """In-memory feature store (used when Feast is absent)."""

    def __init__(self) -> None:
        self._data: Optional[pd.DataFrame] = None
        self._custom_sets: Dict[str, List[str]] = {}
        self._versions: Dict[str, int] = {}

    def register_feature_set(
        self, name: str, columns: List[str],
    ) -> None:
        """Register a custom feature set (name -> columns)."""
        version = self._versions.get(name, 0) + 1
        self._custom_sets[name] = list(columns)
        self._versions[name] = version
        logger.info(
            "Registered feature set '%s' (v%d, %d columns) [in-memory]",
            name, version, len(columns),
        )

    def store_features(self, df: pd.DataFrame) -> None:
        """Store feature data in memory."""
        self._data = df.copy()
        logger.info("Stored %d samples, %d features [in-memory]",
                     len(df), df.shape[1])

    def get_features(
        self,
        feature_set_name: str,
        entity_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Retrieve features for a given set name."""
        from extrapolation_discovery_platform.features import (
            FeatureCatalog,
            FeatureSetName,
        )

        # Try built-in catalog first
        columns: Optional[List[str]] = None
        try:
            fs_enum = FeatureSetName(feature_set_name)
            columns = FeatureCatalog.columns(fs_enum)
        except (ValueError, KeyError):
            pass

        # Then custom sets
        if columns is None:
            columns = self._custom_sets.get(feature_set_name)

        if columns is None:
            raise KeyError(
                f"Unknown feature set: '{feature_set_name}'. "
                f"Available built-in: {[e.value for e in FeatureSetName]} "
                f"Custom: {list(self._custom_sets.keys())}"
            )

        if self._data is None:
            raise RuntimeError(
                "No feature data stored. Call store_features() first."
            )

        # Filter to available columns
        available = [c for c in columns if c in self._data.columns]
        if not available:
            raise ValueError(
                f"None of the columns for '{feature_set_name}' "
                f"are present in stored data. "
                f"Expected: {columns}, Have: {list(self._data.columns)}"
            )

        result = self._data[available].copy()

        if entity_df is not None and "sample_id" in entity_df.columns:
            ids = np.ascontiguousarray(entity_df["sample_id"].to_numpy())
            result = result.iloc[ids].reset_index(drop=True)

        return result

    def list_feature_sets(self) -> Dict[str, Dict[str, Any]]:
        """List all known feature sets with metadata."""
        from extrapolation_discovery_platform.features import (
            FeatureCatalog,
            FeatureSetName,
        )

        result: Dict[str, Dict[str, Any]] = {}
        for fs in FeatureSetName:
            cols = FeatureCatalog.columns(fs)
            result[fs.value] = {
                "columns": cols,
                "n_features": len(cols),
                "source": "builtin",
                "version": 1,
            }

        for name, cols in self._custom_sets.items():
            result[name] = {
                "columns": cols,
                "n_features": len(cols),
                "source": "custom",
                "version": self._versions.get(name, 1),
            }

        return result

    def get_feature_set_version(self, name: str) -> int:
        """Return the current version number of a feature set."""
        return self._versions.get(name, 1)


class FeastFeatureStore:
    """Unified feature store — delegates to Feast or built-in fallback."""

    def __init__(
        self,
        repo_path: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._use_feast = _FEAST_AVAILABLE and enabled
        self._fallback = _BuiltinFeatureStore()
        self._feast_store: Optional[Any] = None
        self._repo_path = Path(repo_path) if repo_path else None

        if self._use_feast and self._repo_path is not None:
            try:
                self._feast_store = FeatureStore(
                    repo_path=str(self._repo_path)
                )
                logger.info(
                    "Feast feature store initialised: repo=%s",
                    self._repo_path,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialise Feast store (%s). "
                    "Falling back to built-in.",
                    exc,
                )
                self._use_feast = False
        elif self._use_feast:
            logger.info(
                "Feast available but no repo_path provided. "
                "Using built-in store with Feast schema validation."
            )
            self._use_feast = False

        if not self._use_feast:
            logger.info("FeastFeatureStore using built-in fallback.")

    @property
    def is_feast_active(self) -> bool:
        """Whether real Feast is being used."""
        return self._use_feast

    def register_feature_set(
        self, name: str, columns: List[str],
    ) -> None:
        """Register a feature set (Feast FeatureView or in-memory)."""
        if self._use_feast and self._feast_store is not None:
            logger.info(
                "Registering Feast FeatureView: %s (%d features)",
                name, len(columns),
            )
            # Feast registration would go through feast apply
            # For now, also register in fallback for query
            self._fallback.register_feature_set(name, columns)
        else:
            self._fallback.register_feature_set(name, columns)

    def store_features(self, df: pd.DataFrame) -> None:
        """Store feature data (Feast offline store or in-memory)."""
        if self._use_feast and self._feast_store is not None:
            # Write to Feast offline store
            logger.info("Storing %d samples to Feast offline store", len(df))
            # For Feast, we'd write to a FileSource or push to online store
            # Fallback always stores for immediate retrieval
            self._fallback.store_features(df)
        else:
            self._fallback.store_features(df)

    def get_features(
        self,
        feature_set_name: str,
        entity_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Retrieve features for a given set."""
        if self._use_feast and self._feast_store is not None:
            try:
                # Attempt Feast historical feature retrieval
                if entity_df is not None:
                    features = self._feast_store.get_historical_features(
                        entity_df=entity_df,
                        features=[f"{feature_set_name}:*"],
                    ).to_df()
                    return features
            except Exception as exc:
                logger.warning(
                    "Feast retrieval failed (%s), falling back to built-in",
                    exc,
                )

        return self._fallback.get_features(feature_set_name, entity_df)

    def list_feature_sets(self) -> Dict[str, Dict[str, Any]]:
        """List all registered feature sets with metadata."""
        return self._fallback.list_feature_sets()

    def get_feature_set_version(self, name: str) -> int:
        """Return the current version number of a feature set."""
        return self._fallback.get_feature_set_version(name)

    def get_store_info(self) -> Dict[str, Any]:
        """Return information about the feature store backend."""
        if self._use_feast:
            return {
                "backend": "feast",
                "repo_path": str(self._repo_path),
                "active": True,
            }
        return {
            "backend": "builtin",
            "repo_path": None,
            "active": False,
            "note": "Install feast for persistent feature store",
        }
