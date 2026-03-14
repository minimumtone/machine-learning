"""MInt workflow adapter — wraps built-in workflows for MInt-style execution."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)



@dataclass
class MIntWorkflowConfig:
    """Declarative configuration for a MInt workflow."""
    name: str
    workflow_type: str = "wrapped"
    wrapped_workflow: Optional[str] = None
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate configuration; return list of error messages."""
        errors: List[str] = []
        if self.workflow_type != "wrapped":
            errors.append(
                f"Only 'wrapped' workflow_type is supported, got '{self.workflow_type}'."
            )
        if not self.wrapped_workflow:
            errors.append("wrapped_workflow name is required.")
        return errors


class MIntWorkflowAdapter:
    """Execute a MInt workflow (wraps a built-in workflow)."""

    def __init__(self, config: MIntWorkflowConfig) -> None:
        errors = config.validate()
        if errors:
            raise ValueError(
                f"Invalid MInt workflow config '{config.name}': "
                + "; ".join(errors)
            )
        self._config = config
        self._wrapped_wf = self._load_wrapped_workflow(
            config.wrapped_workflow or "WF-LIN"
        )
        logger.info("MInt adapter initialised: %s", config.name)

    @property
    def config(self) -> MIntWorkflowConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name

    def _load_wrapped_workflow(self, workflow_name: str) -> Any:
        """Load a built-in workflow by name."""
        from extrapolation_discovery_platform.workflows import get_workflow

        extra = dict(self._config.extra_params)
        return get_workflow(workflow_name, **extra)

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int = 42,
        **kwargs: Any,
    ) -> Any:
        """Execute the wrapped workflow and tag the result."""
        if self._wrapped_wf is None:
            raise RuntimeError("Wrapped workflow not loaded")

        result = self._wrapped_wf.run(
            X_train, y_train, X_test, y_test, seed=seed, **kwargs,
        )
        result.workflow = self._config.name
        result.params["mint_adapter"] = self._config.name
        result.params["mint_base_workflow"] = (
            self._config.wrapped_workflow or "unknown"
        )
        return result

    def check_connectivity(self) -> Dict[str, Any]:
        """Test whether the wrapped workflow is loaded."""
        return {
            "reachable": self._wrapped_wf is not None,
            "message": "Wrapped workflow loaded"
            if self._wrapped_wf is not None
            else "Wrapped workflow not loaded",
            "latency_ms": 0.0,
        }


class MIntWorkflowRegistry:
    """Registry of MInt workflow configurations."""

    def __init__(self) -> None:
        self._configs: Dict[str, MIntWorkflowConfig] = {}
        self._adapters: Dict[str, MIntWorkflowAdapter] = {}

    def register(self, config: MIntWorkflowConfig) -> None:
        """Register a MInt workflow configuration."""
        errors = config.validate()
        if errors:
            raise ValueError(
                f"Invalid config '{config.name}': " + "; ".join(errors)
            )
        self._configs[config.name] = config
        # Invalidate cached adapter
        self._adapters.pop(config.name, None)
        logger.info("Registered MInt workflow: %s", config.name)

    def unregister(self, name: str) -> None:
        """Remove a workflow from the registry."""
        self._configs.pop(name, None)
        self._adapters.pop(name, None)

    def get_config(self, name: str) -> MIntWorkflowConfig:
        """Get configuration by name."""
        if name not in self._configs:
            raise KeyError(
                f"MInt workflow '{name}' not registered. "
                f"Available: {list(self._configs.keys())}"
            )
        return self._configs[name]

    def get_adapter(self, name: str) -> MIntWorkflowAdapter:
        """Get (or create) a cached adapter for a registered workflow."""
        if name not in self._adapters:
            config = self.get_config(name)
            self._adapters[name] = MIntWorkflowAdapter(config)
        return self._adapters[name]

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all registered workflows with their metadata."""
        result = []
        for name, cfg in self._configs.items():
            result.append({
                "name": name,
                "type": cfg.workflow_type,
                "description": cfg.description,
                "tags": dict(cfg.tags),
                "wrapped_workflow": cfg.wrapped_workflow,
            })
        return result

    def find_by_tag(self, key: str, value: str) -> List[str]:
        """Find workflows matching a tag."""
        return [
            name for name, cfg in self._configs.items()
            if cfg.tags.get(key) == value
        ]

    def check_all_connectivity(self) -> Dict[str, Dict[str, Any]]:
        """Check connectivity for all registered workflows."""
        results: Dict[str, Dict[str, Any]] = {}
        for name in self._configs:
            adapter = self.get_adapter(name)
            results[name] = adapter.check_connectivity()
        return results

    # ----- Serialization -----

    def to_json(self, path: Path) -> None:
        """Export registry to JSON file."""
        import dataclasses
        data = {
            name: dataclasses.asdict(cfg)
            for name, cfg in self._configs.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Exported %d MInt configs to %s", len(data), path)

    @classmethod
    def from_json(cls, path: Path) -> "MIntWorkflowRegistry":
        """Load registry from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        registry = cls()
        for name, cfg_dict in data.items():
            cfg = MIntWorkflowConfig(**cfg_dict)
            registry.register(cfg)
        logger.info("Loaded %d MInt configs from %s", len(data), path)
        return registry

    # ----- Default workflows -----

    @classmethod
    def create_default(cls) -> "MIntWorkflowRegistry":
        """Create a registry with default wrapped workflows."""
        registry = cls()
        defaults = [
            MIntWorkflowConfig(
                name="MInt-LIN",
                workflow_type="wrapped",
                wrapped_workflow="WF-LIN",
                description="Linear regression via MInt adapter "
                            "(Ridge, coefficient analysis)",
                tags={"family": "linear", "complexity": "low"},
            ),
            MIntWorkflowConfig(
                name="MInt-XGB",
                workflow_type="wrapped",
                wrapped_workflow="WF-XGB",
                description="XGBoost via MInt adapter "
                            "(hyperparameter optimisation)",
                tags={"family": "tree", "complexity": "medium"},
                extra_params={"quick": True},
            ),
            MIntWorkflowConfig(
                name="MInt-ENS",
                workflow_type="wrapped",
                wrapped_workflow="WF-ENS",
                description="Seed-varied ensemble via MInt adapter "
                            "(uncertainty quantification)",
                tags={"family": "ensemble", "complexity": "high"},
                extra_params={"n_members": 3, "quick": True},
            ),
        ]
        for cfg in defaults:
            registry.register(cfg)
        return registry
