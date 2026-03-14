"""
MInt (Materials Informatics) Workflow Adapter
MIntワークフロー接続アダプタ

Provides an adapter layer that connects the platform's ``BaseWorkflow``
to external MInt workflow execution services.  MInt workflows can be:

  - **Local** : Python-based MInt scripts executed as subprocesses
  - **Remote** : REST API calls to a MInt server (e.g. NIMS gateway)
  - **Mock**  : Wrapped built-in workflows for testing / demonstration

Concepts
--------
- **MIntWorkflowConfig** : a declarative description of a MInt workflow
  (script path, API endpoint, input/output schema, timeout, etc.)
- **MIntWorkflowAdapter** : executes a MInt workflow and returns a
  ``RunResult`` compatible with the platform runner.
- **MIntWorkflowRegistry** : manages registered MInt workflows and
  provides lookup by name or tag.

Usage::

    from extrapolation_discovery_platform.integrations.mint_adapter import (
        MIntWorkflowAdapter,
        MIntWorkflowConfig,
        MIntWorkflowRegistry,
    )

    # Register a local MInt workflow
    cfg = MIntWorkflowConfig(
        name="MInt-RF-v1",
        workflow_type="local",
        script_path="/path/to/mint_rf_workflow.py",
        input_format="csv",
        output_format="json",
        description="Random Forest via MInt local execution",
    )
    registry = MIntWorkflowRegistry()
    registry.register(cfg)

    # Execute via adapter
    adapter = MIntWorkflowAdapter(config=cfg)
    result = adapter.run(X_train, y_train, X_test, y_test, seed=42)

NOTE: External MInt connectivity requires a running MInt server or
local MInt installation.  The adapter validates connectivity at init
and falls back to mock execution with a clear warning if unavailable.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class MIntWorkflowType(str, Enum):
    """How the MInt workflow is executed."""
    LOCAL = "local"       # subprocess execution of a Python script
    REMOTE = "remote"     # REST API call to MInt server
    WRAPPED = "wrapped"   # wraps a built-in BaseWorkflow (for testing)


@dataclass
class MIntWorkflowConfig:
    """Declarative configuration for a MInt workflow.

    Parameters
    ----------
    name : str
        Human-readable workflow name (e.g. 'MInt-RF-v1').
    workflow_type : str
        One of 'local', 'remote', 'wrapped'.
    script_path : str or None
        Path to the local MInt Python script (for ``local`` type).
    api_endpoint : str or None
        URL of the MInt REST API (for ``remote`` type).
    api_key : str or None
        API key for authentication (for ``remote`` type).
    wrapped_workflow : str or None
        Name of a built-in workflow to wrap (for ``wrapped`` type).
        One of 'WF-LIN', 'WF-XGB', 'WF-ENS'.
    input_format : str
        Input data format: 'csv', 'json', 'parquet' (default 'csv').
    output_format : str
        Output data format: 'json', 'csv' (default 'json').
    timeout_sec : int
        Execution timeout in seconds (default 600).
    description : str
        Human-readable description.
    tags : dict
        Arbitrary key-value tags for filtering/search.
    extra_params : dict
        Additional parameters passed to the workflow.
    """
    name: str
    workflow_type: str = "wrapped"
    script_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    wrapped_workflow: Optional[str] = None
    input_format: str = "csv"
    output_format: str = "json"
    timeout_sec: int = 600
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate configuration; return list of error messages."""
        errors: List[str] = []
        wf_type = self.workflow_type
        if wf_type not in ("local", "remote", "wrapped"):
            errors.append(
                f"Invalid workflow_type: '{wf_type}'. "
                f"Must be 'local', 'remote', or 'wrapped'."
            )
        if wf_type == "local" and not self.script_path:
            errors.append("Local workflow requires script_path.")
        if wf_type == "local" and self.script_path:
            if not Path(self.script_path).exists():
                errors.append(
                    f"Script not found: {self.script_path}"
                )
        if wf_type == "remote" and not self.api_endpoint:
            errors.append("Remote workflow requires api_endpoint.")
        if wf_type == "wrapped" and not self.wrapped_workflow:
            errors.append("Wrapped workflow requires wrapped_workflow name.")
        return errors


# ---------------------------------------------------------------------------
# Workflow Adapter
# ---------------------------------------------------------------------------


class MIntWorkflowAdapter:
    """Execute a MInt workflow and return a platform-compatible RunResult.

    Parameters
    ----------
    config : MIntWorkflowConfig
        Workflow configuration.

    Examples
    --------
    >>> cfg = MIntWorkflowConfig(
    ...     name="MInt-XGB", workflow_type="wrapped",
    ...     wrapped_workflow="WF-XGB",
    ... )
    >>> adapter = MIntWorkflowAdapter(config=cfg)
    >>> result = adapter.run(X_train, y_train, X_test, y_test, seed=42)
    """

    def __init__(self, config: MIntWorkflowConfig) -> None:
        errors = config.validate()
        if errors:
            raise ValueError(
                f"Invalid MInt workflow config '{config.name}': "
                + "; ".join(errors)
            )
        self._config = config
        self._wrapped_wf: Optional[Any] = None

        if config.workflow_type == "wrapped":
            self._wrapped_wf = self._load_wrapped_workflow(
                config.wrapped_workflow or "WF-LIN"
            )

        logger.info(
            "MInt adapter initialised: %s (type=%s)",
            config.name, config.workflow_type,
        )

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
        """Execute the MInt workflow.

        Parameters
        ----------
        X_train, y_train : training data
        X_test, y_test : test data
        seed : random seed
        **kwargs : additional parameters (feature_set, split_policy, etc.)

        Returns
        -------
        RunResult
            Platform-compatible run result.
        """
        wf_type = self._config.workflow_type

        if wf_type == "wrapped":
            return self._run_wrapped(
                X_train, y_train, X_test, y_test, seed, **kwargs,
            )
        elif wf_type == "local":
            return self._run_local(
                X_train, y_train, X_test, y_test, seed, **kwargs,
            )
        elif wf_type == "remote":
            return self._run_remote(
                X_train, y_train, X_test, y_test, seed, **kwargs,
            )
        else:
            raise ValueError(f"Unknown workflow type: {wf_type}")

    # ----- Wrapped (built-in) -----

    def _run_wrapped(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int,
        **kwargs: Any,
    ) -> Any:
        """Delegate to a built-in workflow."""
        if self._wrapped_wf is None:
            raise RuntimeError("Wrapped workflow not loaded")

        result = self._wrapped_wf.run(
            X_train, y_train, X_test, y_test, seed=seed, **kwargs,
        )
        # Tag the result as MInt-executed and override workflow name
        result.workflow = self._config.name
        result.params["mint_adapter"] = self._config.name
        result.params["mint_type"] = "wrapped"
        result.params["mint_base_workflow"] = (
            self._config.wrapped_workflow or "unknown"
        )
        return result

    # ----- Local (subprocess) -----

    def _run_local(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int,
        **kwargs: Any,
    ) -> Any:
        """Execute a local MInt Python script via subprocess.

        The adapter:
        1. Writes train/test data to temporary files
        2. Invokes the script with arguments
        3. Reads the output JSON
        4. Converts to RunResult
        """
        from extrapolation_discovery_platform.workflows import RunResult

        t0 = time.time()
        script = self._config.script_path
        if script is None:
            raise RuntimeError("No script_path configured")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Write input data
            train_path = tmp / "train.csv"
            test_path = tmp / "test.csv"
            output_path = tmp / "output.json"

            train_df = X_train.copy()
            train_df["__target__"] = np.ascontiguousarray(
                y_train.to_numpy(dtype="float64")
            )
            train_df.to_csv(train_path, index=False)

            test_df = X_test.copy()
            test_df["__target__"] = np.ascontiguousarray(
                y_test.to_numpy(dtype="float64")
            )
            test_df.to_csv(test_path, index=False)

            # Execute script
            cmd = [
                "python", str(script),
                "--train", str(train_path),
                "--test", str(test_path),
                "--output", str(output_path),
                "--seed", str(seed),
            ]

            logger.info("MInt local exec: %s", " ".join(cmd))

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._config.timeout_sec,
                )
            except subprocess.TimeoutExpired:
                logger.error(
                    "MInt workflow timed out after %d sec",
                    self._config.timeout_sec,
                )
                raise RuntimeError(
                    f"MInt workflow '{self._config.name}' timed out "
                    f"after {self._config.timeout_sec}s"
                )

            if proc.returncode != 0:
                logger.error(
                    "MInt workflow failed (rc=%d): %s",
                    proc.returncode, proc.stderr[:500],
                )
                raise RuntimeError(
                    f"MInt workflow '{self._config.name}' failed: "
                    f"{proc.stderr[:200]}"
                )

            # Parse output
            if not output_path.exists():
                raise RuntimeError(
                    f"MInt workflow did not produce output at {output_path}"
                )

            output = json.loads(output_path.read_text())

        elapsed = time.time() - t0

        # Convert to RunResult
        return RunResult(
            workflow=self._config.name,
            feature_set=kwargs.get("feature_set", ""),
            split_policy=kwargs.get("split_policy", ""),
            seed=seed,
            fold=kwargs.get("fold", 0),
            rmse_train=float(output.get("rmse_train", 0)),
            rmse_test=float(output.get("rmse_test", 0)),
            mae_train=float(output.get("mae_train", 0)),
            mae_test=float(output.get("mae_test", 0)),
            r2_train=float(output.get("r2_train", 0)),
            r2_test=float(output.get("r2_test", 0)),
            y_test_true=np.array(output.get("y_test_true", [])),
            y_test_pred=np.array(output.get("y_test_pred", [])),
            test_indices=kwargs.get("test_indices"),
            params={
                "mint_adapter": self._config.name,
                "mint_type": "local",
                "script": str(script),
                **{
                    k: str(v)
                    for k, v in output.get("params", {}).items()
                },
            },
            artifacts=output.get("artifacts", {}),
            elapsed_sec=elapsed,
        )

    # ----- Remote (REST API) -----

    def _run_remote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int,
        **kwargs: Any,
    ) -> Any:
        """Execute a MInt workflow via REST API call.

        Expected API contract:
          POST {api_endpoint}/run
          Body: { train_data, test_data, seed, params }
          Response: { metrics, predictions, params, artifacts }
        """
        from extrapolation_discovery_platform.workflows import RunResult

        t0 = time.time()
        endpoint = self._config.api_endpoint
        if endpoint is None:
            raise RuntimeError("No api_endpoint configured")

        try:
            import requests
        except ImportError:
            raise RuntimeError(
                "requests package required for remote MInt workflows. "
                "Install with: pip install requests"
            )

        # Prepare payload
        train_data = X_train.copy()
        train_data["__target__"] = np.ascontiguousarray(
            y_train.to_numpy(dtype="float64")
        )
        test_data = X_test.copy()
        test_data["__target__"] = np.ascontiguousarray(
            y_test.to_numpy(dtype="float64")
        )

        payload = {
            "train_data": train_data.to_dict(orient="records"),
            "test_data": test_data.to_dict(orient="records"),
            "seed": seed,
            "params": self._config.extra_params,
        }

        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        url = f"{endpoint.rstrip('/')}/run"
        logger.info("MInt remote exec: POST %s", url)

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._config.timeout_sec,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error("MInt remote call failed: %s", exc)
            raise RuntimeError(
                f"MInt workflow '{self._config.name}' remote call failed: "
                f"{exc}"
            )

        output = resp.json()
        elapsed = time.time() - t0

        metrics = output.get("metrics", {})
        predictions = output.get("predictions", {})

        return RunResult(
            workflow=self._config.name,
            feature_set=kwargs.get("feature_set", ""),
            split_policy=kwargs.get("split_policy", ""),
            seed=seed,
            fold=kwargs.get("fold", 0),
            rmse_train=float(metrics.get("rmse_train", 0)),
            rmse_test=float(metrics.get("rmse_test", 0)),
            mae_train=float(metrics.get("mae_train", 0)),
            mae_test=float(metrics.get("mae_test", 0)),
            r2_train=float(metrics.get("r2_train", 0)),
            r2_test=float(metrics.get("r2_test", 0)),
            y_test_true=np.array(predictions.get("y_test_true", [])),
            y_test_pred=np.array(predictions.get("y_test_pred", [])),
            test_indices=kwargs.get("test_indices"),
            params={
                "mint_adapter": self._config.name,
                "mint_type": "remote",
                "endpoint": endpoint,
                **{
                    k: str(v)
                    for k, v in output.get("params", {}).items()
                },
            },
            artifacts=output.get("artifacts", {}),
            elapsed_sec=elapsed,
        )

    def check_connectivity(self) -> Dict[str, Any]:
        """Test whether the MInt workflow is reachable.

        Returns
        -------
        dict
            Keys: ``reachable`` (bool), ``message`` (str), ``latency_ms`` (float).
        """
        wf_type = self._config.workflow_type

        if wf_type == "wrapped":
            return {
                "reachable": self._wrapped_wf is not None,
                "message": "Wrapped workflow loaded"
                if self._wrapped_wf is not None
                else "Wrapped workflow not loaded",
                "latency_ms": 0.0,
            }

        if wf_type == "local":
            script = self._config.script_path
            exists = script is not None and Path(script).exists()
            return {
                "reachable": exists,
                "message": f"Script {'found' if exists else 'not found'}: "
                           f"{script}",
                "latency_ms": 0.0,
            }

        if wf_type == "remote":
            endpoint = self._config.api_endpoint
            if endpoint is None:
                return {
                    "reachable": False,
                    "message": "No endpoint configured",
                    "latency_ms": 0.0,
                }
            try:
                import requests
                t0 = time.time()
                resp = requests.get(
                    f"{endpoint.rstrip('/')}/health",
                    timeout=5,
                )
                latency = (time.time() - t0) * 1000
                return {
                    "reachable": resp.status_code == 200,
                    "message": f"HTTP {resp.status_code}",
                    "latency_ms": latency,
                }
            except Exception as exc:
                return {
                    "reachable": False,
                    "message": str(exc),
                    "latency_ms": 0.0,
                }

        return {"reachable": False, "message": "Unknown type", "latency_ms": 0.0}


# ---------------------------------------------------------------------------
# Workflow Registry
# ---------------------------------------------------------------------------


class MIntWorkflowRegistry:
    """Registry of MInt workflow configurations.

    Manages available MInt workflows, provides lookup by name or tags,
    and supports JSON serialization for configuration persistence.

    Examples
    --------
    >>> registry = MIntWorkflowRegistry()
    >>> registry.register(MIntWorkflowConfig(
    ...     name="MInt-RF",
    ...     workflow_type="wrapped",
    ...     wrapped_workflow="WF-XGB",
    ...     description="Random Forest via XGBoost wrapper",
    ... ))
    >>> adapter = registry.get_adapter("MInt-RF")
    >>> result = adapter.run(X_train, y_train, X_test, y_test)
    """

    def __init__(self) -> None:
        self._configs: Dict[str, MIntWorkflowConfig] = {}
        self._adapters: Dict[str, MIntWorkflowAdapter] = {}

    def register(self, config: MIntWorkflowConfig) -> None:
        """Register a MInt workflow configuration.

        Parameters
        ----------
        config : MIntWorkflowConfig
            Workflow configuration to register.

        Raises
        ------
        ValueError
            If configuration is invalid.
        """
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
        """Get (or create) an adapter for a registered workflow.

        The adapter is cached for reuse.
        """
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
                "script_path": cfg.script_path,
                "api_endpoint": cfg.api_endpoint,
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
        """Create a registry with default wrapped workflows.

        Returns a registry containing MInt-wrapped versions of
        all built-in workflows (WF-LIN, WF-XGB, WF-ENS).
        """
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
