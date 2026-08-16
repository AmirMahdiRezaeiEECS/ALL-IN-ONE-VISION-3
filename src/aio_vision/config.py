"""
Configuration loading for AIO VISION 3.

Design:
    - All experiment configuration lives in YAML files under configs/.
    - An experiment config may reference a dataset config by name
      (dataset: coco8), which is loaded from configs/datasets/ and
      merged in under the "dataset" key.
    - OmegaConf handles merging, interpolation (${...}), and typed access.

This module is intentionally small: it does not build a custom
configuration framework, it wires OmegaConf into the repo's conventions.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATASETS_CONFIG_DIR = CONFIGS_DIR / "datasets"


def _load_yaml(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


def load_experiment_config(config_path: str | Path) -> DictConfig:
    """
    Load an experiment config and resolve its dataset reference.

    An experiment YAML may contain a top-level `dataset` key that is
    either:
      - a string, treated as the name of a file in configs/datasets/
        (e.g. "coco8" -> configs/datasets/coco8.yaml), or
      - an inline mapping, used as-is.

    Returns a fully merged, resolved OmegaConf DictConfig.
    """
    config_path = Path(config_path)
    cfg = _load_yaml(config_path)

    dataset_ref = cfg.get("dataset")
    if isinstance(dataset_ref, str):
        dataset_cfg_path = DATASETS_CONFIG_DIR / f"{dataset_ref}.yaml"
        cfg.dataset = _load_yaml(dataset_cfg_path)

    OmegaConf.resolve(cfg)
    return cfg


def to_container(cfg: DictConfig) -> dict:
    """Convert an OmegaConf config to a plain dict (e.g. for MLflow logging)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
