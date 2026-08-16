import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio_vision.config import load_experiment_config  # noqa: E402


def test_load_experiment_config_resolves_dataset():
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "experiments" / "yolo11n_coco8_example.yaml"

    cfg = load_experiment_config(cfg_path)

    assert cfg.name == "yolo11n_coco8_example"
    assert cfg.model.weights == "yolo11n.pt"
    assert cfg.dataset.name == "coco8"
    assert cfg.dataset.path == "coco8.yaml"
    assert cfg.training.epochs == 20
    assert cfg.tracking.experiment_name == "aio-vision-3"
