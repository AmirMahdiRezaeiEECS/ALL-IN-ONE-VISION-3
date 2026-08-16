# AIO VISION 3

A simple, low-code, experiment-oriented repository for training, evaluating,
and deploying YOLO models using [Ultralytics](https://docs.ultralytics.com/).

**Philosophy:** Ultralytics already solves the hard ML problems (architectures,
training loop, augmentation, export). This repo does not reimplement any of
that. It exists to make experiments **reproducible, comparable, and easy to
run** via configuration + a small number of scripts. A new experiment is a new
YAML file, not a new Python program.

```
experiment.yaml -> scripts/train.py -> Ultralytics -> MLflow -> checkpoints/metrics
```

## Repository layout

```
configs/
  datasets/       # one YAML per dataset (path, class names, format notes)
  experiments/     # one YAML per experiment (dataset + model + training + tracking)
scripts/
  prepare_dataset.py    # raw images/labels -> Ultralytics-style data.yaml
  validate_dataset.py   # sanity-check a dataset before burning GPU time on it
  train.py               # config -> Ultralytics training -> MLflow
  evaluate.py             # run val/test metrics for a checkpoint, logged to MLflow
  predict.py               # inference on an image/dir/video
  export.py                 # export a checkpoint to ONNX/TorchScript/etc.
  deploy.py                  # minimal FastAPI server around a checkpoint
src/aio_vision/
  config.py       # OmegaConf loading + dataset-reference resolution (~50 lines)
  tracking.py     # MLflow run wrapper: logs config + artifacts (~40 lines)
data/             # datasets live here (gitignored)
tests/            # tests for the (small) custom code, not for Ultralytics/MLflow
```

That's the whole custom codebase — under 200 lines outside of scripts. Everything
else is Ultralytics, MLflow, and YAML.

## Quickstart

```bash
# 1. Install
pip install -e ".[dev]"          # add "[serve]" too if you want scripts/deploy.py

# 2. Sanity-check the example experiment (uses Ultralytics' built-in coco8 sample,
#    no dataset prep needed for this first run)
python scripts/train.py configs/experiments/yolo11n_coco8_example.yaml

# 3. Inspect runs
mlflow ui --backend-store-uri mlruns
```

## Bringing your own dataset

```bash
# Your raw data should already be organized as:
#   <root>/images/train/*.jpg   <root>/labels/train/*.txt
#   <root>/images/val/*.jpg     <root>/labels/val/*.txt
python scripts/prepare_dataset.py \
    --name my_dataset --root data/my_dataset --classes person,car,bicycle

python scripts/validate_dataset.py configs/datasets/my_dataset.yaml
```

Then add `configs/datasets/my_dataset.yaml`:

```yaml
name: my_dataset
format: yolo
path: data/my_dataset/data.yaml
num_classes: 3
```

And point an experiment config at it with `dataset: my_dataset`.

## Running an experiment grid

Because everything is config-driven, sweeping `dataset x model x hyperparams`
is just multiple YAML files plus a shell loop — no new source code:

```bash
for cfg in configs/experiments/*.yaml; do
    python scripts/train.py "$cfg"
done
```

Every run's exact resolved config, metrics, and best/last checkpoints are
logged to MLflow, so any run is traceable back to the config, dataset, and
code that produced it.

## Design notes / what was deliberately left out

- **No custom training framework.** `train.py` calls `model.train(**cfg.training)`
  directly — Ultralytics owns the training loop, losses, schedulers, and
  checkpointing.
- **No custom config framework.** OmegaConf (an established library) handles
  loading and merging; `config.py` only adds the repo's dataset-reference
  convention on top of it.
- **No custom experiment-tracking abstraction.** `tracking.py` is a ~40-line
  context manager around `mlflow.start_run`, not a wrapper API.
- **No dataset-conversion code.** `prepare_dataset.py` assumes YOLO-format
  input; converting from COCO JSON/Pascal VOC/etc. should use an existing
  mature converter (Ultralytics' own JSON2YOLO utilities, FiftyOne, Roboflow)
  as a preprocessing step before this script, not something reimplemented here.
- **Deployment kept minimal on purpose.** `scripts/export.py` hands off to
  Ultralytics' exporters (ONNX, TorchScript, TensorRT, CoreML, TFLite, ...);
  `scripts/deploy.py` is a thin FastAPI wrapper for quick local serving, not a
  production inference server — for that, export and serve via Triton/TorchServe/
  ONNX Runtime Server instead.

## Notes on this scaffold

This was generated as a starting scaffold, not battle-tested against a live
Ultralytics/MLflow install (no network access in the environment that built
it). Before relying on it:
- Run `pip install -e ".[dev]"` and confirm `pytest tests/` passes.
- Run the example training command end-to-end once to confirm the Ultralytics
  and MLflow API surfaces used here (`model.train`, `model.val`, `results.save_dir`,
  `metrics.box.map*`) match the installed Ultralytics version — these have been
  stable across recent releases but are worth a one-time smoke test.
