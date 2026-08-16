# **AIO VISION 3 — Architecture & Implementation Direction**
I am designing **ALL-IN-ONE-VISION 3 (AIO VISION 3)**.
The goal is to build a **simple, low-code, config-driven repository for training YOLO models using Ultralytics**.
The most important principle is:
**Users should be able to create and run new experiments primarily by creating or modifying configuration files, without modifying the source code.**
## **1. Core Scope**
AIO VISION 3 is **not** intended to reimplement YOLO or build another computer-vision framework.
Ultralytics already provides:
- YOLO model architectures
- pretrained models
- training functionality
- validation
- inference
- augmentation
- optimizers
- schedulers
- losses
- many other model-level components
**Do not reimplement these.**
The repository should mainly provide a clean engineering layer around Ultralytics for:
1. **Data pipeline**
2. **Configuration management**
3. **Training orchestration**
4. **Experiment tracking**
5. **Reproducible experiment execution**
The data pipeline, training orchestration, and experiment tracking are the main engineering concerns.
---
## **2. Low-Code Philosophy**
Keep the repository intentionally small.
Training a YOLO model with Ultralytics does not require a large amount of custom code, so **do not create abstractions merely for the sake of abstraction**.
Use established packages whenever they already solve the problem well.
**Prefer configuration and composition over custom code.**
A good implementation should feel almost boring.
---
## **3. Config-Driven Design**
Configuration is one of the most important architectural requirements.
A new experiment should normally require:
**Adding or modifying a config — not modifying Python source code.**
For example, changing:
- dataset
- model
- pretrained checkpoint
- image size
- batch size
- epochs
- optimizer
- learning rate
- augmentation
- hardware/device
- experiment name
- MLflow settings
- output paths
should be possible through configuration.
The architecture should make it easy to run many combinations of:
**dataset × model × training configuration × experiment**
without creating new Python scripts for every experiment.
However, do not create a complicated configuration framework if a simple approach using YAML/Python/dataclasses or an established configuration library is sufficient.
Choose the **simplest approach that satisfies the requirements**.
---
## **4. Data Pipeline**
The **data pipeline is a major part of AIO VISION 3**.
Design it so that datasets can be added and configured without modifying the training logic.
The pipeline should clearly separate:
- dataset configuration
- dataset preparation/validation
- dataset loading
- dataset metadata
- training configuration
Do not build a custom dataset framework unless there is a concrete requirement for one.
Use Ultralytics’ supported dataset formats and functionality wherever possible.
---
## **5. Training**
The training loop should be a **thin orchestration layer around Ultralytics**.
Do not recreate Ultralytics’ training loop.
The custom training code should mainly:
1. Load configuration
2. Resolve the dataset
3. Resolve the model/checkpoint
4. Initialize the Ultralytics model(I prefer find tuning/transfer learning instead of random initialization & training form scratch. )
5. Start training with the configured parameters
6. Capture results
7. Log the experiment to MLflow
8. Save/record useful artifacts
If Ultralytics already provides functionality, use it directly.
The training code should be small and easy to understand.
---
## **6. Experiment Tracking**
**MLflow is an important architectural component from the beginning.**
Experiments should be trackable without requiring users to manually log everything.
At minimum, consider tracking:
- experiment name
- run name
- configuration
- model information
- dataset information
- hyperparameters
- training metrics
- validation metrics
- relevant artifacts
- checkpoints/results when appropriate
The exact MLflow integration should remain simple.
Do not build a custom experiment-tracking abstraction over MLflow unless there is a real requirement for it.
---
## **7. Reproducibility**
The repository should make experiments reproducible.
A run should be associated with the configuration that produced it.
Avoid situations where someone has to remember:
“Which command-line arguments did I use for that experiment?”
The configuration should provide the source of truth.
---
## **8. Repository Philosophy**
Think of the project as:
```
Configuration
      ↓
Data Pipeline
      ↓
Ultralytics
      ↓
Training
      ↓
MLflow
      ↓
Artifacts / Metrics / Results
```
The repository should **orchestrate existing tools rather than replace them**.
The value of AIO VISION 3 is not another YOLO implementation.
The value is:
**A clean, reproducible, config-driven workflow for running many YOLO experiments with minimal custom code.**
---
## **9. Important Constraint**
Do **not** over-engineer this project.
Before adding a component, ask:
“Does this solve a real problem that cannot be solved more simply with an existing package or a small amount of code?”
If the answer is no, do not add it.
The repository should be understandable by someone who opens it and immediately sees:
- where configurations live
- where datasets are defined
- where training starts
- where MLflow is integrated
- where results go
