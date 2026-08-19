# **AIO VISION 3 — Architecture, Implementation & Repository Research Direction**

I am designing **ALL-IN-ONE-VISION 3 (AIO VISION 3)**.

---

# **1. What AIO VISION 3 Actually Is**

AIO VISION 3 is intended to be a **simple, low-code, experiment-oriented repository for training, evaluating, and deploying YOLO models using Ultralytics**.

**Reuse mature source code. Build the workflow around it.**

Ultralytics already provides most of the difficult ML functionality.

Therefore, the value of AIO VISION 3 should primarily come from:

1. **Scripts**
2. **Configuration**
3. **Project structure**
4. **Data workflow**
5. **Experiment management**
6. **Reproducibility**
7. **Execution/environment management**
8. **Deployment**
9. **Integration of mature tools**

Custom source code should be kept as small as reasonably possible.

---

# **2. The Most Important Principle: Scripts + Configuration**

### **Configuration defines WHAT should happen.**

For example:

- dataset
- model
- pretrained checkpoint
- training parameters
- image size
- batch size
- epochs
- optimizer
- learning rate
- augmentation
- device
- experiment name
- tracking settings
- output locations
- deployment settings

### **Scripts define HOW the workflow is executed.**

For example:

```
scripts/
├── prepare_dataset.py
├── validate_dataset.py
├── train.py
├── evaluate.py
├── predict.py
├── export.py
└── deploy.py
```

The important point is:

**A new experiment should normally be created by changing configuration and/or invoking an existing script, not by writing a new Python program.**

---

# **3. Custom Source Code Is NOT the Main Product**

This is extremely important.

Do not assume that AIO VISION 3 needs a large `src/` architecture.

We should actively ask:

“Can this be done entirely by an existing mature package plus a small script?”

If yes, prefer that.

For example, if Ultralytics already provides:

- model architectures
- pretrained models
- transfer learning
- training
- validation
- inference
- augmentation
- optimizers
- schedulers
- losses
- checkpoint handling
- export

then **use Ultralytics directly**.

Do not reimplement these capabilities.

The ideal custom source code may be surprisingly small.

That is a feature, not a weakness.

---

# **4. Model Training Philosophy**

Model training should primarily be:

**Scripts + Configuration + Ultralytics**

rather than:

**Custom Training Framework + Many Python Classes**

The training layer should be a thin operational layer around Ultralytics.

A typical workflow might conceptually be:

```
experiment.yaml
       ↓
train.py
       ↓
Ultralytics
       ↓
training
       ↓
MLflow
       ↓
checkpoints / metrics / artifacts
```

The project should favor **transfer learning and fine-tuning of pretrained models** rather than assuming random initialization and training from scratch.

Training from scratch should be possible when genuinely useful, but it should not be the conceptual center of the repository.

---

# **5. Scripts Are First-Class Architecture**

Treat scripts as an important architectural component, not as miscellaneous helper files.

Study how strong ML repositories use scripts for:

- dataset preparation
- dataset validation
- downloading data
- preprocessing
- training
- evaluation
- inference
- benchmarking
- model export
- experiment execution
- deployment
- environment setup
- reproducibility

The goal is to understand how a repository can expose a powerful ML workflow through a **small number of clear executable entry points**.

We want commands that are easy to understand, for example:

```bash
python scripts/train.py configs/experiments/yolo11_dataset_a.yaml
```

or:

```bash
python scripts/evaluate.py configs/experiments/yolo11_dataset_a.yaml
```

Do not assume this exact CLI design is required. Research good existing patterns first.

---

# **6. Configuration Is the Declarative Layer**

Configuration should make experiments easy to reproduce and vary.

A new experiment should generally be achievable through configuration rather than source-code modification.

Potential configuration categories include:

```
configs/
├── datasets/
├── experiments/
├── training/
├── deployment/
└── ...
```

But do not blindly adopt this structure.

Study real repositories and choose the simplest structure that works.

Avoid building a complicated configuration framework unless there is a concrete reason.

YAML, TOML, Python configuration, dataclasses, Hydra, OmegaConf, or another established solution may be appropriate.

**Research before deciding.**

---

# **7. Data Pipeline**

The data workflow is one of the major engineering concerns.

AIO VISION 3 should make it straightforward to:

- add datasets
- validate datasets
- prepare datasets
- define dataset metadata
- configure dataset locations
- convert formats when necessary
- reproduce dataset preparation
- connect datasets to experiments

However:

Do not build a custom dataset framework unless the real requirements justify it.

Use existing dataset tooling and Ultralytics-supported formats wherever practical.

The important thing is the **workflow and organization**, not creating abstractions for their own sake.

---

# **8. Experiment Management**

AIO VISION 3 should make running many experiments practical.

Conceptually:

```
Dataset A × Model A × Training Config A
Dataset A × Model A × Training Config B
Dataset A × Model B × Training Config A
Dataset B × Model A × Training Config A
...
```

The user should be able to run these combinations primarily through:

- configuration
- existing scripts
- command-line execution

rather than creating new source files.

The repository should make it obvious:

- what experiment was run
- which dataset was used
- which model/checkpoint was used
- which configuration was used
- what metrics were produced
- where artifacts were saved

---

# **9. Experiment Tracking**

**MLflow is an important component from the beginning.**

The system should automatically or conveniently track relevant information such as:

- experiment name
- run name
- configuration
- dataset information
- model/checkpoint
- hyperparameters
- training metrics
- validation metrics
- artifacts
- checkpoints/results where appropriate

But do not create a large custom experiment-tracking abstraction over MLflow.

Use MLflow directly where practical.

The configuration used for a run should be preserved with the experiment.

---

# **10. Reproducibility**

AIO VISION 3 should make experiments reproducible.

A run should be traceable back to:

```
Experiment
   ↓
Configuration
   ↓
Dataset
   ↓
Model / Checkpoint
   ↓
Code / Environment
   ↓
Results / Artifacts
```

The repository should reduce reliance on undocumented terminal commands and personal memory.

Someone should be able to look at the repository and understand how an experiment was executed.

---

# **11. Project Structure Matters**

The repository structure itself is an important part of the project.

Study mature ML repositories to understand how they organize:

- scripts
- configs
- datasets
- experiments
- outputs
- checkpoints
- source code
- tests
- documentation
- environments
- Docker
- deployment
- CI/CD where appropriate

Do not copy a structure simply because it looks professional.

Prefer structures that make the workflow immediately understandable.

A person opening the repository should quickly be able to answer:

Where are the configurations?

What script runs training?

How do I prepare the dataset?

How do I evaluate a model?

Where do experiment results go?

How do I reproduce an experiment?

How do I deploy the resulting model?

---

# **12. Environment & Deployment Are Part of the Architecture**

Do not focus only on training.

Study how mature repositories handle:

- Python environments
- dependency management
- CPU/GPU execution
- Docker
- deployment
- model export
- inference environments
- configuration of runtime environments
- reproducible execution

The project should eventually make it straightforward to move from:

```
Development
    ↓
Training
    ↓
Evaluation
    ↓
Export
    ↓
Deployment
```

Deployment does not necessarily need to be fully implemented immediately.

However, the architecture should not make deployment an afterthought.

---

# **13. Prefer Mature Tools Over Custom Implementations**

Before implementing anything, ask:

**Does a mature package already solve this problem?**

If yes, strongly prefer using it.

Potential examples include:

- Ultralytics → YOLO/model training/inference/export
- PyTorch → deep-learning infrastructure
- MLflow → experiment tracking
- established dataset tooling → data preparation/validation
- established configuration libraries → configuration management
- Docker → reproducible environments/deployment

These are examples, not requirements.

Research what strong real-world repositories actually use.

The project should **compose mature software rather than compete with it**.

---

Lets go . write the code!