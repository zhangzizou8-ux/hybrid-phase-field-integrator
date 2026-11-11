# Project resources

This repository contains source files, configuration files, and model checkpoints.

## Directory overview

| Directory | Contents |
|---|---|
| `src/` | Source modules |
| `configs/` | Configuration files |
| `checkpoints/` | Model checkpoints |
| `data_generation/` | Data preparation scripts and notebooks |
| `validation/` | Verification workflows |
| `evaluation/` | Evaluation workflows |
| `tools/` | Utility scripts |
| `provenance/` | Environment records |
| `archived_workflows/` | Archived scripts |

## Requirements

Python, NumPy, SciPy, PyTorch, CuPy, tqdm, and psutil. Notebook workflows also require Jupyter/IPython. Selected evaluation and plotting scripts use pandas, scikit-image, and Matplotlib. GPU-dependent workflows require an NVIDIA CUDA environment. Environment records are provided in `provenance/`.

## Setup

1. Review the environment records and install the dependencies required by the selected script or notebook.
2. Update input and output paths for your local environment and resolve any local module dependencies before execution.
3. Use each checkpoint with its corresponding configuration and normalization settings.

Some scripts retain original module names and environment-specific paths. Review these dependencies when preparing a local working copy.

## Resource availability

Datasets are not included in this repository. Some evaluation workflows require sampling records or archived outputs that are not distributed. Exact reproduction of all archived outputs is not guaranteed.
