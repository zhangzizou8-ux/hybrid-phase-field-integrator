# hybrid-phase-field-integrator
Code and pretrained models for **Hierarchical hybrid learning-based coarse-time integration framework for long-time phase-field dynamics**.

The method combines learned coarse-time propagation, local physical correction, and intermittent full-order model (FOM) advancement.

## Contents

| Folder | Description |
|---|---|
| `data_generation/` | Phase-field data generation |
| `src/hybrid/` | Hybrid training and inference |
| `src/baselines/` | U-Net and FNO baselines |
| `src/ablations/` | No-local, No-geometry and No-PDE ablations |
| `checkpoints/` | Six pretrained checkpoints |
| `validation/` | Temporal and spatial refinement |
| `evaluation/` | FWHM and historical evaluation |
| `configs/`, `provenance/` | Configuration and environment records |
| `archived_workflows/`, `tools/` | Archived workflows and visualization |

## Requirements

Python, NumPy, SciPy, PyTorch, CuPy, tqdm and psutil. Notebooks require Jupyter/IPython; evaluation and plotting also use pandas, scikit-image and Matplotlib. The FOM requires an NVIDIA CUDA GPU. Environment records are in `provenance/`.

## Usage

1. Run `data_generation/generate_phase_field.ipynb` in a Kaggle GPU environment. Set `RUN_START` and `RUN_END`; the default generates Run00 only.
2. Train on Run01–Run24, validate on Run25–Run30, and evaluate on Run00 and Run31–Run35. Run00 also served development diagnostics.
3. Train using the corresponding code in `src/`, or load a supplied checkpoint from `checkpoints/` for inference. Keep each checkpoint's own normalization settings.
4. Compare predictions with FOM references at matching physical steps.

Datasets are not included. Generate them using the phase-field notebook. The default Hybrid rollout uses ten history frames, 80 learned advances and two 2,000-step FOM bursts, ending at fine step 504,000.

## Implementation notes

Files were renamed and organized without changing their contents. Original imports, filename dependencies and absolute paths remain. Configure paths and restore expected dependency names in a local working copy before running; for example, Hybrid inference is imported as `run03.py`, and No-PDE inference loads `01_train_no_pde_keep_mean.py` beside itself.

Archived evaluators require sampling records or historical outputs that are not distributed. Bitwise reproduction of all historical results is not guaranteed.

---

# 相场动力学的混合粗时间步积分

论文 **Hierarchical hybrid learning-based coarse-time integration framework for long-time phase-field dynamics** 的代码与预训练模型。

该方法结合学习式粗时间步推进、局部物理修正与间歇性全阶模型（FOM）推进。

## 文件结构

| 目录 | 说明 |
|---|---|
| `data_generation/` | 相场数据生成 |
| `src/hybrid/` | Hybrid训练与推理 |
| `src/baselines/` | U-Net与FNO基线 |
| `src/ablations/` | No-local、No-geometry、No-PDE消融 |
| `checkpoints/` | 六种预训练模型权重 |
| `validation/` | 时间与空间细化验证 |
| `evaluation/` | FWHM与历史评价代码 |
| `configs/`, `provenance/` | 配置与环境记录 |
| `archived_workflows/`, `tools/` | 归档流程与可视化 |

## 运行环境

依赖Python、NumPy、SciPy、PyTorch、CuPy、tqdm和psutil。Notebook需要Jupyter/IPython，评价与绘图还需要pandas、scikit-image和Matplotlib。FOM求解器需要支持CUDA的NVIDIA GPU。已有环境记录见 `provenance/`。

## 基本使用

1. 在Kaggle GPU环境运行 `data_generation/generate_phase_field.ipynb`，设置 `RUN_START` 与 `RUN_END`；默认只生成Run00。
2. Run01–Run24用于训练，Run25–Run30用于验证，Run00及Run31–Run35用于评价；Run00也用于开发诊断。
3. 使用 `src/` 中对应代码训练，或加载 `checkpoints/` 中的模型进行推理。保留各模型自身的归一化设置。
4. 按实际物理步数对齐预测结果与FOM参考结果。

本仓库不提供数据集，请通过相场代码自行生成。默认Hybrid使用10帧历史输入、80次学习推进和两次各2,000细步的FOM推进，终点为第504,000细步。

## 使用说明

文件仅重新命名和分类，内容保持不变。原始导入关系、依赖文件名及绝对路径仍保留。运行前须在本地工作副本中设置路径并恢复依赖名称，例如Hybrid推理模块需命名为 `run03.py`，No-PDE推理还需在同目录放置 `01_train_no_pde_keep_mean.py`。

归档评价脚本依赖未随仓库提供的采样记录或历史输出。本仓库不保证逐位复现全部历史结果。
