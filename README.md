# 基于 Transformer 的新能源汽车动力电池多故障预警

本仓库为毕业论文《基于 Transformer 的新能源汽车动力电池多故障预警方法研究》的配套代码，面向新能源汽车动力电池运行数据中的多类故障识别与提前预警任务。

项目围绕真实车端数据中标签稀缺、噪声较大、多故障并发和故障演化隐蔽等问题，构建了弱监督标签生成、结构化样本构建、基线模型对比和共享 Transformer 双任务模型训练流程。

## 研究内容

- 故障类型：自放电异常、突发型内短路、连接异常、采样异常、绝缘失效。
- 标签构造：基于故障机理规则重构逐帧标签，并通过时间平滑、邻域一致性和质量控制降低短时噪声。
- 样本构建：默认采用 30 帧时间窗口、3 帧滑动步长和 60 秒预警提前量，输出 `samples_master.csv`、`features_all.csv` 和 `dataset_pack.npz`。
- 模型结构：共享 Transformer 时序编码器提取多故障共性演化表示，故障专属统计特征进入专家分支，分别输出识别与预警结果。
- 对比实验：包含 Threshold Trend、LightGBM、LSTM、Transformer 和本文方法。

论文实验中，本文方法在五类故障识别任务上取得 F1=0.858、Recall=0.847；在五类故障预警任务上取得 Warning F1=0.790、Warning Recall=0.782、FPR=0.026。三类核心故障场景下，识别 F1 达到 0.922，预警 F1 达到 0.856。

## 仓库结构

```text
.
├── README.md
├── .gitignore
└── code
    ├── battery_thesis          # 核心模块：配置、标签、特征、模型、训练、指标和结果写出
    ├── assets                  # 故障案例图脚本使用的小型参考资产
    ├── scripts                 # 数据构建、标签构造、训练评估、图表导出脚本
    ├── tests                   # 单元测试与 CLI 合约测试
    ├── requirements.txt
    └── pytest.ini
```

## 环境安装

建议使用 Python 3.10 或更高版本，并在虚拟环境中安装依赖。

```powershell
cd code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux 或服务器环境可使用：

```bash
cd code
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 数据说明

原始车辆运行数据和实验生成结果体积较大，且可能包含隐私或授权限制，默认不随仓库上传。代码支持两类数据入口：

- 结构化数据目录：默认示例为 `F:\Data_set`，可通过 `--structured-root` 指定。
- 原始 RAW 数据目录：默认示例为 `F:\RAW_DATA\data`，可通过 `--raw-root` 指定。

运行时建议显式传入自己的数据路径，避免依赖默认路径。

## 运行流程

1. 构建数据集元信息。

```bash
python scripts/build_dataset_meta.py --structured-root /path/to/Data_set --raw-root /path/to/RAW_DATA/data
```

2. 重构故障标签。

```bash
python scripts/build_labels.py --structured-root /path/to/Data_set --raw-root /path/to/RAW_DATA/data
```

3. 构建窗口样本和张量包。

```bash
python scripts/build_samples.py --mode full --structured-root /path/to/Data_set --raw-root /path/to/RAW_DATA/data
```

4. 运行基线模型。

```bash
python scripts/run_threshold_warning_baseline.py
python scripts/run_lightgbm_recognition.py --task identification
python scripts/run_lightgbm_recognition.py --task warning
python scripts/run_sequence_baseline.py --architecture lstm --task identification --device auto
python scripts/run_sequence_baseline.py --architecture transformer --task identification --device auto
```

5. 运行本文双任务模型。

```bash
python scripts/run_dual_task_model.py --device auto --epochs-stage1 5 --epochs-joint 5
```

6. 运行消融实验。

```bash
python scripts/run_dual_task_model.py --ablation no_fault_specific_features --device auto
python scripts/run_dual_task_model.py --ablation no_expert_heads --device auto
python scripts/run_dual_task_model.py --ablation no_warning_task --device auto
python scripts/run_dual_task_model.py --ablation no_label_quality_control --device auto
```

## 测试

```bash
cd code
pytest
```

若只想快速检查命令行入口和核心模块：

```bash
pytest tests/test_cli_contracts.py tests/test_models_and_metrics.py tests/test_training.py
```

部分测试会检查完整实验产物、论文图表或运行手册；这些大文件未纳入仓库时会自动跳过，不影响核心代码测试。

## 主要代码入口

- `battery_thesis/models.py`：共享 Transformer 编码器、故障专家分支和 LSTM/Transformer 基线模型。
- `battery_thesis/training.py`：张量包读取、样本对齐、归一化、损失函数和训练预测工具。
- `battery_thesis/rule_reconstruction.py`：五类故障规则重构逻辑。
- `battery_thesis/samples.py`：窗口样本、统计特征和张量包构建逻辑。
- `scripts/run_dual_task_model.py`：本文主模型训练、验证阈值选择、测试集评估和结果写出。
- `scripts/run_lightgbm_recognition.py`：LightGBM 识别与预警基线。
- `scripts/run_sequence_baseline.py`：LSTM 和 Transformer 时序基线。

## 复现实验注意事项

- 预警任务默认使用验证集搜索 F1 最优阈值，并固定到测试集评估。
- `samples_master.csv` 与 `dataset_pack.npz` 的 `sample_id` 顺序会在读取时进行硬校验，顺序不一致时会直接报错。
- 正式实验建议在 GPU 或服务器环境运行；本地可通过 `--max-train-samples`、`--max-val-samples`、`--max-test-samples` 做小规模链路验证。
- 论文指标来自固定测试集结果，重新运行时可能因数据划分、依赖版本和随机种子产生轻微差异。
