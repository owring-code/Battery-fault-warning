# 🔋 新能源汽车电池故障预警系统 (Battery Fault Warning)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformer-ee4c2c.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Baseline-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-WebUI-orange.svg)
![LangChain](https://img.shields.io/badge/LangChain-Agent-black.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-teal.svg)
![Pytest](https://img.shields.io/badge/Tests-pytest-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 项目概述

本项目致力于**新能源汽车动力电池的多故障识别、风险提前预警与智能诊断交互**。

当前仓库保留原有 **Agent + RAG 知识库检索 + Gradio Web UI** 能力，同时将 `code/` 目录更新为论文《基于 Transformer 的新能源汽车动力电池多故障预警方法研究》的新版实验代码。系统能够覆盖从**弱监督标签构造**、**多故障识别与预警建模**到**维修规范查询与智能问答诊断**的一体化流程。

项目整体面向以下目标：

- 对电池运行状态进行多维度故障识别
- 对潜在风险进行固定提前量预警
- 基于真实车端数据构造弱监督训练标签
- 对比规则、机器学习与深度时序模型
- 结合维修手册知识库输出可解释诊断建议
- 提供实验代码、命令行工具与 Web UI 交互入口

---

## 📂 项目结构与核心模块

项目当前分为两大能力层：

### 🛠️ 一、电池故障识别与预警算法模块 (`code/`)

#### 数据处理与标签构造

| 文件名 / 目录 | 功能描述 | 核心能力 |
| :--- | :--- | :--- |
| `battery_thesis/field_mapping.py` | 字段映射与传感器序列解析 | 统一结构化数据与 RAW 数据字段 |
| `battery_thesis/rule_reconstruction.py` | 故障规则重构 | 基于机理规则生成逐帧弱监督标签 |
| `battery_thesis/label_pipeline.py` | 标签流水线 | 标签平滑、质量控制与事件级合并 |
| `battery_thesis/samples.py` | 样本构建 | 时间窗口切分、统计特征构造、张量包生成 |
| `scripts/build_dataset_meta.py` | 数据集元信息构建 | 车辆统计、故障覆盖与划分约束 |
| `scripts/build_labels.py` | 标签生成入口 | 批量输出每车标签文件 |
| `scripts/build_samples.py` | 样本生成入口 | 输出 `samples_master.csv`、`features_all.csv`、`dataset_pack.npz` |

#### 模型训练与评估

| 文件名 / 目录 | 功能描述 | 核心算法 / 方法 |
| :--- | :--- | :--- |
| `battery_thesis/models.py` | 主模型与序列基线模型 | Transformer、LSTM、故障专家分支 |
| `battery_thesis/training.py` | 训练工具函数 | 归一化、损失函数、DataLoader、预测得分 |
| `battery_thesis/metrics.py` | 指标计算 | F1、Recall、Precision、PR-AUC、ROC-AUC、FPR、Mean Lead Time |
| `scripts/run_dual_task_model.py` | 本文主模型训练入口 | 共享 Transformer 编码器 + 专家分支 + 双任务学习 |
| `scripts/run_sequence_baseline.py` | 时序基线入口 | LSTM / Transformer 单任务基线 |
| `scripts/run_lightgbm_recognition.py` | 机器学习基线入口 | LightGBM 识别与预警实验 |
| `scripts/run_threshold_warning_baseline.py` | 规则基线入口 | Threshold Trend 预警基线 |

#### 结果导出与测试

| 文件名 / 目录 | 功能描述 |
| :--- | :--- |
| `battery_thesis/results.py` | 统一写出 summary、prediction、data points 等结果文件 |
| `scripts/export_recognition_figures.py` | 导出故障识别实验图 |
| `scripts/export_warning_figures.py` | 导出故障预警实验图 |
| `scripts/export_ablation_figures.py` | 导出消融实验图 |
| `scripts/export_fault_case_polished_figures.py` | 导出典型故障案例可视化 |
| `tests/` | 单元测试、CLI 合约测试、训练流程测试 |

---

### 🤖 二、智能诊断 Agent 模块 (`agent/`)

| 文件名 / 目录 | 功能描述 |
| :--- | :--- |
| `agent/run_agent.py` | 命令行版智能诊断 Agent |
| `agent/ui.py` | 基于 Gradio 的 Web 可视化交互界面 |
| `agent/build_rag_vectorstore.py` | 构建维修手册 PDF 向量知识库 |
| `agent/manuals/` | 原始 PDF 手册与维修规范文档 |
| `agent/faiss_industrial_index/` | 本地 FAISS 向量索引库 |
| `agent/tests/` | Agent 相关测试代码 |
| `agent/docs/` | 设计文档、实现计划与开发记录 |

---

## ✨ 功能特性详解

### 1. 🔋 五类动力电池故障建模

系统面向论文中定义的五类典型故障：

| 故障缩写 | 故障类型 | 数据表现 |
| :--- | :--- | :--- |
| `sd` | 自放电异常 | 最弱单体电压持续偏移、静置压降异常 |
| `isc` | 突发型内短路 | 电压突降、温升异常、压差温差扩大 |
| `conn` | 连接异常 | 相邻单体响应差异增大、电压一致性变弱 |
| `samp` | 采样异常 | 传感器跳变、局部残差异常、采样点失真 |
| `ins` | 绝缘失效 | 绝缘电阻降低、低绝缘状态持续 |

### 2. 🏷️ 弱监督标签构造

- **应用场景**：真实车辆运行数据往往缺少精确故障标签。
- **技术原理**：根据故障机理重构逐帧规则标签，再进行时间平滑和质量控制。
- **输出结果**：生成适合模型训练的 `label_final_*` 标签列。

该流程能够降低孤立噪声点、短时误报和异常值对训练监督信号的影响。

### 3. 🧩 多粒度样本构建

- 默认窗口长度：`30` 帧
- 默认滑动步长：`3` 帧
- 默认预警提前量：`60` 秒
- 输出文件：`samples_master.csv`、`features_all.csv`、`dataset_pack.npz`

样本同时保留**原始时序特征**和**故障专属统计特征**，便于模型同时学习时序演化规律与机理特征。

### 4. 🧠 共享 Transformer 编码器

主模型使用共享 Transformer 编码器提取多类故障共同的时序演化特征，例如压差扩大、温度变化、绝缘水平变化和电压波动增强等。

相比独立建模，共享编码器能够减少参数冗余，并提升多故障任务下的特征利用效率。

### 5. 🛠️ 故障专家分支

不同故障的成因和数据表现差异明显，因此模型在共享表示之后为每类故障设置独立专家分支。

每个专家分支接收：

- Transformer 提取的共享时序表示
- 当前故障对应的专属统计特征

并分别输出：

- 当前状态识别结果
- 未来风险预警结果

### 6. 🔮 识别与预警双任务学习

模型将故障识别任务和故障预警任务统一训练：

- **识别任务**：判断当前窗口是否已经处于某类故障状态
- **预警任务**：判断当前尚未故障的窗口是否会在未来固定提前量内进入故障状态

双任务学习能够增强模型对故障前兆信息的捕捉能力。

### 7. 📚 RAG 手册知识库检索

- **应用场景**：查询维修规范、安全隔离流程、故障排查步骤。
- **技术原理**：对维修手册 PDF 切块、向量化，并使用 FAISS 建立本地知识库索引。
- **交互入口**：`agent/run_agent.py` 与 `agent/ui.py`。

### 8. 💬 智能诊断 Agent

Agent 会根据用户问题自动组织工具调用流程：

- 查询维修手册与规范
- 结合上传数据做数据画像
- 触发故障风险评估工具
- 输出结构化诊断建议

### 9. 🖥️ Web 可视化交互界面

`agent/ui.py` 基于 Gradio 构建，支持：

- 上传电池数据文件
- 多轮对话与历史会话记录
- 多文件会话管理
- 快捷指令触发常见诊断任务
- 结合数据分析与维修知识库给出综合结论

---

## 🧪 论文实验结果

论文实验表明，本文方法在五类故障识别与预警任务上均优于各类对比方法。

### 故障识别任务

| 实验场景 | F1 | Recall | PR-AUC |
| :--- | :---: | :---: | :---: |
| 五类故障识别 | 0.858 | 0.847 | 0.949 |
| 三类核心故障识别 | 0.922 | 0.932 | 0.979 |

### 故障预警任务

| 实验场景 | Warning F1 | Warning Recall | FPR |
| :--- | :---: | :---: | :---: |
| 五类故障预警 | 0.790 | 0.782 | 0.026 |
| 三类核心故障预警 | 0.856 | 0.872 | 0.030 |

### 消融实验结论

| 消融设置 | ID-F1 | Warn-F1 | 主要影响 |
| :--- | :---: | :---: | :--- |
| Full model | 0.858 | 0.790 | 完整模型 |
| No fault-specific features | 0.733 | 0.639 | 故障机理特征缺失 |
| No expert branches | 0.689 | 0.600 | 多故障差异表达不足 |
| No warning task | 0.744 | 0.497 | 前兆建模能力下降 |
| No label-quality control | 0.664 | 0.559 | 标签噪声显著影响上限 |

---

## 🧠 技术架构

系统采用“**Transformer 多故障预警模型 + RAG 检索 + LLM Agent**”的组合架构：

1. **数据处理层**：完成字段统一、规则重构、标签质量控制与窗口样本构建
2. **算法模型层**：完成故障识别、风险预警、基线对比与消融实验
3. **知识库层**：对维修手册和规范文档进行向量化存储
4. **Agent 调度层**：根据用户问题自动选择检索工具、数据分析工具或诊断工具
5. **交互层**：通过命令行或 Gradio Web UI 提供统一使用入口

这种架构既保留了故障机理规则和模型指标的可解释性，也增强了系统在复杂诊断场景下的交互能力与知识支撑能力。

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/owring-code/Battery-fault-warning.git
cd Battery-fault-warning
```

### 2. 安装算法实验依赖

Windows PowerShell：

```powershell
cd code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux / 服务器：

```bash
cd code
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 3. 运行核心测试

```bash
pytest
```

若只想快速检查核心代码与命令行入口：

```bash
pytest tests/test_cli_contracts.py tests/test_models_and_metrics.py tests/test_training.py
```

部分测试依赖完整实验产物、论文图表或运行手册；这些大文件未纳入仓库时会自动跳过，不影响核心代码测试。

---

## 📦 数据准备与样本构建

原始车辆运行数据和实验生成结果体积较大，且可能包含隐私或授权限制，默认不随仓库上传。

运行时建议显式传入自己的数据路径：

```bash
python scripts/build_dataset_meta.py \
  --structured-root /path/to/Data_set \
  --raw-root /path/to/RAW_DATA/data
```

生成故障标签：

```bash
python scripts/build_labels.py \
  --structured-root /path/to/Data_set \
  --raw-root /path/to/RAW_DATA/data
```

构建窗口样本：

```bash
python scripts/build_samples.py \
  --mode full \
  --structured-root /path/to/Data_set \
  --raw-root /path/to/RAW_DATA/data
```

---

## 🧠 模型训练与评估

### 1. 规则预警基线

```bash
python scripts/run_threshold_warning_baseline.py
```

### 2. LightGBM 基线

```bash
python scripts/run_lightgbm_recognition.py --task identification
python scripts/run_lightgbm_recognition.py --task warning
```

### 3. LSTM / Transformer 时序基线

```bash
python scripts/run_sequence_baseline.py --architecture lstm --task identification --device auto
python scripts/run_sequence_baseline.py --architecture transformer --task identification --device auto
```

### 4. 本文双任务模型

```bash
python scripts/run_dual_task_model.py \
  --device auto \
  --epochs-stage1 5 \
  --epochs-joint 5
```

### 5. 消融实验

```bash
python scripts/run_dual_task_model.py --ablation no_fault_specific_features --device auto
python scripts/run_dual_task_model.py --ablation no_expert_heads --device auto
python scripts/run_dual_task_model.py --ablation no_warning_task --device auto
python scripts/run_dual_task_model.py --ablation no_label_quality_control --device auto
```

---

## 🤖 Agent 模块使用方式

如果需要使用智能诊断 Agent，请进入 `agent/` 目录运行。

### 1. 安装 Agent 依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

Windows PowerShell 示例：

```powershell
$env:API_KEY="your_api_key"
$env:DEEPSEEK_API_KEY="your_api_key"
```

Linux / macOS 示例：

```bash
export API_KEY="your_api_key"
export DEEPSEEK_API_KEY="your_api_key"
```

### 3. 启动命令行 Agent

```bash
cd agent
python run_agent.py
```

### 4. 启动 Web UI

```bash
cd agent
python ui.py
```

---

## 📚 构建本地知识库

若补充了新的维修 PDF 文档，可放入 `agent/manuals/` 后重新构建向量库：

```bash
cd agent
python -c "from build_rag_vectorstore import build_industrial_knowledge_base; build_industrial_knowledge_base('./manuals','./faiss_industrial_index')"
```

---

## 📊 结果导出

导出识别实验图：

```bash
python scripts/export_recognition_figures.py
```

导出预警实验图：

```bash
python scripts/export_warning_figures.py
```

导出消融实验图：

```bash
python scripts/export_ablation_figures.py
```

聚合实验表格：

```bash
python scripts/aggregate_round1_tables.py
```

---

## 💡 使用示例

你可以向 Agent 输入类似问题：

```text
电池包日志在 D:\data\battery_pack.csv，请评估内短路风险并给出隔离步骤。
```

或者：

```text
针对“电池突发型内短路”警报，请检索维修规范并给出标准高压断电流程。
```

算法实验中也可以通过限制样本数进行快速链路验证：

```bash
python scripts/run_dual_task_model.py --device cpu --max-train-samples 500 --max-val-samples 200 --max-test-samples 200
```

---

## 🌟 项目亮点

- 覆盖 **5 类动力电池典型故障**
- 支持 **故障识别 + 风险预警** 双任务建模
- 融合 **弱监督标签构造 + 机理特征分组**
- 使用 **共享 Transformer 编码器 + 故障专家分支**
- 内置 **Threshold、LightGBM、LSTM、Transformer** 多类基线
- 保留 **RAG 知识库检索 + 大模型 Agent + Gradio Web UI**
- 支持 **消融实验、图表导出与自动化测试**

---

## 📌 适用场景

- 新能源汽车动力电池故障诊断研究
- 电池安全预警模型原型验证
- 多标签时序分类与提前预警实验
- RAG + Agent 在工业诊断场景中的应用实验
- 毕业论文、课程设计和科研复现实验

---

## 🔭 后续优化方向

- 接入更多真实车辆运行数据进行外部验证
- 增强在线推理与实时预警能力
- 增加模型解释性分析与注意力可视化
- 将新版 Transformer 模型封装为 Agent 可直接调用的诊断工具
- 支持自动生成诊断报告
- 将训练流程封装为更完整的实验配置系统

---

## 📄 License

本项目采用 MIT License，欢迎学习、研究与二次开发。
