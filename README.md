# 🔋 新能源汽车电池故障预警系统 (Battery Fault Warning)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-WebUI-orange.svg)
![LangChain](https://img.shields.io/badge/LangChain-Agent-black.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-teal.svg)

## 📖 项目概述

本项目致力于**新能源汽车电池的故障检测、风险预警与智能诊断**。  
系统在原有 **5 种故障识别算法** 与 **1 种时间序列预测算法** 的基础上，进一步融合了 **大模型 Agent + RAG 知识库检索 + Web 可视化交互界面**，能够实现从**异常检测**到**维修规范查询**再到**智能问答诊断**的一体化分析流程。

项目整体面向以下目标：

- 对电池运行状态进行多维度故障识别
- 对潜在风险进行提前预警
- 结合维修手册知识库输出可解释诊断建议
- 提供命令行与 Web UI 两种交互方式，便于演示、研究与扩展

---

## 📂 项目结构与核心模块

项目当前分为两大能力层：

### 🛠️ 一、电池故障检测与预警算法模块

#### 故障检测模块
| 文件名 | 功能描述 | 核心算法 / 判定方法 |
| :--- | :--- | :--- |
| `Connection_Fault.py` | **连接异常检测** | 基于 ICC（组内相关系数） |
| `Failure_of_Insulation.py` | **绝缘失效检测** | 基于充电状态的动态阈值 |
| `Sampling_Fault.py` | **采样异常检测** | 滑动窗口 + 分位数方法 |
| `Self_Discharge_Fault.py` | **自放电异常检测** | 滚动窗口平均 + IQR 四分位距 |
| `Sudden_internal_Fault.py` | **突发内部短路检测** | V/T 综合分析 + SDO 谱密度异常算法 |

#### 故障预警模块
| 文件名 | 功能描述 | 核心算法 / 判定方法 |
| :--- | :--- | :--- |
| `TCN-GAWO.py` | **电池故障预测** | 基于时间序列的深度学习预警模型 |

---

### 🤖 二、智能诊断 Agent 模块

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

### 1. 🔌 连接故障检测 (`Connection_Fault.py`)
- **应用场景**：检测电池单元（Cell）之间是否存在连接松动或阻抗异常。
- **技术原理**：通过分析相邻电压序列的相关性来评估连接健康度。
- **判定标准**：使用 **ICC（组内相关系数）** 进行计算，当 ICC 值 **< 0.8** 时，系统自动标记为连接异常。

### 2. 🛡️ 绝缘失效故障检测 (`Failure_of_Insulation.py`)
- **应用场景**：实时监测电池包绝缘电阻，防止漏电及高压风险。
- **技术原理**：根据电池当前所处的**充电状态（State 1-4）**，动态设定不同绝缘安全阈值。
- **判定标准**：动态识别绝缘失效模式，有效降低误报率。

### 3. 📊 采样故障检测 (`Sampling_Fault.py`)
- **应用场景**：识别 BMS 数据采集过程中因传感器异常或通信抖动产生的错误信号。
- **技术原理**：利用**滑动窗口 (Sliding Window)** 扫描电压数据流。
- **判定标准**：基于**分位数方法**自动过滤并标记异常采样点。

### 4. 🔋 自放电故障检测 (`Self_Discharge_Fault.py`)
- **应用场景**：检测电芯内部微短路、漏电或异常掉电引起的自放电问题。
- **技术原理**：采用**滚动窗口平均方法**平滑电压变化序列。
- **判定标准**：结合 **IQR（四分位距）** 识别电压下降速率的异常特征。

### 5. ⚠️ 突发内部短路检测 (`Sudden_internal_Fault.py`)
- **应用场景**：识别高危内部短路与热失控前兆。
- **技术原理**：综合分析**电压**与**温度**双维度特征变化。
- **判定标准**：利用 **SDO（谱密度异常）算法** 快速捕捉短路预警信号。

### 6. 🔮 故障预测算法 (`TCN-GAWO.py`)
- **应用场景**：基于历史时序数据对未来故障风险进行超前预警。
- **技术原理**：引入时间序列建模与优化策略，从“被动检测”升级到“主动预警”。

### 7. 📚 RAG 手册知识库检索 (`agent/build_rag_vectorstore.py`)
- **应用场景**：当用户需要查询维修规范、安全隔离流程、故障排查步骤时，系统可自动检索本地 PDF 手册。
- **技术原理**：对维修文档进行切块、向量化，并使用 **FAISS** 建立本地知识库索引。

### 8. 💬 智能诊断 Agent (`agent/run_agent.py`)
- **应用场景**：让系统根据用户问题自动判断应执行“故障诊断”“规范查询”还是“综合分析”。
- **技术原理**：基于 **LangChain Agent** 组织工具调用流程，结合大模型输出结构化诊断建议。

### 9. 🖥️ Web 可视化交互界面 (`agent/ui.py`)
- **应用场景**：用于课程答辩、系统演示、快速测试与多轮交互分析。
- **支持能力**：
  - 上传电池数据文件
  - 多轮对话与历史会话记录
  - 快捷指令触发常见诊断任务
  - 结合数据分析与维修知识库给出综合结论

---

## 🧠 技术架构

系统采用“**故障算法 + RAG 检索 + LLM Agent**”的组合架构：

1. **底层算法层**：负责完成传统故障检测与时间序列预警
2. **知识库层**：对维修手册和规范文档进行向量化存储
3. **Agent 调度层**：根据用户问题自动选择调用检索工具或诊断工具
4. **交互层**：通过命令行或 Gradio Web UI 提供统一使用入口

这种架构既保留了传统算法的可解释性，也增强了系统在复杂诊断场景下的交互能力与知识支撑能力。

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/owring-code/Battery-fault-warning.git
cd Battery-fault-warning
```

### 2. 运行传统故障检测算法

以连接异常检测为例：

```bash
python Connection_Fault.py
```

如需运行其他算法，可直接执行对应脚本，例如：

```bash
python Self_Discharge_Fault.py
python Sudden_internal_Fault.py
python TCN-GAWO.py
```

---

## 🤖 Agent 模块使用方式

如果你希望使用新增的智能诊断 Agent，请进入 `agent` 目录后运行。

### 1. 安装依赖

```bash
pip install gradio pandas langchain langchain-community langchain-core langchain-openai langchain-huggingface faiss-cpu pdfplumber sentence-transformers
```

### 2. 配置环境变量

Windows PowerShell 示例：

```powershell
$env:API_KEY="your_api_key"
$env:DEEPSEEK_API_KEY="your_api_key"
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

## 📦 构建本地知识库

若你补充了新的维修 PDF 文档，可在 `agent/manuals/` 中放入手册后重新构建向量库：

```bash
cd agent
python -c "from build_rag_vectorstore import build_industrial_knowledge_base; build_industrial_knowledge_base('./manuals','./faiss_industrial_index')"
```

---

## 💡 使用示例

你可以向 Agent 输入类似问题：

```text
电池包日志在 /data/battery_pack.csv，请评估内短路风险并给出隔离步骤。
```

或者：

```text
针对“电池突发型内短路”警报，请检索维修规范并给出标准高压断电流程。
```

---

## 🌟 项目亮点

- 集成 **5 类故障检测算法 + 1 类故障预测模型**
- 融合 **RAG 知识库检索** 与 **大模型智能诊断**
- 支持 **本地维修手册问答**
- 支持 **Web 界面交互、文件上传、多轮会话**
- 兼顾 **算法研究价值** 与 **系统展示效果**

---

## 📌 适用场景

- 新能源汽车电池故障检测研究
- 电池健康管理与预警系统原型
- RAG + Agent 在工业场景中的应用实验

---

## 🔭 后续优化方向

- 接入更真实的电池时序预测模型
- 增加故障结果可视化图表
- 支持更多新能源设备与手册类型
- 完善异常处理与自动化测试
- 增强诊断报告导出能力

---

## 📄 License

本项目采用 MIT License，欢迎学习、研究与二次开发。
