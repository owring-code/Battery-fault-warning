# 新能源汽车故障预警
## 项目概述
本项目是一个用于新能源汽车电池故障检测，包含五种不同类型的故障识别算法。以及一种预测算法用于故障预警。

## 文件说明
项目包含以下五个核心识别模块：

Connection_Fault.py - 连接异常检测

Failure_of_Insulation.py - 绝缘失效检测

Sampling_Fault.py - 采样异常检测

Self_Discharge_Fault.py - 自放电异常检测

Sudden_internal_Fault.py - 突发内部短路检测

一种预测算法

TCN-GAWO.py


## 功能特性
1. 连接故障检测 (Connection_Fault.py)
* 检测电池单元之间的连接异常

* 使用ICC（组内相关系数）分析相邻电压序列的相关性

* 当ICC值低于0.8时标记为连接异常

2. 绝缘失效故障检测 (Failure_of_Insulation.py)
* 检测电池绝缘电阻故障

* 根据充电状态（1-4）设置不同的绝缘电阻阈值

* 动态识别绝缘失效模式

3. 采样故障检测 (Sampling_Fault.py)
* 识别数据采集过程中的异常

* 使用滑动窗口分析电压数据

* 基于分位数方法检测异常采样

4. 自放电故障检测 (Self_Discharge_Fault.py)
* 检测电池自放电异常

* 使用滚动窗口平均方法分析电压变化

* 基于IQR（四分位距）方法识别异常点

5. 突发内部短路故障检测 (Sudden_internal_Fault.py)
* 检测突发性内部短路故障

* 结合电压和温度数据进行综合分析

* 使用SDO（谱密度异常）算法检测异常
