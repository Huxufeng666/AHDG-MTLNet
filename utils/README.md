# utils 说明

本目录存放数据处理、评估、损失、可视化和日志等公共工具。

## 文件说明

- `__init__.py`: 工具包初始化文件。
- `Early_Stopp.py`: 早停逻辑实现。
- `convert_busi_to_nnunet.py`: 将 BUSI 数据整理为 nnU-Net 风格结构的转换脚本。
- `data.py`: 数据相关辅助定义，偏旧。
- `data_loading.py`: 通用图像/掩码读取与 `BasicDataset` 实现。
- `dice_score.py`: Dice 相关指标实现。
- `evaluate.py`: 通用验证函数，适配原始 UNet 风格推理。
- `get_data.py`: 当前 BUSI 数据集主要读取入口。
- `get_text_data.py`: 文本/多模态相关数据读取辅助。
- `hubconf.py`: 兼容 PyTorch Hub 的模型导出入口。
- `logger.py`: 训练日志 CSV 记录工具。
- `logger_text_mtl.py`: 文本多任务版本的日志工具。
- `loss.py`: 损失函数集合，包括 Dice 与边界损失。
- `plot.py`: 训练曲线可视化工具。
- `plot_text_mtl.py`: 文本多任务版本绘图工具。
- `predict.py`: 原始 UNet 风格单图/批量预测脚本。
- `save.py`: 最优权重和 Top-K 模型保存逻辑。
- `tools.py`: 通用辅助函数集合。
- `tools_text.py`: 文本多任务相关辅助工具。
- `utils.py`: 其他基础辅助函数。
