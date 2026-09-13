# 自定义数据 - 多变量时序预测（Multivariate Forecasting）

与 `scripts/multivariate_forecasting/` 下脚本一致：**多变量预测**（`--features M`），模型同时预测所有变量，而非只预测单一目标列。

## 数据格式

- **CSV**：第一列为 `date`（日期时间），其余列为数值型变量。
- `TARGET`：填 CSV 中**存在的任意列名**（如最后一列或 `OT`），仅用于 DataLoader 列顺序；在 M 模式下所有列都会参与输入与预测。

## 使用步骤

1. 将 CSV 放到数据目录，例如 `dataset/custom/your_data.csv`。
2. 编辑 `train_custom.sh`，修改：
   - `ROOT_PATH`：CSV 所在目录（如 `./dataset/custom/`）
   - `DATA_PATH`：CSV 文件名（如 `your_data.csv`）
   - `NUM_VARIATES`：**变量数 = CSV 列数 - 1**（去掉 date）
   - `TARGET`：任意一个存在的列名（如最后一列）
3. 在 **iTransformer 项目根目录** 下执行：
   ```bash
   bash ./scripts/custom_data/train_custom.sh
   ```
   Windows 可用 Git Bash 或 WSL。

脚本会依次跑 4 组预测长度：**96 / 192 / 336 / 720**（与 Traffic、ECL 等脚本一致），对应 `model_id`：`custom_96_96`、`custom_96_192`、`custom_96_336`、`custom_96_720`。

## 数据划分

`Dataset_Custom` 按 **70% 训练、10% 验证、20% 测试** 自动划分。
