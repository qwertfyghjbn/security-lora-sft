# HTTP 安全流量分类器 —— LoRA 微调实验

基于 Qwen2.5-1.5B-Instruct + 4-bit QLoRA，对 HTTP 请求进行攻击类型与风险等级分类。

---

## 项目结构

```
security_lora_sft/
├── dataset/                    # 原始标注数据（6 类，各 30/60 条）
│   ├── normal_dataset.jsonl        (60)
│   ├── sql_injection_dataset.jsonl (30)
│   ├── xss_dataset.jsonl           (30)
│   ├── path_traversal_dataset.jsonl(30)
│   ├── ssti_dataset.jsonl          (30)
│   └── command_injection_dataset.jsonl (30)
├── data/                       # 划分后的训练/验证集
│   ├── train.jsonl             (176 条)
│   └── val.jsonl               (34 条)
├── output/
│   └── lora_adapter/           # 训练好的 LoRA adapter
├── prepare_data.py             # 数据集划分脚本
├── train.py                    # 微调训练脚本
├── inference.py                # 推理 & 验证集评估
├── eval_baseline.py            # 基线模型（未微调）评估
├── install.sh                  # 环境安装脚本（uv）
└── requirements.txt
```

---

## 数据集

**总量**：210 条标注样本，6 类攻击类型

| 类别                | 样本数 |
|---------------------|--------|
| normal              | 60     |
| sql_injection       | 30     |
| xss                 | 30     |
| path_traversal      | 30     |
| ssti                | 30     |
| command_injection   | 30     |

**数据格式**（每条 JSONL）：

```json
{
  "prompt": "你是一个安全告警分类助手。...\n\nHTTP 请求或日志:\nGET /search?q=<script>alert(1)</script> HTTP/1.1\nHost: example.com\n\n输出:",
  "completion": "{\"attack_type\": \"xss\", \"risk\": \"medium\", \"reason\": \"...\"}"
}
```

**划分策略**：按类别分层抽样，85% / 15% → train 176 条 / val 34 条。

---

## 环境配置

**硬件**：NVIDIA RTX 4060 Laptop GPU（8 GB VRAM），WSL2，CUDA 12.7

**依赖安装**（首次）：

```bash
bash install.sh
source .venv/bin/activate
```

`install.sh` 按顺序执行：
1. `uv venv` 创建 Python 3.10 虚拟环境
2. 安装 PyTorch 2.6.0（cu124 wheels）
3. 安装 Unsloth
4. 安装 `requirements.txt` 其余依赖

**关键版本**：

| 包             | 版本         |
|----------------|--------------|
| Python         | 3.10         |
| PyTorch        | 2.6.0+cu124  |
| Unsloth        | 2026.4.8     |
| TRL            | 0.24.0       |
| Transformers   | 5.5.0        |
| PEFT           | 0.19.1       |
| bitsandbytes   | ≥0.43.0      |

> **注意**：不安装 torchao。torchao 0.9.0 与 PEFT 0.19+ 存在版本检查冲突；torchao ≥0.16.0 需要 torch≥2.7.0（cu124 不可用）。Transformers 和 PEFT 在 torchao 缺失时均安全跳过，无功能损失。

---

## 训练配置

**模型**：`unsloth/Qwen2.5-1.5B-Instruct`（通过 Unsloth 加载 4-bit NF4 量化版本）

**LoRA 配置**：

| 参数            | 值                                                    |
|-----------------|-------------------------------------------------------|
| r               | 16                                                    |
| lora_alpha      | 32                                                    |
| lora_dropout    | 0.05                                                  |
| target_modules  | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 可训练参数      | 18.4M / 1.56B（1.18%）                                |

**训练超参**：

| 参数                   | 值         |
|------------------------|------------|
| 精度                   | bf16       |
| batch_size             | 2          |
| gradient_accumulation  | 4（等效 8）|
| epochs                 | 5          |
| learning_rate          | 2e-4       |
| lr_scheduler           | cosine     |
| warmup_ratio           | 0.1        |
| max_seq_length         | 512        |
| 总训练步数             | 110        |

**损失计算**：全序列损失（prompt + completion），TRL 0.24.0 移除了 `train_on_responses_only`，ChatML 文本格式不支持 `completion_only_loss=True`。

---

## 快速开始

```bash
source .venv/bin/activate

# 1. 评估基线（首次需下载模型约 3GB）
python eval_baseline.py

# 2. 微调训练
python train.py

# 3. 评估微调后模型
python inference.py --eval

# 单条请求推理
python inference.py --request "GET /search?q=<script>alert(1)</script> HTTP/1.1
Host: example.com"
```

---

## 实验结果

### 验证集评估（34 条）

| 指标              | 基线（未微调） | 微调后（5 epochs） | 变化      |
|-------------------|---------------:|-------------------:|-----------|
| attack_type 准确率 | 64.7%（22/34） | **94.1%（32/34）** | +29.4 pp  |
| risk 准确率        | 52.9%（18/34） | **79.4%（27/34）** | +26.5 pp  |
| JSON 解析错误      | 3/34（8.8%）   | 0/34               | 消除      |

### 各类别 attack_type 准确率

| 类别               | 基线       | 微调后     | 说明                      |
|--------------------|:----------:|:----------:|---------------------------|
| normal             | 50%        | **100%**   | 基线误报严重              |
| sql_injection      | 50%        | **100%**   |                           |
| xss                | 83%        | **100%**   |                           |
| path_traversal     | **100%**   | 80%        | 见下方分析                |
| ssti               | 33%        | **100%**   | 基线几乎无法识别          |
| command_injection  | 50%        | **83%**    |                           |

---

## 实验分析

### 整体提升

微调后 attack_type 准确率从 64.7% 提升至 94.1%，提升近 30 个百分点，对模型参数量仅为 1.5B 且有效训练数据仅 176 条而言，效果显著。其中 normal / sql_injection / xss / ssti 四类达到 100% 准确率，说明少量高质量标注样本足以引导模型建立清晰的分类边界。

基线模型 JSON 解析错误率 8.8%（3/34），原因是基础模型倾向于在 JSON 外包裹 markdown 代码块（` ```json ``` `）或在 `reason` 字段中嵌套花括号，导致 `json.loads` 失败。微调后模型直接输出纯 JSON，解析错误降为 0。

### path_traversal 准确率下降分析

基线 100% → 微调后 80%，误分类了 1 条（验证集共 5 条 path_traversal）。

**具体误分类样本**：

```
POST /api/archive/extract HTTP/1.1
Content-Type: application/json

{"entry":"../../../../tmp/pwned.txt"}
```

- **预测**：`command_injection`（risk: high）
- **实际**：`path_traversal`（risk: medium）
- **模型给出的 reason**：JSON 字段 entry 中包含 `../` 并指向 tmp 目录下 pwned.txt，属于命令注入载荷

**分析**：

该样本具有两个容易混淆的特征：路径穿越特征（`../../../../`）叠加了解压接口语义（`/archive/extract`）。`/archive/extract` 端点上的路径穿越在安全领域被称为 Zip Slip，模型可能将"解压到任意路径"的语义联想为命令执行。其余 4 条 path_traversal 均为标准 GET 参数形式（`?file=../../../etc/passwd`），模型全部正确识别。

**是否为系统性问题**：不是。验证集 path_traversal 仅 5 条，1 条出错即 20% 的精度损失，统计方差本身很大。这是一个孤立的边界混淆案例，不代表模型对 path_traversal 整体退步。

**如需修复**：在训练集中补充 2-3 条"POST 接口 + JSON body 中含 `../` 路径"形态的 path_traversal 样本，明确与 command_injection 的边界。

### risk 准确率未达预期（79.4%）

risk 标签分布不均匀（high 占多数，low 样本稀少），且 risk 判断带有主观性，不同攻击类型下相同风险级别的语义差异较大。176 条训练数据中 risk 标签的多样性不足，导致模型在 high/medium 边界处存在偏差。如需提升，可对 risk 标注进行更细粒度的特征描述，或适当增加各风险级别的覆盖样本。
