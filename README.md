# 发电机轴承磨损弱监督诊断模型

本目录已按《发电机轴承磨损故障诊断模型实施方案（可执行版）》搭建第一阶段模型框架：输入单测点完整波形，仅训练和输出总体正常/异常弱监督二分类，不启用四部件分类头。

## 已实现

- 波形文件名解析、JSONL 样本结构和非破坏性数据准入；
- 按完整时间块切分，禁止同一 `sample_group_id` 泄漏；
- 训练集全局绝对幅值 P99.5、普通谱、全频 Hilbert 包络谱和冻结 Hz 网格；
- 外环、内环、滚动体、保持架的 1～5 阶连续机理证据、有效掩码及可靠度；
- 多尺度时域 CNN、双通道频谱 CNN、共享机理 MLP、辅助头和可靠度门控；
- A/B/C/D 四种固定实验：`spectrum_only`、`mechanism_only`、`concat`、`gated`；
- 标签 1:1、测点/时间块限额和同长度分桶采样，AdamW、梯度裁剪、早停、验证集最大 F1 阈值和模型卡；
- 单波形推理的严格外部输出：

```json
{"sample_id":"sample_001","abnormal_probability":0.92,"component_probabilities":null}
```

## 环境

训练环境默认且强制使用 NVIDIA GPU。依赖文件固定安装 CUDA 11.8 版
PyTorch；训练命令不会在 CUDA 不可用时静默退回 CPU。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -m pytest
```

上面的 CUDA 检查必须输出 `True`。如果 `nvidia-smi` 或 CUDA 检查失败，
先修复 NVIDIA 驱动，不要启动正式训练。

## 数据准入

已确认的原始数据可通过以下命令生成可复现的弱标签样本清单、拒绝清单、
范围记录、固定时间块切分和 SHA-256 数据快照。清单保留原始 CSV 的绝对
路径，不复制或修改源文件：

```powershell
bearing-diagnosis build-manifest `
  --raw-root "D:\CMS故障诊断数据" `
  --output-root "D:\Generator_diagnosis\dataset_root" `
  --objects-config configs\objects.json `
  --sources-config configs\data_sources.json
```

RPM 分箱固定为半直驱 20 RPM、双馈 100 RPM。F14 清单统一标记为
`external_test`；评估时直接按记录中的 RPM 分别统计 `<=1200` 和 `>1200`
两组，不拆分同一采集时间块。

样本清单字段遵循实施方案第 3.2 节，`dataset_split` 应为 `train`、`validation` 或测试集标识。先执行：

```powershell
bearing-diagnosis validate-manifest --manifest dataset_root\splits\weak_supervised_split.jsonl --output-dir admission
```

退出码 `2` 表示存在拒绝样本；详情写入 `rejected_samples.jsonl`，源波形不会被修改。

## 训练

分别使用 `configs/semi_direct.yaml` 和 `configs/dfig.yaml`。正式实验需将 `experiment` 依次冻结为四种模式并复用同一切分及种子。

```powershell
bearing-diagnosis train `
  --config configs\dfig.yaml `
  --manifest dataset_root_dfig_expanded\splits\weak_supervised_split.jsonl `
  --run-dir runs\dfig_gated_seed2026 `
  --seed 2026 `
  --device cuda
```

当前批处理要求同一 batch 中波形长度相同。若不同测点长度不同，应按 `object_id + sensor_position + waveform_length` 分桶采样；不能裁剪、补零或把内部窗口随机分到不同集合。
双馈完整波形在 6 GB RTX A2000 上使用 `batch_size: 8`；`batch_size: 32`
会超过显存容量。

半直驱阶次频谱与竞争门控 v2 当前位于实验分支，验证通过前不替换 v1。配置、三种子
验证流程、测试集使用限制与回滚方法见
[`模型修改记录_阶次频谱与竞争门控_v2.md`](模型修改记录_阶次频谱与竞争门控_v2.md)。

## 推理

```powershell
bearing-diagnosis infer `
  --run-dir runs\sd_gated_2026 `
  --sample-id sample_001 `
  --waveform path\to\waveform.csv `
  --sampling-rate-hz 25600 `
  --rpm 215 `
  --orders-json '{"outer_race":13.156,"inner_race":14.844,"rolling_element":8.2638,"cage":0.4698}' `
  --internal-log internal.json
```

测试对象的固定阶次和时间范围已录入 `configs/objects.json`。真实数据缺失时训练会直接停止，不生成虚假模型或指标。
