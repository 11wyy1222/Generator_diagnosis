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
  --output-root "D:\Generator_diagnosis\dataset_root_dfig_diagnosis_20260914" `
  --objects-config configs\objects_dfig_diagnosis_v2_20260915.json `
  --sources-config configs\data_sources_dfig_diagnosis_20260914.json
```

当前双馈数据 v2 中，F14 用于开发，内黄 12 用于跨风机测试，武威 F1
和半岛北 A1 用于外部测试；RPM 分箱和切分角色以对象配置为准。

样本清单字段遵循实施方案第 3.2 节，`dataset_split` 应为 `train`、`validation` 或测试集标识。先执行：

```powershell
bearing-diagnosis validate-manifest --manifest dataset_root_dfig_diagnosis_20260914\splits\diagnosis_v2_20260915.jsonl --output-dir admission
```

退出码 `2` 表示存在拒绝样本；详情写入 `rejected_samples.jsonl`，源波形不会被修改。

### 2026年9月新增数据

新增数据使用完整正常与完整故障范围，已生成版本化准入目录
`dataset_root_new_20260910`。德州和裴桥的右侧映射为 `3点`、左侧
映射为 `9点`；武威源文件为 FJ14/F14，继续使用对象
`DFIG_TEST_F14`，避免重复登记。

故障诊断模型使用：

`dataset_root_new_20260910\splits\diagnosis_split.jsonl`

异常检测模型使用：

`dataset_root_new_20260910\splits\anomaly_split.jsonl`

完整对象、准入数量、拒绝原因、快照哈希和切分统计见
`dataset_root_new_20260910\README.md`；机器可读索引见
`configs\dataset_catalog.json`。

## 训练

### 当前正式主线

双馈正式主线固定为：

- 模型配置：`configs/dfig_model_v1_spectrum_only_data_v2_20260918.yaml`
- 模型算法：`v1_sample_bce`
- 模型分支：`spectrum_only`
- 信号处理：`native`
- 标签/切分定义：`configs/objects_dfig_diagnosis_v2_20260915.json`
- 数据源定义：`configs/data_sources_dfig_diagnosis_20260914.json`
- 固定数据清单：`dataset_root_dfig_diagnosis_20260914/splits/diagnosis_v2_20260915.jsonl`

这里的“数据 v2”仅表示标签和训练/测试角色划分版本，不是模型算法 v2。
历史模型、消融实验和早期数据配置保存在 `configs/archive/`。

半直驱频谱主线使用 `configs/semi_direct_spectrum_only.yaml`。

```powershell
bearing-diagnosis train `
  --config configs\dfig_model_v1_spectrum_only_data_v2_20260918.yaml `
  --manifest dataset_root_dfig_diagnosis_20260914\splits\diagnosis_v2_20260915.jsonl `
  --run-dir runs\dfig_model_v1_spectrum_only_data_v2_20260918_seed2026 `
  --seed 2026 `
  --device cuda
```

### 服务器训练（Linux）

先把当前源码，以及 `artifacts/server_bundles/server_bundle_20260920/` 中的
`training_metadata.zip` 和 `cms_waveforms.zip` 上传到服务器项目目录。以下
命令按项目位于 `/home/ubuntu/Generator_diagnosis` 编写：

```bash
cd /home/ubuntu/Generator_diagnosis
unzip -q training_metadata.zip -d .
mkdir -p data/cms
unzip -q cms_waveforms.zip -d data/cms

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest
```

CUDA 检查必须输出 `True`。服务器清单使用 Linux 绝对路径；如果项目不在
`/home/ubuntu/Generator_diagnosis`，必须先把清单中的 `waveform_path` 改成
实际路径。正式训练前先做数据准入校验：

```bash
mkdir -p admission runs logs
bearing-diagnosis validate-manifest \
  --manifest manifests/dfig_diagnosis_v2_20260915_server.jsonl \
  --output-dir admission/dfig_mainline
```

确认没有拒绝样本后，在 0 号 GPU 后台启动当前双馈正式主线：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 bearing-diagnosis train \
  --config configs/dfig_model_v1_spectrum_only_data_v2_20260918.yaml \
  --manifest manifests/dfig_diagnosis_v2_20260915_server.jsonl \
  --run-dir runs/dfig_model_v1_spectrum_only_data_v2_20260918_seed2026 \
  --seed 2026 \
  --device cuda \
  > logs/dfig_model_v1_spectrum_only_data_v2_20260918_seed2026.log 2>&1 &
echo $! > logs/dfig_model_v1_spectrum_only_data_v2_20260918_seed2026.pid

tail -f logs/dfig_model_v1_spectrum_only_data_v2_20260918_seed2026.log
```

训练完成后检查运行目录中的 `model.pt`、`preprocess.json`、
`mechanism_scaler.npz`、`metrics_validation.json`、`training_history.json` 和
`model_card.md`。每次训练必须使用新的 `run-dir`；程序不会覆盖已有目录。
半直驱服务器训练需另外解压 `legacy_waveforms.zip` 到 `data/legacy`，并改用
`configs/semi_direct_spectrum_only.yaml` 和
`manifests/semi_direct_diagnosis_server.jsonl`。

当前批处理要求同一 batch 中波形长度相同。若不同测点长度不同，应按 `object_id + sensor_position + waveform_length` 分桶采样；不能裁剪、补零或把内部窗口随机分到不同集合。
双馈完整波形在 6 GB RTX A2000 上使用 `batch_size: 8`；`batch_size: 32`
会超过显存容量。

### v1 信号处理模式

v1 配置支持两种统一的训练、测试和推理预处理模式：

```yaml
signal_processing_mode: native               # 原有数据处理
signal_processing_mode: vibration_analysis   # 去直流 + Hann 窗 RMS 频谱 + Hilbert 包络 RMS 频谱
```

未配置时默认使用 `native`，兼容已有模型。模式会冻结到
`preprocess.json`，推理时自动复用。切换到 `vibration_analysis`
会改变频谱幅值分布，必须重新训练 v1 模型，不能直接复用旧权重。

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
