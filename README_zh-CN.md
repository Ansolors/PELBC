# PELBC 海豚遭遇级行为预测软件

**PELBC**（Predicting Encounter-Level Behavioral Context）从一次遭遇的原始录音中自动检测并提取 whistle，将片段按遭遇组成集合，再通过 Acoustic LR 输出**摄食、定向行进、徘徊、社交**四类行为背景分数。整个流程在本地运行。

```text
遭遇原始录音 → whistle 自动检测 → 跨窗口事件合并 → 原采样率片段导出
→ 遭遇级声学特征汇总 → 四类行为评分
```

仓库只维护一套 PELBC 实现。原始录音输入与已有片段输入是同一软件的两种入口。公开内容为软件源码、安装与使用说明、人工合成示例和必要的软件测试；论文、实验与训练脚本、研究结果及私有模型导出流程另行保存，不随本仓库发布。

## 安装与模型

需要 Python 3.10–3.13。在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install .
```

公开仓库只分发源码，不包含保密录音、标注、遭遇元数据、研究派生数据、预测结果或训练后的模型参数。运行时通过 `--model` 指定兼容的本地 Acoustic LR JSON 文件；原始录音模式还需通过 `--detector-model` 指定针对 whistle 训练的本地检测权重。通用目标检测权重不能直接替代 whistle 检测模型。软件不会自动下载这两个模型。

同一源码支持在本地安装中随附模型，但公开分发包不随附模型。检测依赖 Ultralytics 8.4.102，默认依次选择可用的 MPS、CUDA 或 CPU，也可指定 `--device cpu`。

## 从原始录音预测

```bash
pelbc-predict \
  --recording /path/to/encounter_001.wav \
  --model /path/to/private/acoustic_model.json \
  --detector-model /path/to/private/whistle_detector.pt \
  --work-dir /path/to/local/encounter_001_results \
  --output /path/to/local/prediction.json
```

同一次遭遇有多条录音时，将 `--recording` 换成 `--recordings-dir /path/to/encounter_001_recordings`。可用 `--encounter-id` 指定遭遇编号。全部子目录也属于同次遭遇时可加 `--recursive`。每次调用只处理一次遭遇。

输入为未压缩、单声道、16 位 PCM WAV，采样率至少 80 kHz。检测使用 3 秒窗口、1 秒步长，默认置信度阈值 0.62；跨窗合并后，每段前后各保留 0.25 秒上下文，按源录音的原采样率和 PCM 样点导出。

每次运行新建 `run_...` 目录，保存 `whistles/` 片段、`extraction_manifest.json` 提取清单和 `prediction.json` 行为结果。清单记录源文件、样点位置、检测置信度和模型哈希。未指定 `--work-dir` 时，输出在输入同级的 `<名称>_pelbc` 文件夹；输出目录须位于源录音文件夹之外。

`--output prediction.csv` 可导出简表；不指定 `--output` 时，JSON 同时打印到标准输出。`--quiet` 隐藏进度。未检出 whistle 时，状态为 `no_whistles_detected`，行为分数为 `null`，不表示四类行为均未发生。

## 从已有 whistle 片段预测

```bash
pelbc-predict \
  --input-dir /path/to/encounter_001_whistles \
  --model /path/to/private/acoustic_model.json \
  --output /path/to/local/prediction.json
```

这一入口跳过检测，使用相同的声学特征提取和行为模型汇总同次遭遇的片段，无需检测权重。Python 接口见 [英文说明](README.md#python-api)。

## 结果解释与检查

论文报告的主实验最佳模型为 **Acoustic LR**，raw macro-AP 为 **0.710**，raw macro ROC-AUC 为 0.815，校准后 macro Brier 为 0.145。研究包含 **4 类行为、234 次遭遇和 4,126 段 whistle**。这些是论文评估汇总，不构成对用户另行提供的模型文件的独立验证。

输出描述遭遇级群体行为背景，不识别发声个体，也不提供单条 whistle 的行为真值。四个分数独立计算，总和不要求为 1；尚未针对新调查进行前瞻概率校准，软件不设置通用二元判定阈值。

```bash
python -m unittest discover -s tests -v
```

公开测试只使用人工构造的声音和模型参数，流程检查使用模拟检测器，无需私有数据或权重。它们检查软件功能，不构成检测准确率或生物学预测效能验证。详细说明见 [模型说明](MODEL_CARD.md)、[片段提取方法](docs/DETECTOR_SOURCE.md)、[输出格式](docs/OUTPUT_SCHEMA.md) 和 [引用信息](CITATION.cff)。
