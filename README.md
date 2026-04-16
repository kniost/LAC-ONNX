# LAC-ONNX

百度 [LAC](https://github.com/baidu/lac)（Lexical Analysis of Chinese）模型的 ONNX 版本，支持中文分词、词性标注和命名实体识别（NER）。

原始模型来自 PaddleNLP 的 LAC 任务（`Taskflow('ner', mode='fast')`），使用 Paddle 3.x 静态图导出后转换为 ONNX 格式。CRF 解码层使用 numpy 实现，无需依赖 PaddlePaddle。

## 特性

- **零 Paddle 依赖** — 仅需 `onnxruntime` + `numpy`
- **轻量** — 模型文件约 30 MB
- **完整功能** — 分词 + 词性标注 + NER，与原始 LAC 结果完全一致
- **即开即用** — 无需下载模型，无需联网

## 快速开始

### 安装依赖

```bash
pip install onnxruntime numpy
```

### 使用

```python
from example import predict

results = predict('张三在北京市工作')
for word, tag in results:
    print(f'{word}\t{tag}')
```

输出：

```
张三    PER
在      p
北京市  LOC
工作    n
```

### 直接运行示例

```bash
python example.py
```

## 文件说明

| 文件 | 大小 | 说明 |
|---|---|---|
| `lac_encoder.onnx` | 30 MB | ONNX 模型（Embedding + BiGRU + FC） |
| `lac_crf_transitions.npy` | 14 KB | CRF 转移矩阵（numpy 格式） |
| `word.dic` | 745 KB | 字符词表（58224 字符 → ID 映射） |
| `tag.dic` | 425 B | 标签表（57 个 BIO 标签） |
| `q2b.dic` | 44 KB | 全角→半角字符映射 |
| `paddle_static/` | 30 MB | 原始 Paddle 静态图模型（供参考） |

## 模型架构

```
输入字符 → 字符 ID 编码 → Embedding(128d)
  → 2层双向 GRU(hidden=128) → FC(256→59)  ← ONNX 模型
  → CRF Viterbi 解码                       ← numpy 实现
  → BIO 标签序列 → 分词 + 标注结果
```

## 常见标签

| 标签 | 含义 | 标签 | 含义 |
|---|---|---|---|
| PER | 人名 | n | 名词 |
| LOC | 地名 | v | 动词 |
| ORG | 机构名 | a | 形容词 |
| TIME | 时间 | m | 数词 |
| nz | 专有名词 | w | 标点符号 |
| p | 介词 | c | 连词 |
| u | 助词 | d | 副词 |

完整标签集参见 `tag.dic`。

## 转换过程

1. 使用 PaddleNLP `Taskflow('ner', mode='fast')` 下载 LAC 静态图模型
2. 从 `inference.pdiparams` 提取权重，重建 Paddle 动态图模型（`nn.Embedding` + `nn.GRU` + `nn.Linear`）
3. 通过 `paddle.jit.save` 导出新的静态图（不含 `viterbi_decode` 算子）
4. 使用 `paddle2onnx` 转换为 ONNX 格式
5. CRF 转移矩阵单独保存为 numpy 文件，运行时用 numpy 实现 Viterbi 解码

## 许可证

原始 LAC 模型由百度发布，采用 [Apache License 2.0](https://github.com/baidu/lac/blob/master/LICENSE) 许可。

本仓库的转换代码和示例同样采用 Apache License 2.0。
