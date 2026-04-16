"""LAC ONNX 推理示例 — 中文词法分析（分词 + 词性标注 + NER）"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_DIR = Path(__file__).parent


def load_dict(path, key_col=1, val_col=0):
    """加载 TSV 词典，返回 {key_col: val_col} 映射。"""
    d = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if '\t' in line:
                parts = line.split('\t', 1)
                d[parts[key_col]] = int(parts[val_col]) if val_col == 0 else parts[val_col]
    return d


# 加载资源
vocab = load_dict(MODEL_DIR / 'word.dic', key_col=1, val_col=0)  # char → id
id2tag = load_dict(MODEL_DIR / 'tag.dic', key_col=0, val_col=1)  # id → tag
id2tag = {int(k): v for k, v in id2tag.items()}
OOV_ID = vocab.get('OOV', 0)

q2b = {}
q2b_path = MODEL_DIR / 'q2b.dic'
if q2b_path.exists():
    with open(q2b_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if '\t' in line:
                full, half = line.split('\t', 1)
                q2b[full] = half

crf_transitions = np.load(MODEL_DIR / 'lac_crf_transitions.npy')
session = ort.InferenceSession(str(MODEL_DIR / 'lac_encoder.onnx'))


def viterbi_decode(emissions, transitions, seq_len):
    """Viterbi 解码，返回最优标签 ID 序列。"""
    n_tags = emissions.shape[1]
    dp = emissions[0].copy()
    backpointers = []
    for t in range(1, seq_len):
        scores = dp[:, None] + transitions  # (prev, cur)
        bp = np.argmax(scores, axis=0)
        dp = scores[bp, np.arange(n_tags)] + emissions[t]
        backpointers.append(bp)
    best = int(np.argmax(dp))
    path = [best]
    for bp in reversed(backpointers):
        path.append(int(bp[path[-1]]))
    path.reverse()
    return path


def predict(text):
    """对文本进行词法分析，返回 [(word, tag), ...] 列表。

    常见标签：
      PER=人名, LOC=地名, ORG=机构名, TIME=时间,
      n=名词, v=动词, a=形容词, m=数词, w=标点, ...
    """
    chars = list(text)
    if not chars:
        return []

    ids = np.array(
        [[vocab.get(q2b.get(c, c), OOV_ID) for c in chars]],
        dtype=np.int64,
    )
    length = np.array([len(chars)], dtype=np.int64)

    logits = session.run(None, {'token_ids': ids, 'length': length})[0]  # (1, seq, 59)
    tag_ids = viterbi_decode(logits[0], crf_transitions, len(chars))

    # 解码 BIO 标签
    results = []
    i = 0
    while i < len(chars):
        raw_tag = id2tag.get(tag_ids[i], 'n-B')
        label = raw_tag.rsplit('-', 1)[0]
        word = chars[i]
        i += 1
        while i < len(chars):
            if id2tag.get(tag_ids[i], 'n-B').endswith('-I'):
                word += chars[i]
                i += 1
            else:
                break
        results.append((word, label))
    return results


if __name__ == '__main__':
    texts = [
        '张三在北京市工作，手机号13812345678',
        '李四和欧阳明月去了上海交通大学',
        '中国人民银行发布了2024年第一季度货币政策报告',
    ]
    for text in texts:
        print(f'\n>>> {text}')
        for word, tag in predict(text):
            print(f'  {word:12s} {tag}')
