"""LAC 推理核心：ONNX 模型 + numpy Viterbi 解码"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

_DATA_DIR = Path(__file__).parent

# 标签含义速查
TAG_NAMES = {
    'PER': '人名', 'LOC': '地名', 'ORG': '机构名', 'TIME': '时间',
    'n': '名词', 'nz': '专有名词', 'nw': '作品名',
    'v': '动词', 'vd': '副动词', 'vn': '名动词',
    'a': '形容词', 'ad': '副形词', 'an': '名形词',
    'd': '副词', 'f': '方位词', 'r': '代词', 's': '处所词', 't': '时间词',
    'm': '数词', 'q': '量词', 'p': '介词', 'c': '连词', 'u': '助词',
    'xc': '其他虚词', 'w': '标点',
}


def _load_tsv(path):
    """加载 id\\tvalue 格式的词典文件。"""
    d = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if '\t' in line:
                k, v = line.split('\t', 1)
                d[k] = v
    return d


def _viterbi_decode(emissions, transitions, seq_len):
    """Viterbi 解码，返回最优标签 ID 列表。"""
    n_tags = emissions.shape[1]
    dp = emissions[0].copy()
    backpointers = []
    for t in range(1, seq_len):
        scores = dp[:, None] + transitions
        bp = np.argmax(scores, axis=0)
        dp = scores[bp, np.arange(n_tags)] + emissions[t]
        backpointers.append(bp)
    best = int(np.argmax(dp))
    path = [best]
    for bp in reversed(backpointers):
        path.append(int(bp[path[-1]]))
    path.reverse()
    return path


def _decode_bio(chars, tag_ids, id2tag):
    """将 BIO 标签序列解码为 (word, tag) 列表。"""
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


class LAC:
    """中文词法分析器（分词 + 词性标注 + NER）。

    基于百度 LAC 模型，使用 ONNX Runtime 推理。

    Args:
        model_dir: 模型文件目录，默认使用包内自带模型。
            目录下需包含: lac_encoder.onnx, lac_crf_transitions.npy,
            word.dic, tag.dic, q2b.dic

    Examples::

        lac = LAC()

        # 单句
        lac.run('张三在北京市工作')
        # [('张三', 'PER'), ('在', 'p'), ('北京市', 'LOC'), ('工作', 'n')]

        # 批量
        lac.run(['张三在北京', '李四去上海'])
        # [[('张三', 'PER'), ...], [('李四', 'PER'), ...]]
    """

    def __init__(self, model_dir=None):
        data_dir = Path(model_dir) if model_dir else _DATA_DIR

        # 字符词表
        raw = _load_tsv(data_dir / 'word.dic')
        self._vocab = {v: int(k) for k, v in raw.items()}
        self._oov_id = self._vocab.get('OOV', 0)

        # 标签表
        raw_tags = _load_tsv(data_dir / 'tag.dic')
        self._id2tag = {int(k): v for k, v in raw_tags.items()}

        # 全角→半角
        self._q2b = {}
        q2b_path = data_dir / 'q2b.dic'
        if q2b_path.exists():
            raw_q2b = _load_tsv(q2b_path)
            self._q2b = dict(raw_q2b)

        # CRF 转移矩阵
        self._transitions = np.load(data_dir / 'lac_crf_transitions.npy')

        # ONNX 会话
        self._session = ort.InferenceSession(
            str(data_dir / 'lac_encoder.onnx'),
            providers=ort.get_available_providers(),
        )

    def run(self, texts):
        """对文本进行词法分析。

        Args:
            texts: 单个字符串或字符串列表。

        Returns:
            单个字符串输入时返回 [(word, tag), ...]；
            列表输入时返回 [[(word, tag), ...], ...]。
        """
        if isinstance(texts, str):
            return self._predict_one(texts)
        return [self._predict_one(t) for t in texts]

    def _predict_one(self, text):
        chars = list(text)
        if not chars:
            return []

        ids = np.array(
            [[self._vocab.get(self._q2b.get(c, c), self._oov_id) for c in chars]],
            dtype=np.int64,
        )
        length = np.array([len(chars)], dtype=np.int64)

        logits = self._session.run(
            None, {'token_ids': ids, 'length': length},
        )[0]  # (1, seq_len, n_tags)

        tag_ids = _viterbi_decode(logits[0], self._transitions, len(chars))
        return _decode_bio(chars, tag_ids, self._id2tag)
