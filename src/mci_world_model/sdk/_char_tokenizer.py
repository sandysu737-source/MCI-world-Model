from __future__ import annotations

"""CharTokenizer — 轻量字符级中文 tokenizer (v4.4.0)."""


import numpy as np


class CharTokenizer:
    """字符级中文 tokenizer。
    词表: 常用汉字 (U+4E00-U+9FFF ~21K) + ASCII 128 + 特殊 token 4 ≈ 22K
    """

    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    CJK_START = 0x4E00
    CJK_END = 0x9FFF
    SPECIAL_TOKENS = {PAD_IDX: "<pad>", UNK_IDX: "<unk>", BOS_IDX: "<bos>", EOS_IDX: "<eos>"}

    def __init__(self) -> None:
        cjk_count = self.CJK_END - self.CJK_START + 1
        self._vocab_size = 4 + cjk_count + 128
        self._char_to_id: dict[str, int] = {}
        # Build mapping: CJK chars get IDs 4..4+cjk_count-1, ASCII gets 4+cjk_count..4+cjk_count+127
        for i, cp in enumerate(range(self.CJK_START, self.CJK_END + 1)):
            self._char_to_id[chr(cp)] = 4 + i
        for i in range(128):
            self._char_to_id[chr(i)] = 4 + cjk_count + i

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, max_len: int = 64) -> np.ndarray:
        ids = []
        for ch in text:
            tid = self._char_to_id.get(ch, self.UNK_IDX)
            ids.append(tid)
            if len(ids) >= max_len:
                break
        # Pad
        if len(ids) < max_len:
            ids += [self.PAD_IDX] * (max_len - len(ids))
        return np.array(ids[:max_len], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        result = []
        for tid in ids:
            if tid < 4:
                result.append(self.SPECIAL_TOKENS.get(int(tid), ""))
            elif tid < 4 + (self.CJK_END - self.CJK_START + 1):
                cp = self.CJK_START + tid - 4
                result.append(chr(cp))
            elif tid < self._vocab_size:
                ascii_idx = tid - 4 - (self.CJK_END - self.CJK_START + 1)
                if 0 <= ascii_idx < 128:
                    result.append(chr(ascii_idx))
        return "".join(result)


class SimpleTextEmbedderV2:
    """升级版文本嵌入器: TF-IDF + char n-gram 混合。

    替换原始 SimpleTextEmbedder 的纯 char 3-gram hash。
    """

    def __init__(self, output_dim: int = 128, seed: int = 42):
        self._dim = output_dim
        self._rng = np.random.RandomState(seed)
        self._idf: dict[str, float] = {}
        self._corpus_size = 0
        self._tokenizer = CharTokenizer()

    def fit(self, corpus: list[str]) -> None:
        self._corpus_size = len(corpus)
        df: dict[str, int] = {}
        for text in corpus:
            seen = set()
            for i in range(len(text) - 1):
                bigram = text[i : i + 2]
                if bigram not in seen:
                    df[bigram] = df.get(bigram, 0) + 1
                    seen.add(bigram)
        for bigram, count in df.items():
            self._idf[bigram] = np.log((self._corpus_size + 1) / (count + 1)) + 1

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float64)
        for i in range(len(text) - 1):
            bigram = text[i : i + 2]
            idf = self._idf.get(bigram, 1.0)
            h = hash((bigram, i)) % (2**31 - 1)
            dim = abs(h) % self._dim
            vec[dim] += idf
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec.astype(np.float32)
