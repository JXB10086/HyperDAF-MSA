"""
CMU-MOSI Dataset  (第一阶段 · Step 2)
--------------------------------------------------
严格基于实测数据结构实现 (见 scripts/inspect_data.py 输出):
  顶层 dict keys = ['train', 'valid', 'test']
  每个 split 含字段 = ['vision', 'labels', 'text', 'audio', 'id']
    text   : (N, 50, 300)  float32
    audio  : (N, 50,  5)   float64
    vision : (N, 50, 20)   float64
    labels : (N,  1,  1)   float64
    id     : (N,  3)       bytes(|S14) -> [clip_id, start, end]

设计要点:
  1. dtype 不统一 -> 全部转 float32 (PyTorch 默认)
  2. labels (N,1,1) -> squeeze 成 (N,)
  3. id bytes 三元组 -> decode, 取 clip_id 作为 sample_id
  4. 不修改 train/valid/test 划分, 不改序列长度, 不删数据
  5. 第一阶段【不加入 mask】, 仅返回原始五要素

运行方式见 scripts/test_dataloader.py
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

# 项目根目录 (datasets/ 的上一级), 保证任意工作目录都能定位数据
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(ROOT, "data", "mosi", "mosi_data.pkl")

SPLITS = ("train", "valid", "test")


def load_mosi_raw(data_path=None):
    """一次性加载完整 pkl (train/valid/test 都在同一文件), 避免重复读 147MB。"""
    path = data_path or DEFAULT_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 MOSI 数据文件: {path}")
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return raw


def _decode_id(id_item):
    """
    id_item: 形如 array([b'03bSnISJMiM_11', b'95.05', b'97.20'], dtype='|S14')
    返回: clip_id 字符串 (唯一样本标识)
    """
    first = id_item[0]
    if isinstance(first, bytes):
        return first.decode("utf-8")
    return str(first)


class MOSIDataset(Dataset):
    """单个 split 的 MOSI 数据集。传入已取出的 split dict (来自 load_mosi_raw)。"""

    def __init__(self, split_data):
        # --- 统一 dtype 到 float32, 保持原始维度/序列长度不变 ---
        self.text = np.ascontiguousarray(split_data["text"], dtype=np.float32)     # (N,50,300)
        self.audio = np.ascontiguousarray(split_data["audio"], dtype=np.float32)   # (N,50,5)
        self.vision = np.ascontiguousarray(split_data["vision"], dtype=np.float32) # (N,50,20)
        self.label = split_data["labels"].reshape(-1).astype(np.float32)           # (N,)
        self.ids = [_decode_id(x) for x in split_data["id"]]                       # list[str]

        # --- 样本数一致性断言 (防止错位) ---
        n = self.label.shape[0]
        assert self.text.shape[0] == n, "text 样本数与 label 不一致"
        assert self.audio.shape[0] == n, "audio 样本数与 label 不一致"
        assert self.vision.shape[0] == n, "vision 样本数与 label 不一致"
        assert len(self.ids) == n, "id 样本数与 label 不一致"
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # 第一阶段: 仅返回原始五要素, 不加 mask
        return {
            "text": torch.from_numpy(self.text[idx]),                       # (50,300)
            "audio": torch.from_numpy(self.audio[idx]),                     # (50,5)
            "vision": torch.from_numpy(self.vision[idx]),                   # (50,20)
            "label": torch.tensor(self.label[idx], dtype=torch.float32),    # scalar
            "id": self.ids[idx],                                            # str
        }


def build_mosi_datasets(data_path=None):
    """加载一次 pkl, 返回 {'train':Dataset, 'valid':Dataset, 'test':Dataset}。"""
    raw = load_mosi_raw(data_path)
    datasets = {}
    for split in SPLITS:
        if split not in raw:
            raise KeyError(f"数据文件缺少 split: {split}, 实际 keys={list(raw.keys())}")
        datasets[split] = MOSIDataset(raw[split])
    return datasets
