"""
Transformer 路径级时序/功耗回归（PyTorch 实现）
------------------------------------------------
用途：
  给定一条路由路径的离散序列（如 tile / pip / wire / edge 等），
  以及可选的数值特征（如 mux_size、是否长线等），
  预测该路径的总延迟（和/或功耗）。

特点：
  - 纯 PyTorch，无外部依赖（仅可选 pandas/pyarrow 用于读 CSV/Parquet）。
  - 支持可变长序列，padding + attention mask。
  - 支持“多头输出”：delay / power 任一或二者同时训练（多任务）。
  - 类别特征采用哈希嵌入（无需完整字典），数值特征采用线性投影。
  - 自带合成数据生成器，便于快速跑通；也可切换到真实数据。

数据格式（两种方式）
===================
1) 合成数据（默认）：--use_synth 1
   自动生成 (N, L) 的 pip_id / tile_id / num_feat，和标签 delay / power。

2) 真实数据：--data_path 指向 CSV 或 Parquet，需包含：
   - pip_seq:  以空格分隔的整数ID（或字符串）序列（每一行是一条路径）
   - tile_seq: 同上
   - num_seq:  以分号分隔的数值序列，每个token可多维，用逗号分隔
               例如："0.5,1,3; 1.0,0,2; ..."
   - delay_ns: 路径级总延迟（浮点）
   - power_mw: （可选）路径级功耗（浮点）

快速开始：
==========
# 1) 合成数据训练（演示）
python train_transformer_path_reg.py --use_synth 1 --epochs 5

# 2) 真实数据训练（CSV）
python train_transformer_path_reg.py \
  --data_path my_paths.csv \
  --epochs 20 --batch_size 64 --predict_power 0 --predict_delay 1

# 3) 真实数据训练（Parquet）
python train_transformer_path_reg.py \
  --data_path my_paths.parquet \
  --epochs 20 --batch_size 64 --predict_power 1 --predict_delay 1

导出：
  - 最优模型权重: outputs/best.pt
  - 训练日志:      outputs/log.txt

提示：
  - 若类别原始ID是字符串，可先用稳定哈希（本脚本自带）映射到 0..(vocab_hash_size-1)。
  - 数值序列维度由 --num_feat_dim 指定；若每个token数值维度不一致，请先在数据预处理时对齐。
"""


import argparse
import math
import os
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils import set_seed



