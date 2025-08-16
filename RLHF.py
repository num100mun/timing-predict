import torch
import torch.nn as nn
import torch.nn.functional as F



#有个智能体，在环境里
'''
每一步它看到一个状态，选一个动作，环境给奖励并跳到下个状态
目标：长期积累计奖励尽可能大
两条主线：学值（Q-learning)：学在某个状态做某动作有多好（Q值），然后选Q值最大的
    学策略：直接学，在这个状态以多大概率做那个动作（策略网络），用奖励来推着策略变好
探索和利用，
'''

N_STATES = 5
TERMINAL = 4
def step(state, action):
    ns = state + (1 if action == 1 else -1)

'''
RLHF的关键公式与数据流
监督微调（SFT）
    data:prompt, ideal_response
    target: 最大化参考答案的似然，得到参考策略Π_ref（我们也复制一份做初步策略πθ）
奖励模型（RM）
    data:prompt, resp_A, resp_B + 人类排序
    train: Bradly-Terry排序损失， 让RM对更好的样本打更高的分
PPO（强化学习）
    用当前策略πθ生成大难->RM打分得到reawrd
    总奖励=RM 分数 - β * KL(πθ || π_ref)
    用 PPO 更新 πθ（带裁剪），同时训练价值头 V(s) 做基线。

'''
import math, random, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0); random.seed(0)

############################################
# 0) 小数据与字符词表（玩具任务：礼貌回复）
############################################
prompts = [
    "hi", "hello", "help", "who are you", "answer me", "tell me", "thanks"
]
# SFT 理想回复：礼貌、简洁
ideal = {
    "hi":"hello!", "hello":"hello!", "help":"sure, how can I help?",
    "who are you":"I'm a helpful assistant.",
    "answer me":"I'd be happy to help.",
    "tell me":"sure, what would you like to know?",
    "thanks":"you're welcome!"
}


# 假设“人类偏好”：礼貌 > 粗鲁。构造一些对比样本
# pair: (better, worse)
prefs = [
    ("hi",      "hello!",              "what?"),
    ("help",    "sure, how can I help?","no"),
    ("answer me","I'd be happy to help.","shut up"),
    ("thanks",  "you're welcome!",     "k"),
    ("who are you","I'm a helpful assistant.","none")
]

# 字符级词表（简单粗暴）
def build_vocab(texts):
    chars = sorted(set("".join(texts)))
    stoi = {c:i+1 for i,c in enumerate(chars)}  # 0留给PAD
    stoi["<eos>"]=len(stoi)+1
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

all_text = prompts + list(ideal.values()) + [b for _,b,_ in prefs] + [w for _,_,w in prefs]
stoi, itos = build_vocab(all_text)
PAD, EOS = 0, stoi["<eos>"]
vocab_size = len(stoi)+1

def encode(s, max_len=64):
    ids = [stoi[c] for c in s]
    ids = ids[:max_len-1] + [EOS]
    return torch.tensor(ids, dtype=torch.long)

############################################
# 1) 生成模型：极简字符级 GRU + 两个头
#    - policy 头：下一个字符分布
#    - value 头：当前序列的标量价值
############################################

class TinyGRUPolicy(nn.Module):
    def __init__(self, vocab, emb=64, hid=128):
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.policy_head = nn.Linear(hid, vocab)
        self.value_head = nn.Linear(hid, 1)
    def forward(self, x, h0=None):
        e =self.emb(x)
        h, _ = self.gru(e, h0)
        logits = self.policy_head(h)