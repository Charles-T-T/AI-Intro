import copy
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from langconv import Converter
from transformer_model import *
from nltk import word_tokenize
from torch.autograd import Variable


PAD = 0  # padding占位符的索引
UNK = 1  # 未登录词标识符的索引
BATCH_SIZE = 256  # 批次大小
EPOCHS = 60  # 训练轮数
LAYERS = 4  # transformer中encoder、decoder层数
H_NUM = 4  # 多头注意力个数
D_MODEL = 256  # 输入、输出词向量维数
D_FF = 512  # feed forward全连接层维数
DROPOUT = 0.1  # dropout比例
MAX_LENGTH = 60  # 语句最大长度

TRAIN_FILE = "nmt/en-cn/train.txt"  # 训练集
DEV_FILE = "nmt/en-cn/dev.txt"  # 验证集
SAVE_FILE = "save/model.pt"  # 模型保存路径


def seq_padding(X, padding=PAD):
    """
    按批次（batch）对数据填充、长度对齐
    """
    # 计算该批次各条样本语句长度
    L = [len(x) for x in X]
    # 获取该批次样本中语句长度最大值
    ML = max(L)
    # 遍历该批次样本，如果语句长度小于最大长度，则用padding填充
    return np.array(
        [
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
            for x in X
        ]
    )


def cht_to_chs(sent):
    sent = Converter("zh-hans").convert(sent)
    sent.encode("utf-8")
    return sent


class PrepareData:
    def __init__(self, train_file, dev_file):
        # 读取数据、分词
        self.train_en, self.train_cn = self.load_data(train_file)
        self.dev_en, self.dev_cn = self.load_data(dev_file)
        # 构建词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(
            self.train_en
        )
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(
            self.train_cn
        )
        # 单词映射为索引
        self.train_en, self.train_cn = self.word2id(
            self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict
        )
        self.dev_en, self.dev_cn = self.word2id(
            self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict
        )
        # 划分批次、填充、掩码
        self.train_data = self.split_batch(self.train_en, self.train_cn, BATCH_SIZE)
        self.dev_data = self.split_batch(self.dev_en, self.dev_cn, BATCH_SIZE)

    def load_data(self, path):
        """
        读取英文、中文数据
        对每条样本分词并构建包含起始符和终止符的单词列表
        形式如：en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
                cn = [['BOS', '我', '爱', '你', 'EOS'], ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        en = []
        cn = []
        with open(path, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                sent_en, sent_cn = line.strip().split("\t")
                sent_en = sent_en.lower()
                sent_cn = cht_to_chs(sent_cn)
                sent_en = ["BOS"] + word_tokenize(sent_en) + ["EOS"]
                # 中文按字符切分
                sent_cn = ["BOS"] + [char for char in sent_cn] + ["EOS"]
                en.append(sent_en)
                cn.append(sent_cn)
        return en, cn

    def build_dict(self, sentences, max_words=5e4):
        """
        构造分词后的列表数据
        构建单词-索引映射（key为单词，value为id值）
        """
        # 统计数据集中单词词频
        word_count = Counter([word for sent in sentences for word in sent])
        # 按词频保留前max_words个单词构建词典
        # 添加UNK和PAD两个单词
        ls = word_count.most_common(int(max_words))
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict["UNK"] = UNK
        word_dict["PAD"] = PAD
        # 构建id2word映射
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def word2id(self, en, cn, en_dict, cn_dict, sort=True):
        """
        将英文、中文单词列表转为单词索引列表
        `sort=True`表示以英文语句长度排序，以便按批次填充时，同批次语句填充尽量少
        """
        length = len(en)
        # 单词映射为索引
        out_en_ids = [[en_dict.get(word, UNK) for word in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(word, UNK) for word in sent] for sent in cn]

        # 按照语句长度排序
        def len_argsort(seq):
            """
            传入一系列语句数据(分好词的列表形式)，
            按照语句长度排序后，返回排序后原来各语句在数据中的索引下标
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 按相同顺序对中文、英文样本排序
        if sort:
            # 以英文语句长度排序
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[idx] for idx in sorted_index]
            out_cn_ids = [out_cn_ids[idx] for idx in sorted_index]
        return out_en_ids, out_cn_ids

    def split_batch(self, en, cn, batch_size, shuffle=True):
        """
        划分批次
        `shuffle=True`表示对各批次顺序随机打乱
        """
        # 每隔batch_size取一个索引作为后续batch的起始索引
        idx_list = np.arange(0, len(en), batch_size)
        # 起始索引随机打乱
        if shuffle:
            np.random.shuffle(idx_list)
        # 存放所有批次的语句索引
        batch_indexs = []
        for idx in idx_list:
            """
            形如[array([4, 5, 6, 7]),
                 array([0, 1, 2, 3]),
                 array([8, 9, 10, 11]),
                 ...]
            """
            # 起始索引最大的批次可能发生越界，要限定其索引
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        # 构建批次列表
        batches = []
        for batch_index in batch_indexs:
            # 按当前批次的样本索引采样
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # 对当前批次中所有语句填充、对齐长度
            # 维度为：batch_size * 当前批次中语句的最大长度
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            # 将当前批次添加到批次列表
            # Batch类用于实现注意力掩码
            batches.append(Batch(batch_en, batch_cn))
        return batches


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


def get_std_opt(model):
    return NoamOpt(
        model.src_embed[0].d_model,
        2,
        4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )


def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.0
    total_loss = 0.0
    tokens = 0.0

    step = 0
    progress_bar = tqdm(data, ncols=100)
    for batch in progress_bar:
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        progress_bar.set_description_str(
            f"Epoch {epoch+1} Batch: {step+1} Loss: {loss / batch.ntokens:.4f}"
        )
        step += 1

    return total_loss / total_tokens


def train(data, model, criterion, optimizer):
    """
    训练并保存模型
    """
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_dev_loss = 1e5

    # 保存loss变化记录
    train_losses = []
    dev_losses= []

    for epoch in range(EPOCHS):
        # 模型训练
        model.train()
        train_loss = run_epoch(
            data.train_data,
            model,
            SimpleLossCompute(model.generator, criterion, optimizer),
            epoch,
        )
        train_losses.append(train_loss.cpu().item())

        # 在dev集上进行loss评估
        model.eval()
        print(">>>>> Evaluating on dev set...")
        dev_loss = run_epoch(
            data.dev_data,
            model,
            SimpleLossCompute(model.generator, criterion, None),
            epoch,
        )
        print("<<<<< Evaluate loss: %f" % dev_loss)
        dev_losses.append(dev_loss.cpu().item())

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), SAVE_FILE)
            best_dev_loss = dev_loss
            print("****** Save model done! ******")

        print()

    # 绘制loss曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Training Loss")
    plt.plot(range(1, EPOCHS + 1), dev_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def do_train():
    # 数据预处理
    data = PrepareData(TRAIN_FILE, DEV_FILE)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % src_vocab)
    print("tgt_vocab %d" % tgt_vocab)

    # 初始化模型
    model = make_model(src_vocab, tgt_vocab, LAYERS, D_MODEL, D_FF, H_NUM, DROPOUT)

    # 训练
    print(">>>>>>> Training starts!\n")
    train_start = time.time()
    criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=0.0)
    optimizer = NoamOpt(
        D_MODEL,
        1,
        2000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )

    train(data, model, criterion, optimizer)
    print(f"<<<<<<< Training finished. Cost {time.time()-train_start:.4f} seconds.")


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    # TODO 拓展 - question 2: 训练过程和测试过程decoder的输入有何不同？
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(
            memory,
            src_mask,
            Variable(ys),
            Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
        )
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def translate():
    # 初始化模型
    data = PrepareData(TRAIN_FILE, DEV_FILE)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    # print("src_vocab %d" % src_vocab)
    # print("tgt_vocab %d" % tgt_vocab)

    model = make_model(src_vocab, tgt_vocab, LAYERS, D_MODEL, D_FF, H_NUM, DROPOUT)
    # 加载模型
    model.load_state_dict(torch.load(SAVE_FILE, weights_only=True))
    with torch.no_grad():
        while True:
            # 输入待翻译的英文语句
            sentence_en = input("请输入英文: ")
            # 打印待翻译的英文语句
            sentence_en = sentence_en.lower()
            # TODO 基础 - task 3: 将输入的英文语句分词并转为单词 id 表示
            sentence_en = sentence_en.split()  # 分词
            sentence_en_ids = [
                data.en_word_dict.get(word, data.en_word_dict["UNK"])
                for word in sentence_en
            ]

            en_sent = " ".join([data.en_index_dict[w] for w in sentence_en_ids])
            # print("\n" + en_sent)

            # 将当前以单词id表示的英文语句数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(np.array(sentence_en_ids)).long().to(DEVICE)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(
                model,
                src,
                src_mask,
                max_len=MAX_LENGTH,
                start_symbol=data.cn_word_dict["BOS"],
            )
            # 初始化一个用于存放模型翻译结果语句单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = data.cn_index_dict[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前语句的翻译结果列表
                if sym != "EOS":
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            # 打印模型翻译输出的中文语句结果
            print("译文: %s\n" % " ".join(translation))


if __name__ == "__main__":

    # Three settings of the lrate hyperparameters.
    # opts = [NoamOpt(512, 1, 4000, None),
    #         NoamOpt(512, 1, 8000, None),
    #         NoamOpt(256, 1, 4000, None)]
    # plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    # plt.legend(["512:4000", "512:8000", "256:4000"])
    # plt.show()

    do_train()

    translate()
