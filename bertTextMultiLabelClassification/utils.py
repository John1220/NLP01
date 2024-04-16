# -*- coding:utf-8 -*-
import os.path
import pickle

import torch
from tqdm import tqdm
import time
import re
import json
import random
from datetime import timedelta
import numpy as np
import pandas as pd

PAD, CLS = '[PAD]', '[CLS]'  # padding符号


def format_content(content):
    url_regex = re.compile("(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#*]*[\w\-\@?^=%&/~\+#])?",
                           re.A)
    content = url_regex.sub(' <url> ', content)

    # num_regex = re.compile(
    #     '-[1-9]\d*,\d+\.?\d*|-[1-9]\d*\.?\d*|-0\.\d*[0-9]|[1-9]\d*,\d+\.?\d*|0\.\d*[0-9]')
    # content = re.sub(num_regex, '<num>', content)

    # pat_str = "[1-9]\d{0,8}\.\d{1,4}|\d{4,10}"
    # content = re.sub(pat_str, "", content)
    return content


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.tokenizer = config.tokenizer
        with open(config.cat2id_path, 'r', encoding='utf8') as r:
            self.cat2id = json.load(r)

    def pandas_func(self, content, cat, scene, pad_size):
        # content = format_content(content)
        tokenizer = self.tokenizer
        cat2id = self.cat2id
        token = tokenizer.tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = tokenizer.convert_tokens_to_ids(token)
        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size

        scene_length = 12
        cat_length = 37
        labels = np.eye(cat_length)[cat2id.get(cat, 0)].tolist() + np.eye(scene_length)[cat2id.get(scene, cat_length) - cat_length].tolist()
        return (token_ids, labels, seq_len, mask)

    def load_dataset(self, path, pad_size=32):
        contents = []
        data = pd.read_csv(path, sep='|', encoding='utf8', engine='python', quoting=3)
        print(f"data size: {data.shape}")
        # data = data.head(10)
        data['c'] = data.apply(lambda x: self.pandas_func(x['content'], x['cat'], x['scene'], pad_size=pad_size), axis=1)
        contents = data['c'].tolist()
        return contents


def build_dataset(config, pad_size=32, rebuild=False):
    def convert_cat_multi_id(df, cat_to_id):
        """
        生成多标签 label_ids
        :param df: 数据集，含有cat、scene 2个标签
        :param cat_to_id: 类别索引字典
        :return:
        """
        # scene_length = Config.scene_length  # 场景类别个数
        scene_length = 12
        # cat_length = Config.cat_length  # 行为类别个数
        cat_length = 37
        cats = df['cat'].map(cat_to_id).tolist()
        scenes = df['scene'].apply(lambda x: cat_to_id[x] - cat_length).tolist()
        return np.concatenate([np.eye(cat_length)[cats], np.eye(scene_length)[scenes]], 1)

    if os.path.exists(config.datasetpkl) and not rebuild:
        with open(config.datasetpkl, "rb") as r:
            dic = pickle.load(r)
        train, dev, test = dic["train"], dic["dev"], dic["test"]
        print(f"load from {config.datasetpkl}")
    else:
        ds = Dataset(config)
        train = ds.load_dataset(config.train_path, config.pad_size)
        dev = ds.load_dataset(config.dev_path, config.pad_size)
        test = ds.load_dataset(config.test_path, config.pad_size)

        with open(config.datasetpkl, 'wb') as w:
            pickle.dump({"train": train, "dev": dev, "test": test}, w)
            print(f"write into {config.datasetpkl}")

    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        print(len(batches), batch_size)
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas])
        y = torch.Tensor([_[1] for _ in datas])

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.Tensor([_[2] for _ in datas])
        mask = torch.Tensor([_[3] for _ in datas])
        return (x, seq_len, mask), y

    # def _to_tensor(self, datas):
    #     x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
    #     y = torch.Tensor([_[1] for _ in datas]).to(self.device)
    #
    #     # pad前的长度(超过pad_size的设为pad_size)
    #     seq_len = torch.Tensor([_[2] for _ in datas]).to(self.device)
    #     mask = torch.Tensor([_[3] for _ in datas]).to(self.device)
    #     return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, shuffle=False):
    if shuffle:
        random.seed(2023)
        random.shuffle(dataset)
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
