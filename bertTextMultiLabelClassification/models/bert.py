# -*- coding:utf-8 -*-
import json
import os

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

BASE_DIR = r'bertTextMultiLabelClassification'


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train_20230419.txt'  # 训练集
        self.dev_path = dataset + '/data/valid_20230419.txt'  # 验证集
        self.test_path = dataset + '/data/test_20230419.txt'  # 测试集
        self.cat2id_path = dataset + '/resources/cat_to_id.json'     # 类别字典
        self.datasetpkl = dataset + '/data/dataset_tch.pkl'
        with open(dataset + '/resources/cat_to_id.json', 'r', encoding='utf8') as r:
            self.class_dict = json.load(r)
        # self.class_dict = [x.strip() for x in open(
        #     dataset + '/data/classes.txt', 'r', encoding='utf8').readlines()]  # 类别名单
        os.makedirs(dataset + '/saved_dict/') if not os.path.exists(dataset + '/saved_dict/') else None
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_dict)
        self.cat_length = 37  # 类别个数
        self.scene_length = 12  # 场景个数
        self.num_epochs = 50
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 160
        self.learning_rate = 1e-4  # 学习率
        # self.bert_path = os.path.join(BASE_DIR, 'roberta')
        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hf_model_name = "uer/chinese_roberta_L-4_H-256"
        self.tokenizer = BertTokenizer.from_pretrained(self.hf_model_name)
        self.hidden_size = 256


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # self.bert = BertModel.from_pretrained(config.bert_path)
        self.bert = BertModel.from_pretrained(config.hf_model_name)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, return_dict=False)
        out = self.fc(pooled)
        out = torch.sigmoid(out)
        return out
