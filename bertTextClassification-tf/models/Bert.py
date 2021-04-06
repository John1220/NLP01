#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import tensorflow as tf
from transformers import BertTokenizer, BertConfig, TFBertModel
from tensorflow.keras.layers import Dense

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # 文件路径
print("BASE_DIR", BASE_DIR)


class Config(object):
    """
    配置参数
    """

    def __init__(self, dataset):
        self.model_name = 'Bert'
        self.output_dir = dataset + '/data'
        # 训练集
        self.train_path = dataset + '/data/train.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        # 校验集
        self.dev_path = dataset + '/data/dev.txt'
        # dataset
        self.datasetpkl = dataset + '/data/dataset.pkl'
        # 类别
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        # 模型训练结果
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.h5'

        # 类别数
        # self.num_classes = len(self.class_list)
        self.num_classes = 5
        # epoch数
        self.num_epochs = 30
        # batch_size
        self.batch_size = 8
        # 每句话处理的长度(短填，长切）
        self.max_len = 320
        # 学习率
        self.learning_rate = 1e-5
        # bert预训练模型位置
        self.bert_path = os.path.join(BASE_DIR, 'bert_pretrain')
        self.bert_model_config_path = os.path.join(self.bert_path, 'bert-base-chinese-config.json')
        self.bert_model_weights_path = os.path.join(self.bert_path, 'bert-base-chinese-tf_model.h5')
        self.bert_model_vocab_path = os.path.join(self.bert_path, 'bert-base-chinese-vocab.txt')
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_vocab_path)
        # bert隐层层个数
        self.hidden_size = 768


class MyModel(tf.keras.Model):

    def __init__(self, config):
        super(MyModel, self).__init__()
        self.bert_model_config = BertConfig.from_pretrained(config.bert_model_config_path)
        self.bert_model = TFBertModel.from_pretrained(config.bert_model_weights_path,
                                                      config=self.bert_model_config)
        self.fc = Dense(config.num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        embedding, cls_token2 = self.bert_model(inputs)
        cls_token = embedding[:, 0, :]
        output = self.fc(cls_token)

        return output
