"""
本模块为深度学习：
    1、构造各种分类模型，供 model_trainer 调用
    2、两种初始化方式，一种是随机初始化模型，另外一种是加载预训练的参数
"""

import os
import tensorflow as tf
from config import Config
from Utils.io_utils import IO_Utils

# 控制台仅输出 warning 和 error 信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def variable_summaries(var, name):
    """
    记录网络各层变量的均值(mean)和标准差(std)
    :param var: 变量
    :param name: 名称
    """
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


class Layers(object):
    """深度学习中的各种层，全连接层、卷积层、词嵌入层等"""

    def __init__(self):
        pass

    # 全连接层
    @staticmethod
    def dense(input, output_dim):
        """
        全连接层
        """
        # 权重和偏置
        weights = tf.Variable(tf.truncated_normal([input.get_shape().as_list()[1], output_dim], stddev=0.1))
        biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
        layer = tf.matmul(input, weights) + biases

        # 将网络权重加入tf.summary.scalar
        variable_summaries(weights, "Dense_weights")
        variable_summaries(biases, "Dense_weights")
        return layer

    # 一维卷积层
    @staticmethod
    def conv_1d(input_tensor, conv_width, output_channels):
        """
        一维卷积层
        """
        filter = tf.Variable(
            tf.truncated_normal([conv_width, input_tensor.get_shape().as_list()[2], output_channels], stddev=0.1))
        conv_biases = tf.Variable(tf.constant(0.0, shape=[output_channels]))
        conv = tf.nn.conv1d(input_tensor, filter, stride=1, padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

        # 将权重记入tf.summary.scalar
        variable_summaries(filter, "conv_weights")
        variable_summaries(conv_biases, "conv_biases")
        return relu

    # 词嵌入层
    @staticmethod
    def embedding(input_tensor, vocab_size, emb_size):
        """词嵌入层"""
        embedding = tf.Variable(tf.truncated_normal([vocab_size, emb_size], stddev=0.1))
        variable_summaries(embedding, "embedding")
        return tf.nn.embedding_lookup(embedding, input_tensor)

    # 全局最大池化层
    @staticmethod
    def global_max_pooling(input_tensor):
        """全局最大平均"""
        pool = tf.layers.max_pooling1d(input_tensor, pool_size=input_tensor.get_shape().as_list()[1], strides=1)
        return pool


class textcnn(object):
    """
    textcnn文本分类模型，参考：
        Convolutional Neural Networks for Sentence Classification https://arxiv.org/pdf/1408.5882.pdf
    """

    def __init__(self, word_dict_path, cat_id_path):
        """
        :param word_dict_path: 词索引字典文件路径
        :param cat_id_path: 类索引字典文件路径
        """
        self.embedding_size = Config.embedding_size
        self.word_length = len(IO_Utils.load_json(word_dict_path))  # 获取字典的规模
        self.output_dim = len(IO_Utils.load_json(cat_id_path))

    # 计算前向传播
    def inference(self, x_input):
        with tf.name_scope("network"):
            # x_input = tf.layers.dropout(x_input, 0.1) # 输入dropout
            embed = Layers.embedding(x_input, self.word_length, Config.embedding_size)
            cnn1 = Layers.conv_1d(embed, 2, output_channels=256)
            cnn1 = Layers.global_max_pooling(cnn1)
            # emb1 = tf.reduce_mean(embed, axis=1, keepdims=True)
            # cnn1 = tf.concat([emb1, cnn1], axis=1)
            flat = tf.layers.flatten(cnn1)
            drop = tf.layers.dropout(flat, 0.5)
            output = Layers.dense(drop, self.output_dim)
        return output
