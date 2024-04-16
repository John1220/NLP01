import os, re
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import graph_util
from tensorflow.python.platform import gfile

from Utils.io_utils import IO_Utils
from Utils.timer_utils import timer
from Utils.MultiLabelUtil import inverse_transform
from config import Config
from segmenter import segmenter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dl_model_path = "Trained_model/Release/textcnn_cpu_0.997752.pb"  # .pb深度学习模型路径
#dl_model_path = "Trained_model/4-21-16-1/0.997823/textcnn_cpu_0.9978.ckpt.meta"  # .meta深度学习模型路径

word2id_dict = Config.word_vocab_file  # 分词转索引字典
cat2id_dict = Config.cat_vocab_file  # 类别转索引字典


class DLClassifier(object):
    """深度学习分类器，读取训练好的深度学习模型，输入文本文本，输出文本的场景+类别"""

    def __init__(self, model_path=dl_model_path, word2id_dict=word2id_dict, cat2id_dict=cat2id_dict):
        """
        :param model_path: 保存模型的路径，自动加载最优模型
        :param x_test:  测试集样本
        :param y_test:  测试集标签
        """
        if not model_path.endswith('pb'):
            self.trained_classifier = self._get_trained_ckpt(model_path)
        else:
            self.trained_classifier = self._get_trained_model(model_path)
        self.word2id, self.id2cat = self._read_vocab(word2id_dict, cat2id_dict)

    def _get_trained_ckpt(self, model_path):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver = tf.train.import_meta_graph(model_path)
        saver.restore(sess, re.sub('.meta$', '', model_path))
        return sess

    def inverse_transform(self, probs, id2cat, cat_length):
        """
        根据模型输出概率 转换成label
        :param probs: 模型输出的每个类别概率
        :param id2cat: 类别索引字典
        :param cat_length: 行为标签类别个数
        :return:
        """
        cat_probs, scene_probs = probs[:, :cat_length], probs[:, cat_length:]
        # 对概率向量分段，行为段和场景段，每段概率最大值作为预测标签
        cat_indexs = np.concatenate([np.expand_dims(np.argmax(cat_probs, 1), 1),
                                     cat_length + np.expand_dims(np.argmax(scene_probs, 1), 1)], 1)
        cats = np.vectorize(id2cat.get)(cat_indexs)
        confidences = [prob.take(idx).tolist() for idx, prob in zip(cat_indexs, probs)]
        confidences = np.round(confidences, 4)
        # cats[np.where(confidences[:, 1] < 0.1)[0], 1] = '其他场景'
        # cats[np.where(confidences[:, 0] < 0.1)[0], 0] = '其他信息'
        return cats, confidences

    @timer
    def predict_multi_label(self, messages, max_batch=20000):
        samples = self._texts_to_matrix(messages)  # 将输入的文本转换成矩阵

        g = tf.get_default_graph()
        prob = g.get_tensor_by_name('Sigmoid:0')
        # cat_id = g.get_tensor_by_name("predict:0")
        x = g.get_tensor_by_name("input/x-input:0")

        # 预测
        cat_list = []
        conf_list = []
        pred_epochs = math.ceil(len(messages) / max_batch)
        for i in range(pred_epochs):
            probs = self.trained_classifier.run(prob, feed_dict={x: samples[i * max_batch:(i + 1) * max_batch]})
            # 概率转换成标签
            cats, confs = self.inverse_transform(probs, self.id2cat, Config.cat_length)
            cat_list += cats.tolist()
            conf_list += confs.tolist()
        return cat_list, conf_list

    def _read_vocab(self, word2id_dict, cat2id_dict):
        """
        获取：（1）分词转索引字典，(2)类别索引转类别名称字典，后者用于将深度学习预测得到类别索引，转换为对应的类别
        :param word2id_dict: 分词转索引字典文件
        :param cat2id_dict: 类别转索引字典文件
        :return: 分词转索引字典，类别索引转类别字典
        """
        # 读取分词词典({分词1:1,分词2:2,...}) 和类别索引词典({类别1:1,类别2:2,...})
        with open(word2id_dict, "r", encoding='utf-8') as wi:
            word2id = json.load(wi)
        with open(cat2id_dict, "r", encoding='utf-8') as ci:
            cat2id = json.load(ci)
            id2cat = {value: key for key, value in cat2id.items()}
        return word2id, id2cat

    def _get_trained_model(self, model_path):
        """
        :param model_path: 模型保存路径
        :return: 模型的会话 sess，通过 sess.run() 实现预测
        """
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        return sess

    # 模型的预测
    # @timer
    def predict(self, messages, max_batch=20000):
        """
        :param messages: 文本列表
        :return: 预测的类别列表，和文本列表中的文本一一对应
        """
        samples = self._texts_to_matrix(messages)  # 将输入的文本转换成矩阵

        g = tf.get_default_graph()
        pre = g.get_tensor_by_name("Softmax:0")
        cat_id = g.get_tensor_by_name("predict:0")
        x = g.get_tensor_by_name("input/x-input:0")

        # 预测
        cat_list = []
        conf_list = []
        pred_epochs = math.ceil(len(messages) / max_batch)
        for i in range(pred_epochs):
            cat_indexs, confidence = self.trained_classifier.run([cat_id, pre], feed_dict={
                x: samples[i * max_batch:(i + 1) * max_batch]})
            cat_list = cat_list + [self.id2cat[cat_index] for cat_index in cat_indexs]
            confidence = list(np.max(confidence, axis=1))
            conf_list += confidence
        return cat_list, conf_list

    # 文本列表转矩阵
    def _texts_to_matrix(self, messages):
        """
        将文本列表转换为矩阵，每一条文本转换为等长的向量，作为矩阵的一行
        :param messages: 文本列表
        :return: 转换之后的矩阵
        """
        cuts = [segmenter.cut_words(message) for message in messages]
        samples = self._convert_word_id(cuts, self.word2id)
        return samples

    def _convert_word_id(self, cuts, word_to_id):
        """
        将词列表转换为数值列表
        :param cuts: 词列表，list
        :param word_to_id: 词索引字典，可获得词对应的索引值，从文件中加载得到，dict
        :return: 样本数值矩阵，np.array,每一行为一个样本
        """
        filter_cut_list = []
        converted_ids = []
        for cut in cuts:
            word_list = str(cut).split()
            cut = [word for word in word_list if word in word_to_id.keys()]
            padded_cut = self._pad_sequence(cut)  
            filter_cut_list.append(" ".join(padded_cut))
            converted_cut = [word_to_id[word] for word in padded_cut]
            converted_ids.append(converted_cut)
        return np.array(converted_ids)

    def _pad_sequence(self, cut, max_length=Config.sequence_length, pad_str="<PAD>"):
        """
        词列表填充函数，对字列表进行填充或者截断，以达到指定长度。
        :param cut: 词列表 list
        :param max_length:  序列长度
        :param pad_str: 填充的字符，默认为 "<PAD>"
        :return: 返回填充后的词列表
        """
        if len(cut) >= max_length:
            pad_cut_list = cut[:max_length]
        else:
            pad_strs = [pad_str] * (max_length - len(cut))
            pad_cut_list = cut + pad_strs
        return pad_cut_list


def predict_csv():
    """
    测试用
    :return:
    """
    data_path = r"C:\repositories\TextSample\output\dropedtext-169w-hm10-0412-15_00.txt"
    contents = IO_Utils.read_texts(data_path)
    # s, e = 2, len(text_lines)
    # contents = text_lines[s * 10000:]
    df = pd.DataFrame({'content': contents})
    cat, confds = dl_classifier.predict_multi_label(messages=contents)
    # app = pd.read_excel(r'C:\Users\12209\Documents\文件\Gama渠道数据-国内\gama_samples\app_count.xlsx').dropna()
    # app_dict = app[['app_name', 'scene']].set_index('app_name').to_dict()
    # df['sc'] = df['app_name'].apply(lambda x: app_dict['scene'].get(x, x))
    df['scene_pre'] = [it[1] for it in cat]
    df['cat_pre'] = [it[0] for it in cat]
    df['scene_prob'] = [it[1] for it in confds]
    df['cat_prob'] = [it[0] for it in confds]
    df.to_csv(os.path.join("output", f"pred_hm10-0412-15_00.csv"), index=False)


def predict_label_data():
    """
    测试用
    :return:
    """
    df_cat = pd.read_excel('data/labeld_message_gm_mb_jd.xlsx')
    # df_cat = df_cat.loc[(df_cat['verif'] != 1) & (df_cat['enterprise'] != 1) &
    #                     (df_cat['scene_fix'].isin(used))]
    df_cat = df_cat[['content', 'scene_fix', 'cat_fix']]

    df_cat_1 = pd.read_excel('data/labeld_message_gamma.xlsx')
    df_cat_1 = df_cat_1[['content', 'scene_fix', 'cat_fix']]

    df_cat_2 = pd.read_excel('data/labeld_message_gm_mb_4.xlsx')
    df_cat_2 = df_cat_2.loc[df_cat_2['scene_fix'] != '借贷old', ['content', 'scene_fix', 'cat_fix']]
    # df_gm['cat_fix'] = df_gm['cat_fix'].apply(lambda x: cm.cat_map(x))
    df = pd.concat([df_cat, df_cat_1, df_cat_2], ignore_index=True)

    # df = pd.read_excel(r'data/labeld_message_gm_mb.xlsx')
    # df_cat_2 = pd.read_excel('data/labeld_message_gm_mb_2.xlsx')
    # df = pd.concat([df, df_cat_2], ignore_index=True)
    # df.to_excel(r'data/labeld_message_gamma.xlsx', index=False)
    # df = df.drop(['scene', 'cat'], axis=1)
    # df = df.rename(columns={'cat_fix': 'cat', 'scene_fix': 'scene'})

    cat, confds = dl_classifier.predict_multi_label(df['content'].tolist())
    df['scene_pre'] = [it[1] for it in cat]
    df['same_sc'] = df.apply(lambda x: '1' if x['scene_pre'] == x['scene_fix'] else '0', axis=1)
    df['cat_pre'] = [it[0] for it in cat]
    df['same_cat'] = df.apply(lambda x: '1' if x['cat_pre'] == x['cat_fix'] else '0', axis=1)
    df['scene_prob'] = [it[1] for it in confds]
    df['cat_prob'] = [it[0] for it in confds]
    df.to_excel('output/labeld_message_gamma_pred.xlsx', index=False)

def convert_model(model_folder, pb_model_path):
    """
    将ckpt文件转换为.pb文件
    """
    checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path  #得ckpt文件路径
    
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices =True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(sess=sess, input_graph_def=input_graph_def,
                                                                      output_node_names=["predict"])
        with tf.gfile.GFile(pb_model_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("ckpt transform to pb done!")    


# 模块单例
dl_classifier = DLClassifier()

if __name__ == "__main__":
    # predict_label_data()
    convert_model(os.path.join(os.getcwd(), "Trained_model", "9-22-12-4", "0.997752"), os.path.join(os.getcwd(), "Trained_model", "Release", "textcnn_cpu_0.997752.pb"))
    message = []
    #result = dl_classifier.predict_multi_label(message)
    #print(result)