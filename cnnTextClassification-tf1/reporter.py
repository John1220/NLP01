
"""
本模块为预测报告模块，并输出一系列报告：
    1、加载保存的最优模型
    2、生成 classification report 包含各类别的分类精度，查准率，查全率，F1值
    3、生成误分类样本文件，包含误分类类别，实际类别，原始文本
    4、生成混淆矩阵
"""
import time
import os
import re
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.metrics import classification_report, confusion_matrix
#from tensorflow_core.python.framework import graph_util

from Utils.timer_utils import timer
from scene_classifier import SceneClassifier
from Utils.MultiLabelUtil import inverse_transform
from dl_further_classifier import DLFurtherClassifier

from dataset import WordDataset
from config import Config, tc_logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))


# 深度学习分类器
class DLReporter(object):
    def __init__(self, model_path, multi_label=False):
        """
        初始化，载入保存的最佳模型和类别标签
        :param model_path: 模型保存的路径
        """
        self.multi_label = multi_label
        self.trained_classifier = self._get_best_trained_classifier(model_path)
        self.target_names = self._get_target_names()  # 将索引标签按顺序还原成文本标签，用于生成报告的补充信息
        word2id_dict = Config.word_vocab_file  # 分词转索引字典
        cat2id_dict = Config.cat_vocab_file  # 类别转索引字典
        with open(cat2id_dict, "r", encoding='utf-8') as ci:
            cat2id = json.load(ci)
            self.id2cat = {value: key for key, value in cat2id.items()}
        # self.word2id, self.id2cat = self._read_vocab(word2id_dict, cat2id_dict)

    def _get_best_trained_classifier(self, model_path):
        """
        从模型的路径中找到最优的模型并读取,返回读取后的会话
        :param model_path: 模型保存路径
        :return: 最优模型，文件名包含了测试精度
        """
        tc_logger.info("加载已保存最优模型")
        # 获取最优的模型文件
        
        file_list = os.listdir(model_path)
        if not file_list:
            raise ValueError
        model_dirs = sorted(map(lambda x: float(x), file_list))
        print(model_dirs)
        
        best_model_path = os.path.join(model_path, str(model_dirs[-1]))
        best_model_path = os.path.join(best_model_path,
                                       [meta for meta in os.listdir(best_model_path) if meta.endswith('meta')][0])
        # 加载最优模型
        tc_logger.info("加载已保存最优模型 %s" % best_model_path)
        saver = tf.train.import_meta_graph(best_model_path)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver.restore(sess, re.sub('.meta$', '', best_model_path))
        return sess

    @timer
    def predict(self, x_test):
        g = tf.get_default_graph()
        if self.multi_label:
            prob = g.get_tensor_by_name('Sigmoid:0')
            cat_id = g.get_tensor_by_name("predict:0")
            x = g.get_tensor_by_name('input/x-input:0')
            probs = self.trained_classifier.run(prob, feed_dict={x: x_test})
            return probs
        else:
            pre = g.get_tensor_by_name("Softmax:0")
            cat_id = g.get_tensor_by_name("predict:0")
            x = g.get_tensor_by_name("input/x-input:0")
            '''Compute the prediction value by loading the trained model.'''
            cat_indexs, confident = self.trained_classifier.run([cat_id, pre], feed_dict={x: x_test})
            return cat_indexs, confident

    def gen_cls_report_m(self, x_test, y_test, report_suffix):
        """
        生成classification_report
        格式如下：
                          precision    recall  f1-score   support
            <BLANKLINE>
                 class 0       0.50      1.00      0.67         1
                 class 1       0.00      0.00      0.00         1
                 class 2       1.00      0.67      0.80         3
            <BLANKLINE>
               micro avg       0.60      0.60      0.60         5
               macro avg       0.50      0.56      0.49         5
            weighted avg       0.70      0.60      0.61         5
            <BLANKLINE>
        """
        probs = self.predict(x_test)
        cats, confs = inverse_transform(probs, self.id2cat, Config.cat_length)
        cats, confs = cats.tolist(), confs.tolist()
        scenes = [it[-1] for it in cats]
        ops = [it[0] for it in cats]
        y_test, _ = inverse_transform(y_test, self.id2cat, Config.cat_length)
        y_scenes = [it[-1] for it in y_test]
        y_ops = [it[0] for it in y_test]

        # y_pred, _ = self.predict(x_test)

        true_pred_s = [int(scenes[i] == y_scenes[i]) for i in range(len(scenes))]
        true_pred_o = [int(ops[i] == y_ops[i]) for i in range(len(ops))]
        print("Test accuracy scene is %f !" % (sum(true_pred_s) / len(true_pred_s)))
        print("Test accuracy operation is %f !" % (sum(true_pred_o) / len(true_pred_o)))

        report_df_scene = pd.DataFrame(
            classification_report(y_scenes, scenes, output_dict=True, digits=3)).transpose()
        report_df_op = pd.DataFrame(
            classification_report(y_ops, ops, output_dict=True, digits=3)).transpose()
        with pd.ExcelWriter(os.path.join(Config.dl_reports_path, f"cls_reports_{report_suffix}.xlsx")) as w:
            report_df_scene.to_excel(w, sheet_name='scene')
            report_df_op.to_excel(w, sheet_name='operations')

    def get_error_cls_samples_m(self, x_test, y_test, text_dataframe, report_suffix):
        probs = self.predict(x_test)
        cats, confs = inverse_transform(probs, self.id2cat, Config.cat_length)
        cats, confs = cats.tolist(), confs.tolist()
        scenes = [it[-1] for it in cats]
        ops = [it[0] for it in cats]
        conf_scenes = [it[-1] for it in confs]
        conf_cats = [it[0] for it in confs]
        y_test, _ = inverse_transform(y_test, self.id2cat, Config.cat_length)
        y_scenes = [it[-1] for it in y_test]
        y_ops = [it[0] for it in y_test]

        error_pred_s = [scenes[i] != y_scenes[i] for i in range(len(scenes))]
        error_pred_o = [ops[i] != y_ops[i] for i in range(len(ops))]
        terms_name_list = ["right_pred", "pred_cat", "scene", "cat", "conf", "content", "filtered_cut"]

        error_terms_list = []
        right_terms_list = []
        for i in range(len(error_pred_s)):
            if error_pred_s[i]:
                error_terms_list.append(
                    ["N", scenes[i], y_scenes[i], y_ops[i], conf_scenes[i], text_dataframe["content"][i],
                     text_dataframe["filtered_cut"][i]])
            else:
                right_terms_list.append(
                    ["Y", scenes[i], y_scenes[i], y_ops[i], conf_scenes[i], text_dataframe["content"][i],
                     text_dataframe["filtered_cut"][i]])
        terms_df_s = pd.DataFrame(data=error_terms_list + right_terms_list, columns=terms_name_list)
        terms_df_s['filtered_cut'] = terms_df_s['filtered_cut'].apply(lambda x: x.replace(' ', " \ "))
        error_terms_list = []
        right_terms_list = []
        for i in range(len(error_pred_o)):
            if error_pred_o[i]:
                error_terms_list.append(
                    ["N", ops[i], y_scenes[i], y_ops[i], conf_cats[i], text_dataframe["content"][i],
                     text_dataframe["filtered_cut"][i]])
            else:
                right_terms_list.append(
                    ["Y", ops[i], y_scenes[i], y_ops[i], conf_cats[i], text_dataframe["content"][i],
                     text_dataframe["filtered_cut"][i]])
        terms_df_o = pd.DataFrame(data=error_terms_list + right_terms_list, columns=terms_name_list)
        terms_df_o['filtered_cut'] = terms_df_o['filtered_cut'].apply(lambda x: x.replace(' ', " \ "))
        with pd.ExcelWriter(os.path.join(Config.dl_reports_path, f"error_pred_messages_{report_suffix}.xlsx")) as w:
            terms_df_s.to_excel(w, sheet_name='scene', index=False)
            terms_df_o.to_excel(w, sheet_name='operations', index=False)

    def get_error_cls_samples(self, x_test, y_test, text_dataframe, report_suffix):
        """
        :param text_dataframe: Dataset.text_data_frame，用于分类的原始数据，包含 cat content cut 信息
        :return: 空，直接写入txt文件
        """
        if '打标结果' not in text_dataframe.columns:
            text_dataframe['打标结果'] = text_dataframe['cat']

        tc_logger.info("输出误分类文本文件，路径 %s" % os.path.join(Config.dl_reports_path, "error_pred_messages.xlsx"))

        y_pred, confidence = self.predict(x_test)
        confidence = np.max(confidence, axis=1)

        error_pred = [y_pred[i] != y_test[i] for i in range(len(y_pred))]  # 误分类索引列表

        # 需要存储的信息:是否预测正确、预测类别、实际类别、文本内容、过滤后的分词
        terms_name_list = ["right_pred", "confidence", "pred_cat", "cat", "打标结果", "content", "filtered_cut"]

        error_terms_list = []
        right_terms_list = []

        for i in range(len(error_pred)):
            if error_pred[i]:
                error_terms_list.append(
                    ["N", "%.2f" % confidence[i], self.target_names[y_pred[i]],
                     text_dataframe["cat"][i], text_dataframe["打标结果"][i],
                     text_dataframe["content"][i], text_dataframe["filtered_cut"][i]])
            else:
                right_terms_list.append(
                    ["Y", "%.2f" % confidence[i], self.target_names[y_pred[i]],
                     text_dataframe["cat"][i], text_dataframe["打标结果"][i],
                     text_dataframe["content"][i], text_dataframe["filtered_cut"][i]])
        right_terms_list.sort(key=lambda x: x[1])
        terms_list = error_terms_list + right_terms_list

        terms_dict = {}
        for i in range(len(terms_name_list)):
            terms_dict[terms_name_list[i]] = [term[i] for term in terms_list]

        # 将预测结果写入报告
        terms_df = pd.DataFrame(terms_dict)
        terms_df['filtered_cut'] = terms_df['filtered_cut'].apply(lambda x: x.replace(' ', " \ "))
        terms_df.to_excel(os.path.join(Config.dl_reports_path, f"error_pred_messages_{report_suffix}.xlsx"),
                          index=False)

    # 获取分类的混淆矩阵
    def gen_confusion_matrix(self, x_test, y_test, report_suffix):
        tc_logger.info("生成混淆矩阵,路径为 %s" % os.path.join(Config.dl_reports_path, "confusion_matrix.xlsx"))
        y_pred, _ = self.predict(x_test)
        cf_matrix = confusion_matrix(y_test, y_pred)

        cat_set = set(y_pred).union(set(y_test))
        target_names = [self.target_names[i] for i in cat_set]

        cf_matrix_df = pd.DataFrame(cf_matrix, index=target_names, columns=target_names)
        cf_matrix_df.to_excel(os.path.join(Config.dl_reports_path, f"confusion_matrix_{report_suffix}.xlsx"),
                              encoding="utf_8_sig")
        tc_logger.info(cf_matrix)
        return cf_matrix_df

    # 通过类别索引字典，按序获取文本类别列表[]
    def _get_target_names(self):
        with open(Config.cat_vocab_file, "r", encoding='utf8') as li:
            cat_to_id = json.load(li)
        sorted_cats = sorted(cat_to_id.items(), key=lambda x: x[1])
        target_names, _ = zip(*sorted_cats)
        return list(target_names)


# 场景分类器 Reporter
class SceneReporter(object):
    def __init__(self):
        """
        报告针对的分类器，默认为深度学习分类器。
        :param defualt_report:
        :param model_path:
        """
        self.scene_classifier = SceneClassifier()

    def predict(self, x_test):
        return [self.scene_classifier.scene_predict(message=x[1], app_name=x[0]) for x in x_test]

    def gen_cls_report(self, dataset_df):
        """
        生成classification_report
        格式如下：
                          precision    recall  f1-score   support
            <BLANKLINE>
                 class 0       0.50      1.00      0.67         1
                 class 1       0.00      0.00      0.00         1
                 class 2       1.00      0.67      0.80         3
            <BLANKLINE>
               micro avg       0.60      0.60      0.60         5
               macro avg       0.50      0.56      0.49         5
            weighted avg       0.70      0.60      0.61         5
            <BLANKLINE>
        """
        # tc_logger.info("生成 classification report,路径 %s" % os.path.join(Config.scene_reports_path, "cls_reports.xlsx"))

        y_pred = self.predict(dataset_df[["app_name", "content"]].values)
        y_test = list(dataset_df["场景"])

        true_pred = [int(y_pred[i] == y_test[i]) for i in range(len(y_pred))]
        print("Test accuracy is %f !" % (sum(true_pred) / len(true_pred)))
        target_names = list(set(y_pred).union(set(y_test)))
        y_pred_idx = [target_names.index(y) for y in y_pred]
        y_test_idx = [target_names.index(y) for y in y_test]

        report = classification_report(y_test_idx, y_pred_idx, target_names=list(target_names), output_dict=True,
                                       digits=3)
        report_df = pd.DataFrame(report).transpose()
        # tc_logger.info(report_df)

        # 写入文件
        report_df.to_excel(os.path.join(Config.scene_reports_path, f"cls_reports_{current_time}.xlsx"),
                           encoding="utf_8_sig")

    def get_error_cls_samples(self, dataset_df):
        """
        :param text_dataframe: Dataset.text_data_frame，用于分类的原始数据，包含 cat content cut 信息
        :return: 空，直接写入txt文件
        """
        # tc_logger.info("输出误分类文本文件，路径 %s" % os.path.join(Config.scene_reports_path, "error_pred_messages.xlsx"))

        y_pred = self.predict(dataset_df[["app_name", "content"]].values)
        y_test = list(dataset_df["场景"])

        error_pred = [y_pred[i] != y_test[i] for i in range(len(y_pred))]  # 误分类索引list

        # 需要存储的信息:是否预测正确、预测类别、实际类别、文本内容、过滤后的分词
        terms_name_list = ["right_pred", "pred_scene", "scene", "app_name", "content"]

        error_terms_list = []
        right_terms_list = []

        for i in range(len(error_pred)):
            if error_pred[i]:
                error_terms_list.append(
                    ["N", y_pred[i], y_test[i], dataset_df["app_name"][i], dataset_df["content"][i]])
            else:
                right_terms_list.append(
                    ["Y", y_pred[i], y_test[i], dataset_df["app_name"][i], dataset_df["content"][i]])

        terms_list = error_terms_list + right_terms_list

        terms_dict = {}
        for i in range(len(terms_name_list)):
            terms_dict[terms_name_list[i]] = [term[i] for term in terms_list]

        # 将预测结果写入报告
        terms_df = pd.DataFrame(terms_dict)
        terms_df.to_excel(os.path.join(Config.scene_reports_path, f"error_pred_messages_{current_time}.xlsx"),
                          index=False)

    # 获取分类的混淆矩阵
    def gen_confusion_matrix(self, dataset_df):
        # tc_logger.info("生成混淆矩阵，路径为：%s" % os.path.join(Config.scene_reports_path, "confusion_matrix.xlsx"))
        y_pred = self.predict(dataset_df[["app_name", "content"]].values)
        y_test = dataset_df["场景"]
        target_names = list(set(y_pred).union(set(y_test)))

        y_pred_idx = [target_names.index(y) for y in y_pred]
        y_test_idx = [target_names.index(y) for y in y_test]
        cf_matrix = confusion_matrix(y_test_idx, y_pred_idx)

        cf_matrix_df = pd.DataFrame(cf_matrix, index=target_names, columns=target_names)
        cf_matrix_df.to_excel(os.path.join(Config.scene_reports_path, f"confusion_matrix_{current_time}.xlsx"),
                              encoding="utf_8_sig")
        tc_logger.info(cf_matrix)
        return cf_matrix_df



if __name__ == "__main__":
    file_dir = os.path.dirname(__file__)

    # 1、scene/dl_classifier reporter
    model_path = os.path.join(file_dir, "Trained_model/9-22-12-4")
    dl_reporter = DLReporter(model_path, multi_label=True)
    test_path = Config.test_data_file
    report_suffix = "_aplha_v2"
    # report_suffix = "5.2_v2"
    test_path = os.path.join(file_dir, test_path)
    test_dataset = WordDataset(test_path, multi_cls=True)
    test_dataset.generate_dataset(one_hot=False)
    dl_reporter.gen_cls_report_m(test_dataset.samples, test_dataset.labels, report_suffix)
    dl_reporter.get_error_cls_samples_m(test_dataset.samples, test_dataset.labels, test_dataset.text_data_frame,
                                        report_suffix)
    # # dl_reporter.gen_confusion_matrix(test_dataset.samples, test_dataset.labels, report_suffix)

    
    # 2、dl_further_classifier reporter
    # dataset_df = pd.read_excel("Dataset/labeled_messages_beta_v1.21.xlsx")
    # dataset_df['content'] = dataset_df['content'].apply(lambda x: re.sub(r'\s', '', x))
    # dl_further_reporter = DlFutherReporter()
    # scene_name = "撤保"
    # dl_further_reporter.gen_cls_report(scene_name, dataset_df)
    # dl_further_reporter.get_error_cls_samples(scene_name, dataset_df)
    # dl_further_reporter.gen_confusion_matrix(scene_name, dataset_df)
