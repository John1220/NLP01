
"""
模型训练模块：
1、加载模型库 text_classifier 中的指定模型进行训练
2、指定训练超参数，如epochs,batch_size,optimizer,loss_function 等
3、Tensorboard可视化设置
4、模型存储设置，如存储最优模型
"""

import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow import graph_util
from sklearn.metrics import accuracy_score

import dataset
from dl_model import textcnn
from config import Config, tc_logger, tensorboard_path
from Utils.syn_replacer import SynReplacer
from Utils.timer_utils import timer


model_folder = os.path.join(os.getcwd(), "Trained_model", "9-22-10-6", "0.997588")
checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
input_checkpoint = checkpoint.model_checkpoint_path  #得ckpt文件路径 

class ModelTrainer(object):
    """深度学习分类器训练模块"""

    def __init__(self, classifier, need_syn_replace=False):
        self.classifier = classifier
        self.need_syn_replace = need_syn_replace
        self.syn_replacer = SynReplacer("Resources/vocabulary/syn_index_dict.json") if need_syn_replace else None
        self._init_training_paras()

    def _init_training_paras(self):
        """初始化模型参数"""
        self.input_node = Config.sequence_length
        self.output_node = Config.cat_length + Config.scene_length
        self.batch_size = Config.batch_size
        self.training_steps = 100000

    def _get_random_block_from_data(self, datasets, data_labels, batch_size):
        """
        从训练集中以设定的 batch_size 随机获取样本
        :param datasets: 样本集
        :param data_labels: 样本标签
        :param batch_size: 批量大小
        :return: 随机获取的一批样本
        """
        sampels_index = np.random.randint(0, datasets.shape[0], batch_size)
        batch_samples, batch_labels = datasets[sampels_index], data_labels[sampels_index]
        if self.need_syn_replace:
            batch_samples = self.syn_replacer.generate_syn_samples(batch_samples)
        return batch_samples, batch_labels

    @timer
    def train(self, train_dataset, val_dataset, test_dataset=None, multi_label=False):
        """
        模型训练
        :param classifier:分类器模型
        :param train_dataset: 训练集
        :param val_dataset: 验证集
        :param test_dataset 测试集
        :return: 无返回，保存训练中最优的模型
        """
 
        
        # 模型的输入样本，输入样本的标签
        with tf.name_scope("input"):
            x = tf.placeholder(tf.int32, [None, self.input_node], name="x-input")
            y_ = tf.placeholder(tf.float32, [None, self.output_node], name="y-input")

        output = self.classifier.inference(x)

        # multi label
        # 模型输出
        if multi_label:
            logits = tf.nn.sigmoid(output)
            zero = tf.zeros_like(logits)
            one = tf.ones_like(logits)
            y_output = tf.where(logits < 0.5, x=zero, y=one, name='predict')
        else:
            soft_output = tf.nn.softmax(output)
            y_output = tf.arg_max(soft_output, 1, name="predict")      

        # 损失函数
        with tf.name_scope("loss_function"):
            if multi_label:
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y_)
            else:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y_)
            loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', loss)
           
        
        # 模型训练
        #train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
        with tf.name_scope("train_step"):
            train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
          
        # 预测
        # with tf.name_scope("accuracy"):
        #     correct_prediction = tf.equal(y_output, tf.argmax(y_, 1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #     tf.summary.scalar("accuracy", accuracy)

        # 创建会话
        with tf.Session() as sess:
            for j in range(1):  # 可设置多轮训练

                tf.global_variables_initializer().run()
                
                
                
                # 加载预训练模型权重
                saver = tf.train.Saver()
                saver.restore(sess, input_checkpoint)
                
                
                # 每次训练根据当前的时间创建文件夹，避免多次训练生成的文件混在一起
                now = datetime.now()
                now_str = "%d-%d-%d-%d" % (now.month, now.day, now.hour, now.minute)
                os.mkdir("Trained_model/" + now_str)

                merged = tf.summary.merge_all()

                # 创建 Tensorboard 可视化写入器，写入计算图和各种监控指标
                writer = tf.summary.FileWriter(tensorboard_path, tf.get_default_graph())

                # 验证集输入
                validation_feed = {x: val_dataset.samples, y_: val_dataset.labels}
                if test_dataset:
                    test_feed = {x: test_dataset.samples, y_: test_dataset.labels}

                best_acc = 0
                val_acc_list, test_acc_list = [], []

                for i in range(self.training_steps):
                    train_samples, train_labels = self._get_random_block_from_data(train_dataset.samples,
                                                                                   train_dataset.labels,
                                                                                   batch_size=self.batch_size)
                    train_feed = {x: train_samples, y_: train_labels}

                    # 每1000步存储一次训练可视化数据
                    if i % 1000 == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(train_step, feed_dict=train_feed, options=run_options, run_metadata=run_metadata)
                        writer.add_run_metadata(run_metadata, "step%03d" % i)
                        print("run mata_data is saved!")
                    else:
                        sess.run(train_step, feed_dict=train_feed)

                    # 每训练10步评估测试
                    if i % 20 == 0:
                        summary, train_loss = sess.run([merged, loss], feed_dict=train_feed)
                        writer.add_summary(summary, i)
                        
                        validation_acc = accuracy_score(val_dataset.labels, sess.run(y_output, feed_dict=validation_feed))
                        validation_acc = round(validation_acc, 6)
                        
                        tc_logger.info("after %d training steps,the train_loss is %f" % (i, train_loss)) 
                        tc_logger.info("after %d training steps,the validation accuracy is %f" % (i, validation_acc))

                    # 如果模型的准确率超过一定的值，且优于之前的最佳准确率，则保存该模型
                    if validation_acc > best_acc and validation_acc >= 0.996:
                        tc_logger.info("the best acc raises from %.4f to %.4f" % (best_acc, validation_acc))
                        best_acc = validation_acc
                        val_acc_list.append(validation_acc)
                        if test_dataset:
                            test_acc = accuracy_score(sess.run(y_output, feed_dict=test_feed), test_dataset.labels)
                            test_acc_list.append(test_acc)

                        # 保存模型
                        saver = tf.train.Saver()
                        model_name = "/textcnn_cpu_" + "%.4f" % (best_acc)
                        save_path = "Trained_model/" + now_str + f"/{best_acc}/" + model_name + ".ckpt"
                        saver.save(sess, save_path)

                        # 保存模型，保存之前先把计算图中的变量转换为常量，可以减小模型的大小，同时加快模型部署之后的推理速度
                        # save_path = "Trained_model/" + now_str + "/textcnn_cpu_" + "%.4f" % (best_acc) + ".pb"
                        # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                        #                                                            output_node_names=["predict"])
                        # with tf.gfile.FastGFile(save_path, mode='wb') as f:
                        #     f.write(constant_graph.SerializeToString())

                print("val_acc --> test_acc")
                [print("%.4f --> %.4f" % (val_, test_)) for val_, test_ in zip(val_acc_list, test_acc_list)]


if __name__ == "__main__":
    

    # 训练集加载
    train_dataset = dataset.WordDataset(Config.train_data_file, multi_cls=True)
    #train_dataset.generate_dataset(sample_balance=True)
    train_dataset.generate_dataset()

    # 验证集加载
    val_dataset = dataset.WordDataset(Config.val_data_file, multi_cls=True)
    val_dataset.generate_dataset()

    # 测试集加载
    test_dataset = dataset.WordDataset(Config.test_data_file, multi_cls=True)
    test_dataset.generate_dataset()

    # 初始化分类器和分类器训练器
    textcnn_classifier = textcnn(Config.word_vocab_file, Config.cat_vocab_file)
    model_trainer = ModelTrainer(textcnn_classifier, need_syn_replace=False)

    # 训练
    model_trainer.train(train_dataset, val_dataset, test_dataset, multi_label=True)
