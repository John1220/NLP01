"""
系统全局配置模块：
    1. log 日志配置
    2. 模型训练参数配置
    3. 系统资源：数据集，词典 路径配置
    4、程序生成文件配置：可视化Tensorboard文件，预测报告等
"""

import os
import logging.config
import yaml

def check_path(path):
    """检查是否存在路径，若不存在则创建路径"""
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
    return path


# 获取系统各资源
project_path = os.path.dirname(__file__)

dataset_path = check_path(os.path.join(project_path, "Dataset"))  # 数据集路径
vocabulary_path = check_path(os.path.join(project_path, "Resources/vocabulary"))  # 词典路径，用于将词和分类映射为数值
log_path = check_path(os.path.join(project_path, "Logs"))  # 日志路径

tensorboard_path = check_path(os.path.join(project_path, "Tensorboard"))  # Tensorboard 可视化相关资源
segmenter_resources_path = check_path(os.path.join(project_path, "Resources/segmenter_resources"))  # 分词器相关资源

# 创建名为"tc_logger"的日志对象，导入yaml配置
with open(os.path.join(log_path, "log_config.yaml"), 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
tc_logger = logging.getLogger("tc_logger")


# 全局配置参数，包括训练配置，后续可添加其他配置
class Config(object):
    # 模型参数
    cat_length = 71  # 类别个数 
    scene_length = 22  # 场景个数 
    sequence_length = 40  # 统一样本序列长度，大于则截断，小于则填充
    batch_size = 256
    embedding_size = 200  # 词嵌入的长度

    # 训练集，验证集，测试集
    train_data_file = os.path.join(dataset_path, "alpha_v4", "dataset_2022826/train_2022826.txt")  # 训练数据集文件
    val_data_file = os.path.join(dataset_path, "alpha_v4", "dataset_2022826/valid_2022826.txt")  # 验证数据集文件
    test_data_file = os.path.join(dataset_path, "alpha_v4", "dataset_2022826/test_2022826.txt")  # 测试数据集文件

    # 分词和类别索引文件
    word_vocab_file = os.path.join(vocabulary_path, "alpha/word_to_id_20220826.json")  # 词索引字典，将每一个词转换成一个索引
    cat_vocab_file = os.path.join(vocabulary_path, "alpha/cat_to_id_20220826.json")  # 类索引字典，将每一个类别转换为数字

    # 模型评估报告相关资源
    scene_reports_path = check_path(os.path.join(project_path, "Reports/scene_cls_report/"))  # 场景分类器预测报告保存路径
    dl_reports_path = check_path(os.path.join(project_path, "Reports/dl_cls_report/"))  # 深度学习分类器预测报告保存路径
    dl_further_reports_path = check_path(os.path.join(project_path, "Reports/dl_further_cls_report/"))  # 细分分类器预测报告保存路径
    bert_report_path = check_path(os.path.join(project_path, "Reports/bert_cls_report/"))

    # 分词相关资源
    stopwords_file = os.path.join(segmenter_resources_path, "stopwords.txt")  # 停用词文件，存储停用词
    user_dict_file = os.path.join(segmenter_resources_path, "user_dict.txt")  # 用户词典,用于分词模块
