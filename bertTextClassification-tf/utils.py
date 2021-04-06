#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import tensorflow as tf
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import re
from random import shuffle


def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return (vTEXT)


def build_dataset(config):
    def load_dataset(path):
        lables = []
        input_ids, token_type_ids, attention_masks = [], [], []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                print(lin)
                try:
                    label, title, content = lin.split('\t')
                    content = remove_urls(content)
                    input_id, token_type_id, attention_mask = convert_content_to_inputs(title, content, config)
                    input_ids.append(input_id)
                    token_type_ids.append(token_type_id)
                    attention_masks.append(attention_mask)
                    lables.append(int(label))
                except:
                    continue
        return (
            list(map(lambda x: np.asarray(x, dtype=np.int32), [input_ids, attention_masks, token_type_ids])), lables)

    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        train = dataset['train']
        dev = dataset['dev']
        test = dataset['test']
    else:
        train = load_dataset(config.train_path)
        dev = load_dataset(config.dev_path)
        test = load_dataset(config.test_path)
        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        dataset['test'] = test
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))
    return train, dev, test


def build_net_data(dataset, config):
    data_x = dataset[0]
    label_y = dataset[1]
    label_y = tf.keras.utils.to_categorical(label_y, num_classes=config.num_classes)
    return data_x, label_y


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def convert_content_to_inputs(title, content, config):
    tokenized_outputs = config.tokenizer.encode_plus(title, content, add_special_tokens=True,
                                                     max_length=config.max_len,
                                                     truncation=True,
                                                     truncation_strategy='longest_first',
                                                     pad_to_max_length=True)
    input_id = tokenized_outputs['input_ids']
    token_type_id = tokenized_outputs['token_type_ids']
    attention_mask = tokenized_outputs['attention_mask']
    # print(input_id.shape,token_type_id.shape,attention_mask.shape)
    return input_id, token_type_id, attention_mask


def save_file(config):
    """
    将分类目录下文件整合并存到3个文件中,并按比例划分训练集、测试集、验证集，文件内容格式:  类别\t内容
    Args:
        input_dir: 原数据目录名
        output_dir: 训练集、测试集、训练集保存目录
    return:
        train_count：训练集个数
    """
    input_dir = config.input_dir
    out_dir = config.output_dir
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    f_train = open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(out_dir, 'test.txt'), 'w', encoding='utf-8')
    f_val = open(os.path.join(out_dir, 'dev.txt'), 'w', encoding='utf-8')
    f_classes = open(os.path.join(out_dir, 'classes.txt'), 'w', encoding='utf-8')

    train_count = 0
    tag_2_ids = {}
    ii = 0
    for category in os.listdir(input_dir):  # 分类目录
        tag_2_ids[category] = ii
        cat_dir = os.path.join(input_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        filescounts = len(files)

        if filescounts == 0:
            raise Exception("""No data under this category""")
        if filescounts < 20:  # 某类别下至少有两条数据
            # raise Exception("""No enough data under this category, Please put at least two data """)
            continue

        f_classes.write(category + "\n")
        count = 0
        shuffle(files)  # 打乱某类别文档顺序
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            # title,_ = os.path.splitext(cur_file)#文件名去后缀
            try:

                file = open(filename, 'r', encoding='utf-8')
                data = file.read().strip().split('\n')
                # data.remove('\n')
                if len(data) == 1:
                    # title='$'
                    title = ' '
                    # content=''.join([i.strip().replace('/\s+/g','') for i in data])
                    content = data[0]
                else:
                    title = data[0].strip()
                    content = ''.join([i.strip().replace('/\s+/g', '') for i in data[2:]])
                file.close()
            except:
                continue

            if count < int(filescounts * 0.8):
                f_train.write(str(ii) + '\t' + title + '\t' + content + '\n')
                train_count += 1
            # elif count < int(filescounts*0.92):
            #    f_test.write(category + '\t' + title+'。'+ content + '\n')
            else:
                f_val.write(str(ii) + '\t' + title + '\t' + content + '\n')
                f_test.write(str(ii) + '\t' + title + '\t' + content + '\n')  # 测试集与验证集相同
            count += 1
        ii += 1

    # print('Finished:', category)
    f_classes.close()
    f_train.close()
    f_test.close()
    f_val.close()
    print(tag_2_ids)
