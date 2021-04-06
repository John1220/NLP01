#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import time
from tensorflow.keras.models import model_from_json
from importlib import import_module
from bertTextClassification.utils import build_dataset, get_time_dif, build_net_data, convert_content_to_inputs
import argparse
import os
from tqdm import tqdm
import numpy as np
import re
import csv

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', default="BruceBert", type=str,
                    help='choose a model: BruceBert, BruceBertCNN,BruceBertRNN')
args = parser.parse_args()


def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return (vTEXT)


def load_dataset(config):
    lables = []
    input_ids, token_type_ids, attention_masks = [], [], []
    f_classes = open(os.path.join(config.output_dir, 'classes.txt'), 'r', encoding='utf-8').readlines()
    vob_2_int = {vab.strip(): index for index, vab in enumerate(f_classes)}
    with open(config.dev_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            label, title, content = lin.split('\t')
            content = remove_urls(content)
            input_id, token_type_id, attention_mask = convert_content_to_inputs(title, content, config)
            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            attention_masks.append(attention_mask)
            lables.append(int(label))
    return (list(map(lambda x: np.asarray(x, dtype=np.int32), [input_ids, attention_masks, token_type_ids])), lables)


int_2_vab = {0: '产业经济动态', 1: '兄弟省市举措', 2: '国家部委政策动态', 3: '重大项目信息', 4: '其它'}


def predict(model, title, content, config):
    content = remove_urls(content)
    input_id, token_type_id, attention_mask = convert_content_to_inputs(title, content, config)
    sample = list(map(lambda x: np.asarray([x], dtype=np.int32), [input_id, token_type_id, attention_mask]))
    score = model.predict(sample)
    prob = list(np.max(score, 1))
    pre_class = list(np.argmax(score, 1))
    return [int_2_vab[i] for i in pre_class], prob
