# -*- coding:utf-8 -*-
import time
import torch
import numpy as np
from importlib import import_module
from train_eval import train
from utils import build_dataset, build_iterator, get_time_dif

if __name__ == '__main__':
    dataset = ''
    model_name = "bert"
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    config.batch_size = 64
    config.pad_size = 160
    config.bert_path = '../bertTextMultiLabelClassification/roberta'

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config, use_pkl=False, pad_size=config.pad_size)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    # if os.path.exists(config.save_path):
    #   model.load_state_dict(torch.load(config.save_path))
    # test(config, model, test_iter)
    train(config, model, train_iter, dev_iter, test_iter)