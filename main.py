# -*- coding: utf-8 -*-
"""
This implementation is based on 
    - AutoGluon https://auto.gluon.ai/stable/index.html 
    - GluonCV https://cv.gluon.ai/api/data.datasets.html

Note that although we are only borrowing the AutoML library, this project is not about AutoML. 
"""
from gluoncv.auto.estimators import ImageClassificationEstimator
from gluoncv.auto.tasks import ImageClassification
import autogluon.core as ag
import pandas as pd
import time
import numpy as np
import logging
import argparse
from utils import save_acc_pkl, plot_acc
import os
os.environ['NUMEXPR_MAX_THREADS'] = '32'
import mxnet as mx
mx.nd.waitall()

parser = argparse.ArgumentParser(description='Experiment')
parser.add_argument('--exp-dir', type=str, required=True)
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--result-dir', type=str, default='result')
args = parser.parse_args()


EXP_ROOT_PATH = args.exp_dir 
DATA_ROOT_PATH = os.path.join(EXP_ROOT_PATH, args.data_dir) 
RESULTS_PATH = os.path.join(EXP_ROOT_PATH, args.result_dir) 
os.makedirs(RESULTS_PATH, exist_ok = True)
time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
pkl_result_path = os.path.join(RESULTS_PATH, "eval-acc-{}.pkl".format(time_str)) 
plot_result_path = os.path.join(RESULTS_PATH, "plot-{}.jpg".format(time_str)) 

"""
Datasets in Pytorch Image Folder Format: https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html 

1. Training Dataset: 
    - os.path.join(DATA_ROOT_PATH, 'train')

2. Validation Datasets (each would have an evaluation acc):
    - os.path.join(DATA_ROOT_PATH, 'majority-val')
    - os.path.join(DATA_ROOT_PATH, 'minority-val')
"""

train_data          = ImageClassification.Dataset.from_folder( os.path.join(DATA_ROOT_PATH, 'train') )
majority_val_data  = ImageClassification.Dataset.from_folder( os.path.join(DATA_ROOT_PATH, 'majority-val') ) 
minority_val_data = ImageClassification.Dataset.from_folder( os.path.join(DATA_ROOT_PATH, 'minority-val') )
val_data = majority_val_data

"""
Execution flow:
1. Search space specification. initialize a configuration search space of 500 entries. 
2. Single/Parallel execution of training; Evaluation for majority subpopulation and minority subpopulation. During the process, we maintain a table where each entry corresponds to a configuration, and update the table every time we finish training a model.
3. Join and merge results. After this step, we get a full table with evaluation results of each configuration.
"""

@ag.args( # 5 models * 5 lr * 5 batch_size * 4 epochs = 500 configurations
    model = ag.space.Categorical(
        'mobilenetv3_small', 
        'resnet18_v1b', 
        'resnet50_v1', 
        'mobilenetv3_large', 
        'resnet101_v2', 
        ),
    lr = ag.space.Categorical(0.01, 0.005, 0.001, 0.0005, 0.0001), 
    batch_size = ag.space.Categorical(8, 16, 32, 64, 128), 
    epochs = ag.space.Categorical(1, 5, 10, 25)
    )
    
def train_fn(args, reporter):
    # Wrap parameterizable classes and functions
    print('args: {}'.format(args))
    model_keys = ['model', 'batch_norm', 'last_gamma', 'use_gn', 'use_pretrained', 'use_se']
    model_args = {k: v for k, v in args.items() if k in model_keys}
    train_args = {k: v for k, v in args.items() if k not in model_keys+['task_id']}
    classifier = ImageClassificationEstimator({
        'train': train_args, 
        'img_cls': model_args, 
        'valid': {'batch_size': 512}, 
        'gpus': [0] }) 

    # Train and evaluate
    train_results = classifier.fit(train_data, val_data)
    majority_eval_acc = classifier.evaluate(majority_val_data)[0]
    minority_eval_acc = classifier.evaluate(minority_val_data)[0]
    
    # Report results
    reporter(
        args=args,
        train_results=train_results,
        majority_eval_acc=majority_eval_acc,
        minority_eval_acc=minority_eval_acc,
        )
    return

def train_job_callback(training_history, time, config_history, state_dict):
    eval_df = save_acc_pkl(pkl_result_path, training_history)
    plot_acc(plot_result_path, eval_df)
    return
    
# -------- training scheduler -------- #
scheduler = ag.scheduler.FIFOScheduler(train_fn,
                                       resource={'num_cpus': 2, 'num_gpus': 1},
                                       num_trials=500,
                                       reward_attr='accuracy',
                                       time_attr='epoch',
                                       training_history_callback=train_job_callback)
try:
    scheduler.run()
except Exception as e:
    pass
print('-------- scheduler finished --------')
scheduler.join_jobs()

