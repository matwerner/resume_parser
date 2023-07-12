import argparse
import json
import os
import random
import sys
from argparse import Namespace
from datetime import datetime
from logging import Logger
from typing import Any, Dict, List

import logging
import numpy as np
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from resume_parser.layout.dataset import load_data_module
from resume_parser.layout.model import load_model
from resume_parser.layout.utils import (get_logger, combine_dfs, generate_output_dirpath,
                                        generate_model_experiment_name)
from resume_parser.layout.evaluation import ClassificationEvaluation

# Reproducibility
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
cudnn.benchmark = True
random.seed(0)
np.random.seed(0)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

# GPU or CPU
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(args: Namespace,
             model: nn.Module,
             data_loader: DataLoader,
             criterion: nn.Module,
             evaluation: ClassificationEvaluation,
            ):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    evaluation_start = datetime.now()
    with torch.no_grad():
        # for i, data in tqdm(enumerate(data_loader), total=len(data_loader), desc='Evaluating'):
        for i, data in enumerate(data_loader):
            image, labels = data
            image = image.to(args.device)
            labels = labels.to(args.device)

            # forward pass
            outputs = model(image)

            # calculate the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    y_pred = [args.labels[int(t)] for t in y_pred]
    y_true = [args.labels[int(t)] for t in y_true]

    epoch_loss = total_loss / len(data_loader)
    running_time = (datetime.now() - evaluation_start).total_seconds()
    metrics = evaluation.compute_metrics(y_true, y_pred, epoch_loss, running_time)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': metrics
    }


def run_split(args: Namespace, logger: Logger) -> Dict[str, Any]:
    # Load model
    model, training_args = load_model(args)
    args.image_size = training_args.image_size
    args.image_mean = training_args.image_mean
    args.image_std = training_args.image_std
    args.labels = training_args.labels
    args.num_labels = training_args.num_labels

    # Load dataset
    dataset = load_data_module(args, 'test')

    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()

    # To device
    model = model.to(args.device)

    logger.info("Testing model")

    criterion = nn.CrossEntropyLoss()

    train_evaluation = ClassificationEvaluation()
    train_output = evaluate(args, model, train_loader, criterion, train_evaluation)
    train_metrics = train_output['metrics']

    test_evaluation = ClassificationEvaluation()
    test_output = evaluate(args, model, test_loader, criterion, test_evaluation)
    test_metrics = test_output['metrics']

    # Results to logger
    def to_log(metrics: Dict, set_name: str):
        msg = '{}:\nReport:\n{}\nConfusion Matrix:\n{}\n'
        report_df, cm_df = metrics["classification_report"], metrics['confusion_matrix']
        logger.info(msg.format(set_name, str(report_df), str(cm_df)))
    to_log(train_metrics, 'TRAIN')
    to_log(test_metrics, 'TEST')

    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,

        'train_evaluation': train_evaluation,
        'test_evaluation': test_evaluation,
    }


def main(args: Namespace):
    args_filepath = os.path.join(args.experiment_dir, 'args.json')
    experiment_args = json.load(open(args_filepath, 'r', encoding='utf-8'))
    experiment_args = Namespace(**experiment_args)
    experiment_args.device = args.device

    experiment_name = generate_model_experiment_name(experiment_args, True)
    output_dirpath = generate_output_dirpath(__file__, 'output/model', prefix=experiment_name)

    args_filepath = output_dirpath + '/args.json'
    with open(args_filepath, mode='w', encoding='utf-8') as fp:
        json.dump(vars(experiment_args), fp, indent=4)

    # Log
    log_filepath = output_dirpath + '/log.txt'
    logger = get_logger(log_filepath)
    if experiment_args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    args_txt = json.dumps(vars(experiment_args), indent=4)
    logger.info("console arguments:\n%s", args_txt)
    logger.info("device: %s", experiment_args.device)

    results = []
    for i in range(experiment_args.num_splits):
        logger.info("##### SPLIT %d #####", i)
        experiment_args.split = i
        experiment_args.output_dir = os.path.join(args.experiment_dir, f"split_{i}")
        experiment_args.dataset_file = experiment_args.dataset_split.format(i)

        result = run_split(experiment_args, logger)
        results.append(result)

    train_metrics_list = [r['train_metrics'] for r in results]
    test_metrics_list = [r['test_metrics'] for r in results]
    def to_log(metrics_list: List[Dict[str, Any]], set_name: str):
        report_dfs = [metrics['classification_report'] for metrics in metrics_list]
        cm_dfs = [metrics['confusion_matrix'] for metrics in metrics_list]

        report_df = combine_dfs(report_dfs)
        cm_df = combine_dfs(cm_dfs)

        msg = '%s:\nReport:\n%s\nConfusion Matrix:\n%s\n'
        logger.info(msg, set_name, str(report_df), str(cm_df))
    to_log(train_metrics_list, 'TRAIN')
    to_log(test_metrics_list, 'TEST')


def get_args():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default=None,
        help='Experiment directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
        help='Device (cpu or cuda)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(get_args())
