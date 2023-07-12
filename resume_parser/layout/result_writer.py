from argparse import Namespace
from datetime import datetime
from logging import Logger
from typing import Tuple, List, Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sn


# DISK

def dump_result_split_to_disk(filename: str,
                              args: Namespace,
                              training_result: Dict[str, Any],
                              best_metric_name: Optional[str] = 'accuracy',
                             ) -> Dict[str, Any]:
    train_evaluation = training_result['train_evaluation']
    valid_evaluation = training_result['valid_evaluation']

    summary = vars(args)
    summary['timestamp'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    summary['is_early_stopping'] = training_result['is_early_stopping']
    summary['training_running_time'] = training_result['training_running_time']
    summary['evaluating_running_time'] = training_result['evaluating_running_time']
    summary['total_running_time'] = training_result['total_running_time']

    best_epoch = valid_evaluation.get_best_epoch(best_metric_name)
    summary['best_epoch'] = best_epoch

    def to_float(value: Any):
        if isinstance(value, list):
            return [round(float(v), 4) for v in value]
        return round(float(value), 4)

    skip_metric_names = ['classification_report', 'confusion_matrix']
    for prefix, evaluation in [('train', train_evaluation),
                               ('valid', valid_evaluation)]:
        # Dump metrics
        summary[prefix] = {}
        for metric_name, values in evaluation.history.items():
            if metric_name in skip_metric_names or not values:
                continue

            summary[prefix][metric_name] = to_float(values)
            if prefix == 'valid':
                best_value = to_float(values[best_epoch])
                summary[prefix][f'best_{metric_name}'] = best_value

        # Dump running times
        running_times = evaluation.history['running_time']
        running_times = np.array(to_float(running_times))
        num_examples = evaluation.history['num_examples']
        num_examples = np.array(to_float(num_examples))
        summary[prefix]['total_running_time'] = running_times.sum()
        summary[prefix]['epoch_running_time'] = running_times.mean()
        summary[prefix]['example_running_time'] = running_times.sum() / num_examples.sum()

    # Dump classification report and confusion matrix
    # Only best and last epoches for better readability
    summary['valid']['best_classification_report'] = \
        str(valid_evaluation.get_value('classification_report', best_epoch))
    summary['valid']['best_confusion_matrix'] = \
        str(valid_evaluation.get_value('confusion_matrix', best_epoch))
    summary['valid']['last_classification_report'] = \
        str(valid_evaluation.get_value('classification_report'))
    summary['valid']['last_confusion_matrix'] = \
        str(valid_evaluation.get_value('confusion_matrix'))

    summary_filepath = os.path.join(args.output_dir, f'{filename}.json')
    with open(summary_filepath, mode='w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=4)


def dump_result_cv_tensorboard_to_disk(output_dir: str,
                                       filename: str,
                                       training_results: List[Dict[str, Any]],
                                      ) -> None:
    row_dicts = []
    for result in training_results:
        row_dict = {}
        row_dict.update(result['tensorboard_hparam_dict'])
        row_dict.update(result['tensorboard_metric_dict'])
        row_dicts.append(row_dict)
    df = pd.DataFrame(row_dicts)
    df.to_csv(os.path.join(output_dir, f'{filename}.csv'), index=False)


# TENSORBOARD


def dump_metrics_to_tensorboard(writer: SummaryWriter,
                                eval_name: str,
                                score_map: Dict[str, Any],
                                epoch: int,
                                dump_figures: bool=False
                               ) -> None:

    for metric_name, score in score_map.items():
        if metric_name in ['classification_report', 'confusion_matrix']:
            continue
        tag_name = f'{metric_name}/{eval_name}'
        writer.add_scalar(tag_name, score, epoch)

    if dump_figures:
        report_df = score_map['classification_report']
        report_tag_name = f'classification_report/{eval_name}'
        fig = plt.figure(figsize=(12, 7))
        report_fig = fig.text(0.5, 0.5, report_df.to_string(), ha='center', va='center', size=20).get_figure()
        writer.add_figure(report_tag_name, report_fig, epoch)

        cm_df = score_map['confusion_matrix']
        cm_tag_name = f'confusion_matrix/{eval_name}'
        plt.figure(figsize=(12, 7))
        cm_fig = sn.heatmap(cm_df, annot=True, fmt='d').get_figure()
        writer.add_figure(cm_tag_name, cm_fig, epoch)


def dump_result_to_tensorboard(writer: SummaryWriter,
                               args: Namespace,
                               is_early_stopping: bool,
                               total_running_time: float,
                               best_epoch: int,
                               best_train_metrics: Dict[str, float],
                               best_valid_metrics: Dict[str, float],
                               last_train_metrics: Dict[str, float],
                               last_valid_metrics: Dict[str, float],
                              ) -> Tuple[Dict[str, float], Dict[str, float]]:
    hparam_dict = {
        'split': args.split,
        'dicretized_image': args.discretized_image,
        'model': args.model,
        'pretrained': args.pretrained if args.pretrained else 'scratch',
        'unfreeze_blocks': args.unfreeze_blocks,
        'optimizer': args.optimizer,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'batch_size': args.train_batch_size,
        'epochs': args.num_train_epochs,
    }

    metric_dict = {
        'hp/early_stopping': is_early_stopping,
        'hp/running_time': total_running_time,
        'hp/best_epoch': best_epoch,
    }
    def update_hparam_dict(metrics, prefix):
        metric_dict[f'hp/{prefix}_loss'] = metrics['loss']
        metric_dict[f'hp/{prefix}_acc'] = metrics['accuracy']
        metric_dict[f'hp/{prefix}_m_f1'] = metrics['macro_f1_score']
        metric_dict[f'hp/{prefix}_w_f1'] = metrics['weighted_f1_score']
    update_hparam_dict(best_train_metrics, 'best_train')
    update_hparam_dict(best_valid_metrics, 'best_valid')
    update_hparam_dict(last_train_metrics, 'last_train')
    update_hparam_dict(last_valid_metrics, 'last_valid')

    writer.add_hparams(
        hparam_dict = hparam_dict,
        metric_dict = metric_dict,
        run_name=args.output_dir
    )

    # remove the prefix 'hp/' from the metric names
    # this is only required to display the metrics in tensorboard
    metric_dict = {k.rsplit('/', 1)[-1]: v for k, v in metric_dict.items()}
    return hparam_dict, metric_dict
