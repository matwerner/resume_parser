import argparse
import json
import os
import random
import sys
from argparse import Namespace
from datetime import datetime
from logging import Logger
from typing import Any, Dict, Optional, List

import logging
import numpy as np
import torch

from torch import nn, optim
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import trange

from resume_parser.layout.dataset import (FeatureExtractorDataModule, load_data_module,
                                          compute_image_mean_std_from_train)
from resume_parser.layout.model import (get_model_image_size, initialize_model,
                                        train_last_blocks, save_model)
from resume_parser.layout.utils import (get_logger, matplotlib_imshow, combine_dfs,
                                        generate_model_experiment_name, generate_output_dirpath)
from resume_parser.layout.evaluation import ClassificationEvaluation
from resume_parser.layout.result_writer import (
    dump_result_split_to_disk,
    dump_result_cv_tensorboard_to_disk,
    dump_metrics_to_tensorboard,
    dump_result_to_tensorboard
)

# Reproducibility
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
cudnn.benchmark = True
random.seed(0)
np.random.seed(0)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

# GPU or CPU
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

# Creates once at the beginning of training
scaler = torch.cuda.amp.GradScaler()


def prepare_criterion_optimizer_and_scheduler(args: Namespace,
                                              model: nn.Module,
                                              for_classifier_layer: bool = False,
                                             ):
    # Loss function
    criterion = nn.CrossEntropyLoss()
    if for_classifier_layer:
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        scheduler = lr_scheduler.LinearLR(optimizer, 1, 1, total_iters=30)
    else:
        # Observe that all parameters are being optimized
        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                momentum=args.momentum, nesterov=True)
        else:
            print("Invalid optimizer name, exiting...")
            sys.exit()

        # Decay LR by a cosine factor
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_epochs)

    return criterion, optimizer, scheduler


def train_epoch(args: Namespace,
                model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: lr_scheduler._LRScheduler,
                evaluation: ClassificationEvaluation,
               ):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    training_start = datetime.now()
    for i, data in enumerate(train_loader):
        image, labels = data
        image = image.to(DEVICE)
        labels = labels.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()
        # Casts operations to mixed precision
        with torch.cuda.amp.autocast():
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)

        step_loss = loss.detach().item()
        total_loss += step_loss

        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        y_pred.extend(preds.detach().tolist())
        y_true.extend(labels.detach().tolist())

        # Scales the loss, and calls backward()
        # to create scaled gradients
        scaler.scale(loss).backward()

        # Unscales gradients and calls
        # or skips optimizer.step()
        scaler.step(optimizer)

        # Updates the scale for next iteration
        scaler.update()

    # update the scheduler parameters
    scheduler.step()

    y_pred = [args.labels[int(t)] for t in y_pred]
    y_true = [args.labels[int(t)] for t in y_true]

    # loss and accuracy for the complete epoch
    epoch_loss = total_loss / len(train_loader)
    running_time = (datetime.now() - training_start).total_seconds()
    metrics = evaluation.compute_metrics(y_true, y_pred, epoch_loss, running_time)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': metrics
    }


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
        for i, data in enumerate(data_loader):
            image, labels = data
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)

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


def train(args: Namespace,
          model: nn.Module,
          train_loader: DataLoader,
          valid_loader,
          train_evaluation: ClassificationEvaluation,
          valid_evaluation: ClassificationEvaluation,
          num_train_epochs: int,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: lr_scheduler._LRScheduler,
          writer: SummaryWriter = None,
          best_metric_name: Optional[str] = 'accuracy',
         ):
    # lists to keep track of losses and accuracies
    best_epoch = -1
    best_score = -1
    last_score = float('inf')
    best_train_metrics = None
    best_valid_metrics = None
    num_consecutive_worse = 0
    is_early_stopping = False

    # Tensorboard already sample images
    # Do not need to save images every epoch -> Time and Storage
    num_prints = min(5., num_train_epochs)
    print_every = int(num_train_epochs / num_prints)

    training_running_time = 0.0
    evaluating_running_time = 0.0

    def format_tqdm_metric(epoch_metrics, best_metrics, metric_name, fmt: str='{:.3f}') -> str:
        value = epoch_metrics[metric_name]
        best_value = best_metrics[metric_name]
        if value == best_value:
            return (fmt + '*').format(value)
        return (fmt + ' (' + fmt + '*)').format(value, best_value)

    # start the training
    running_start = datetime.now()

    epoch_tqdm = trange(int(num_train_epochs), desc="Epoch")
    for epoch in epoch_tqdm:
        is_print_figures = epoch % print_every == print_every - 1

        # Train
        training_start = datetime.now()
        train_output = train_epoch(args, model, train_loader, criterion,
                                   optimizer, scheduler, train_evaluation)
        training_delta = datetime.now() - training_start
        training_running_time += training_delta.total_seconds()

        # Evaluate
        evaluate_start = datetime.now()
        valid_output = evaluate(args, model, valid_loader,  criterion, valid_evaluation)
        evaluate_delta = datetime.now() - evaluate_start
        evaluating_running_time += evaluate_delta.total_seconds()

        train_metrics = train_output['metrics']
        valid_metrics = valid_output['metrics']
        epoch_score = valid_metrics[best_metric_name]
        if best_score < epoch_score:
            best_epoch = epoch
            best_score = epoch_score
            best_train_metrics = train_metrics
            best_valid_metrics = valid_metrics
            if args.save_model:
                save_model(model, args)

        stats = {
            'trn_loss': format_tqdm_metric(train_metrics, best_train_metrics, 'loss'),
            'val_loss': format_tqdm_metric(valid_metrics, best_valid_metrics, 'loss'),
            'trn_score': format_tqdm_metric(train_metrics, best_train_metrics, best_metric_name),
            'val_score': format_tqdm_metric(valid_metrics, best_valid_metrics, best_metric_name),
        }
        epoch_tqdm.set_postfix(stats)
        epoch_tqdm.refresh()

        # Results to tensorboard
        if writer:
            dump_metrics_to_tensorboard(writer, 'train', train_metrics, epoch, is_print_figures)
            dump_metrics_to_tensorboard(writer, 'valid', valid_metrics, epoch, is_print_figures)

        if last_score > epoch_score:
            num_consecutive_worse += 1
        else:
            num_consecutive_worse = 0
        last_score = epoch_score

        if num_consecutive_worse >= args.num_consecutive_worse_threshold:
            is_early_stopping = True
            break

    total_running_time = (datetime.now() - running_start).total_seconds()

    hparam_dict, metric_dict = None, None
    if writer:
        hparam_dict, metric_dict = dump_result_to_tensorboard(
            writer, args, is_early_stopping, total_running_time, best_epoch,
            best_train_metrics, best_valid_metrics, train_metrics, valid_metrics
        )

    return {
        'is_early_stopping': is_early_stopping,
        'training_running_time': training_running_time,
        'evaluating_running_time': evaluating_running_time,
        'total_running_time': total_running_time,

        'train_evaluation': train_evaluation,
        'valid_evaluation': valid_evaluation,

        'best_epoch': best_epoch,
        'best_train_metrics': best_train_metrics,
        'best_valid_metrics': best_valid_metrics,
        'last_train_metrics': train_metrics,
        'last_valid_metrics': valid_metrics,

        'tensorboard_hparam_dict': hparam_dict,
        'tensorboard_metric_dict': metric_dict,
    }


def train_split(args: Namespace, logger: Logger) -> Dict[str, Any]:
    is_pretrained = args.pretrained is not None and args.pretrained != ''

    args.image_size = get_model_image_size(args.model)

    if is_pretrained:
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
    else:
        image_mean, image_std = compute_image_mean_std_from_train(args)
    args.image_mean = image_mean
    args.image_std = image_std

    # Load dataset
    dataset = load_data_module(args, 'fit')
    args.labels = dataset.train.classes
    args.num_labels = len(args.labels)

    train_loader = dataset.train_dataloader()
    valid_loader = dataset.val_dataloader()

    # Load model
    model = initialize_model(args.model, args.num_labels, args.pretrained)

    writer = SummaryWriter(args.output_dir)

    # get some random training images
    data_iter = iter(train_loader)
    example_images, _ = data_iter.next()

    # Create model
    writer.add_graph(model, example_images)

    # create grid of images
    img_grid = make_grid(example_images)
    matplotlib_imshow(img_grid, one_channel=False)
    writer.add_image('resume_layout_images', img_grid)

    # To device
    model = model.to(DEVICE)

    # Fit classifier ~> Avoid 'noise' when updating weights
    if is_pretrained:
        logger.info("Training classifier layer")

        # Generate feature vectors
        model_fe = initialize_model(args.model, -1, args.pretrained)
        model_fe = model_fe.to(DEVICE)
        dataset_fe = FeatureExtractorDataModule(dataset, model_fe, DEVICE,
                                                args.train_batch_size,
                                                args.test_batch_size,
                                                use_uniform_sampler=True)
        dataset_fe.setup('fit')
        train_loader_fe = dataset_fe.train_dataloader()
        valid_loader_fe = dataset_fe.val_dataloader()

        # Train classifier
        criterion, optimizer, scheduler = \
            prepare_criterion_optimizer_and_scheduler(args, model, True)
        classifier_model = model.fc if args.model.startswith('resnet') else model.classifier
        train_evaluation = ClassificationEvaluation()
        test_evaluation = ClassificationEvaluation()
        classifier_result = train(args, classifier_model, train_loader_fe, valid_loader_fe,
                                  train_evaluation, test_evaluation, 30,
                                  criterion, optimizer, scheduler)
        dump_result_split_to_disk('classifier_result', args, classifier_result, 'accuracy')

    if is_pretrained:
        train_last_blocks(model, args.model, args.unfreeze_blocks, True)
    else:
        train_last_blocks(model, args.model, -1, False)

    logger.info("Training model")

    criterion, optimizer, scheduler = prepare_criterion_optimizer_and_scheduler(args, model)

    train_evaluation = ClassificationEvaluation()
    valid_evaluation = ClassificationEvaluation()
    model_result = train(args, model, train_loader, valid_loader,
                         train_evaluation, valid_evaluation, args.num_train_epochs,
                         criterion, optimizer, scheduler, writer)
    dump_result_split_to_disk('model_result', args, model_result, 'accuracy')

    best_train_metrics = model_result['best_train_metrics']
    best_valid_metrics = model_result['best_valid_metrics']

    # Results to logger
    logger.info(f'Best epoch: {model_result["best_epoch"] + 1} (of {args.num_train_epochs})')
    def to_log(metrics: Dict, set_name: str):
        report_df, cm_df = metrics["classification_report"], metrics['confusion_matrix']
        msg = '%s:\nReport:\n%s\nConfusion Matrix:\n%s\n'
        logger.info(msg, set_name, str(report_df), str(cm_df))
    to_log(best_train_metrics, 'TRAIN')
    to_log(best_valid_metrics, 'VALID')

    writer.close()

    return model_result


def main(args: Namespace):
    # The output directory where the model checkpoints
    # and predictions will be written.
    experiment_name = generate_model_experiment_name(args)
    output_dirpath = generate_output_dirpath(__file__, 'output/model', prefix=experiment_name)

    # Dump configuration to directory
    args_filepath = output_dirpath + '/args.json'
    with open(args_filepath, mode='w', encoding='utf-8') as fp:
        json.dump(vars(args), fp, indent=4)

    # Log
    log_filepath = output_dirpath + '/log.txt'
    logger = get_logger(log_filepath)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    args_txt = json.dumps(vars(args), indent=4)
    logger.info("console arguments:\n%s", args_txt)
    logger.info("output dir: %s", output_dirpath)
    logger.info("device: %s", DEVICE)

    results = []
    for i in range(args.num_splits):
        logger.info("##### SPLIT %d #####", i)
        args.split = i
        args.output_dir = os.path.join(output_dirpath, f"split_{i}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        args.dataset_file = args.dataset_split.format(i)

        result = train_split(args, logger)
        results.append(result)

    # Dump results
    dump_result_cv_tensorboard_to_disk(output_dirpath, 'tensorboard_results', results)

    train_metrics_list = [r['best_train_metrics'] for r in results]
    valid_metrics_list = [r['best_valid_metrics'] for r in results]
    def to_log(metrics_list: List[Dict[str, Any]], set_name: str):
        report_dfs = [metrics['classification_report'] for metrics in metrics_list]
        cm_dfs = [metrics['confusion_matrix'] for metrics in metrics_list]

        report_df = combine_dfs(report_dfs)
        cm_df = combine_dfs(cm_dfs)

        msg = '%s:\nReport:\n%s\nConfusion Matrix:\n%s\n'
        logger.info(msg, set_name, str(report_df), str(cm_df))
    to_log(train_metrics_list, 'TRAIN')
    to_log(valid_metrics_list, 'VALID')


def get_args():
    # construct the argument parser
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--dataset_split", default=None, type=str, required=True,
        help="JSON containing dataset path and split")
    parser.add_argument("--num_splits", default=5, type=int,
        help="Number of dataset split from [0, n-1].")
    parser.add_argument('--discretized_image', type=str, default='False',
        help='Whether to use original or discretized page image')
    parser.add_argument('--save_model', type=str, default='False',
        help="If true, the best model will be saved in the output directory")
    parser.add_argument("--verbose", action='store_true',
        help="If true, all of the warnings related to data processing will be printed.")

    # Model
    parser.add_argument('--model', type=str, default='efficientnet_b0',
        help='Model to train')
    parser.add_argument('--pretrained', type=str, default='imagenet',
        help='Whether to use ImageNet weights')
    parser.add_argument('--unfreeze_blocks', type=int, default=-1,
        help='Number of layer blocks to unfreeze')
    parser.add_argument('--optimizer', type=str, default='adam',
        help='Whether to use Adam or SGD optimizer')
    parser.add_argument('--lr', type=float, default=1e-3,
        help='Inital learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
        help='Wight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
        help='Momentum')

    # Training parameters
    parser.add_argument("--num_consecutive_worse_threshold", default=5, type=int,
        help='Number of consecutive epochs to stop training if validation score is not improving')
    parser.add_argument("--num_train_epochs", default=25, type=int,
        help="Total number of training epochs to perform.")
    parser.add_argument('--train_batch_size', type=int, default=32,
        help='Batch of instances to train in parallel')
    parser.add_argument('--test_batch_size', type=int, default=256,
        help='Batch of instances to train in parallel')
    args = parser.parse_args()

    # Parse some parameters
    if args.pretrained.lower().strip() in ['none', '']:
        args.pretrained = None
    args.discretized_image = args.discretized_image.lower() == 'true'
    args.save_model = args.save_model.lower() == 'true'
    return args


if __name__ == '__main__':
    main(get_args())
