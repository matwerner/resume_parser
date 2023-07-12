import argparse
import json
import random
import logging
import os
import numpy as np
import pandas as pd
import torch

from argparse import Namespace
from logging import Logger
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Any, Dict, List, Optional, Type, Tuple

from resume_parser.layout.dataset import ResumeLayoutDataModule, FeatureExtractorDataModule
from resume_parser.layout.evaluation import ClassificationEvaluation
from resume_parser.layout.model import initialize_model, get_model_image_size
from resume_parser.layout.utils import (get_logger, combine_dfs, generate_output_dirpath,
                                        generate_classifier_experiment_name)

# Reproducibility
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
random.seed(0)
np.random.seed(0)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

# GPU or CPU
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')


def load_data_module(args: Namespace, model_fe: Type[torch.nn.Module],
                     stage: Optional[str] = None):

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(args.image_mean, args.image_std),
    ])

    dataset = ResumeLayoutDataModule(
        config_filepath=args.dataset_file,
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        train_transform=transform,
        test_transform=transform,
        use_uniform_sampler=False,
        is_anonymized=args.discretized_image
    )
    dataset.setup(stage)

    model_fe = model_fe.to(DEVICE)
    dataset_fe = FeatureExtractorDataModule(
        data_module=dataset,
        model=model_fe,
        device=DEVICE,
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        use_uniform_sampler=False
    )
    dataset_fe.setup(stage)

    return dataset_fe


def get_feature_vectors(data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    x_batches, y_batches = [], []
    for outputs, labels in data_loader:
        x_batches.append(outputs)
        y_batches.append(labels)
    return torch.cat(x_batches, 0).numpy(), torch.cat(y_batches, 0).numpy()


def grid_search(args: Namespace,
                train_loader: DataLoader,
                valid_loader: DataLoader,
               ) -> float:
    X_train, y_train = get_feature_vectors(train_loader)
    X_test, y_test = get_feature_vectors(valid_loader)

    def my_range(a, b, steps):
        ran = np.linspace(a, b, steps)
        return ran[np.nonzero(ran)]
    params = {"C": [10**c for c in my_range(-6, 5, 45)]}

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    ps = PredefinedSplit(test_fold=[-1] * len(X_train) + [0] * len(X_test))

    regression = LogisticRegression(class_weight='balanced', solver='liblinear')
    classifier = GridSearchCV(regression, params, scoring='accuracy', cv=ps,
                              n_jobs=-3, refit=False)
    classifier.fit(X, y)

    # Save results
    df = pd.DataFrame(classifier.cv_results_)
    df['split'] = args.split
    df['discretized_image'] = args.discretized_image
    df['model_name'] = args.model
    df['pretrained_name'] = args.pretrained
    columns = ['split', 'discretized_image', 'model_name', 'pretrained_name',
               'param_C', 'split0_test_score']
    df[columns].to_excel(f'{args.output_dir}/gridsearch_result.xlsx', index=False)

    # Show best results
    return classifier.best_params_["C"]


def train_classifier(args: Namespace, train_loader: DataLoader, C: float) -> LogisticRegression:
    X_train, y_train = get_feature_vectors(train_loader)
    regression = LogisticRegression(class_weight='balanced', solver='liblinear', C=C)
    return regression.fit(X_train, y_train)


def evaluate_classifier(args: Namespace,
                        classifier: LogisticRegression,
                        data_loader: DataLoader,
                        evaluation: ClassificationEvaluation
                       ) -> None:
    X, y_true = get_feature_vectors(data_loader)
    start_time = datetime.now()
    y_pred = classifier.predict(X)
    running_time = (datetime.now() - start_time).total_seconds()

    y_true = [args.labels[i] for i in y_true]
    y_pred = [args.labels[i] for i in y_pred]
    metrics = evaluation.compute_metrics(y_true, y_pred, -1, running_time)
    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred,
    }


def run_split(args: Namespace, logger: Logger) -> Dict[str, Any]:
    args.image_size = get_model_image_size(args.model)
    args.image_mean = [0.485, 0.456, 0.406]
    args.image_std = [0.229, 0.224, 0.225]

    # Load model
    model = initialize_model(args.model, -1, args.pretrained)
    model = model.to(DEVICE)

    stage = 'test' if args.test_mode else 'fit'

    # Load dataset
    dataset = load_data_module(args, model, stage)
    args.labels = dataset.classes
    args.num_labels = len(args.labels)

    train_loader = dataset.train_dataloader()
    if args.test_mode:
        test_loader = dataset.test_dataloader()
    else:
        test_loader = dataset.val_dataloader()

    training_args_filepath = os.path.join(args.output_dir, 'training_args.json')
    if args.test_mode:
        training_args_filepath = json.load(open(training_args_filepath, 'r', encoding='utf-8'))
        args.C = training_args_filepath['C']
    else:
        # Grid search
        logger.info("Grid search")
        args.C = grid_search(args, train_loader, test_loader)
        with open(training_args_filepath, mode='w', encoding='utf-8') as fp:
            json.dump(vars(args), fp, indent=4)

    # Train model
    logger.info("Training model")
    model = train_classifier(args, train_loader, args.C)

    logger.info("Testing model")

    train_evaluation = ClassificationEvaluation()
    train_output = evaluate_classifier(args, model, train_loader, train_evaluation)
    train_metrics = train_output['metrics']

    test_evaluation = ClassificationEvaluation()
    test_output = evaluate_classifier(args, model, test_loader, test_evaluation)
    test_metrics = test_output['metrics']

    # Results to logger
    def to_log(metrics: Dict, set_name: str):
        msg = '{}:\nReport:\n{}\nConfusion Matrix:\n{}\n'
        report_df, cm_df = metrics["classification_report"], metrics['confusion_matrix']
        logger.info(msg.format(set_name, str(report_df), str(cm_df)))
    to_log(train_metrics, 'TRAIN')
    to_log(test_metrics, 'TEST' if args.test_mode else 'VALID')

    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,

        'train_evaluation': train_evaluation,
        'test_evaluation': test_evaluation,
    }


def main(args: Namespace):
    # experiment_name = generate_experiment_name(args)
    experiment_name = generate_classifier_experiment_name(args, args.test_mode)
    output_dirpath = generate_output_dirpath(__file__, 'output/classifier', prefix=experiment_name)

    # Dump configuration to directory
    args_filepath = output_dirpath + '/args.json'
    with open(args_filepath, mode='w', encoding='utf-8') as fp:
        json.dump(vars(args), fp, indent=4)

    # Log
    log_filepath = output_dirpath + '/log.txt'
    logger = get_logger(log_filepath)
    logger.setLevel(logging.INFO)

    args_txt = json.dumps(vars(args), indent=4)
    logger.info("console arguments:\n%s", args_txt)
    logger.info("output dir: %s", output_dirpath)
    logger.info("device: %s", DEVICE)

    results = []
    for i in range(args.num_splits):
        logger.info("##### SPLIT %d #####", i)
        args.split = i
        args.dataset_file = args.dataset_split.format(i)

        if args.test_mode:
            args.output_dir = os.path.join(args.experiment_dir, f"split_{i}")
        else:
            args.output_dir = os.path.join(output_dirpath, f"split_{i}")
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        result = run_split(args, logger)
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
    to_log(test_metrics_list, 'TEST' if args.test_mode else 'VALID')


def get_args():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_split", default=None, type=str,
        help="JSON containing dataset path and split")
    parser.add_argument("--num_splits", default=5, type=int,
        help="Number of dataset split from [0, n-1].")
    parser.add_argument('--discretized_image', type=str, default='False',
        help='Whether to use original or discretized page image')
    parser.add_argument('--experiment_dir', type=str, default=None,
        help='Experiment directory. If provided, test mode is enabled '
        'and the model will be loaded from this directory.')

    parser.add_argument('--model', type=str, default='efficientnet_b0',
        help='Model to train')
    parser.add_argument('--pretrained', type=str, default='imagenet',
        help='Whether to use ImageNet weights')
    parser.add_argument('--batch_size', type=int, default=256,
        help='Batch of instances to train in parallel')
    args = parser.parse_args()

    # Parse some parameters
    if args.pretrained.lower().strip() in ['none', '']:
        args.pretrained = None
    args.discretized_image = args.discretized_image.lower() == 'true'
    args.test_mode = False

    # Enable test mode
    if args.experiment_dir is not None:
        experiment_args_filepath = os.path.join(args.experiment_dir, 'args.json')
        experiment_args = json.load(open(experiment_args_filepath, 'r', encoding='utf-8'))
        experiment_args = Namespace(**experiment_args)
        experiment_args.experiment_dir = args.experiment_dir
        experiment_args.test_mode = True
        return experiment_args
    return args


if __name__ == '__main__':
    main(get_args())
