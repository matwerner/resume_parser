# For logging
import argparse
import logging
from argparse import Namespace
from datetime import datetime
from typing import List, Dict, Any, Optional

# For data
from collections import Counter
import glob
import os
import random
import pickle
import json
import shutil
import numpy as np
import pandas as pd

# For preprocessing
from nltk.corpus import stopwords

# For model
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,  PredefinedSplit

import sklearn_crfsuite
from sklearn_crfsuite import metrics as sequence_metrics

from resume_parser.segmenter.utils import generate_output_dirpath
from resume_parser.segmenter.utils import get_logger
from resume_parser.segmenter.crf.custom_transformer import (
    MultiLineFeatureExtractor,
    DictFeatureUnion,
    SelectTfidfFeatureExtractor,
    TextPreprocessor
)
from resume_parser.segmenter.crf.dataset import ResumeSegmenterDatasetForCRF
from resume_parser.segmenter.evaluation import SequenceEvaluation
from resume_parser.segmenter.evaluation import SequenceEvaluationDebugger

NUM_SEED = 0
STOP_WORDS = stopwords.words('portuguese')

# Reproducibility
random.seed(NUM_SEED)
np.random.seed(NUM_SEED)

def save_model(args: Namespace, model: Pipeline):
    model_filepath = os.path.join(args.output_dir, 'model.pickle')
    with open(model_filepath, 'wb') as fp:
        pickle.dump(model, fp)


def load_model(args: Namespace) -> Pipeline:
    model_filepath = os.path.join(args.input_dir, 'model.pickle')
    with open(model_filepath, 'rb') as fp:
        model = pickle.load(fp)
    return model


def get_model(args: Namespace) -> Pipeline:
    # Feature extractor parameters
    section_names_map = json.load(open(args.section_names_file, encoding='utf-8'))

    # Feature extractor transformers

    # Segment related features
    use_vocabulary = not args.disable_vocabulary
    use_text       = not args.disable_text
    use_visual     = not args.disable_visual
    use_spatial    = not args.disable_spatial
    seg_extractor = MultiLineFeatureExtractor(
        section_names_map,
        STOP_WORDS,
        use_vocabulary=use_vocabulary,
        use_text=use_text,
        use_visual=use_visual,
        use_spatial=use_spatial
    )
    seg_features = Pipeline([("seg_vectorizer", seg_extractor)])

    # Classification related features
    cls_features = Pipeline([
        ("cls_preprocessor", TextPreprocessor(is_stemm=True,
            stop_words=STOP_WORDS)),
        ("cls_vectorizer", SelectTfidfFeatureExtractor(max_features=500,
            max_features_label=50, ngram_range=(1,2), debug=False)),
    ])

    # Feature union
    feature_extractor = DictFeatureUnion([('seg_features', seg_features),
                                          ('cls_features', cls_features)])

    # Segmenter model
    segmenter = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100,
                                     all_possible_transitions=True)

    return Pipeline([
        ("feature_extractor", feature_extractor),
        ("segmenter", segmenter)
    ])


def evaluate(args: Namespace,
             model: Pipeline,
             dataset: ResumeSegmenterDatasetForCRF
            ) -> Dict[str, Any]:
    # Predict
    start = datetime.now()
    X = [example.word_sequence for example in dataset]
    y = [[e.label for e in example.word_sequence] for example in dataset]
    y_pred = model.predict(X)
    running_time = (datetime.now() - start).total_seconds()

    # Evaluate
    evaluation = SequenceEvaluation()
    metrics = evaluation.compute_metrics(y, y_pred, None, running_time)

    return {
        'y_true': y,
        'y_pred': y_pred,
        'metrics': metrics
    }


def get_dataset(args: Namespace,
                dataset_dir: str,
                filenames: Optional[List[str]]=None
               ) -> ResumeSegmenterDatasetForCRF:
    if filenames is None:
        pattern = os.path.join(dataset_dir, '*.csv')
        filenames = [os.path.basename(filepath) for filepath in glob.glob(pattern)]
    return ResumeSegmenterDatasetForCRF(dataset_dir, filenames, args.section_label_only)


def test_split(args: Namespace, logger: logging.Logger) -> Dict[str, Any]:
    # Load model
    model = load_model(args)

    # Get dataset
    with open(args.dataset_file, encoding='utf-8') as fp:
        config = json.load(fp)
    root_dirpath = config['root_dirpath']
    train_dataset = get_dataset(args, root_dirpath, config['train'])
    if args.eval_dir:
        test_dataset = get_dataset(args, args.eval_dir)
    else:
        test_dataset = get_dataset(args, root_dirpath, config['test'])

    evaluating_start = datetime.now()
    train_result = evaluate(args, model, train_dataset)
    test_result = evaluate(args, model, test_dataset)
    evaluating_running_time = (datetime.now() - evaluating_start).total_seconds()

    def results_to_log(metrics: Dict[str, Any], prefix_text: str, log_level: int):
        pk = metrics['pk']
        wd = metrics['wd']
        seg_equal = metrics['seg_equal']
        cls_equal = metrics['cls_equal']
        report = metrics['classification_report']
        logger.log(log_level, f'{prefix_text}:\n{str(report)}')
        logger.log(log_level, f'{prefix_text}:\nPk: {pk}\nWD: {wd}\nSeg equal: {seg_equal}\nCls equal: {cls_equal}\n')
    results_to_log(train_result['metrics'], "TRAIN REPORT", logging.INFO)
    results_to_log(test_result['metrics'], "TEST REPORT", logging.INFO)

    # Dump evaluation
    logger.info('Dumping evalution...')
    filenames = test_dataset.filenames

    lines_list = [[line for line in example.word_sequence] for example in test_dataset]
    evaluation_filepath = os.path.join(args.output_dir, 'test_evaluation.xlsx')
    SequenceEvaluationDebugger().dump_results(evaluation_filepath, filenames, lines_list,
                                              test_result['y_true'], test_result['y_pred'])

    return {
        'training_running_time': 0,
        'evaluating_running_time': evaluating_running_time,
        'train_metrics': train_result['metrics'],
        'test_metrics': test_result['metrics'],
    }


def train_split(args: Namespace, logger: logging.Logger) -> Dict[str, Any]:
    # Get model
    model = get_model(args)

    # Get dataset
    with open(args.dataset_file, encoding='utf-8') as fp:
        config = json.load(fp)
    root_dirpath = config['root_dirpath']
    train_dataset = get_dataset(args, root_dirpath, config['train'])
    valid_dataset = get_dataset(args, root_dirpath, config['valid'])

    # Convert dataset
    logger.info('Start extracting features for training data')
    X_train = [example.word_sequence for example in train_dataset]
    y_train = [[e.label for e in example.word_sequence]
               for example in train_dataset]

    X_valid = [example.word_sequence for example in valid_dataset]
    y_valid = [[e.label for e in example.word_sequence]
               for example in valid_dataset]

    # Precompute these steps to reduce memory and computational usage
    model.named_steps.feature_extractor.fit(X_train, y_train)
    X_train = model.named_steps.feature_extractor.transform(X_train)
    X_valid = model.named_steps.feature_extractor.transform(X_valid)

    X = X_train + X_valid
    y = y_train + y_valid

    logger.info('Started training CRF model')

    # Grid search
    # params_space = {
    #     'c1': scipy.stats.expon(scale=0.5),
    #     'c2': scipy.stats.expon(scale=0.05),
    # }
    params = 10**np.linspace(-3, 2, args.num_iter)
    params_space = {'c1': params, 'c2': params}

    # Scorer
    labels = np.unique([label for labels in y_train for label in labels])
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    f1_scorer = make_scorer(sequence_metrics.flat_f1_score, average='micro', labels=sorted_labels)

    # Train instances = -1
    # Validation instances = 0
    split_index = [-1] * len(X_train) + [0] * len(X_valid)

    # search
    training_start = datetime.now()
    ps = PredefinedSplit(test_fold=split_index)
    # rs = RandomizedSearchCV(
    rs = GridSearchCV(
        model.named_steps.segmenter,
        params_space,
        cv           = ps,
        n_jobs       = args.num_jobs,
        # n_iter       = args.num_iter,
        scoring      = f1_scorer,
        refit        = False,
        # random_state = NUM_SEED,
        verbose      = False,
    )
    rs.fit(X, y)
    training_running_time = (datetime.now() - training_start).total_seconds()

    logger.info('best params:   %s', rs.best_params_)
    logger.info('best CV score: %f', rs.best_score_)

    logger.info('Start refit model to best parameters')
    model.named_steps.segmenter.set_params(**rs.best_params_)
    model.named_steps.segmenter.fit(X_train, y_train)

    logger.debug('num features:  %d', model.named_steps.segmenter.num_attributes_)
    logger.debug('model size:    %.2fM', model.named_steps.segmenter.size_ / 10**6)

    def print_state_features(state_features):
        return '\n'.join(["%0.6f %-8s %s" % (weight, label, attr)
                          for (attr, label), weight in state_features])

    state_features_counter = Counter(model.named_steps.segmenter.state_features_).most_common()
    positive_features = print_state_features(state_features_counter[:30])
    logger.debug('Top positive features:\n' + positive_features)

    negative_features = print_state_features(state_features_counter[-30:])
    logger.debug('Top negative features:\n' + negative_features)

    # Test - Predict from zero
    logger.info('Started testing model')
    evaluating_start = datetime.now()
    train_result = evaluate(args, model, train_dataset)
    valid_result = evaluate(args, model, valid_dataset)
    evaluating_running_time = (datetime.now() - evaluating_start).total_seconds()

    def results_to_log(metrics: Dict[str, Any], prefix_text: str, log_level: int):
        pk = metrics['pk']
        wd = metrics['wd']
        seg_equal = metrics['seg_equal']
        cls_equal = metrics['cls_equal']
        report = metrics['classification_report']
        logger.log(log_level, f'{prefix_text}:\n{str(report)}')
        logger.log(log_level, f'{prefix_text}:\nPk: {pk}\nWD: {wd}\nSeg equal: {seg_equal}\nCls equal: {cls_equal}\n')
    results_to_log(train_result['metrics'], "TRAIN REPORT", logging.DEBUG)
    results_to_log(valid_result['metrics'], "VALID REPORT", logging.INFO)

    # Dump evaluation
    logger.info('Dumping evalution...')
    filenames = valid_dataset.filenames

    lines_list = [[line for line in example.word_sequence] for example in valid_dataset]
    evaluation_filepath = os.path.join(args.output_dir, 'valid_evaluation.xlsx')
    SequenceEvaluationDebugger().dump_results(evaluation_filepath, filenames, lines_list,
                                              valid_result['y_true'], valid_result['y_pred'])

    if args.save_model:
        logger.info('Saving model')
        save_model(args, model)

    return {
        'training_running_time': training_running_time,
        'evaluating_running_time': evaluating_running_time,
        'train_metrics': train_result['metrics'],
        'test_metrics': valid_result['metrics'],
    }


def compile_result(args: Namespace, training_result: Dict[str, Any]) -> Dict[str, Any]:
    training_running_time = training_result['training_running_time']
    evaluating_running_time = training_result['evaluating_running_time']
    train_metrics = training_result['train_metrics']
    test_metrics = training_result['test_metrics']

    results = vars(args)
    results['timestamp'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    results['training_running_time'] = training_running_time
    results['evaluating_running_time'] = evaluating_running_time

    def to_float(value):
        if isinstance(value, list):
            return [float(val) for val in value]
        return float(value)

    skip_metric_names = ['classification_report']
    for prefix, metrics in [('train', train_metrics),
                            ('test' if args.test_mode else 'valid', test_metrics)]:
        results[prefix] = {}
        for metric_name, values in metrics.items():
            if metric_name in skip_metric_names or not values:
                continue
            results[prefix][metric_name] = to_float(values)
        running_time = metrics['running_time']
        num_examples = metrics['num_examples']
        results[prefix]['total_running_time'] = running_time
        results[prefix]['example_running_time'] = running_time / num_examples

    results['classification_report'] = str(test_metrics['classification_report'])

    return results


def generate_cv_report_to_log(logger: logging.Logger, results: List[Dict]) -> None:
    reports = [r['classification_report'] for r in results]

    df_concat = pd.concat(reports)
    group_df = df_concat.groupby(level=0)

    df_means = group_df.mean().applymap('{:.4f}'.format)
    df_stds = group_df.std().applymap('{:.4f}'.format)
    df = df_means.combine(df_stds, lambda x1, x2: x1 + ' ± ' + x2)
    logger.info(f'CROSS-VALIDATION REPORT:\n{str(df)}')

    def get_metric_output(metric_name):
        values = np.array([r[metric_name] for r in results])
        return f'{values.mean():.4f} ± {values.std():.4f}'

    logger.info(
        'CROSS-VALIDATION SEGMENTATION:\n'
        f'Pk: {get_metric_output("pk")}\n'
        f'WD: {get_metric_output("wd")}\n'
        f'Seg equal: {get_metric_output("seg_equal")}\n'
        f'Cls equal: {get_metric_output("cls_equal")}\n'
    )


def main(args: Namespace):
    # The output directory where the model checkpoints
    # and predictions will be written.
    output_dirpath = generate_output_dirpath(__file__)

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

    # Get resources
    sections_filepath = args.section_names_file

    # Copy to output
    shutil.copy2(sections_filepath, output_dirpath)

    results = []
    for i in range(args.num_splits):
        logger.info("##### SPLIT %d #####", i)
        args.split = i
        args.output_dir = os.path.join(output_dirpath, f"split_{i}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        args.dataset_file = args.dataset_split.format(i)

        if args.test_mode:
            args.input_dir = os.path.join(args.experiment_dir, f"split_{i}")
            result = test_split(args, logger)
        else:
            result = train_split(args, logger)

        compiled_result = compile_result(args, result)
        result_filepath = args.output_dir + '/result.json'
        with open(result_filepath, mode='w', encoding='utf-8') as fp:
            json.dump(compiled_result, fp, indent=4)
        results.append(result)

    test_results = [result['test_metrics'] for result in results]
    generate_cv_report_to_log(logger, test_results)


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--labels_file', default=None, type=str,
                        help="File with all NER classes to be considered, one "
                        "per line.")
    parser.add_argument('--section_names_file', default=None, type=str,
                        help="File with all section names in a JSON format")
    parser.add_argument("--dataset_split", default=None, type=str,
                        help="JSON containing dataset path and split")
    parser.add_argument("--num_splits", default=5, type=int,
                        help="Number of dataset split from [0, n-1].")
    parser.add_argument("--section_label_only", action='store_true',
                        help="If true, the model will be trained to detect "
                        "only the resume sections")
    parser.add_argument("--save_model", action='store_true',
                        help="If true, the best model will be saved in the "
                        "output directory")
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the debug related info will be "
                        " printed.")

    # Model related
    parser.add_argument('--disable_vocabulary', action='store_true',
                        help="Whether to disable section name features.")
    parser.add_argument('--disable_text', action='store_true',
                        help="Whether to disable textual features.")
    parser.add_argument('--disable_visual', action='store_true',
                        help="Whether to disable visual (Bold, ...) features.")
    parser.add_argument('--disable_spatial', action='store_true',
                        help="Whether to disable spatial (axis) features.")

    # Training related
    parser.add_argument("--num_jobs", default=-5, type=int,
                        help="Number of cores to use while running the CRF.")
    parser.add_argument("--num_iter", default=21, type=int,
                        help="Number of logarithmically spaced values in the "
                        "interval [10^-3, 10^2] to be sampled for c1 and c2.")

    # Testing related
    parser.add_argument('--experiment_dir', type=str, default=None,
                        help='Experiment directory. If provided, test mode is enabled '
                        'and the model will be loaded from this directory.')
    parser.add_argument('--eval_dir', type=str, default=None,
                        help='Data to be evaluated. If not provided, evaluation '
                        'will be performed on the test set.')
    args = parser.parse_args()

    # Enable test mode
    args.test_mode = False
    if args.experiment_dir is not None:
        experiment_args_filepath = os.path.join(args.experiment_dir, 'args.json')
        experiment_args = json.load(open(experiment_args_filepath, 'r', encoding='utf-8'))
        experiment_args = Namespace(**experiment_args)
        experiment_args.experiment_dir = args.experiment_dir
        experiment_args.eval_dir = args.eval_dir
        experiment_args.test_mode = True
        return experiment_args
    return args


if __name__ == '__main__':
    main(get_args())
