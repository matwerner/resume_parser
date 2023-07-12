# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file defines the `main` function that handles BERT, BERT-CRF,
BERT-LSTM and BERT-LSTM-CRF training and evaluation on NER task.

The `main` function should be imported and called by another script that passes
functions to 1) load and preprocess input data and 2) define metrics evaluate
the model during training or testing phases.

For further information, see `main` function docstring and the ArgumentParser
arguments.

The code was inspired by Huggingface Tranformers' script for training and
evaluating BERT on SQuAD dataset.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import glob
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from logging import Logger
from typing import Any, Dict, List, Optional, Union, Type

import logging
import numpy as np
import pandas as pd
import torch

from transformers import get_linear_schedule_with_warmup
from torch import nn
from torch.backends import cudnn
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange

from resume_parser.segmenter.bert.dataset import (ResumeSegmenterDataModuleForBert, 
                                                  ResumeSegmenterDataModuleForBertPerExample,
                                                  FeatureExtractorDataModule)
from resume_parser.segmenter.bert.model import load_model, save_model
from resume_parser.segmenter.bert.dataset_utils import NERTagEncoder
from resume_parser.segmenter.bert.dataset_utils import InputComposer
from resume_parser.segmenter.bert.dataset_utils import OutputComposer
from resume_parser.segmenter.evaluation import SequenceEvaluationDebugger, SequenceEvaluation
from resume_parser.segmenter.utils import generate_output_dirpath, get_logger
from resume_parser.segmenter.dataset import TextElement


# Format
pd.options.display.float_format = '{:.4f}'.format

# Reproducibility
torch.manual_seed(0)
if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)
cudnn.benchmark = True
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

# GPU or CPU
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

class RunningAccumulator:
    """Loss value running accumulator."""

    def __init__(self):
        self.total = 0
        self.num_values = 0

    def accumulate(self, value: Union[torch.Tensor, float]):
        if torch.is_tensor(value):
            with torch.no_grad():
                self.total += value.item()
        else:
            self.total += value
        self.num_values += 1

    def mean(self) -> float:
        return self.total / self.num_values


def prepare_optimizer_and_scheduler(args: Namespace,
                                    model: nn.Module,
                                    num_batches: int,
                                    logger: Logger
                                    ):
    """Configures BERT's AdamW optimizer and WarmupLinearSchedule learning rate
    scheduler. Divides parameters into two learning rate groups, with higher
    learning rate for non-BERT parameters (classifier model)."""
    t_total = (num_batches // args.gradient_accumulation_steps *
               args.num_train_epochs)

    logger.info("  Total optimization steps = %d", t_total)

    # Prepare optimizer
    param_optimizer = list(
        filter(lambda p: p[1].requires_grad, model.named_parameters()))

    no_decay = ['bias', 'LayerNorm.weight']
    higher_lr = ['classifier', 'crf', 'lstm']

    def is_classifier_param(param_name: str) -> bool:
        return any(hl in param_name for hl in higher_lr)

    def ignore_in_weight_decay(param_name: str) -> bool:
        return any(nd in param_name for nd in no_decay)

    optimizer_grouped_parameters = [
        # Remaining network
        {'params': [p for name, p in param_optimizer
                    if not ignore_in_weight_decay(name)
                    and not is_classifier_param(name)],
         'weight_decay': 0.01},
        # Classification layer
        {'params': [p for name, p in param_optimizer
                    if not ignore_in_weight_decay(name)
                    and is_classifier_param(name)],
         'weight_decay': 0.01,
         'lr': args.classifier_lr},
        # Bias and Normalization Layer
        {'params': [p for name, p in param_optimizer
                    if ignore_in_weight_decay(name)
                    and not is_classifier_param(name)],
         'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_warmup_steps = t_total * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)

    return optimizer, scheduler


def convert_spans_to_examples(data_loader: DataLoader,
                              example_map: defaultdict(list),
                              output_composer: OutputComposer,
                             ) -> Dict[str, List]:
    filenames = []
    # Word-level data
    words_list = []
    words_y_true = []
    words_y_pred = []
    # Line-level data
    lines_list = []
    lines_y_true = []
    lines_y_pred = []
    for filename, span_outputs in example_map.items():
        filenames.append(filename)

        span_outputs = sorted(span_outputs, key=lambda x: x[0])
        outputs = [output[1] for output in span_outputs]

        example = data_loader.dataset.get_example(filename)
        spans = data_loader.dataset.get_example_spans(filename)
        word_sequence = example.word_sequence

        # Get example output
        example_y_true = [word.label for word in example.word_sequence]
        example_y_pred = output_composer.get_example_output(spans, outputs)

        assert len(example_y_true) == len(example_y_pred),\
            f"Output mismatch: {filename}"

        # Word-level
        words = [word.text for word in word_sequence]

        words_list.append(words)
        words_y_true.append(example_y_true)
        words_y_pred.append(example_y_pred)

        # Line-level
        def build_line(words: List[TextElement]) -> TextElement:
            text = ''.join(w.text + (w.tail if i+1 != len(words) else '')
                    for i, w in enumerate(words))
            tail = words[-1].tail
            start = min(w.start for w in words)
            end = max(w.end for w in words)
            return TextElement(text, start, end, -1, False, False, -1, -1, -1, -1, -1, -1, tail=tail)

        line_mask = [i == 0 or word.line_idx != word_sequence[i-1].line_idx
                     for i, word in enumerate(word_sequence)]

        lines = []
        line = []
        for i, (is_begin, word) in enumerate(zip(line_mask, word_sequence)):
            if is_begin:
                if i > 0:
                    lines.append(build_line(line))
                line = []
            line.append(word)
        if len(line) > 0:
            lines.append(build_line(line))

        lines_list.append(lines)
        lines_y_true.append(np.array(example_y_true)[line_mask].tolist())
        lines_y_pred.append(np.array(example_y_pred)[line_mask].tolist())

    return {
        'filenames': filenames,
        'words': words_list,
        'words_y_true': words_y_true,
        'words_y_pred': words_y_pred,
        'lines': lines_list,
        'lines_y_true': lines_y_true,
        'lines_y_pred': lines_y_pred,
    }


def train_epoch(args: Namespace,
                model: Type[nn.Module],
                train_dl: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LambdaLR,
                loss_accumulator: RunningAccumulator,
                scaler: torch.cuda.amp.GradScaler,
                output_composer: OutputComposer,
                evaluation: SequenceEvaluation,
               ):
    model.train()
    example_map = defaultdict(list)

    training_start = datetime.now()
    for step, batch in enumerate(train_dl):
        # Unpack batch
        if model.is_bert_features_precomputed:
            input_ids = batch['bert_features'].to(DEVICE)
        else:
            input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)
        label_ids = batch['label_ids'].to(DEVICE)
        context_mask = batch['context_mask'].to(DEVICE)
        prediction_mask = batch['prediction_mask'].to(DEVICE)
        extra_classifier_features = batch['extra_classifier_features'].to(DEVICE)

        # Casts operations to mixed precision
        if args.fp16:
            with torch.cuda.amp.autocast():
                outs = model(input_ids, extra_classifier_features,
                             token_type_ids, attention_mask,
                             label_ids, context_mask, prediction_mask)
        else:
            outs = model(input_ids, extra_classifier_features,
                         token_type_ids, attention_mask,
                         label_ids, context_mask, prediction_mask)

        filenames = batch['filename']
        span_indices = batch['span_index']
        sequence_lengths = batch['sequence_length']
        for filename, span_index, length, y_pred in zip(filenames,
                                                        span_indices,
                                                        sequence_lengths,
                                                        outs['y_pred']):
            # Ignore padding
            example_map[filename].append((span_index, y_pred[:length]))

        # No instance in the batch is computing anything. Motive:
        # A. Spliting instance into spans (Memory constrain)
        # B. Only predicts first token in each line
        # Thus, it's possible that a batch only contains spans composed
        # of incomplete last lines
        if not prediction_mask.any():
            print('not prediction_mask.any()')
            continue

        loss = outs['loss']
        # Skip this batch, when loss is float or undefined
        # TODO: how to handle this event? Exit?
        if isinstance(loss, float) or torch.isnan(loss).any():
            continue

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss_accumulator.accumulate(loss.item())
        # running_mean_loss = loss_accumulator.mean()
        # train_tqdm.set_postfix({'loss': running_mean_loss})

        if args.fp16:
            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()

            # Unscale the gradients before clipping
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Perform gradient clipping
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    clip_grad_norm_(p, 1)

            if args.fp16:
                # Unscales gradients and calls or skips optimizer.step()
                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

    examples = convert_spans_to_examples(train_dl, example_map, output_composer)
    filenames = examples['filenames']
    lines = examples['lines']
    lines_y_true = examples['lines_y_true']
    lines_y_pred = examples['lines_y_pred']

    loss = loss_accumulator.mean()
    running_time = (datetime.now() - training_start).total_seconds()
    metrics = evaluation.compute_metrics(lines_y_true, lines_y_pred, loss, running_time)

    return {
        'filenames': filenames,
        'tokens': lines,
        'y_true': lines_y_true,
        'y_pred': lines_y_pred,
        'metrics': metrics
    }


def evaluate(args: Namespace,
             model: nn.Module,
             data_loader: DataLoader,
             output_composer: OutputComposer,
             evaluation: SequenceEvaluation
             ) -> Dict[str, Any]:
    """Runs a model forward pass on the entire dataloader to compute predictions
    for all examples. Final predictions are gathered in `output_composer`,
    combining the max-context tokens of each forward pass. Returns the
    metrics dict computed by `sequence_metrics.calculate_metrics()`."""
    # Evaluate
    model.eval()

    evaluation_start = datetime.now()
    losses = []
    example_map = defaultdict(list)
    for batch in data_loader:
        # Unpack batch
        if model.is_bert_features_precomputed:
            input_ids = batch['bert_features'].to(DEVICE)
        else:
            input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)
        label_ids = batch['label_ids'].to(DEVICE)
        context_mask = batch['context_mask'].to(DEVICE)
        prediction_mask = batch['prediction_mask'].to(DEVICE)
        extra_classifier_features = batch['extra_classifier_features'].to(DEVICE)

        with torch.no_grad():
            if args.fp16:
                with torch.cuda.amp.autocast():
                    outs = model(input_ids, extra_classifier_features,
                                 token_type_ids, attention_mask,
                                 label_ids, context_mask, prediction_mask)
            else:
                outs = model(input_ids, extra_classifier_features,
                             token_type_ids, attention_mask,
                             label_ids, context_mask, prediction_mask)

            filenames = batch['filename']
            span_indices = batch['span_index']
            sequence_lengths = batch['sequence_length']
            for filename, span_index, length, y_pred in zip(filenames,
                                                            span_indices,
                                                            sequence_lengths,
                                                            outs['y_pred']):
                # Ignore padding
                example_map[filename].append((span_index, y_pred[:length]))

        loss = outs.get('loss')
        if loss is not None:
            loss = loss.item()
            losses.append(loss)

    examples = convert_spans_to_examples(data_loader, example_map, output_composer)
    filenames = examples['filenames']
    lines = examples['lines']
    lines_y_true = examples['lines_y_true']
    lines_y_pred = examples['lines_y_pred']

    loss = float(np.mean(losses)) if losses else None
    running_time = (datetime.now() - evaluation_start).total_seconds()
    metrics = evaluation.compute_metrics(lines_y_true, lines_y_pred, loss, running_time)

    return {
        'filenames': filenames,
        'tokens': lines,
        'y_true': lines_y_true,
        'y_pred': lines_y_pred,
        'metrics': metrics
    }


def train(args: Namespace,
          model: torch.nn.Module,
          train_dl: DataLoader,
          valid_dl: DataLoader,
          output_composer: OutputComposer,
          train_evaluation: SequenceEvaluation,
          valid_evaluation: SequenceEvaluation,
          evaluation_debugger: SequenceEvaluationDebugger,
          logger: Logger,
          best_metric_name: Optional[str] = 'micro_f1_score',
         ) -> None:
    """Train routine."""

    optimizer, scheduler = prepare_optimizer_and_scheduler(
        args, model, len(train_dl), logger)

    if args.fp16:
        # Creates once at the beginning of training
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_epoch = -1
    best_score = -1
    last_score = float('inf')
    best_train_metrics = None
    best_valid_metrics = None
    num_consecutive_worse = 0
    is_early_stopping = False

    training_running_time = 0
    evaluating_running_time = 0

    def format_tqdm_metric(epoch_metrics, best_metrics, metric_name, fmt: str='{:.3f}') -> str:
        value = epoch_metrics[metric_name]
        best_value = best_metrics[metric_name]
        if value == best_value:
            return (fmt + '*').format(value)
        return (fmt + ' (' + fmt + '*)').format(value, best_value)

    # start the training
    running_start = datetime.now()

    try:
        epoch_tqdm = trange(int(args.num_train_epochs), desc="Epoch")
        loss_accum = RunningAccumulator()
        for epoch in epoch_tqdm:
            training_start = datetime.now()
            train_output = train_epoch(args, model, train_dl, optimizer, scheduler,
                                       loss_accum, scaler, output_composer, train_evaluation)
            training_running_time += (datetime.now() - training_start).total_seconds()

            # Evaluate
            evaluate_start = datetime.now()
            valid_output = evaluate(args, model, valid_dl, output_composer, valid_evaluation)
            evaluating_running_time += (datetime.now() - evaluate_start).total_seconds()

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

                filepath = os.path.join(args.output_dir, 'best_valid_results.xlsx')
                evaluation_debugger.dump_results(
                    output_filepath = filepath,
                    filenames       = valid_output['filenames'],
                    tokens_list     = valid_output['tokens'],
                    y_true_list     = valid_output['y_true'],
                    y_pred_list     = valid_output['y_pred']
                )

            stats = {
                'trn_loss': format_tqdm_metric(train_metrics, best_train_metrics, 'loss'),
                'val_loss': format_tqdm_metric(valid_metrics, best_valid_metrics, 'loss'),
                'trn_score': format_tqdm_metric(train_metrics, best_train_metrics, best_metric_name),
                'val_score': format_tqdm_metric(valid_metrics, best_valid_metrics, best_metric_name),
            }
            epoch_tqdm.set_postfix(stats)
            epoch_tqdm.refresh()

            if last_score > epoch_score:
                num_consecutive_worse += 1
            else:
                num_consecutive_worse = 0
            last_score = epoch_score

            if num_consecutive_worse >= args.num_consecutive_worse_threshold:
                is_early_stopping = True
                break

    except KeyboardInterrupt:
        action = ''
        while action.lower() not in ('y', 'n'):
            action = input('\nInterrupted. Continue execution to save model '
                           'weights? [Y/n]')
            if action == 'n':
                sys.exit()

    # results_to_log(train_evaluation, -1, "[LAST] TRAIN", logging.DEBUG)
    # results_to_log(valid_evaluation, -1, "[LAST] VALID", logging.DEBUG)
    filepath = os.path.join(args.output_dir, 'last_valid_results.xlsx')
    evaluation_debugger.dump_results(
        output_filepath = filepath,
        filenames       = valid_output['filenames'],
        tokens_list     = valid_output['tokens'],
        y_true_list     = valid_output['y_true'],
        y_pred_list     = valid_output['y_pred']
    )

    total_running_time = (datetime.now() - running_start).total_seconds()

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
    }


def dump_compiled_result_split(filename: str,
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

    skip_metric_names = ['classification_report']
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
    summary['valid']['last_classification_report'] = \
        str(valid_evaluation.get_value('classification_report'))

    summary_filepath = os.path.join(args.output_dir, f'{filename}.json')
    with open(summary_filepath, mode='w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=4)


def test_split(args: Namespace,
               input_composer: InputComposer,
               output_composer: OutputComposer,
               logger: Logger
              ) -> Dict[str, Any]:
    data_module = ResumeSegmenterDataModuleForBertPerExample(
        args.dataset_file,
        input_composer     = input_composer,
        section_label_only = args.section_label_only,
        is_cache_data      = False,
        train_batch_size   = 1,
        test_batch_size    = 1,
    )
    stage = "predict" if args.eval_dir else "test"
    data_module.setup(stage)

    # Load a pretrained model
    if args.extra_classifier_features:
        args.num_extra_classifier_features = input_composer.num_extra_classifier_features()
    model = load_model(args, args.input_dir)
    model.to(DEVICE)

    # Training loop
    train_dl = data_module.train_dataloader()
    if args.eval_dir:
        pattern = os.path.join(args.eval_dir, '*.csv')
        filenames = [os.path.basename(filepath) for filepath in glob.glob(pattern)]
        test_dl = data_module.build_external_dataloader(args.eval_dir, filenames)
    else:
        test_dl = data_module.test_dataloader()

    evaluating_start = datetime.now()
    train_evaluation = SequenceEvaluation()
    train_output = evaluate(args, model, train_dl, output_composer, train_evaluation)
    train_metrics = train_output['metrics']

    test_evaluation = SequenceEvaluation()
    test_output = evaluate(args, model, test_dl, output_composer, test_evaluation)
    test_metrics = test_output['metrics']
    evaluating_running_time = (datetime.now() - evaluating_start).total_seconds()

    def results_to_log(metrics: Dict[str, Any], prefix_text: str, log_level: int):
        pk = metrics['pk']
        wd = metrics['wd']
        seg_equal = metrics['seg_equal']
        cls_equal = metrics['cls_equal']
        report = metrics['classification_report']
        logger.log(log_level, f'{prefix_text}:\n{str(report)}')
        logger.log(log_level, f'{prefix_text}:\nPk: {pk}\nWD: {wd}\nSeg equal: {seg_equal}\nCls equal: {cls_equal}\n')
    results_to_log(train_metrics, "TRAIN REPORT", logging.INFO)
    results_to_log(test_metrics, "TEST REPORT", logging.INFO)

    # Dump evaluation
    logger.info('Dumping evalution...')
    evaluation_filepath = os.path.join(args.output_dir, 'test_evaluation.xlsx')
    SequenceEvaluationDebugger().dump_results(evaluation_filepath,
                                              test_output['filenames'], test_output['tokens'],
                                              test_output['y_true'], test_output['y_pred'])

    testing_result = {
        'is_early_stopping': None,
        'training_running_time': 0,
        'evaluating_running_time': evaluating_running_time,
        'total_running_time': evaluating_running_time,

        'train_evaluation': train_evaluation,
        'valid_evaluation': test_evaluation,

        'best_epoch': 0,
        'best_train_metrics': train_metrics,
        'best_valid_metrics': test_metrics,
        'last_train_metrics': train_metrics,
        'last_valid_metrics': test_metrics,
    }
    dump_compiled_result_split('model_result', args, testing_result)
    return testing_result


def train_split(args: Namespace,
                input_composer: InputComposer,
                output_composer: OutputComposer,
                logger: Logger
               ) -> Dict[str, Any]:

    # Load a pretrained model
    if args.extra_classifier_features:
        args.num_extra_classifier_features = input_composer.num_extra_classifier_features()
    model = load_model(args, args.bert_model)
    model.to(DEVICE)

    data_module = FeatureExtractorDataModule(
        args.dataset_file,
        input_composer     = input_composer,
        model              = model,
        device             = DEVICE,
        fp16               = args.fp16,
        section_label_only = args.section_label_only,
        train_batch_size   = args.train_batch_size,
        test_batch_size    = args.train_batch_size,
    )
    # data_module = ResumeSegmenterDataModuleForBert(
    #     args.dataset_file,
    #     input_composer     = input_composer,
    #     section_label_only = args.section_label_only,
    #     is_cache_data      = False,
    #     train_batch_size   = args.train_batch_size,
    #     test_batch_size    = args.train_batch_size,
    # )
    data_module.setup("fit")

    # Evaluation
    evaluation_debugger = SequenceEvaluationDebugger()

    # Training loop
    train_dl = data_module.train_dataloader()
    valid_dl = data_module.val_dataloader()

    logger.info("## TRAIN: CLASSIFIER ONLY ##")
    model.freeze_bert(True)
    model.precomputed_bert_features(True)
    train_evaluation = SequenceEvaluation()
    valid_evaluation = SequenceEvaluation()
    training_result = train(args, model, train_dl, valid_dl, output_composer,
                            train_evaluation, valid_evaluation,
                            evaluation_debugger, logger)
    dump_compiled_result_split('classifier_result', args, training_result)

    if not args.fit_classifier_only:
        logger.info("## TRAIN: ALL ##")
        model.freeze_bert(False)
        model.precomputed_bert_features(False)
        # Avoid overfitting
        args.classifier_lr = args.learning_rate
        train_evaluation = SequenceEvaluation()
        valid_evaluation = SequenceEvaluation()
        training_result = train(args, model, train_dl, valid_dl, output_composer,
                                train_evaluation, valid_evaluation,
                                evaluation_debugger, logger)
        dump_compiled_result_split('model_result', args, training_result)

    # Results to logger
    logger.info(f'Best epoch: {training_result["best_epoch"] + 1} (of {args.num_train_epochs})')
    def results_to_log(metrics: Dict[str, Any], prefix_text: str, log_level: int):
        pk = metrics['pk']
        wd = metrics['wd']
        seg_equal = metrics['seg_equal']
        cls_equal = metrics['cls_equal']
        report = metrics['classification_report']
        logger.log(log_level, f'{prefix_text}:\n{str(report)}')
        logger.log(log_level, f'{prefix_text}:\nPk: {pk}\nWD: {wd}\nSeg equal: {seg_equal}\nCls equal: {cls_equal}\n')
    results_to_log(training_result['best_train_metrics'], "TRAIN REPORT", logging.DEBUG)
    results_to_log(training_result['best_valid_metrics'], "VALID REPORT", logging.INFO)

    return training_result


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
    logger.info("device: %s, 16-bits training: %s", DEVICE, args.fp16)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: "
            f"{args.gradient_accumulation_steps}, should be >= 1.")

    # Instantiate NER Tag encoder
    tag_encoder = NERTagEncoder.from_labels_file(args.labels_file)
    args.num_labels = tag_encoder.num_labels

    tokenizer_path = args.tokenizer_model or args.bert_model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, do_lower_case=args.do_lower_case)

    input_composer = InputComposer(
        tag_encoder               = tag_encoder,
        tokenizer                 = tokenizer,
        max_seq_length            = args.max_seq_length,
        seq_context_length        = args.doc_stride,
        is_training               = True,
        extra_classifier_features = args.extra_classifier_features
    )

    output_composer = OutputComposer(tag_encoder, args.section_label_only)

    results = []
    for i in range(args.num_splits):
        logger.info("##### SPLIT %d #####", i)
        args.output_dir = os.path.join(output_dirpath, f"split_{i}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        args.dataset_file = args.dataset_split.format(i)

        if args.test_mode:
            args.input_dir = os.path.join(args.experiment_dir, f"split_{i}")
            result = test_split(args, input_composer, output_composer, logger)
        else:
            result = train_split(args, input_composer, output_composer, logger)

        results.append(result)

    valid_results = [r['best_valid_metrics'] for r in results]
    generate_cv_report_to_log(logger, valid_results)

    # Save tokenizer
    if args.save_model:
        tokenizer.save_pretrained(output_dirpath)


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    # Model and hyperparameters
    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model name or path to a "
                        "checkpoint directory.")
    parser.add_argument("--tokenizer_model", default=None, type=str,
                        help="Path to tokenizer files. If empty, defaults to "
                        "--bert_model.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to lower case the input text. True for "
                        "uncased models, False for cased models.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after "
                        "WordPiece tokenization. Sequences longer than this "
                        "will be split into multiple spans, and sequences "
                        "shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, "
                        "how much stride to take between chunks.")
    parser.add_argument('--pooler',
                        default='last',
                        help='Pooling strategy for extracting BERT encoded '
                        'features from last BERT layers. '
                        'One of "last", "sum" or "concat".')
    parser.add_argument("--fit_classifier_only",
                        action='store_true',
                        help="Whether to only train the classifier layer "
                        "while using BERT as a feature extractor.")

    # General
    parser.add_argument('--labels_file', default=None, type=str,
                        help="File with all NER classes to be considered, one "
                        "per line.")
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
                        help="If true, all of the warnings related to data "
                        "processing will be printed.")

    # Model related
    parser.add_argument('--use_lstm', action='store_true',
                        help='Add a BiLSTM layer at the top of the model.')
    parser.add_argument("--lstm_hidden_size", default=100, type=int,
                        help="LSTM hidden size.")
    parser.add_argument("--lstm_layers", default=1, type=int,
                        help="Number of LSTM layers.")
    parser.add_argument('--use_crf', action='store_true',
                        help='Add a CRF layer at the top of the model.')
    parser.add_argument('--extra_classifier_features', action='store_true',
                        help="Whether to use additional features in the "
                        "classifier layer.")

    # Training related
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--classifier_lr', default=1e-3, type=float,
                        help='Learning rate of the classifier and CRF layers.')
    parser.add_argument("--num_consecutive_worse_threshold", default=3, type=int,
                        help='Number of consecutive epochs to stop training if '
                             'validation score is not improving')
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear "
                             "learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help="Number of updates steps to accumulate before "
                             "performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', default=1., type=float,
                        help="Maximum value of gradient norm on update.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

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

if __name__ =='__main__':
    main(get_args())
