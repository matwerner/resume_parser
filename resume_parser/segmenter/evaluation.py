from collections import Counter
from itertools import chain
from typing import List, Dict, Any, Optional
from nltk.metrics import segmentation
from sklearn import metrics

import numpy as np
import pandas as pd

#TODO: Change to another package or copy metrics script
from sklearn_crfsuite import metrics as sequence_metrics

from resume_parser.segmenter.dataset import TextElement

def _flatten(y: List[List]) -> List:
    return list(chain.from_iterable(y))


# https://github.com/sebastianarnold/SECTOR/blob/master/src/main/java/de/datexis/sector/eval/SegmentationEvaluation.java
class SegmentationMetrics(object):


    def compute_pk(self, y_true: List[str], y_pred: List[str], k: int) -> float:
        b_true = self._encode_boundary(y_true)
        b_pred = self._encode_boundary(y_pred)
        return segmentation.pk(b_true, b_pred, k=k, boundary=True)


    def compute_wd(self, y_true: List[str], y_pred: List[str], k: int) -> float:
        b_true = self._encode_boundary(y_true)
        b_pred = self._encode_boundary(y_pred)
        return segmentation.windowdiff(b_true, b_pred, k=k, boundary=True)


    def compute_is_boundaries_equal(self, y_true: List[str], y_pred: List[str]) -> int:
        b_true = self._encode_boundary(y_true)
        b_pred = self._encode_boundary(y_pred)
        return int(b_true == b_pred)


    def compute_is_classes_equal(self, y_true: List[str], y_pred: List[str]) -> int:
        return int(y_true == y_pred)


    def compute_k(self, y_seqs: List[List[str]]) -> int:
        # Convert from BIO to Boundaries
        b_seqs = [self._encode_boundary(seq) for seq in y_seqs]

        length_sum = 0
        length_count = 0
        for seq in b_seqs:
            segment_lengths = self._get_segment_lengths(seq)
            length_sum += sum(segment_lengths)
            length_count += len(segment_lengths)
        length_mean = length_sum / length_count
        return max(2, int(round(length_mean / 2.0)))


    def get_num_boundaries(self, seq: List[str]) -> int:
        return sum(self._encode_boundary(seq))


    def _encode_boundary(self, seq: List[str]) -> List[bool]:
        return [i>0 and seq_i.startswith('B-') for i, seq_i in enumerate(seq)]


    def _get_segment_lengths(self, seq: List[bool]) -> List[int]:
        segment_lengths = []

        length = 0
        for is_boundary in seq:
            if is_boundary:
                segment_lengths.append(length)
                length = 0
            length += 1
        if length > 0:
            segment_lengths.append(length)

        assert sum(segment_lengths) == len(seq)
        return segment_lengths


class SequenceEvaluation(object):
    """Calculates sequence metrics and keeps history of metric values.
    NOTE: Methods `get_best` and `get_best_epoch` assumes a **higher value**
        is better.
    """

    def __init__(self):
        metric_names = ['accuracy', 'classification_report',
                        'pk', 'wd', 'seg_equal', 'cls_equal',
                        'loss', 'running_time', 'num_examples']
        for average in ['micro', 'macro', 'weighted']:
            for metric_name in ['precision', 'recall', 'f1_score']:
                metric_names.append(f'{average}_{metric_name}')

        self.history = {metric_name: [] for metric_name in metric_names}
        self.segmentation_metrics = SegmentationMetrics()


    def clear_history(self) -> None:
        self.history = {k: [] for k in self.history.keys()}


    def get_best(self, metric_name: str) -> Any:
        """Returns the maximum value of the given metric by name."""
        return max(self.history[metric_name])


    def get_best_epoch(self, metric_name: str) -> int:
        """Returns the epoch number for which the metric has its highest
        value."""
        return int(np.argmax(self.history[metric_name]))


    def get_value(self, metric_name: str, epoch: Optional[int] = None) -> Any:
        """Returns the value of a metric at a given epoch (defaults to last
        epoch)."""
        if epoch is None:
            epoch = -1
        return self.history[metric_name][epoch]


    def add_value(self, metric_name: str, value: Any) -> None:
        """Manually add last epoch value to a given metric."""
        self.history[metric_name].append(value)


    def compute_metrics(self,
                        y_true: List[List[str]],
                        y_pred: List[List[str]],
                        loss: float,
                        running_time: float,
                       ) -> Dict[str, Any]:
        """Calculates all registered metrics for the gold and predicted tag
        sequences.
        Args:
            y_true: a list of gold tag sequences.
            y_pred: a list of predicted tag sequences.
        Returns:
            A dict of metric names to calculated metric values.
        """
        assert len(y_true) == len(y_pred), ('Number of examples is different '
            f'while computing metrics: {len(y_true)} != {len(y_pred)}')

        f_true = _flatten(y_true)
        f_pred = _flatten(y_pred)

        labels = np.unique(f_true + f_pred).tolist()
        sorted_labels = sorted(labels, key=lambda x: (x[1:], x[0]))

        def get_metrics(average:str):
            return {
                f'{average}_precision': sequence_metrics.flat_precision_score(
                    y_true, y_pred, average=average),
                f'{average}_recall': sequence_metrics.flat_recall_score(
                    y_true, y_pred, average=average),
                f'{average}_f1_score': sequence_metrics.flat_f1_score(
                    y_true, y_pred, average=average),
            }

        score_map = {
            'loss': loss,
            'num_examples': len(y_true),
            'running_time': running_time,
            'accuracy': sequence_metrics.flat_accuracy_score(y_true, y_pred),
            'classification_report': pd.DataFrame(
                sequence_metrics.flat_classification_report(y_true, y_pred,
                    labels=sorted_labels, digits=3, output_dict=True)
                ).transpose()
        }
        for average in ['micro', 'macro', 'weighted']:
            score_map.update(get_metrics(average))

        k = self.segmentation_metrics.compute_k(y_true)
        pks = [self.segmentation_metrics.compute_pk(y_true_i, y_pred_i, k)
               for y_true_i, y_pred_i in zip(y_true, y_pred)]
        wds = [self.segmentation_metrics.compute_wd(y_true_i, y_pred_i, k)
               for y_true_i, y_pred_i in zip(y_true, y_pred)]
        is_seg_equals = [self.segmentation_metrics.compute_is_boundaries_equal(y_true_i, y_pred_i)
                         for y_true_i, y_pred_i in zip(y_true, y_pred)]
        is_cls_equals = [self.segmentation_metrics.compute_is_classes_equal(y_true_i, y_pred_i)
                         for y_true_i, y_pred_i in zip(y_true, y_pred)]
        score_map['pk'] = np.array(pks).mean()
        score_map['wd'] = np.array(wds).mean()
        score_map['seg_equal'] = np.array(is_seg_equals).mean()
        score_map['cls_equal'] = np.array(is_cls_equals).mean()

        for metric_name, score in score_map.items():
            self.history[metric_name].append(score)
        return score_map


class SegmentSequenceEvaluationDebugger(object):

    class Segment(object):
        "Single segment representation"

        def __init__(self,
                     text: str,
                     start: int,
                     end: int,
                     label: Optional[str]=None
                    ) -> None:
            self.text = text
            self.start = start
            self.end = end
            self.label = label


    def __init__(self, jaccard_threshold: float = 0.8) -> None:
        self.jaccard_threshold = jaccard_threshold


    def get_classification_df(self,
                              filenames: List[str],
                              tokens_list: List[List[TextElement]],
                              y_true_list: List[List[str]],
                              y_pred_list: List[List[str]]
                             ) -> pd.DataFrame:
        cummulative_precision_counters = None
        cummulative_recall_counters = None
        df_list = []

        # Per file
        for i, filename in enumerate(filenames):
            tokens = tokens_list[i]
            y_true = y_true_list[i]
            y_pred = y_pred_list[i]
            metrics_df, precision_counters, recall_counters = \
                self.get_metrics(tokens, y_true, y_pred)
            metrics_df.insert(0, 'filename', filename)

            df_list.append(metrics_df)
            if cummulative_precision_counters is None:
                cummulative_precision_counters = precision_counters
            else:
                for label in precision_counters.keys():
                    if label not in cummulative_precision_counters:
                        cummulative_precision_counters[label] = Counter()
                    cummulative_precision_counters[label].update(precision_counters[label])
            if cummulative_recall_counters is None:
                cummulative_recall_counters = recall_counters
            else:
                for label in recall_counters.keys():
                    if label not in cummulative_recall_counters:
                        cummulative_recall_counters[label] = Counter()
                    cummulative_recall_counters[label].update(recall_counters[label])

        # ALL
        labels = sorted({label for label in cummulative_precision_counters.keys()})
        metrics_df = self.compute_metrics_from_counters(
            cummulative_precision_counters, cummulative_recall_counters, labels)
        metrics_df.insert(0, 'filename', 'ALL')
        df_list.append(metrics_df)

        return pd.concat(df_list)


    def get_metrics(self,
                    tokens: List[TextElement],
                    y_true: List[str],
                    y_pred: List[str]
                   ):
        segments_true = self.get_segments(tokens, y_true)
        segments_pred = self.get_segments(tokens, y_pred)
        labels = sorted({s.label for s in segments_true + segments_pred})

        # Compute metrics
        recall_counters = self.get_metric_counters(segments_true, segments_pred, labels)
        precision_counters = self.get_metric_counters(segments_pred, segments_true, labels)
        metrics_df = self.compute_metrics_from_counters(precision_counters, recall_counters, labels)

        return (metrics_df, precision_counters, recall_counters)


    def get_segments(self, tokens: List[TextElement], y: List[str]):
        segments = []
        segment_tokens = []
        segment_label = None
        for token, tag in zip(tokens, y):
            tag = tag.replace('_Item', '')
            prefix, label = tag.split('-')
            if prefix == 'B':
                if segment_label is not None:
                    segment = self.build_segment(segment_tokens, segment_label)
                    segments.append(segment)
                    segment_tokens = []
                    segment_label = None
                segment_tokens.append(token)
                segment_label = label
            elif prefix == 'I':
                segment_tokens.append(token)
        if segment_label is not None:
            segment = self.build_segment(segment_tokens, segment_label)
            segments.append(segment)
        return segments


    def build_segment(self, lines: List[TextElement], label: str) -> Segment:
        start = min(line.start for line in lines)
        end = max(line.end for line in lines)
        text = ''.join(line.text + line.tail for line in lines)
        return self.Segment(text, start, end, label)


    def compute_metrics_from_counters(self,
                                      precision_counters: Dict[str, Dict[str, int]],
                                      recall_counters: Dict[str, Dict[str, int]],
                                      labels: List[str]
                                     ) -> pd.DataFrame:
        metrics_per_label = {}
        for label in labels:
            assert precision_counters[label]['tp'] == recall_counters[label]['tp']
            tp = precision_counters[label]['tp']
            fp = precision_counters[label]['fn']
            fn = recall_counters[label]['fn']
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            metrics_per_label[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': tp + fn
            }
        metrics_per_label['macro'] = {
            'precision': np.mean([metrics_per_label[label]['precision'] for label in labels]),
            'recall': np.mean([metrics_per_label[label]['recall'] for label in labels]),
            'f1_score': np.mean([metrics_per_label[label]['f1_score'] for label in labels]),
            'support': np.sum([metrics_per_label[label]['support'] for label in labels])
        }

        indices = ['precision', 'recall', 'f1_score', 'support']
        df = pd.DataFrame(metrics_per_label, index=indices, columns=labels+['macro']) \
            .transpose().reset_index().rename(columns={'index': 'section'})
        return df


    def get_metric_counters(self,
                            segments_true: List[Segment],
                            segments_pred: List[Segment],
                            labels: List[str]
                           ):
        # Compute metrics
        # counters = {label: {'tp': 0, 'fn': 0} for label in labels}
        counters = {label: Counter() for label in labels}
        for s1 in segments_true:
            best_jaccard = -1
            best_segment = None
            for s2 in segments_pred:
                jaccard = self.compute_jaccard(s1, s2)
                if  jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_segment = s2
            if best_jaccard >= self.jaccard_threshold and s1.label == best_segment.label:
                counters[s1.label]['tp'] += 1
            else:
                counters[s1.label]['fn'] += 1
        return counters


    def compute_jaccard(self, s1: Segment, s2: Segment) -> float:
        """Compute Jaccard similarity between two segments

        Example:
            >>> s1 = Segment('abc', 0, 3) # [0, 1, 2]
            >>> s2 = Segment('bcd', 1, 4) # [1, 2, 3]
            >>> compute_jaccard(s1, s2)
            0.5
        """
        union_start = min(s1.start, s2.start)
        union_end = max(s1.end, s2.end)
        inter_start = max(s1.start, s2.start)
        inter_end = min(s1.end, s2.end)

        if inter_start >= inter_end:
            return 0.0
        return (inter_end - inter_start) / (union_end - union_start)


class SequenceEvaluationDebugger(object):


    def __init__(self) -> None:
        self.segmentation_metrics = SegmentationMetrics()
        self.segment_sequence_evaluation = SegmentSequenceEvaluationDebugger(0.90)


    def dump_results(self,
                     output_filepath: str,
                     filenames: List[str],
                     tokens_list: List[List[TextElement]],
                     y_true_list: List[List[str]],
                     y_pred_list: List[List[str]],
                    ) -> None:
        segmentation_df = self.get_segmentation_df(filenames, y_true_list, y_pred_list)

        # Focus on the segmentation aspect
        def remove_item_suffix_from_seq(seq: List[str]) -> List[str]:
            return [tag.rsplit('_Item')[0] for tag in seq]
        y_true_list = [remove_item_suffix_from_seq(seq) for seq in y_true_list]
        y_pred_list = [remove_item_suffix_from_seq(seq) for seq in y_pred_list]
        output_df = self.get_raw_df(filenames, tokens_list, y_true_list, y_pred_list)
        classification_df = self.get_classification_df(filenames, y_true_list, y_pred_list)
        confusion_matrix_df = self.get_confusion_matrix_df(y_true_list, y_pred_list)
        segment_classification_df = self.segment_sequence_evaluation.get_classification_df(
            filenames, tokens_list, y_true_list, y_pred_list)

        def remove_item_sections_from_seq(seq: List[str]) -> List[str]:
            section_seq = []
            item_sections = ['Work_Experience', 'Education']
            for i, tag in enumerate(seq):
                prefix, label = tag.split('-')
                if i > 0 and prefix == 'B' and label in item_sections:
                    _, prev_label = seq[i-1].split('-')
                    new_prefix = 'I' if label == prev_label else 'B'
                    section_seq.append(f'{new_prefix}-{label}')
                else:
                    section_seq.append(tag)
            return section_seq
        y_true_list = [remove_item_sections_from_seq(seq) for seq in y_true_list]
        y_pred_list = [remove_item_sections_from_seq(seq) for seq in y_pred_list]
        section_segment_classification_df = self.segment_sequence_evaluation.get_classification_df(
            filenames, tokens_list, y_true_list, y_pred_list)

        # Focus on the classification aspect
        def remove_iob_prefix_from_seq(seq: List[str]) -> List[str]:
            return [tag[2:] for tag in seq]
        y_true_list = [remove_iob_prefix_from_seq(seq) for seq in y_true_list]
        y_pred_list = [remove_iob_prefix_from_seq(seq) for seq in y_pred_list]
        classification_no_iob_df = self.get_classification_df(filenames, y_true_list, y_pred_list)
        confusion_matrix_no_iob_df = self.get_confusion_matrix_df(y_true_list, y_pred_list)

        with pd.ExcelWriter(output_filepath, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='raw output',
                index=False, header=True)
            classification_df.to_excel(writer, sheet_name='classification',
                index=False, header=True, float_format="%.4f")
            confusion_matrix_df.to_excel(writer, sheet_name='confusion matrix',
                index=True, header=True, float_format="%.4f")
            segment_classification_df.to_excel(writer, sheet_name='classification (segment)',
                index=False, header=True, float_format="%.4f")
            section_segment_classification_df.to_excel(writer, sheet_name='classification (section segment)',
                index=False, header=True, float_format="%.4f")
            segmentation_df.to_excel(writer, sheet_name='segmentation',
                index=False, header=True, float_format="%.4f")
            classification_no_iob_df.to_excel(writer, sheet_name='classification (no iob)',
                index=False, header=True, float_format="%.4f")
            confusion_matrix_no_iob_df.to_excel(writer, sheet_name='confusion matrix (no iob)',
                index=True, header=True, float_format="%.4f")


    def get_raw_df(self,
                   filenames: List[str],
                   tokens_list: List[List[TextElement]],
                   y_true_list: List[List[str]],
                   y_pred_list: List[List[str]]
                  ) -> pd.DataFrame:
        # Get token outputs
        df_list = []
        for i, filename in enumerate(filenames):
            tokens = tokens_list[i]
            y_true, y_pred = y_true_list[i], y_pred_list[i]
            df = self._get_token_outputs(filename, tokens, y_true, y_pred)
            df_list.append(df)
        return pd.concat(df_list)


    def _get_token_outputs(self,
                           filename: str,
                           tokens: List[TextElement],
                           y_true: List[str],
                           y_pred: List[str]
                          ):
        records = []
        for i, token in enumerate(tokens):
            text = token.text
            is_boundary_true = int(y_true[i].startswith('B-'))
            is_boundary_pred = int(y_pred[i].startswith('B-'))
            records.append((
                filename, i, repr(text),  y_true[i], y_pred[i],
                is_boundary_true, is_boundary_pred
            ))
        columns = ['filename', 'index', 'token_text', 'section_true',
            'section_pred', 'boundary_true', 'boundary_pred']
        return pd.DataFrame.from_records(records, columns=columns)


    def get_classification_df(self,
                              filenames: List[str],
                              y_true_list: List[List[str]],
                              y_pred_list: List[List[str]]
                             ) -> pd.DataFrame:
        df_list = []

        # Per file
        for i, filename in enumerate(filenames):
            df = self._get_classification_metrics(filename, y_true_list[i], y_pred_list[i])
            df_list.append(df)

        # ALL
        df = self._get_classification_metrics_all(filenames, y_true_list, y_pred_list)
        df_list.append(df)

        return pd.concat(df_list)


    def _get_classification_metrics(self,
                                    filename: str,
                                    y_true: List[str],
                                    y_pred: List[str]
                                   ) -> pd.DataFrame:
        return self._get_classification_metrics_internal(filename, y_true, y_pred)


    def _get_classification_metrics_all(self,
                                        filenames: List[str],
                                        y_true_list: List[List[str]],
                                        y_pred_list: List[List[str]]
                                       ) -> pd.DataFrame:
        f_true = _flatten(y_true_list)
        f_pred = _flatten(y_pred_list)
        return self._get_classification_metrics_internal('ALL', f_true, f_pred)


    def _get_classification_metrics_internal(self,
                                             filename: str,
                                             y_true: List[str],
                                             y_pred: List[str]
                                            ) -> pd.DataFrame:
        # Sort labels
        labels = np.unique(y_true + y_pred).tolist()
        is_iob = all(label.startswith('B-') or label.startswith('I-') for label in labels)
        sorted_labels = sorted(labels, key=lambda x: (x[1:], x[0])) if is_iob else sorted(labels)

        # Label metrics
        result = metrics.precision_recall_fscore_support(y_true, y_pred,
            labels=sorted_labels, zero_division=0)

        indices = ['precison', 'recall', 'f1-score', 'support']
        df = pd.DataFrame(result, index=indices, columns=sorted_labels) \
            .transpose().reset_index().rename(columns={'index': 'section'})

        # Average metrics
        accuracy = metrics.accuracy_score(y_true, y_pred)
        df.loc[df.shape[0]] = ['accuracy', accuracy, accuracy, accuracy, accuracy]

        def append_average(metric_name: str, y_t: List[str], y_p: List[str],
                           valid_labels: List[str]):
            precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(
                y_t, y_p, labels=valid_labels, average='macro', zero_division=0)
            df.loc[df.shape[0]] = [metric_name, precision, recall, f1_score, len(y_t)]
        append_average('macro_avg', y_true, y_pred, sorted_labels)

        def filter_tag(y_t: List[str], y_p: List[str], prefix: str) -> List[str]:
            new_y_t, new_y_p = [], []
            for y_t_i, y_p_i in zip(y_t, y_p):
                if y_t_i.startswith(prefix) or y_p_i.startswith(prefix):
                    new_y_t.append(y_t_i)
                    new_y_p.append(y_p_i)
            return new_y_t, new_y_p

        if is_iob:
            b_y_true, b_y_pred = filter_tag(y_true, y_pred, 'B-')
            valid_label = [label for label in sorted_labels if label.startswith('B-')]
            append_average('B-macro_avg', b_y_true, b_y_pred, valid_label)
            i_y_true, i_y_pred = filter_tag(y_true, y_pred, 'I-')
            valid_label = [label for label in sorted_labels if label.startswith('I-')]
            append_average('I-macro_avg', i_y_true, i_y_pred, valid_label)

        df.insert(loc=0, column='filename', value=filename)
        return df


    def get_confusion_matrix_df(self,
                                y_true_list: List[str],
                                y_pred_list: List[str]
                               ) -> pd.DataFrame:
        f_true = _flatten(y_true_list)
        f_pred = _flatten(y_pred_list)

        labels = np.unique(f_true + f_pred).tolist()
        sorted_labels = sorted(labels, key=lambda x: (x[1:], x[0]))

        return pd.DataFrame(
            metrics.confusion_matrix(f_true, f_pred, labels=sorted_labels),
            index=[f'true:{x}' for x in sorted_labels],
            columns=[f'pred:{x}' for x in sorted_labels]
        )


    def get_segmentation_df(self,
                            filenames: List[str],
                            y_true_list: List[List[str]],
                            y_pred_list: List[List[str]]
                           ) -> pd.DataFrame:
        k = self.segmentation_metrics.compute_k(y_true_list)

        df_list = []
        for i, filename in enumerate(filenames):
            df = self.get_segmentation_metrics(filename, y_true_list[i], y_pred_list[i], k)
            df_list.append(df)
        df = pd.concat(df_list)

        df.loc[df.shape[0]] = [
            'ALL',
            df['boundaries_true'].sum(),
            df['boundaries_pred'].sum(),
            df['pk'].mean(),
            df['wd'].mean(),
            df['is_seq_equal'].mean(),
            df['is_cls_equal'].mean(),
            df['is_seq_equal_section'].mean(),
            df['is_cls_equal_section'].mean(),
            df['is_seq_equal_work'].mean(),
            df['is_cls_equal_work'].mean(),
            df['is_seq_equal_edu'].mean(),
            df['is_cls_equal_edu'].mean()
        ]
        return df


    def get_segmentation_metrics(self,
                                 filename: str,
                                 y_true: List[str],
                                 y_pred: List[str],
                                 k: int
                                ) -> pd.DataFrame:
        # Compute value
        true_count = self.segmentation_metrics.get_num_boundaries(y_true)
        pred_count = self.segmentation_metrics.get_num_boundaries(y_pred)

        pk = self.segmentation_metrics.compute_pk(y_true, y_pred, k)
        wd = self.segmentation_metrics.compute_wd(y_true, y_pred, k)

        # Perfect segmentation
        is_seq_equal = self.segmentation_metrics.compute_is_boundaries_equal(y_true, y_pred)
        is_cls_equal = self.segmentation_metrics.compute_is_classes_equal(y_true, y_pred)

        # Perfect segmentation (section only)
        def section_only(y: List[str]) -> List[str]:
            return [f'I-{y_i[2:-5]}' if y_i.endswith('_Item') else y_i for y_i in y]
        y_true_section = section_only(y_true)
        y_pred_section = section_only(y_pred)
        is_seq_equal_section = self.segmentation_metrics.compute_is_boundaries_equal(y_true_section, y_pred_section)
        is_cls_equal_section = self.segmentation_metrics.compute_is_classes_equal(y_true_section, y_pred_section)

        # Perfect section segmentation (item only)
        def section_and_item(y: List[str]) -> List[str]:
            return [y_i.replace('_Item', '') for y_i in y]
        y_true_section_item = section_and_item(y_true)
        y_pred_section_item = section_and_item(y_pred)

        def filter_tag(y_t: List[str], y_p: List[str], tag: str) -> List[str]:
            new_y_t, new_y_p = [], []
            for y_t_i, y_p_i in zip(y_t, y_p):
                if y_t_i[2:].startswith(tag):
                    new_y_t.append(y_t_i)
                    new_y_p.append(y_p_i)
                elif y_p_i[2:].startswith(tag):
                    return [], [], False
            return new_y_t, new_y_p, True

        y_true_work, y_pred_work, is_valid = filter_tag(y_true_section_item, y_pred_section_item, 'Work_Experience')
        if is_valid:
            is_seq_equal_work = self.segmentation_metrics.compute_is_boundaries_equal(y_true_work, y_pred_work)
            is_cls_equal_work = self.segmentation_metrics.compute_is_classes_equal(y_true_work, y_pred_work)
        else:
            is_seq_equal_work = 0
            is_cls_equal_work = 0

        y_true_edu, y_pred_edu, is_valid = filter_tag(y_true_section_item, y_pred_section_item, 'Education')
        if is_valid:
            is_seq_equal_edu = self.segmentation_metrics.compute_is_boundaries_equal(y_true_edu, y_pred_edu)
            is_cls_equal_edu = self.segmentation_metrics.compute_is_classes_equal(y_true_edu, y_pred_edu)
        else:
            is_seq_equal_edu = 0
            is_cls_equal_edu = 0

        # Record
        result = [(filename, true_count, pred_count, pk, wd,
                   is_seq_equal, is_cls_equal,
                   is_seq_equal_section, is_cls_equal_section,
                   is_seq_equal_work, is_cls_equal_work,
                   is_seq_equal_edu, is_cls_equal_edu)]

        columns = ['filename', 'boundaries_true', 'boundaries_pred', 'pk', 'wd',
                   'is_seq_equal', 'is_cls_equal',
                   'is_seq_equal_section', 'is_cls_equal_section',
                   'is_seq_equal_work', 'is_cls_equal_work',
                   'is_seq_equal_edu', 'is_cls_equal_edu']
        return pd.DataFrame.from_records(result, columns=columns)


if __name__ == '__main__':
    pass
