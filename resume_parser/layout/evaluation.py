from typing import List, Dict, Any, Optional
from sklearn import metrics

import numpy as np
import pandas as pd


class ClassificationEvaluation(object):
    """Calculates sequence metrics and keeps history of metric values.
    NOTE: Methods `get_best` and `get_best_epoch` assumes a **higher value**
        is better.
    """

    def __init__(self):
        metric_names = ['accuracy', 'classification_report', 'confusion_matrix',
                        'loss', 'running_time', 'num_examples']
        for average in ['micro', 'macro', 'weighted']:
            for metric_name in ['precision', 'recall', 'f1_score']:
                metric_names.append(f'{average}_{metric_name}')

        self.history = {metric_name: [] for metric_name in metric_names}

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
                        y_true: List[str],
                        y_pred: List[str],
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

        labels = sorted(np.unique(y_true + y_pred).tolist())

        def get_metrics(average:str):
            precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(
                y_true, y_pred, average=average, labels=labels)
            return {
                f'{average}_precision': precision,
                f'{average}_recall': recall,
                f'{average}_f1_score': f1_score,
            }

        score_map = {
            'loss': loss,
            'num_examples': len(y_true),
            'running_time': running_time,
            'accuracy': metrics.accuracy_score(y_true, y_pred),
            'classification_report': pd.DataFrame(
                metrics.classification_report(y_true, y_pred, labels=labels,
                                              digits=3, output_dict=True)
            ).transpose(),
            'confusion_matrix': pd.DataFrame(
                metrics.confusion_matrix(y_true, y_pred),
                index=list(labels), columns=list(labels)
            ),
        }
        for average in ['micro', 'macro', 'weighted']:
            score_map.update(get_metrics(average))

        for metric_name, score in score_map.items():
            self.history[metric_name].append(score)
        return score_map


if __name__ == '__main__':
    pass
