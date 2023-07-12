from typing import Optional, List, Dict
from tqdm.auto import tqdm

import json
import numpy as np
import pandas as pd

from dataset import ResumeSegmenterDataset, InputExample, TextElement


def get_annotations_per_instance(dataset: ResumeSegmenterDataset
                                ) -> List[Dict[str, List[TextElement]]]:
    def get_subtag_label(word: TextElement) -> str:
        label = word.label
        subtag, label = label.split('-', 1)
        label = label.replace('_Item', '')
        return subtag, label

    annotations_per_instance = []
    for instance in dataset:
        annotations_per_label = {}

        annotation = []
        for word in instance.word_sequence:
            word_subtag, word_label = get_subtag_label(word)
            if word_label not in annotations_per_label:
                annotations_per_label[word_label] = []

            if word_subtag == 'B':
                if annotation:
                    previous_label = get_subtag_label(annotation[0])[1]
                    annotations_per_label[previous_label].append(annotation)
                annotation = [word]
            elif word_subtag == 'I':
                annotation.append(word)
        if annotation:
            previous_label = get_subtag_label(annotation[0])[1]
            annotations_per_label[previous_label].append(annotation)
        annotations_per_instance.append(annotations_per_label)
    return annotations_per_instance


def compute_num_lines(word_sequence: List[TextElement]) -> int:
    line_indices = [word.line_idx for word in word_sequence]
    return len(set(line_indices))


def compute_num_words(word_sequence: List[TextElement]) -> int:
    return len(word_sequence)


def get_label_statistics(annotations_per_instance: List[Dict[str, List[TextElement]]],
                         label: str
                        ) -> dict:
    if label.lower() == 'all':
        label_annotations_per_instance = [sum(annotations.values(), [])
                                          for annotations in annotations_per_instance]
    else:
        label_annotations_per_instance = [annotations.get(label, [])
                                          for annotations in annotations_per_instance]

    num_annotations = 0
    num_words_per_annotation = []
    num_lines_per_annotation = []
    for annotations in label_annotations_per_instance:
        num_annotations += len(annotations)
        for annotation in annotations:
            num_words_per_annotation.append(compute_num_words(annotation))
            num_lines_per_annotation.append(compute_num_lines(annotation))

    num_words_per_annotation = np.array(num_words_per_annotation)
    num_lines_per_annotation = np.array(num_lines_per_annotation)

    label_statistics = {
        ('total', 'annotations'): num_annotations,
        ('total', 'words'): num_words_per_annotation.sum(),
        ('total', 'lines'): num_lines_per_annotation.sum(),
        ('words', 'mean'): num_words_per_annotation.mean(),
        ('words', 'std'): num_words_per_annotation.std(),
        ('words', 'min'): num_words_per_annotation.min(),
        ('words', 'max'): num_words_per_annotation.max(),
        ('lines', 'mean'): num_lines_per_annotation.mean(),
        ('lines', 'std'): num_lines_per_annotation.std(),
        ('lines', 'min'): num_lines_per_annotation.min(),
        ('lines', 'max'): num_lines_per_annotation.max(),
    }

    label_statistics = {key: round(value) for key, value in label_statistics.items()}
    return label_statistics


def get_general_statistics(dataset: ResumeSegmenterDataset) -> dict:
    """Get the general statistics of the dataset.

    Args:
        dataset (ResumeSegmenterDataset): The dataset to get the statistics from.

    Returns:
        dict: A dictionary with the general statistics.
    """    

    num_instances = len(dataset)
    num_lines_per_instance = np.array([compute_num_lines(instance.word_sequence)
                                       for instance in dataset])
    num_words_per_instance = np.array([compute_num_words(instance.word_sequence)
                                       for instance in dataset])

    general_statistics = {
        'total': {
            'instances': num_instances,
            'lines': num_lines_per_instance.sum(),
            'words': num_words_per_instance.sum(),
        },
        'lines': {
            'mean': num_lines_per_instance.mean(),
            'std': num_lines_per_instance.std(),
            'min': num_lines_per_instance.min(),
            'max': num_lines_per_instance.max(),
        },
        'words': {
            'mean': num_words_per_instance.mean(),
            'std': num_words_per_instance.std(),
            'min': num_words_per_instance.min(),
            'max': num_words_per_instance.max(),
        },
    }

    general_statistics = {key: { key2: round(value2) for key2, value2 in value.items() }
                          for key, value in general_statistics.items()}
    return general_statistics


def main() -> None:
    resources_dirpath = '/home/mwerner/Git/paper/resume_parser/resources/segmenter'
    config_filepath = f'{resources_dirpath}/split_0.conf'
    with open(config_filepath, encoding='utf-8') as fp:
        config = json.load(fp)

    labels_filepath = f'{resources_dirpath}/labels.txt'
    with open(labels_filepath, encoding='utf-8') as fp:
        labels = [line.strip() for line in fp]
        labels = [label for label in labels if not label.endswith('_Item')]
        labels += ['All']

    root_dirpath = config['root_dirpath']
    filenames = config['train'] + config['valid'] + config['test']
    dataset = ResumeSegmenterDataset(root_dirpath, filenames, section_label_only=False)

    print(json.dumps(get_general_statistics(dataset), indent=4))

    annotations_per_instance = get_annotations_per_instance(dataset)
    label_statistics_list = [get_label_statistics(annotations_per_instance, label)
                             for label in labels]
    
    df = pd.DataFrame.from_records(label_statistics_list, index=labels)
    columns = pd.MultiIndex.from_tuples(list(df.columns))
    df.columns = columns
    print(df)
    # We divide by 2 because we have a row where we account all annotations
    print(df[('total','annotations')].sum() / 2)
    print(df[('total','words')].sum() / 2)
    print(df[('total','lines')].sum() / 2)

    # for word in dataset[0].word_sequence:
    #     print(word.label, word.text)

    # # Check whether all instances load without any issue
    # for _ in tqdm(dataset):
    #     continue


if __name__ == '__main__':
    main()
