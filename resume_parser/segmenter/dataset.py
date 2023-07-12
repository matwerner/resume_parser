from typing import List, Optional

import json
import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from tqdm.auto import tqdm


class TextElement(object):
    "Single item representation"

    def __init__(self,
                 text: str,
                 start: int,
                 end: int,
                 fontsize: int,
                 bold: bool,
                 italic: bool,
                 rgb: int,
                 x_coord: int,
                 y_coord: int,
                 width: int,
                 height: int,
                 num_page: int,
                 word_idx: Optional[int]=-1,
                 line_idx: Optional[int]=-1,
                 tail: Optional[str]='',
                 label: Optional[str]=None
                 ) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.fontsize = fontsize
        self.bold = bold
        self.italic = italic
        self.rgb = rgb
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.width = width
        self.height = height
        self.num_page = num_page
        self.word_idx = word_idx
        self.line_idx = line_idx
        self.tail = tail
        self.label = label


class InputExample(object):
    """A single input example."""

    def __init__(self,
                 filename: str,
                 word_sequence: List[TextElement]
                ):
        self.filename = filename
        self.word_sequence = word_sequence


class ResumeSegmenterDataset(Dataset):
    """Resume layout dataset."""

    def __init__(self, root_dirpath: str,
                       filenames: List[str],
                       section_label_only: Optional[bool] = True,
                       verbose: Optional[bool] = False
                ) -> None:
        self.root_dirpath = root_dirpath
        self.filenames = filenames
        self.section_label_only = section_label_only
        self.verbose = verbose


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx) -> InputExample:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.filenames[idx]
        return self._get_example(filename)


    def _get_example(self, filename: str) -> InputExample:
        filepath = os.path.join(self.root_dirpath, filename)
        # NA values are disturbing the parse of some tokens
        # Since there shouldn't be any NA value, we disable this functionality
        df = pd.read_csv(filepath, na_filter=False)

        word_sequence = [TextElement(**kwargs)
                         for kwargs in df.to_dict(orient='records')]

        # In this project, we assume that a new segment can only begin when
        # starting a new line. Thus, we cannot shift a label in the middle of a
        # line, which is a fair assumption for Resume data. However, rarely it
        # is still possible for 2+ labels to share the same line.
        #
        # In this cases, we  consider the whole line to only be related to the
        # last label. We do this mainly due to the properties that indicate that
        # a shift happens must be in that line, for example, a section marker

        # Group words per line
        line_words_map = {}
        for i, word in enumerate(word_sequence):
            line_idx = word.line_idx
            if line_idx not in line_words_map:
                line_words_map[line_idx] = []
            line_words_map[line_idx].append(i)

        # Check the labels in each line
        for line_idx, word_indices in line_words_map.items():
            line_words = [word_sequence[i] for i in word_indices]
            labels = list(set(word.label for word in line_words))

            # Line is ok: Internal segment
            if len(labels) == 1:
                continue

            b_labels = [label for label in labels if label.startswith('B-')]

            # Line is inconsistant: We found 2+ 'I-' or 'B-' labels
            if len(b_labels) != 1:
                raise Exception('Annotation is inconsistant.\n'
                    f'Filename: {filename}.\nLine idx: {line_idx}.\n'
                    f'Label: {labels}')

            label = b_labels[0]

            # Line is ok: New segment at the start of the line
            if line_words[0].label == b_labels[0]:
                continue

            # Line is not ok: New segment is in the middle of the line
            _, label = label.split('-', 1)

            if self.verbose:
                print('Moving segment to the beginning of the line.\n'
                    f'Filename: {filename}.\nLine idx: {line_idx}.\n'
                    f'Label: {label}')

            # Change line
            for i, word in enumerate(line_words):
                subtag = 'B' if i == 0 else 'I'
                word.label = f'{subtag}-{label}'

        # Add / Remove "item" tagging inside Work_Experience and Education
        #
        # The annotated data is something like:
        # B-Work_Experience Experience
        # I-Work_Experience Company A - Senior Developer
        # I-Work_Experience From 2020 to 2022
        # I-Work_Experience ...
        # B-Work_Experience Company B - Junior Developer
        # I-Work_Experience ...
        #
        # If 'section_label_only' == True:
        # Remove "item" inside Work_Experience and Education
        # Else
        # Create an "Item" tag.
        #
        # This trick is similar to the one behind IOBES tagging.
        # We want the model to differentiate the start of a section of the start
        # of a new item internally
        labels_with_item_segmentation = ['Work_Experience', 'Education']
        for i, word in enumerate(word_sequence):
            subtag, label = word.label.split('-', 1)
            if i == 0 or subtag == 'I'\
                or label not in labels_with_item_segmentation:
                continue

            prev_word = word_sequence[i-1]
            prev_subtag, prev_label = prev_word.label.split('-', 1)
            prev_label = prev_label.replace('_Item', '')
            if prev_label != label:
                continue
            if prev_subtag == 'B' and prev_word.line_idx == word.line_idx:
                raise Exception('Only the first word of each line should start '
                    f'with the "B-" subtag. Filename: {filename}. Word index: '
                    f'{prev_word.word_idx}')

            if self.section_label_only:
                word.label = f'I-{label}'
            else:
                word.label = f'B-{label}_Item'

        return InputExample(filename, word_sequence)


def main() -> None:
    resources_dirpath = '/home/mwerner/Git/resume_parser/resources/segmenter'
    config_filepath = f'{resources_dirpath}/split_0.conf'
    with open(config_filepath, encoding='utf-8') as fp:
        config = json.load(fp)

    root_dirpath = config['root_dirpath']
    train_filenames = config['train']
    train_dataset = ResumeSegmenterDataset(root_dirpath, train_filenames)

    for word in train_dataset[0].word_sequence:
        print(word.label, word.text)

    # Check whether all instances load without any issue
    for _ in tqdm(train_dataset):
        continue

if __name__ == '__main__':
    main()
