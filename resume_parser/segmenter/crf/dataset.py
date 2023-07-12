from collections import Counter
from typing import List

from resume_parser.segmenter.dataset import (
    ResumeSegmenterDataset,
    InputExample,
    TextElement
)


class ResumeSegmenterDatasetForCRF(ResumeSegmenterDataset):
    """Resume layout dataset."""

    def __getitem__(self, idx) -> InputExample:
        example = super().__getitem__(idx)
        return  self._to_line_sequence(example)

    def _to_line_sequence(self, example: InputExample) -> InputExample:
        # Group words per line
        line_words_map = {}
        for word in example.word_sequence:
            line_idx = word.line_idx
            if line_idx not in line_words_map:
                line_words_map[line_idx] = []
            line_words_map[line_idx].append(word)

        # Words -> Line
        lines = [self._merge_elements(example.filename, line_words)
                 for line_words in line_words_map.values()]
        lines = sorted(lines, key=lambda x: x.line_idx)

        # Check possible inconsistences while converting word to line sequence
        for i, line in enumerate(lines):
            if i == 0:
                continue
            prev_line = lines[i-1]

            subtag, label = line.label.split('-', 1)
            _, prev_label = prev_line.label.split('-', 1)

            label = label.replace('_Item', '')
            prev_label = prev_label.replace('_Item', '')
            if prev_label != label and subtag == 'I':
                raise Exception('Annotation is inconsistant.\n'
                    f'Filename: {example.filename}.\nLine idx: {line.line_idx}.\n'
                    f'Previous Label: {prev_label}, Next Label: {label}')

        return InputExample(example.filename, lines)

    def _merge_elements(self,
                        filename: str,
                        elements: List[TextElement]
                       ) -> TextElement:
        # Text
        text = ''.join(e.text + (e.tail if i+1 != len(elements) else '')
                       for i, e in enumerate(elements))
        tail = elements[-1].tail
        start = min(e.start for e in elements)
        end = max(e.end for e in elements)
        # Visual - Get most frequent attribute
        def get_most_common(array: List):
            return Counter(array).most_common(1)[0][0]
        fontsize = get_most_common([e.fontsize for e in elements])
        bold = get_most_common([e.bold for e in elements])
        italic = get_most_common([e.italic for e in elements])
        rgb = get_most_common([e.rgb for e in elements])
        # Bounding box - Top-left orientation
        x_coord = min(e.x_coord for e in elements)
        y_coord = max(e.y_coord for e in elements)
        width = max(e.x_coord + e.width - x_coord for e in elements)
        height = max(e.height + y_coord - e.y_coord for e in elements)
        num_page = min(e.num_page for e in elements)

        line_indices = list(set(e.line_idx for e in elements))
        assert len(line_indices) == 1, \
            f"Elements merged must be from the same line. Text: {text}"
        line_idx = line_indices[0]

        # Only one word in line can have subtag 'B-', so it takes priority.
        labels = list(set(e.label for e in elements))
        b_labels = [label for label in labels if label.startswith('B-')]
        if len(b_labels) == 1:
            label = b_labels[0]
        elif len(labels) == 1:
            label = labels[0]
        else:
            raise Exception('More than one label while merging words.\n'
                f'Filename: {filename}.\nLine idx: {line_idx}.\nLabels: {labels}')

        return TextElement(
            text = text,
            start = start,
            end = end,
            fontsize = fontsize,
            bold = bold,
            italic = italic,
            rgb = rgb,
            x_coord = x_coord,
            y_coord = y_coord,
            width = width,
            height = height,
            num_page = num_page,
            word_idx = -1,
            line_idx = line_idx,
            tail = tail,
            label = label
        )


if __name__ == '__main__':
    pass
