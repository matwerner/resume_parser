from typing import List, Tuple, Optional, Union

import json
import numpy as np

from transformers.tokenization_utils import PreTrainedTokenizer
from nltk.corpus import stopwords

from resume_parser.segmenter.dataset import TextElement, InputExample
from resume_parser.segmenter.bert.feature_extractor import LineFeatureExtractor

STOP_WORDS = stopwords.words('portuguese')

# Reference:
# https://github.com/neuralmind-ai/portuguese-bert/blob/master/ner_evaluation/tag_encoder.py
class NERTagEncoder(object):
    """Handles creation of NER BIO tags for a list of named entity classes and
    conversion of tags to ids and vice versa."""

    VALID_TRANSITIONS = {
        'B': ['B', 'I', 'O'],
        'I': ['B', 'I', 'O'],
        'O': ['B', 'O'],
    }

    def __init__(self, labels: List[str], ignore_index: int = -100):
        if not len(set(labels)) == len(labels):
            raise ValueError("'labels' have duplicate entries.")
        if "O" in labels or "X" in labels:
            raise ValueError("'classes' should not have tag O nor X.")
        if ignore_index >= 0 or not isinstance(ignore_index, int):
            raise ValueError("'ignore_index' should be a negative int.")

        self.labels = tuple(labels)
        self.tags = ["O"]
        self.ignore_index = ignore_index
        self.tag_to_id = {"X": ignore_index}

        for label in labels:
            for subtag in ['B', 'I']:
                self.tags.append(f"{subtag}-{label}")

        for i, tag in enumerate(self.tags):
            self.tag_to_id[tag] = i

    @classmethod
    def from_labels_file(cls, filepath: str, *args, **kwargs):
        """Creates encoder from a file with NER label classes (one class per
        line) and a given scheme."""
        with open(filepath, 'r', encoding='utf-8') as fp:
            labels = [label.strip() for label in fp if label]

        return cls(labels, *args, **kwargs)

    @property
    def num_labels(self) -> int:
        return len(self.tags)

    def convert_tags_to_ids(self, tags: List[str]) -> List[int]:
        """Converts a list of tag strings to a list of tag ids."""
        return [self.tag_to_id[tag] for tag in tags]

    def convert_ids_to_tags(self, tag_ids: List[int]) -> List[str]:
        """Returns a list of tag strings from a list of tag ids."""
        return [self.tags[tag_id] for tag_id in tag_ids]

    def decode_valid(self, tag_sequence: List[str]) -> List[str]:
        """Processes a list of tag strings to remove invalid predictions given
        the valid transitions of the tag scheme, such as "I" tags coming after
        "O" tags."""
        prev_tag = 'O'
        prev_label = 'O'

        final = []
        for tag_and_label in tag_sequence:
            tag, label = tag_and_label.split('-', 1)
            valid_transitions = self.VALID_TRANSITIONS[prev_tag]

            valid_tag = False
            if tag in valid_transitions:
                if tag in ('B', 'O'):
                    valid_tag = True
                elif tag == 'I' and label == prev_label:
                    valid_tag = True

            if valid_tag:
                prev_tag = tag
                prev_label = label
                final.append(tag_and_label)
            else:
                prev_tag = 'O'
                prev_label = 'O'
                final.append('O')

        return final


class InputSpan(object):
    """A single set of features of data."""

    def __init__(self,
                 filename: str,
                 index: int,
                 word_count: int,
                 token_count: int,
                 input_ids: List[int],
                 tokens: List[str],
                 word_ids: List[int],
                 token_type_ids: List[int],
                 attention_mask: List[int],
                 context_mask: List[bool],
                 prediction_mask: List[bool],
                 extra_classifier_features: Optional[List[List[float]]] = None,
                 label_ids: Optional[List[int]] = None,
                 bert_features: Optional[List[np.array]] = None,
                 ):
        self.filename = filename
        self.index = index
        self.word_count = word_count
        self.token_count = token_count
        self.input_ids = input_ids
        self.tokens = tokens
        self.word_ids = word_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.context_mask = context_mask
        self.prediction_mask = prediction_mask
        self.extra_classifier_features = extra_classifier_features
        self.label_ids = label_ids
        self.bert_features = bert_features


class InputComposer(object):
    """Break down the examples into words spans to be feed the model."""

    def __init__(self,
                 tag_encoder: NERTagEncoder,
                 tokenizer: PreTrainedTokenizer,
                 max_seq_length: int,
                 seq_context_length: int,
                 is_training: bool,
                 extra_classifier_features: Optional[bool] = False,
                 prediction_mode: Optional[str] = 'line',
                 verbose: bool = True
                 ) -> None:
        if max_seq_length < seq_context_length:
            raise ValueError("Input spans will ignore tokens!")
        elif prediction_mode.lower() not in {'word', 'line'}:
            raise ValueError("Prediction mode must be 'word' or 'line'.")

        self.tag_encoder = tag_encoder
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.seq_context_length = seq_context_length
        self.is_training = is_training
        self.extra_classifier_features = extra_classifier_features
        self.prediction_mode = prediction_mode.lower()
        self.verbose = verbose

        resources_dirpath = '/home/mwerner/Git/resume_parser/resources/segmenter'
        sections_filepath = resources_dirpath + '/section_names_map.json'
        section_names_map = json.load(open(sections_filepath, encoding='utf-8'))

        self.feature_extractor = LineFeatureExtractor(
            section_names_map=section_names_map,
            stop_words=STOP_WORDS,
            use_vocabulary=True,
            use_text=True,
            use_visual=True,
            use_spatial=True
        )


    def _get_context_prediction_masks(self,
                                      span_idx: int,
                                      word_ids: List[int],
                                      word_sequence: List[TextElement]
                                     ) -> Tuple[List[bool], List[bool]]:
        context_mask = []
        prediction_mask = []
        for i, word_id in enumerate(word_ids):
            # Special token: [CLS], [SEP], ...
            if word_id is None:
                context_mask.append(False)
                prediction_mask.append(False)
                continue

            word = word_sequence[word_id]
            prev_word = word_sequence[word_id-1] if word_id > 0 else None

            is_suffix = i > 0 and word_id == word_ids[i-1]
            is_context_token = span_idx > 0 and i <= self.seq_context_length
            is_new_line = prev_word is None or word.line_idx != prev_word.line_idx

            # If prediction_mode, we only classify:
            # 'line': The first token of each line
            # 'word': The first token of each word
            is_new_line = is_new_line or self.prediction_mode == 'word'

            is_first_line_token = not is_suffix and is_new_line

            context_mask.append(is_first_line_token and is_context_token)
            prediction_mask.append(is_first_line_token and not is_context_token)
        return context_mask, prediction_mask


    def _get_word_token_counts(self,
                               span_idx: int,
                               word_ids: List[int],
                              ) -> Tuple[int, int]:
        # Select valid word ids
        word_ids = [word_id for i, word_id in enumerate(word_ids)
                    if word_id is not None
                    and (i == 0 or word_id != word_ids[i-1])
                    and (span_idx == 0 or i > self.seq_context_length)]
        word_count = len(set(word_ids))
        token_count = len(word_ids)
        return word_count, token_count


    def convert_example_into_spans(self, example: InputExample) -> List[InputSpan]:
        word_sequence = example.word_sequence
        word_texts = [word.text for word in word_sequence]

        #TODO: Create a custom span generator considering start / end of line
        tokenized_text = self.tokenizer(
            word_texts,
            is_split_into_words       = True,
            truncation                = True,
            max_length                = self.max_seq_length,
            stride                    = self.seq_context_length,
            return_overflowing_tokens = True,
        )

        classifier_features = self.feature_extractor.get_feature_vectors(word_sequence)

        global_word_ids = set()
        global_word_count = 0
        input_spans = []
        for span_idx, input_ids in enumerate(tokenized_text['input_ids']):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            word_ids = tokenized_text.word_ids(span_idx)

            # Get which tokens matter while training the model
            context_mask, prediction_mask = self._get_context_prediction_masks(
                span_idx, word_ids, word_sequence)

            # Convert labels
            label_ids = []
            if self.is_training:
                sequence_mask = np.array(context_mask) | np.array(prediction_mask)
                labels = [word_sequence[word_ids[i]].label if pred else 'X'
                          for i, pred in enumerate(sequence_mask)]
                label_ids = self.tag_encoder.convert_tags_to_ids(labels)

            extra_classifier_features = []
            if self.extra_classifier_features:
                length = self.feature_extractor.get_feature_vector_length()
                for word_id in word_ids:
                    if word_id is None:
                        features = length * [0]
                    else:
                        features = classifier_features[word_id]
                    extra_classifier_features.append(features)
                # extra_classifier_features = \
                #     self._get_extra_classifier_features(word_sequence, word_ids)

            # For asserting information
            word_count, token_count = self._get_word_token_counts(span_idx, word_ids)
            global_word_count += word_count
            global_word_ids.update(word_ids)

            # Extra BERT attributes
            attention_mask = [1] * len(tokens)
            token_type_ids = [0] * len(tokens)
            word_ids = [-1 if word_id is None else word_id for word_id in word_ids]

            input_spans.append(InputSpan(
                filename             = example.filename,
                index                = span_idx,
                word_count           = word_count,
                token_count          = token_count,
                input_ids            = np.array(input_ids, dtype=np.int32),
                tokens               = np.array(tokens, dtype=np.object_),
                word_ids             = np.array(word_ids, dtype=np.int32),
                token_type_ids       = np.array(token_type_ids, dtype=np.int32),
                attention_mask       = np.array(attention_mask, dtype=np.int32),
                # For masking a tensor, variable must be Long or Bool
                context_mask         = np.array(context_mask, dtype=np.bool_),
                prediction_mask      = np.array(prediction_mask, dtype=np.bool_),
                # CrossEntropyLoss not implement for Int, so use Long
                label_ids            = np.array(label_ids, dtype=np.int64),
                # Custom input features
                extra_classifier_features = \
                    np.array(extra_classifier_features, dtype=np.float32),
                bert_features        = np.array([])
            ))

        # Check whether all words are covered
        text_word_count = len(word_texts)
        if text_word_count != global_word_count:
            raise ValueError("Word counts of text and spans are not equal:"
                f"\nFilename:      {example.filename};"
                f"\nText count:    {text_word_count};"
                f"\nSpan count:    {global_word_count};"
                f"\nWord id count: {len(global_word_ids - {None})}."
            )

        return input_spans


    def num_extra_classifier_features(self):
        if self.extra_classifier_features:
            return self.feature_extractor.get_feature_vector_length()
        return 0


class OutputComposer(object):
    """Combine words spans into an example to be output to the user."""

    def __init__(self,
                 tag_encoder: NERTagEncoder,
                 section_label_only: Optional[bool] = True,
                 prediction_mode: Optional[str] = 'line',
                 verbose: Optional[bool] = True
                 ) -> None:
        if prediction_mode.lower() not in {'word', 'line'}:
            raise ValueError("Prediction mode must be 'word' or 'line'.")

        self.tag_encoder = tag_encoder
        self.section_label_only = section_label_only
        self.prediction_mode = prediction_mode.lower()
        self.verbose = verbose

    def get_example_output(self,
                           spans: List[InputSpan],
                           span_outputs: Union[List[np.ndarray],
                                               List[List[int]]],
                          ) -> List[str]:
        num_spans = len(spans)
        if num_spans < 1:
            raise ValueError("Input spans is empty.")
        elif num_spans != len(span_outputs):
            raise ValueError("Input spans and Outputs are not the same size.")
        example_filenames = {s.filename for s in spans}
        if len(example_filenames) != 1:
            raise ValueError("Input spans are not from the same example!")
        span_indices = [s.index for s in spans]
        if len(span_indices) != len(set(span_indices)):
            raise ValueError("Input spans have duplicated entries!")
        for i, span_idx in enumerate(span_indices):
            if (i == 0 and span_idx != 0) \
                or (i > 0 and span_indices[i-1] + 1 != span_idx):
                raise ValueError("Input span is missing or spans are not ordered!")

        # Convert to numpy array
        output = span_outputs[0]
        if isinstance(output, list):
            for i in range(num_spans):
                span_outputs[i] = np.array(span_outputs[i], dtype=np.int32)
        elif not isinstance(output, np.ndarray):
            TypeError(f"Invalid type: {type(output)}")

        # Only get predicted tokens
        word_count = 0
        word_ids_to_concat = []
        output_to_concat = []
        for span, output in zip(spans, span_outputs):
            # Remove special, padding and suffix tokens
            mask = np.asarray(span.prediction_mask, dtype=np.bool_)
            partial_word_ids = span.word_ids[mask]
            output = output[mask]

            word_count += span.word_count
            word_ids_to_concat.append(partial_word_ids)
            output_to_concat.append(output)
        word_ids = np.concatenate(word_ids_to_concat)
        output = np.concatenate(output_to_concat)

        # Get prediction at word level
        example_output = -1 * np.ones(word_count, dtype=np.int32)
        example_output.put(word_ids, output)

        # If prediction mode:
        # * 'word': all words were predicted
        # * 'line': Only the first word of each line was predicted
        if self.prediction_mode == 'line':
            tag_id = example_output[0]
            for i in range(1, word_count):
                if example_output[i] >= 0:
                    tag_id = example_output[i]
                else:
                    example_output[i] = tag_id

        # Build full output and convert to tags
        example_output = self.tag_encoder.convert_ids_to_tags(example_output)

        # Word inside line should be 'I-' instead of 'B-'
        if self.prediction_mode == 'line':
            not_mapped_word_ids = set(range(word_count)) - set(word_ids)
            for word_id in not_mapped_word_ids:
                tag = example_output[word_id]
                if tag == 'O':
                    continue
                subtag, label = tag.split('-', 1)
                if subtag == 'B':
                    example_output[word_id] = 'I-' + label

        #
        segment = []
        segments = []
        for tag in example_output:
            if tag == 'O':
                segment.append(tag)
                continue

            subtag, label = tag.split('-', 1)
            if subtag == 'B' and len(segment) > 0:
                segments.append(segment)
                segment = []
            segment.append(label.replace('_Item', ''))
        if len(segment) > 0:
            segments.append(segment)

        labels_with_item_segmentation = ['Work_Experience', 'Education']

        segment_labels = []
        fixed_example_output = []
        for segment in segments:
            vals, counts = np.unique(segment, return_counts=True)
            index = np.argmax(counts)
            label = vals[index]

            last_label = segment_labels[-1] if len(segment_labels) > 0 else None
            for i in range(len(segment)):
                tag = ('B-' if i == 0 else 'I-') + label
                if not self.section_label_only \
                    and label in labels_with_item_segmentation \
                    and i == 0 and label == last_label:
                    tag += '_Item'
                fixed_example_output.append(tag)
            segment_labels.append(label)
        return fixed_example_output
