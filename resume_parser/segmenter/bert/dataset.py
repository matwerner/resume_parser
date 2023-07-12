from typing import List, Dict, Optional, Union
from collections import defaultdict

import json
import os
import numpy as np
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm.auto import tqdm

from resume_parser.segmenter.dataset import ResumeSegmenterDataset
from resume_parser.segmenter.bert.dataset_utils import (
    InputComposer,
    InputExample,
    InputSpan
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _pad_sequences(seqs: List[np.ndarray],
                   pad_value: Optional[Union[int, List[int]]] = 0,
                   dtype: Optional[np.dtype] = np.int32
                  ) -> np.array:
    max_seq_length = max(len(seq) for seq in seqs)

    padded_seqs = []
    for seq in seqs:
        pad_length = max_seq_length-len(seq)
        if pad_length == 0:
            padded_seqs.append(seq)
        else:
            # Only pad first dimension
            npads = [(0, 0)] * len(seq.shape)
            npads[0] = (0, pad_length)
            # padded_seq = np.pad(seq, npads, mode='constant', constant_values=pad_value)
            padded_seq = np.pad(seq, npads, mode='constant', constant_values=0)
            padded_seq[-pad_length:] = pad_value
            padded_seqs.append(padded_seq)

    return np.array(padded_seqs, dtype=dtype)

class ResumeSegmenterDatasetForBert(ResumeSegmenterDataset):
    """Resume layout dataset."""

    def __init__(self,
                 root_dirpath: str,
                 filenames: List[str],
                 input_composer: InputComposer,
                 section_label_only: Optional[bool] = True,
                 is_cache_data: Optional[bool] = False,
                 verbose: Optional[bool] = False
                ) -> None:
        super().__init__(root_dirpath, filenames, section_label_only, verbose)
        self.input_composer = input_composer
        self.is_cache_data = is_cache_data
        self.verbose = verbose

        self.cache_examples = {}
        self.cache_spans = {}
        example_map, span_map = self._build_data_maps()
        self.example_map = example_map
        self.span_map = span_map


    def _get_spans(self, example: InputExample) -> List[InputSpan]:
        return self.input_composer.convert_example_into_spans(example)


    def _build_data_maps(self):
        example_map, span_map = [], []
        for filename in tqdm(self.filenames):
            example = self._get_example(filename)
            spans = self._get_spans(example)
            num_spans = len(spans)
            example_map.append(filename)
            span_map.extend([(filename, i) for i in range(num_spans)])
            if self.is_cache_data:
                self.cache_examples[filename] = example
                self.cache_spans[filename] = spans
        return example_map, span_map


    def __len__(self):
        return len(self.span_map)


    def __getitem__(self, idx) -> InputSpan:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename, span_idx = self.span_map[idx]
        if self.is_cache_data:
            return self.cache_spans[filename][span_idx]
        else:
            example = self._get_example(filename)
            return self._get_spans(example)[span_idx]


    def get_example(self, filename) -> InputExample:
        """"""
        if self.is_cache_data:
            return self.cache_examples[filename]
        return self._get_example(filename)


    def get_example_spans(self, filename) -> List[InputSpan]:
        """"""
        if self.is_cache_data:
            return self.cache_spans[filename]
        example = self._get_example(filename)
        return self._get_spans(example)


class ResumeSegmenterDataModuleForBert(pl.LightningDataModule):
    """DataModule used for semantic segmentation in geometric generalization
    project.
    """

    train: ResumeSegmenterDatasetForBert
    valid: ResumeSegmenterDatasetForBert
    test: ResumeSegmenterDatasetForBert

    def __init__(self,
                 config_filepath: str,
                 input_composer: InputComposer,
                 section_label_only: Optional[bool] = True,
                 is_cache_data: Optional[bool] = False,
                 train_batch_size: Optional[int] = 1,
                 test_batch_size: Optional[int] = 1,
                 verbose: Optional[bool] = False
                ) -> None:
        super().__init__()
        self.config = json.load(open(config_filepath, encoding='utf-8'))
        self.input_composer = input_composer
        self.section_label_only = section_label_only
        self.is_cache_data = is_cache_data
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.verbose = verbose


    def _get_dataset(self, set_name: str) -> ResumeSegmenterDatasetForBert:
        return ResumeSegmenterDatasetForBert(
            root_dirpath       = self.config['root_dirpath'],
            filenames          = self.config[set_name],
            input_composer     = self.input_composer,
            section_label_only = self.section_label_only,
            is_cache_data      = self.is_cache_data,
            verbose            = self.verbose
        )


    def setup(self, stage: Optional[str] = None):
        self.train = self._get_dataset('train')
        if stage == "predict" or stage is None:
            pass
        elif stage == "fit":
            self.valid = self._get_dataset('valid')
        elif stage == "test":
            self.test = self._get_dataset('test')
        else:
            raise ValueError(f'Invalid stage: {stage}')


    def _custom_collate(self, data: List[InputSpan]):
        filenames                       = [d.filename for d in data]
        span_indices                    = [d.index for d in data]
        sequence_lengths                = [len(d.input_ids) for d in data]
        input_ids_list                  = [d.input_ids for d in data]
        tokens_list                     = [d.tokens for d in data]
        token_type_ids_list             = [d.token_type_ids for d in data]
        attention_mask_list             = [d.attention_mask for d in data]
        context_mask_list               = [d.context_mask for d in data]
        prediction_mask_list            = [d.prediction_mask for d in data]
        label_ids_list                  = [d.label_ids for d in data]
        extra_classifier_features_list  = [d.extra_classifier_features for d in data]
        bert_features_list              = [d.bert_features for d in data]

        pad_token_id = self.input_composer.tokenizer.pad_token_id
        pad_token_type_id = self.input_composer.tokenizer.pad_token_type_id
        pad_label_id = -100

        batch = {
            # Meta attributes
            'filename':        np.array(filenames, dtype=np.object_),
            'span_index':      np.array(span_indices, dtype=np.int32),
            'sequence_length': np.array(sequence_lengths, dtype=np.int32),

            # Model attributes
            'input_ids':       _pad_sequences(input_ids_list, pad_token_id, np.int32),
            'tokens':          _pad_sequences(tokens_list, None, np.object_),
            'token_type_ids':  _pad_sequences(token_type_ids_list, pad_token_type_id, np.int32),
            'attention_mask':  _pad_sequences(attention_mask_list, 0, np.int32),

            # Additional attributes
            'extra_classifier_features': _pad_sequences(extra_classifier_features_list, 0., np.float32),

            #  For masking a tensor, variable must be Long or Bool
            'context_mask':    _pad_sequences(context_mask_list, False, np.bool_),
            'prediction_mask': _pad_sequences(prediction_mask_list, False, np.bool_),

            # CrossEntropyLoss not implement for Int, so use Long
            'label_ids':       _pad_sequences(label_ids_list, pad_label_id, np.int64),

            # For BERT
            'bert_features':   _pad_sequences(bert_features_list, 0., np.float32)
        }

        batch = {key: array if array.dtype == object else torch.tensor(array)
                 for key, array in batch.items()}
        return batch


    def train_dataloader(self) -> DataLoader:
        """Train dataset"""
        sampler = RandomSampler(range(len(self.train)))
        return DataLoader(self.train, batch_size=self.train_batch_size,
            num_workers=4, sampler=sampler, collate_fn=self._custom_collate)


    def val_dataloader(self) -> DataLoader:
        """Validation dataset"""
        sampler = SequentialSampler(range(len(self.valid)))
        return DataLoader(self.valid, batch_size=self.test_batch_size,
            num_workers=4, sampler=sampler, collate_fn=self._custom_collate)


    def test_dataloader(self) -> DataLoader:
        """Test dataset"""
        sampler = SequentialSampler(range(len(self.test)))
        return DataLoader(self.test, batch_size=self.test_batch_size,
            num_workers=4, sampler=sampler, collate_fn=self._custom_collate)


    def build_external_dataloader(self,
                                  dataset_dir: str,
                                  filenames: List[str]
                                 ) -> DataLoader:
        dataset = ResumeSegmenterDatasetForBert(
            root_dirpath       = dataset_dir,
            filenames          = filenames,
            section_label_only = self.section_label_only,
            input_composer     = self.input_composer,
            is_cache_data      = self.is_cache_data,
        )
        sampler = SequentialSampler(range(len(dataset)))
        return DataLoader(dataset, batch_size=self.test_batch_size,
            num_workers=4, sampler=sampler, collate_fn=self._custom_collate)


class FeatureExtractorDataset(ResumeSegmenterDatasetForBert):


    def __init__(self,
                 root_dirpath: str,
                 filenames: List[str],
                 input_composer: InputComposer,
                 model: torch.nn.Module,
                 device: torch.device = torch.device('cpu'),
                 fp16: bool = False,
                 batch_size: int = 256,
                 section_label_only: Optional[bool] = True,
                 verbose: Optional[bool] = False
                ) -> None:
        super().__init__(root_dirpath, filenames, input_composer, section_label_only, 
                         True, verbose)
        self.model = model
        self.device = device
        self.fp16 = fp16
        self.batch_size = batch_size
        self._compute_bert_features()


    def _custom_collate(self, data: List[InputSpan]):
        input_ids_list                  = [d.input_ids for d in data]
        token_type_ids_list             = [d.token_type_ids for d in data]
        attention_mask_list             = [d.attention_mask for d in data]

        pad_token_id = self.input_composer.tokenizer.pad_token_id
        pad_token_type_id = self.input_composer.tokenizer.pad_token_type_id

        batch = {
            # Model attributes
            'input_ids':       _pad_sequences(input_ids_list, pad_token_id, np.int32),
            'token_type_ids':  _pad_sequences(token_type_ids_list, pad_token_type_id, np.int32),
            'attention_mask':  _pad_sequences(attention_mask_list, 0, np.int32),
        }

        batch = {key: array if array.dtype == object else torch.tensor(array)
                 for key, array in batch.items()}
        return batch


    def _compute_bert_features(self) -> None:
        self.model.eval()

        with torch.no_grad():
            for _, spans in self.cache_spans.items():
                batch = self._custom_collate(spans)

                # Unpack batch
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)

                if self.fp16:
                    with torch.cuda.amp.autocast():
                        outs = self.model.bert_encode(input_ids, token_type_ids, attention_mask)
                else:
                    outs = self.model.bert_encode(input_ids, token_type_ids, attention_mask)

                features_batch = outs.cpu().detach().clone().numpy()
                for span, features in zip(spans, features_batch):
                    length = len(span.input_ids)
                    span.bert_features = features[:length]


class FeatureExtractorDataModule(ResumeSegmenterDataModuleForBert):
    """DataModule used for semantic segmentation in geometric generalization
    project.
    """

    train: FeatureExtractorDataset
    valid: FeatureExtractorDataset
    test: FeatureExtractorDataset

    def __init__(self,
                 config_filepath: str,
                 input_composer: InputComposer,
                 model: torch.nn.Module,
                 device: torch.device,
                 fp16: bool = False,
                 section_label_only: Optional[bool] = True,
                 train_batch_size: Optional[int] = 1,
                 test_batch_size: Optional[int] = 1,
                 verbose: Optional[bool] = False,
                ) -> None:
        super().__init__(config_filepath, input_composer, section_label_only, True,
                         train_batch_size, test_batch_size, verbose)
        self.model = model
        self.device = device
        self.fp16 = fp16


    def _get_dataset(self, set_name: str) -> FeatureExtractorDataset:
        return FeatureExtractorDataset(
            root_dirpath       = self.config['root_dirpath'],
            filenames          = self.config[set_name],
            input_composer     = self.input_composer,
            model              = self.model,
            device             = self.device,
            fp16               = self.fp16,
            batch_size         = self.test_batch_size,
            section_label_only = self.section_label_only,
            verbose            = self.verbose
        )


class ResumeSegmenterDatasetForBertPerExample(ResumeSegmenterDatasetForBert):


    def __init__(self,
                 root_dirpath: str,
                 filenames: List[str],
                 input_composer: InputComposer,
                 section_label_only: Optional[bool] = True,
                 is_cache_data: Optional[bool] = False,
                 verbose: Optional[bool] = False
                ) -> None:
        super().__init__(root_dirpath, filenames, input_composer, section_label_only,
                         is_cache_data, verbose)


    def __len__(self):
        return len(self.example_map)


    def __getitem__(self, idx) -> List[InputSpan]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.example_map[idx]
        if self.is_cache_data:
            return self.cache_spans[filename]
        example = self._get_example(filename)
        return self._get_spans(example)


class ResumeSegmenterDataModuleForBertPerExample(ResumeSegmenterDataModuleForBert):


    def _get_dataset(self, set_name: str) -> ResumeSegmenterDatasetForBert:
        return ResumeSegmenterDatasetForBertPerExample(
            root_dirpath       = self.config['root_dirpath'],
            filenames          = self.config[set_name],
            input_composer     = self.input_composer,
            section_label_only = self.section_label_only,
            is_cache_data      = self.is_cache_data,
            verbose            = self.verbose
        )


    def _custom_collate(self, data: List[List[InputSpan]]):
        flat_data = []
        for d in data:
            flat_data.extend(d)
        return super()._custom_collate(flat_data)


    def build_external_dataloader(self,
                                  dataset_dir: str,
                                  filenames: List[str]
                                 ) -> DataLoader:
        dataset = ResumeSegmenterDatasetForBertPerExample(
            root_dirpath       = dataset_dir,
            filenames          = filenames,
            section_label_only = self.section_label_only,
            input_composer     = self.input_composer,
            is_cache_data      = self.is_cache_data,
        )
        sampler = SequentialSampler(range(len(dataset)))
        return DataLoader(dataset, batch_size=self.test_batch_size,
            num_workers=4, sampler=sampler, collate_fn=self._custom_collate)


def main() -> None:
    resources_dirpath = '/home/mwerner/Git/resume_parser/resources/segmenter'
    config_filepath = f'{resources_dirpath}/split_0.conf'
    # with open(config_filepath, encoding='utf-8') as fp:
    #     config = json.load(fp)
    # root_dirpath = config['root_dirpath']
    # train_filenames = config['train']

    from resume_parser.segmenter.bert.dataset_utils import NERTagEncoder
    tag_encoder = NERTagEncoder(['Personal_Data', 'Objective', 'About',
        'Education', 'Work_Experience', 'Other'], ignore_index = -100)

    model_name = 'neuralmind/bert-base-portuguese-cased'
    # transformers_cache_dir = '/home/mwerner/.cache/huggingface/transformers'
    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_composer = InputComposer(
	    tag_encoder               = tag_encoder,
	    tokenizer                 = tokenizer,
	    max_seq_length            = 384,
        seq_context_length        = 128,
	    is_training               = True,
        extra_classifier_features = True
	)

    data_module = ResumeSegmenterDataModuleForBert(
        config_filepath,
        input_composer     = input_composer,
        is_cache_data      = False,
        train_batch_size   = 2,
        test_batch_size    = 2,
    )
    data_module.setup("fit")

    train_loader = data_module.train_dataloader()
    print(len(train_loader))

    for batch in tqdm(train_loader, total=len(train_loader)):
        print(batch['input_ids'].shape)
        print(batch['input_ids'][0].shape)
        print(batch['label_ids'])
        filename = batch['filename'][0]
        tokens = batch['tokens'][0]
        label_ids = batch['label_ids'][0]
        classifier_features = batch['extra_classifier_features'][0]
        example = train_loader.dataset.get_example(filename)

        print(filename)
        for i, (label_id, token, features) in enumerate(zip(label_ids, tokens, classifier_features)):
            print(i, label_id, len(features), token)
        print(batch['extra_classifier_features'])
        print(batch['extra_classifier_features'].shape)
        break
        # for key, data in batch.items():
        #     print(key, data.shape)
        # break

    from collections import Counter

    bold_counter = Counter()
    italic_counter = Counter()

    for filename in train_loader.dataset.filenames:
        example = train_loader.dataset.get_example(filename)
        bold_counter.update([word.bold for word in example.word_sequence])
        italic_counter.update([word.italic for word in example.word_sequence])
    print(bold_counter.most_common())
    print(italic_counter.most_common())


if __name__ == '__main__':
    main()
