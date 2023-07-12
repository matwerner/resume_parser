# REFERENEC:
# https://github.com/neuralmind-ai/portuguese-bert/blob/master/ner_evaluation/model.py

"""Implementations of BERT, BERT-CRF, BERT-LSTM and BERT-LSTM-CRF models."""

import logging
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple, Type

import os
import torch
from torch import nn
from transformers import BertConfig, BertForTokenClassification
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from torchcrf import CRF

LOGGER = logging.getLogger(__name__)


def sum_last_4_layers(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Sums the last 4 hidden representations of a sequence output of BERT.
    Args:
    -----
    sequence_output: Tuple of tensors of shape (batch, seq_length, hidden_size).
        For BERT base, the Tuple has length 13.

    Returns:
    --------
    summed_layers: Tensor of shape (batch, seq_length, hidden_size)
    """
    last_layers = sequence_outputs[-4:]
    return torch.stack(last_layers, dim=0).sum(dim=0)


def get_last_layer(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Returns the last tensor of a list of tensors."""
    return sequence_outputs[-1]


def concat_last_4_layers(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Concatenate the last 4 tensors of a tuple of tensors."""
    last_layers = sequence_outputs[-4:]
    return torch.cat(last_layers, dim=-1)


POOLERS = {
    'sum': sum_last_4_layers,
    'last': get_last_layer,
    'concat': concat_last_4_layers,
}


def get_model_and_kwargs_for_args(args: Namespace) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Given the parsed arguments, returns the correct model class and model
    args.

    Args:
        args: a Namespace object (from parsed argv command).
        training: if True, sets a high initialization value for classifier bias
            parameter after model initialization.
    """

    # Model configuration
    model_args = {'pooler': args.pooler}
    if args.extra_classifier_features:
        model_args['num_extra_classifier_features'] = args.num_extra_classifier_features

    # Models available
    if args.use_lstm:
        model_args['lstm_hidden_size'] = args.lstm_hidden_size
        model_args['lstm_layers'] = args.lstm_layers
        if args.use_crf:
            model_class = BertLSTMCRF
        else:
            model_class = BertLSTM
    else:
        if args.use_crf:
            model_class = BertCRFForNERClassification
        else:
            model_class = BertForNERClassification

    return model_class, model_args


def save_model(model: Type[torch.nn.Module], args: Namespace) -> None:
    """Save a trained model and the associated configuration to output dir."""
    model.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


def load_model(args: Namespace, model_path: str) -> torch.nn.Module:
    """Instantiates a pretrained model from parsed argument values.
    Args:
        args: parsed arguments from argv.
        model_path: name of model checkpoint or path to a checkpoint directory.
        training: if True, loads a model with training-specific parameters.
    """

    model_class, model_kwargs = get_model_and_kwargs_for_args(args)
    # logger.info('model: {}, kwargs: {}'.format(model_class.__name__, model_kwargs))

    model = model_class.from_pretrained(
        model_path,
        num_labels=args.num_labels,
        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
        output_hidden_states=True,  # Ensure all hidden states are returned
        **model_kwargs)

    return model


class BertForNERClassification(BertForTokenClassification):
    """BERT model for NER task.

    The number of NER tags should be defined in the `BertConfig.num_labels`
    attribute.

    Args:
        config: BertConfig instance to build BERT model.
        pooler: which pooler configuration to use to pass BERT features to the
            classifier.
    """

    def __init__(self,
                 config: BertConfig,
                 num_extra_classifier_features: Optional[int] = 0,
                 pooler: Optional[str] ='last',
                ):
        super().__init__(config)

        # Build new classifier layer
        if hasattr(self, 'classifier'):
            del self.classifier
        elif hasattr(self, 'cls'):
            del self.cls
        assert num_extra_classifier_features >= 0
        self._build_classifier(config, pooler, num_extra_classifier_features)
        self.loss_fct = torch.nn.CrossEntropyLoss()

        # Pooler
        if pooler not in POOLERS:
            raise ValueError(f"Invalid pooler: {pooler}. "
                f"Pooler must be one of {list(POOLERS.keys())}.")
        self.pooler = POOLERS.get(pooler)
        self.is_bert_frozen = False
        self.is_bert_features_precomputed = False


    def _build_classifier(self, config, pooler, num_extra_classifier_features):
        """Build tag classifier."""
        if pooler in ('last', 'sum'):
            num_input_features = config.hidden_size + num_extra_classifier_features
        else:
            assert pooler == 'concat'
            num_input_features = 4*config.hidden_size + num_extra_classifier_features
        self.classifier = torch.nn.Linear(num_input_features, config.num_labels)


    def freeze_bert(self, is_freeze: bool=True):
        """Whether to freeze all BERT parameters. Only the classifier
        and new embeddings weights will be updated."""
        for p in self.bert.parameters():
            p.requires_grad = not is_freeze
        self.is_bert_frozen = is_freeze


    def precomputed_bert_features(self, is_precomputed: bool=False):
        """Whether inputs were precompute beforehand. If True, the BERT features
        will be considered precomputed. This is useful when training
        multiple models with the same BERT features."""
        self.is_bert_features_precomputed = is_precomputed


    def bert_encode(self,
                    input_ids: torch.LongTensor,
                    token_type_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.LongTensor] = None
                   ):
        """Gets encoded sequence from BERT model and pools the layers accordingly.
        BertModel outputs a tuple whose elements are:
        1- Last encoder layer output. Tensor of shape (B, S, H)
        2- Pooled output of the [CLS] token. Tensor of shape (B, H)
        3- Encoder inputs (embeddings) + all Encoder layers' outputs. This
            requires the flag `output_hidden_states=True` on BertConfig. Returns
            List of tensors of shapes (B, S, H).
        4- Attention results, if `output_attentions=True` in BertConfig.

        This method uses just the 3rd output and pools the layers.
        """

        input_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        output = self.bert(
            inputs_embeds=input_embeddings,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Use the defined pooler to pool the hidden representation layers
        sequence_output = self.pooler(output.hidden_states)

        return sequence_output


    def predict_logits(self,
                       input_ids: torch.LongTensor,
                       extra_classifier_features: Optional[torch.LongTensor] = None,
                       token_type_ids: Optional[torch.LongTensor] = None,
                       attention_mask: Optional[torch.LongTensor] = None,
                      ):
        """Returns the logits prediction from BERT + classifier."""
        if self.is_bert_features_precomputed:
            sequence_output = input_ids
        else:
            sequence_output = self.bert_encode(input_ids, token_type_ids, attention_mask)
        if extra_classifier_features is not None and extra_classifier_features.nelement() > 0:
            sequence_output = torch.cat((sequence_output, extra_classifier_features), dim=2)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # (batch, seq, tags)
        return logits


    def forward(self,
                input_ids: torch.LongTensor,
                extra_classifier_features: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                context_mask: Optional[torch.LongTensor] = None,
                prediction_mask: Optional[torch.LongTensor] = None,
               ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the network.

        If `labels` are not None, it will calculate and return the loss.
        Otherwise, it will return the logits and predicted tags tensors.

        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns a dict with calculated tensors:
          - "logits"
          - "y_pred"
          - "loss" (if `labels` is not `None`)
        """
        outputs = {}
        logits = self.predict_logits(input_ids=input_ids,
                                     extra_classifier_features=extra_classifier_features,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask)
        _, y_pred = torch.max(logits, dim=-1)
        y_pred = y_pred.cpu().numpy()
        outputs['logits'] = logits
        outputs['y_pred'] = y_pred

        if labels is not None:
            # Only keep active parts of the loss
            mask = prediction_mask
            if mask is not None:
                mask = mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[mask]
                active_labels = labels.view(-1)[mask]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs['loss'] = loss

        return outputs


class BertCRFForNERClassification(BertForNERClassification):
    """BERT-CRF model.
    Args:
        config: BertConfig instance to build BERT model.
        kwargs: arguments to be passed to superclass.
    """

    def __init__(self, config: BertConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        del self.loss_fct  # Delete unused CrossEntropyLoss
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)


    def forward(self,
                input_ids: torch.LongTensor,
                extra_classifier_features: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                context_mask: Optional[torch.LongTensor] = None,
                prediction_mask: Optional[torch.LongTensor] = None,
               ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the network.
        If `labels` is not `None`, it will calculate and return the the loss,
        that is the negative log-likelihood of the batch.
        Otherwise, it will calculate the most probable sequence outputs using
        Viterbi decoding and return a list of sequences (List[List[int]]) of
        variable lengths.
        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).
        Returns a dict with calculated tensors:
          - "logits"
          - "loss" (if `labels` is not `None`)
          - "y_pred" (if `labels` is `None`)
        """
        outputs = {}

        logits = self.predict_logits(input_ids=input_ids,
                                     extra_classifier_features=extra_classifier_features,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask)
        outputs['logits'] = logits

        # mask: mask padded sequence and also subtokens, because they must
        # not be used in CRF.
        # mask = prediction_mask
        batch_size = logits.shape[0]
        seq_size = prediction_mask.shape[1]

        if labels is not None:
            # Negative of the log likelihood.
            # Loop through the batch here because of 2 reasons:
            # 1- the CRF package assumes the mask tensor cannot have interleaved
            # zeros and ones. In other words, the mask should start with True
            # values, transition to False at some moment and never transition
            # back to True. That can only happen for simple padded sequences.
            # 2- The first column of mask tensor should be all True, and we
            # cannot guarantee that because we have to mask all non-first
            # subtokens of the WordPiece tokenization.
            # device = logits.get_device()
            # loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss = 0
            for seq_logits, seq_labels, ctx_mask, pred_mask in zip(logits,
                                                                   labels,
                                                                   context_mask,
                                                                   prediction_mask):
                #
                if not pred_mask.any():
                    continue
                seq_mask = ctx_mask | pred_mask

                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(seq_logits, seq_labels, reduction='token_mean')

            loss /= batch_size
            outputs['loss'] = loss

        # Same reasons for iterating
        output_tags = []
        for seq_logits, ctx_mask, pred_mask in zip(logits,
                                                   context_mask,
                                                   prediction_mask):
            seq_tags = torch.zeros(seq_size, dtype=torch.long)
            if pred_mask.any():
                seq_mask = ctx_mask | pred_mask
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                tags = self.crf.decode(seq_logits)
                # Unpack "batch" results
                seq_tags[seq_mask] = torch.tensor(tags[0], dtype=torch.long)
                seq_tags[ctx_mask] = 0
            output_tags.append(seq_tags.unsqueeze(0))

        outputs['y_pred'] = torch.cat(output_tags, dim=0)

        return outputs


class BertLSTM(BertForNERClassification):
    """BERT model with an LSTM model as classifier. This model is meant to be
    used with frozen BERT schemes (feature-based).

    Args:
        config: BertConfig instance to build BERT model.
        lstm_hidden_size: hidden size of LSTM layers. Defaults to 100.
        lstm_layers: number of LSTM layers. Defaults to 1.
        kwargs: arguments to be passed to superclass.
    """

    def __init__(self,
                 config: BertConfig,
                 num_extra_classifier_features: Optional[int] = 0,
                 pooler: Optional[str] ='last',
                 lstm_hidden_size: Optional[int] = 100,
                 lstm_layers: Optional[int] = 1,
                 **kwargs: Any):

        lstm_dropout = 0.2 if lstm_layers > 1 else 0
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        super().__init__(config, num_extra_classifier_features, pooler, **kwargs)

        if pooler in ('last', 'sum'):
            lstm_input_size = config.hidden_size + num_extra_classifier_features
        else:
            assert pooler == 'concat'
            lstm_input_size = 4 * config.hidden_size + num_extra_classifier_features

        self.lstm = torch.nn.LSTM(input_size    = lstm_input_size,
                                  hidden_size   = lstm_hidden_size,
                                  num_layers    = lstm_layers,
                                  dropout       = lstm_dropout,
                                  batch_first   = True,
                                  bidirectional = True)


    def _build_classifier(self, config, pooler, num_extra_classifier_features):
        """Build label classifier."""
        self.classifier = torch.nn.Linear(2 * self.lstm_hidden_size, config.num_labels)


    def _pack_bert_encoded_sequence(self, encoded_sequence, attention_mask):
        """Returns a PackedSequence to be used by LSTM.

        The encoded_sequence is the output of BERT, of shape (B, S, H).
        This method sorts the tensor by sequence length using the
        attention_mask along the batch dimension. Then it packs the sorted
        tensor.

        Args:
        -----
        encoded_sequence (tensor): output of BERT. Shape: (B, S, H)
        attention_mask (tensor): Shape: (B, S)

        Returns:
        --------
        sorted_encoded_sequence (tensor): sorted `encoded_sequence`.
        sorted_ixs (tensor): tensor of indices returned by `torch.sort` when
            performing the sort operation. These indices can be used to unsort
            the output of the LSTM.
        """
        seq_lengths = attention_mask.sum(dim=1)   # Shape: (B,)
        sorted_lengths, sort_ixs = torch.sort(seq_lengths, descending=True)

        sorted_encoded_sequence = encoded_sequence[sort_ixs, :, :]

        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_encoded_sequence,
            sorted_lengths.cpu(),
            batch_first=True)

        return packed_sequence, sort_ixs


    def _unpack_lstm_output(self, packed_sequence, sort_ixs):
        """Unpacks and unsorts a sorted PackedSequence that is output by LSTM.

        Args:
            packed_sequence (PackedSequence): output of LSTM. Shape: (B, S, Hl)
            sort_ixs (tensor): the indexes of be used for unsorting. Shape: (B,)

        Returns:
            The unsorted sequence.
        """
        B = len(sort_ixs)

        # Unpack
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=True)

        assert unpacked.shape <= (B, 512, 2 * self.lstm.hidden_size)

        # Prepare indices for unsort
        sort_ixs = sort_ixs.unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
        # (B, S, Hl)
        sort_ixs = sort_ixs.expand(-1, unpacked.shape[1], unpacked.shape[2])
        # Unsort
        unsorted_sequence = torch.zeros_like(unpacked).scatter_(0, sort_ixs, unpacked)

        return unsorted_sequence


    def forward_lstm(self, bert_encoded_sequence, attention_mask):
        packed_sequence, sorted_ixs = self._pack_bert_encoded_sequence(
            bert_encoded_sequence, attention_mask)

        packed_lstm_out, _ = self.lstm(packed_sequence)
        lstm_out = self._unpack_lstm_output(packed_lstm_out, sorted_ixs)

        return lstm_out


    def forward(self,
                input_ids: torch.LongTensor,
                extra_classifier_features: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                context_mask: Optional[torch.LongTensor] = None,
                prediction_mask: Optional[torch.LongTensor] = None,
               ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the network.

        Computes the logits, predicted tags and if `labels` is not None, it will
        it will calculate and return the the loss, that is, the negative
        log-likelihood of the batch.

        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns:
            A dict with calculated tensors:
            - "logits"
            - "y_pred"
            - "loss" (if `labels` is not `None`)
        """
        outputs = {}

        if self.is_bert_features_precomputed:
            sequence_output = input_ids
        else:
            sequence_output = self.bert_encode(input_ids, token_type_ids, attention_mask)
        if extra_classifier_features is not None and extra_classifier_features.nelement() > 0:
            sequence_output = torch.cat((sequence_output, extra_classifier_features), dim=2)
        sequence_output = self.dropout(sequence_output)

        lstm_out = self.forward_lstm(sequence_output, attention_mask)
        sequence_output = self.dropout(lstm_out)

        logits = self.classifier(sequence_output)
        _, y_pred = torch.max(logits, dim=-1)
        y_pred = y_pred.cpu().numpy()
        outputs['logits'] = logits
        outputs['y_pred'] = y_pred

        if labels is not None:
            # Only keep active parts of the loss
            mask = prediction_mask
            if mask is not None:
                # Adjust mask and labels to have the same length as logits
                mask = mask[:, :logits.size(1)].contiguous()
                labels = labels[:, :logits.size(1)].contiguous()

                mask = mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[mask]
                active_labels = labels.view(-1)[mask]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs['loss'] = loss

        return outputs


class BertLSTMCRF(BertLSTM):
    """BERT model with an LSTM-CRF as classifier. This model is meant to be
    used with frozen BERT schemes (feature-based).

    Args:
        config: BertConfig instance to build BERT model.
        kwargs: arguments to be passed to superclass (see BertLSTM).
    """

    def __init__(self, config: BertConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)


    def forward(self,
                input_ids: torch.LongTensor,
                extra_classifier_features: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                context_mask: Optional[torch.LongTensor] = None,
                prediction_mask: Optional[torch.LongTensor] = None,
               ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the network.

        If `labels` are not None, it will calculate and return the the loss,
        that is the negative log-likelihood of the batch.
        Otherwise, it will calculate the most probable sequence outputs using
        Viterbi decoding and return a list of sequences (List[List[int]]) of
        variable lengths.

        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns:
            A dict with calculated tensors:

            - "logits"
            - "loss" (if `labels` is not `None`)
            - "y_pred" (if `labels` is `None`)
        """
        outputs = {}

        if self.is_bert_features_precomputed:
            sequence_output = input_ids
        else:
            sequence_output = self.bert_encode(input_ids, token_type_ids, attention_mask)        
        if extra_classifier_features is not None and extra_classifier_features.nelement() > 0:
            sequence_output = torch.cat((sequence_output, extra_classifier_features), dim=2)
        sequence_output = self.dropout(sequence_output)

        lstm_out = self.forward_lstm(sequence_output, attention_mask)
        sequence_output = self.dropout(lstm_out)
        logits = self.classifier(sequence_output)
        outputs['logits'] = logits

        # mask: mask padded sequence and also subtokens, because they must
        # not be used in CRF.
        # mask = prediction_mask
        context_mask = context_mask[:, :logits.size(1)].contiguous()
        prediction_mask = prediction_mask[:, :logits.size(1)].contiguous()
        batch_size = logits.shape[0]
        seq_size = prediction_mask.shape[1]

        if labels is not None:
            # Negative of the log likelihood.
            # Loop through the batch here because of 2 reasons:
            # 1- the CRF package assumes the mask tensor cannot have interleaved
            # zeros and ones. In other words, the mask should start with True
            # values, transition to False at some moment and never transition
            # back to True. That can only happen for simple padded sequences.
            # 2- The first column of mask tensor should be all True, and we
            # cannot guarantee that because we have to mask all non-first
            # subtokens of the WordPiece tokenization.
            # device = logits.get_device()
            # loss = torch.tensor(0.0, device=device, requires_grad=True)
            labels = labels[:, :logits.size(1)].contiguous()

            loss = 0
            for seq_logits, seq_labels, ctx_mask, pred_mask in zip(logits,
                                                                   labels,
                                                                   context_mask,
                                                                   prediction_mask):
                #
                if not pred_mask.any():
                    continue
                seq_mask = ctx_mask | pred_mask

                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(seq_logits, seq_labels, reduction='token_mean')

            loss /= batch_size
            outputs['loss'] = loss

        # Same reasons for iterating
        output_tags = []
        for seq_logits, ctx_mask, pred_mask in zip(logits,
                                                   context_mask,
                                                   prediction_mask):
            seq_tags = torch.zeros(seq_size, dtype=torch.long)
            if pred_mask.any():
                seq_mask = ctx_mask | pred_mask
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                tags = self.crf.decode(seq_logits)
                # Unpack "batch" results
                seq_tags[seq_mask] = torch.tensor(tags[0], dtype=torch.long)
                seq_tags[ctx_mask] = 0
            output_tags.append(seq_tags.unsqueeze(0))

        outputs['y_pred'] = torch.cat(output_tags, dim=0)

        return outputs
