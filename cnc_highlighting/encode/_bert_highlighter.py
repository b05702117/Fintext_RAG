"""
Author: Jia-Huei (Dylan) Ju (dylanjootw@gmail.com)
Date: Nov. 23, 2023

The following adjustments are based on:
- Original CNC pipeline 
- Huggingface's `QA Pipeline`

References:
https://github.com/cnclabs/codes.fin.highlight/blob/main/highlighting/models.py
https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/question_answering.py#L225
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertModel, BertPreTrainedModel
from transformers import AutoTokenizer

class BertForHighlightPrediction(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, **model_kwargs):
        super().__init__(config)
        # self.model_args = model_kwargs.pop("model_args", None)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        ## additional linear layer
        self.dropout = nn.Dropout(classifier_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.tokens_clf = nn.Linear(config.hidden_size, config.num_labels)

        ## hyper-parameters
        self.tau = model_kwargs.pop('tau', 1)
        self.gamma = model_kwargs.pop('gamma', 1)
        self.soft_labeling = model_kwargs.pop('soft_labeling', False)
        self.pooling = model_kwargs.pop('pooling', 'max')

        ## inference
        self.tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)


        self.init_weights()

    def forward(self,
                input_ids=None,
                probs=None, # soft-labeling
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        tokens_output = outputs[0]
        highlight_logits = self.tokens_clf(self.dropout(tokens_output))

        loss_ce = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = highlight_logits.view(-1, self.num_labels)
            active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss_ce = loss_fct(active_logits, active_labels)

        return TokenClassifierOutput(
                loss=loss_ce,
                logits=highlight_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
        )

    def _pool_probs(self, probs, word_ids):
        ret = np.zeros(1+max(word_ids))
        for w_id, p in zip(word_ids, probs):
            if self.pooling == 'max':
                ret[w_id] = np.max([ret[w_id], p])
            if self.pooling == 'mean':
                ret[w_id] = np.mean([ret[w_id], p])
        return ret

    def encode(self, 
               text_tgt: List[str], 
               text_ref: Optional[List] = None,
               device: str = 'cpu',
               pretokenized: bool = True,
               return_reference: bool = False,
        ):

        if text_ref is None:
            text_ref = [self.tokenizer.pad_token] * len(text_tgt)

        if pretokenized is False:
            text_tgt = [t.split() for t in text_tgt]
            text_ref = [t.split() for t in text_ref]

        input_tokenized = self.tokenizer(
                text_ref, text_tgt,
                max_length=512,
                truncation=True,
                padding=True,
                is_split_into_words=True,
                return_tensors='pt'
        ).to(device)

        # encode
        with torch.no_grad():
            logits = self.forward(**input_tokenized).logits
            batch_probs = self.softmax(logits)[:, :, 1].detach().cpu()
            batch_probs = batch_probs.numpy()
            # preds = torch.argmax(probs, dim=-1)

        # word importance
        outputs = []
        for i, probs in enumerate(batch_probs):
            mapping = np.array(input_tokenized.word_ids(i))
            sep = np.argwhere(mapping==None).flatten()[1] - 1
            token_probs = probs[mapping!=None] 
            word_ids = mapping[mapping!=None]

            token_probs_ref = token_probs[:sep]
            token_probs_tgt = token_probs[sep:]
            word_ids_ref = word_ids[:sep]
            word_ids_tgt = word_ids[sep:]

            ret = {'words_tgt': text_tgt[i], 
                   'word_probs_tgt': self._pool_probs(
                       token_probs_tgt, word_ids_tgt
                    )}

            if return_reference:
                ret.update({
                    'words_ref': text_ref[i],
                    'word_probs_ref': self._pool_probs(
                        token_probs_ref, word_ids_ref
                    )
                })

            outputs.append(ret)


        return outputs
