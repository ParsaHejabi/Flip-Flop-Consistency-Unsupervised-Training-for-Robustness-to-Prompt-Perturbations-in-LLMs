import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForSeq2Seq:
    """Custom seq2seq data collator that adds support for expanding nested batch lists."""

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    expand_list: bool = False

    def __call__(self, features, return_tensors: Optional[str] = None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        if self.expand_list:
            return self._collate_list(features, return_tensors)

        if isinstance(features[0], list):
            features = [feature for example in features for feature in example]

        if self._is_decoder_only_model():
            features = [self._prepare_decoder_only_feature(feature) for feature in features]
        else:
            features = [self._strip_prompt_metadata(feature) for feature in features]

        labels = [feature["labels"] for feature in features] if "labels" in features[0] else None
        padded_features = self._pad_features(features, labels, return_tensors)

        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=padded_features["labels"])
            padded_features["decoder_input_ids"] = decoder_input_ids

        return padded_features

    def _collate_list(self, list_of_features, return_tensors: str):
        results = []
        # expand_list is only used for nested mini-batches; assume a single nested example per batch
        list_of_features = list_of_features[0]
        for features in list_of_features:
            if self._is_decoder_only_model():
                features = [self._prepare_decoder_only_feature(feature) for feature in features]
            else:
                features = [self._strip_prompt_metadata(feature) for feature in features]
            labels = [feature["labels"] for feature in features] if "labels" in features[0] else None
            padded_features = self._pad_features(features, labels, return_tensors)

            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=padded_features["labels"])
                padded_features["decoder_input_ids"] = decoder_input_ids
            results.append(padded_features)

        return results

    def _pad_features(self, features, labels, return_tensors: str):
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    if padding_side == "right":
                        feature["labels"] = feature["labels"] + remainder
                    else:
                        feature["labels"] = remainder + feature["labels"]
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        return self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

    def _is_decoder_only_model(self) -> bool:
        if self.model is None:
            return False
        return not getattr(getattr(self.model, "config", None), "is_encoder_decoder", False)

    def _prepare_decoder_only_feature(self, feature: dict) -> dict:
        if "prompt_text" in feature and "label_text" in feature:
            return self._prepare_decoder_only_feature_with_chat_template(feature)
        return self._prepare_decoder_only_feature_from_ids(feature)

    def _prepare_decoder_only_feature_from_ids(self, feature: dict) -> dict:
        updated = dict(feature)
        self._strip_prompt_metadata(updated)

        prompt_ids = self._to_list(updated.get("input_ids", []))
        answer_ids = [tok for tok in self._to_list(updated.get("labels", [])) if tok != self.label_pad_token_id]

        attention = self._to_list(updated.get("attention_mask"))
        if not attention:
            attention = [1] * len(prompt_ids)

        input_ids = prompt_ids + answer_ids
        attention_mask = attention + [1] * len(answer_ids)
        labels = [self.label_pad_token_id] * len(prompt_ids) + answer_ids

        updated["input_ids"] = input_ids
        updated["attention_mask"] = attention_mask
        updated["labels"] = labels
        return updated

    def _prepare_decoder_only_feature_with_chat_template(self, feature: dict) -> dict:
        updated = dict(feature)
        prompt_text = updated.pop("prompt_text", None)
        label_text = updated.pop("label_text", None)
        chat_messages = updated.pop("chat_messages", None)

        if prompt_text is None or label_text is None:
            raise ValueError("decoder-only feature missing prompt or label text; cannot apply chat template")

        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not implement apply_chat_template; cannot format decoder-only prompts")

        messages = chat_messages if chat_messages is not None else [{"role": "user", "content": prompt_text}]
        prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prompt_str is None:
            raise ValueError("tokenizer.apply_chat_template returned None; check chat template configuration")

        continuation = self._format_label_continuation(prompt_str, label_text)
        full_text = prompt_str + continuation

        prompt_tokens = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
        rendered = self.tokenizer(full_text, add_special_tokens=False)
        input_ids = self._to_list(rendered.input_ids)
        if not input_ids:
            raise ValueError("Tokenization produced empty input_ids for chat-formatted example")

        label_start = len(prompt_tokens)
        if label_start > len(input_ids):
            raise ValueError("Prompt token length exceeded total sequence length; check chat template spacing")

        labels = [self.label_pad_token_id] * label_start + input_ids[label_start:]

        # Acceptance check: ensure prompt tokens are all ignored and label tokens retained
        if any(tok != self.label_pad_token_id for tok in labels[:label_start]):
            raise AssertionError("Prompt portion of labels is not masked with label_pad_token_id")
        if any(tok == self.label_pad_token_id for tok in labels[label_start:]):
            raise AssertionError("Label span contains masked tokens unexpectedly; verify continuation text")

        attention_mask = [1] * len(input_ids)

        updated["input_ids"] = input_ids
        updated["attention_mask"] = attention_mask
        updated["labels"] = labels

        if os.environ.get("FLIPFLOP_CHAT_DEBUG"):
            label_sample = input_ids[label_start : label_start + min(5, len(input_ids) - label_start)]
            logger.info(
                "chat-template tail=%r first_label_tokens=%r",
                prompt_str[-40:],
                self.tokenizer.decode(label_sample),
            )
        return updated

    @staticmethod
    def _to_list(tensor_like: Optional[Union[List[int], Any]]) -> List[int]:
        if tensor_like is None:
            return []
        if isinstance(tensor_like, list):
            return list(tensor_like)
        if hasattr(tensor_like, "tolist"):
            return list(tensor_like.tolist())
        return list(tensor_like)

    @staticmethod
    def _strip_prompt_metadata(feature: dict) -> dict:
        feature.pop("prompt_text", None)
        feature.pop("label_text", None)
        feature.pop("chat_messages", None)
        return feature

    @staticmethod
    def _format_label_continuation(prompt_str: str, label_text: str) -> str:
        if label_text is None:
            return ""
        if not label_text:
            return label_text
        if label_text[0].isspace():
            return label_text
        if not prompt_str:
            return " " + label_text
        return label_text if prompt_str[-1].isspace() else " " + label_text
