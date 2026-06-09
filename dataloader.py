import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple, NamedTuple

import datasets
import numpy as np
from promptsource.templates import DatasetTemplates
from torch.utils.data import Dataset


class RenderedPrompt(NamedTuple):
    prompt_name: str
    input_text: str
    choices: Tuple[str, ...]
    gold_index: int

logger = logging.getLogger(__name__)


class DatasetByPrompt(Dataset):
    # used for test, maybe need to extend this class for open-ended generation tasks
    def __init__(self, args, cache_dir, tokenizer, split=None, hold_out=-1, random_hold_out=True, testdev_set=False, prompt_names=None, subset_indices=None):
        super().__init__()
        self.cache_dir = cache_dir

        self.split = split
        self.testdev_set = testdev_set  # if True, always use all prompts for ensemble predictions
        self.DATASET_NAME = args.dataset_name
        self.SUBSET_NAME = args.subset_name if args.subset_name != "none" else None
        self.TESTSET_NAME = args.testset_name
        self.PROMPTSET_NAME = args.prompt_set_name  # has subset name?
        self.task_type = args.task_type
        self.hold_out = hold_out
        self.random_hold_out = random_hold_out  # if False, use the first "hold_out" number of samples in the data
        self.cb_surgery = args.cb_surgery
        # Optional explicit subset selection (indices into the loaded split)
        self.subset_indices = subset_indices
        self.load()

        self.tokenizer = tokenizer
        self.abl_nprompts = args.abl_nprompts
        self.prompts = DatasetTemplates(self.PROMPTSET_NAME, self.SUBSET_NAME)
        if prompt_names is None:
            self.original_task_prompts = self.extract_original_task_prompts(check_valid_prompts=self.SUBSET_NAME == "copa")
        else:
            self.original_task_prompts = prompt_names
        self.construct_meta_info()
        self._rendered_prompt_cache: Dict[int, List[RenderedPrompt]] = {}
        self._label_cache: Dict[int, int] = {}
        self._tokenized_cache: Dict[int, Tuple[List[dict], int]] = {}
        self._raw_id_cache: Dict[int, object] = {}
        self.num_choices = len(self.prompts[self.original_task_prompts[0]].get_answer_choices_list(self.dataset[0]))
        print(
            "{} has {} original task prompts, number choices = {}, total examples = {}".format(
                self.DATASET_NAME + ("/" + self.SUBSET_NAME) if self.SUBSET_NAME is not None else "", len(self.original_task_prompts), self.num_choices, len(self)
            )
        )

    def load(self):
        if self.DATASET_NAME == "story_cloze":
            val_path = os.path.join(self.cache_dir, "cloze_2016_val.csv")
            test_path = os.path.join(self.cache_dir, "cloze_2016_test.csv")

            def _load_local_csv(path, split_name):
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Expected local Story Cloze {split_name} file at '{path}', but it was not found."
                    )
                return datasets.load_dataset(
                    "csv",
                    data_files={"train": path},
                    cache_dir=self.cache_dir,
                )["train"]

            split_key = (self.split if self.split is not None else self.TESTSET_NAME or "test").lower()
            if split_key in {"train", "validation", "dev"}:
                self.dataset = _load_local_csv(val_path, "validation")
            elif split_key == "test":
                self.dataset = _load_local_csv(test_path, "test")
            else:
                raise ValueError(
                    f"Unsupported split '{split_key}' for Story Cloze; expected one of 'train', 'validation', 'dev', or 'test'."
                )
        else:
            self.dataset = datasets.load_dataset(self.DATASET_NAME, self.SUBSET_NAME, cache_dir=self.cache_dir)[self.TESTSET_NAME if self.split is None else self.split]

        if self.hold_out > -1 and len(self.dataset) > self.hold_out:
            selected_data = np.random.choice(len(self.dataset), self.hold_out, replace=False) if self.random_hold_out else range(self.hold_out)
            self.dataset = [self.dataset[int(sidx)] for sidx in selected_data]
        # Apply explicit subset selection if provided (ensures disjoint splits when coordinated externally)
        if self.subset_indices is not None:
            self.dataset = [self.dataset[int(sidx)] for sidx in self.subset_indices]

    @property
    def num_prompts(self):
        return len(self.original_task_prompts)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx in self._tokenized_cache:
            return self._tokenized_cache[idx]

        prompt_views, label = self._prepare_rendered_prompts(idx)
        inputs = []
        outputs = []
        for view in prompt_views:
            for answer in view.choices:
                inputs.append(view.input_text)
                outputs.append(answer)

        if self.SUBSET_NAME == "cb" and self.cb_surgery:
            model_inputs = self.tokenizer(inputs, padding=False, truncation=True)
        else:
            model_inputs = self.tokenizer(inputs, padding=False, truncation=True, add_special_tokens=False)
        outputs_ids = self.tokenizer(outputs, padding=False, truncation=True).input_ids
        model_inputs["labels"] = [[l if l != self.tokenizer.pad_token_id else -100 for l in x] for x in outputs_ids]

        results = []
        for input_id, amask, label_ids, prompt_text, answer_text in zip(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["labels"],
            inputs,
            outputs,
        ):
            results.append(
                {
                    "input_ids": input_id,
                    "attention_mask": amask,
                    "labels": label_ids,
                    "prompt_text": prompt_text,
                    "label_text": answer_text,
                }
            )

        cached = (results, label)
        self._tokenized_cache[idx] = cached
        return cached

    def _prepare_rendered_prompts(self, idx):
        if idx not in self._rendered_prompt_cache:
            raw_ex = self.dataset[idx]
            prompt_views: List[RenderedPrompt] = []
            gold_label = None
            for pname in self.original_task_prompts:
                input_template, output_template = self.prompts[pname].apply(raw_ex)
                if self.task_type == "classification":
                    targets = [ans.strip() for ans in self.prompts[pname].get_answer_choices_list(raw_ex)]
                    label_idx = targets.index(output_template.strip())
                    if gold_label is None:
                        gold_label = label_idx
                    else:
                        assert gold_label == label_idx
                    self.set_num_choices(len(targets))
                    prompt_views.append(RenderedPrompt(pname, input_template.strip(), tuple(targets), label_idx))
                else:
                    # todo: for generation
                    pass
            self._rendered_prompt_cache[idx] = prompt_views
            self._label_cache[idx] = gold_label
        return self._rendered_prompt_cache[idx], self._label_cache[idx]

    def get_rendered_prompts(self, idx):
        prompts, _ = self._prepare_rendered_prompts(idx)
        return prompts

    def get_label(self, idx):
        _, label = self._prepare_rendered_prompts(idx)
        return label

    def get_example_id(self, idx):
        if idx not in self._raw_id_cache:
            self._raw_id_cache[idx] = self._extract_id_from_raw(self.dataset[idx])
        return self._raw_id_cache[idx]

    @staticmethod
    def _extract_id_from_raw(raw_ex):
        if raw_ex is None:
            return None
        for key in ["id", "guid", "idx", "pairID", "example_id", "instance_id"]:
            if key in raw_ex:
                return raw_ex[key]
        for k in raw_ex.keys():
            if "id" in k.lower():
                return raw_ex[k]
        return None

    # Provide an explicit iterator to avoid Python's sequence-iteration
    # fallback, which relies on IndexError to signal termination and can
    # surface as noisy (but benign) exceptions in debuggers.
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def construct_meta_info(self):
        answer_groups = defaultdict(list)

        for pidx, pname in enumerate(self.original_task_prompts):
            answer_choices = self.prompts[pname].get_fixed_answer_choices_list()
            answer_choices = "_".join(answer_choices) if answer_choices is not None else None
            answer_groups[answer_choices].append(pidx)
        self.prompt_groups = [answer_groups[key] for key in answer_groups.keys()]

    def set_num_choices(self, n):
        check = True if self.num_choices == -1 else self.num_choices == n
        assert check
        self.num_choices = n

    def extract_original_task_prompts(self, check_valid_prompts=False):
        all_prompt_names = self.prompts.all_template_names
        invalid_names = []
        if check_valid_prompts:
            for pname in all_prompt_names:
                if self.prompts[pname].metadata.original_task:
                    for d in self.dataset:
                        return_value_length = len(self.prompts[pname].apply(d))
                        if return_value_length != 2:
                            invalid_names.append(pname)
                            break
        invalid_names = set(invalid_names)
        prompt_names = [name for name in all_prompt_names if self.prompts[name].metadata.original_task and name not in invalid_names]
        if self.SUBSET_NAME == "cb" and self.cb_surgery:
            return [name for ii, name in enumerate(prompt_names) if ii != 1 and ii != 10]

        if self.abl_nprompts > 0 and not self.testdev_set:
            prompt_names = np.random.choice(prompt_names, self.abl_nprompts, replace=False)
        return prompt_names


class FlipFlopDataset(Dataset):
    def __init__(self, test_dataset, test_args, random_n_prompts, dev_bsz, idx=-1, serialized=None):
        super().__init__()
        action = "Loading cached" if serialized is not None else "Building"
        print(f"{action} FlipFlop training set: {test_args.train_data_source}!")

        self.random_n_prompts = random_n_prompts
        self.dev_bsz = dev_bsz
        self.split_answer_groups = test_args.split_answer_groups

        train_data_form = test_args.train_data_source

        if serialized is not None:
            meta = serialized.get("meta", {})
            self.dataset = serialized.get("dataset", [])
            self.gold_labels = serialized.get("gold_labels", [])
            self.datasize = meta.get("datasize", len(self.gold_labels))
            if self.datasize is not None:
                self.datasize = int(self.datasize)
            self.num_prompts = meta.get("num_prompts")
            if self.num_prompts is not None:
                self.num_prompts = int(self.num_prompts)
            self.num_choices = meta.get("num_choices")
            if self.num_choices is not None:
                self.num_choices = int(self.num_choices)
            self.tot_single_ds_size = meta.get("tot_single_ds_size")
            if self.tot_single_ds_size is not None:
                self.tot_single_ds_size = int(self.tot_single_ds_size)
            self.original_task_prompts = meta.get("original_task_prompts")
            if self.original_task_prompts is not None:
                self.original_task_prompts = list(self.original_task_prompts)
            self.prompt_groups = meta.get("prompt_groups")
            if self.prompt_groups is not None:
                self.prompt_groups = [list(group) for group in self.prompt_groups]
            batches = meta.get("dev_batches")
            if batches is not None:
                self.dev_batches = [tuple(batch) for batch in batches]
            else:
                tot = self.tot_single_ds_size
                self.dev_batches = [
                    (i, i + self.dev_bsz if i + self.dev_bsz < tot else tot)
                    for i in range(0, tot, self.dev_bsz)
                ]
            self.dev_size = meta.get("dev_size", len(self.dev_batches))
            if self.dev_size is not None:
                self.dev_size = int(self.dev_size)
            self.split_answer_groups = meta.get("split_answer_groups", self.split_answer_groups)
            self.random_n_prompts = meta.get("random_n_prompts", self.random_n_prompts)
            self.dev_bsz = meta.get("dev_bsz", self.dev_bsz)
            self.base_raw_dataset = None
            if self.tot_single_ds_size is None and self.num_prompts is not None and self.num_choices is not None:
                self.tot_single_ds_size = self.num_prompts * self.num_choices
        else:
            if train_data_form != "stream":
                assert idx == -1
                self.dataset, self.gold_labels = self.construct_dataset(test_dataset)
                self.datasize = len(test_dataset)
            else:
                self.dataset, gold_label = test_dataset[idx]
                self.gold_labels = [gold_label]
                self.datasize = len(self.gold_labels)

            self.num_choices = test_dataset.num_choices
            self.num_prompts = test_dataset.num_prompts
            self.tot_single_ds_size = self.num_prompts * self.num_choices
            self.original_task_prompts = test_dataset.original_task_prompts
            self.prompt_groups = test_dataset.prompt_groups
            try:
                self.base_raw_dataset = getattr(test_dataset, "dataset", None)
            except Exception:
                self.base_raw_dataset = None

            tot = self.tot_single_ds_size
            self.dev_batches = [
                (i, i + dev_bsz if i + dev_bsz < tot else tot) for i in range(0, tot, dev_bsz)
            ]
            self.dev_size = len(self.dev_batches)

    def construct_dataset(self, dataset: DatasetByPrompt):
        all_data = []
        labels = []
        for examples, label in dataset:
            all_data.extend(examples)
            labels.append(label)
        return all_data, labels

    def serialize(self):
        return {
            "dataset": self.dataset,
            "gold_labels": self.gold_labels,
            "meta": {
                "datasize": self.datasize,
                "num_prompts": self.num_prompts,
                "num_choices": self.num_choices,
                "tot_single_ds_size": self.tot_single_ds_size,
                "original_task_prompts": self.original_task_prompts,
                "prompt_groups": self.prompt_groups,
                "dev_batches": list(self.dev_batches),
                "dev_size": self.dev_size,
                "split_answer_groups": self.split_answer_groups,
                "random_n_prompts": self.random_n_prompts,
                "dev_bsz": self.dev_bsz,
                "version": 1,
            },
        }

    def __len__(self):
        return self.datasize

    def __getitem__(self, idx):
        results = []
        s = idx * self.tot_single_ds_size
        for bidx1, bidx2 in self.dev_batches:
            results.append(self.dataset[s + bidx1 : s + bidx2])

        if self.split_answer_groups:
            for pgroup in self.prompt_groups:
                random_prompts = pgroup if len(pgroup) <= self.random_n_prompts else np.random.choice(pgroup, self.random_n_prompts, replace=False)
                for ans_idx in range(self.num_choices):
                    results.append([self.dataset[s + pid * self.num_choices + ans_idx] for pid in random_prompts])
        else:
            random_prompts = list(range(self.num_prompts)) if self.num_prompts <= self.random_n_prompts else np.random.choice(self.num_prompts, self.random_n_prompts, replace=False)
            for ans_idx in range(self.num_choices):
                results.append([self.dataset[s + pid * self.num_choices + ans_idx] for pid in random_prompts])
        return results
