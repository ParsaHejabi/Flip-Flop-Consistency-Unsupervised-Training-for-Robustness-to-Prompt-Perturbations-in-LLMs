import hashlib
import json
import math
import os
import os.path
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

sys.path.insert(2, "./")

import logging

import datasets
import transformers
from evaluate import load as load_metric
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

from data_collator import DataCollatorForSeq2Seq
from dataloader import DatasetByPrompt, FlipFlopDataset
from flipflop_trainer import DevLossEarlyStoppingCallback, FlipFlopTrainer
from lora_utils import apply_peft_lora_to_decoder_model, apply_peft_lora_to_seq2seq_model
from options import *
from utils import compute_metrics, compute_unsupervised_dev_best_results, compute_unsupervised_metrics, summarize_metrics

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, TestArguments))
    model_args, data_args, training_args, test_args = parser.parse_args_into_dataclasses()
    if "/" not in model_args.model_name_or_path and not os.path.isdir(model_args.model_name_or_path):
        model_args.model_name_or_path = "bigscience/" + model_args.model_name_or_path

    # Setup logging
    log_level = training_args.get_process_log_level()
    desired_root_level = logging.INFO if log_level > logging.INFO else log_level

    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(desired_root_level)
    console_handler.setFormatter(log_formatter)
    handlers = [console_handler]

    log_file = None
    if getattr(training_args, "local_rank", -1) in (-1, 0):
        output_dir = training_args.output_dir or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "log.txt")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(desired_root_level)
        file_handler.setFormatter(log_formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)
        existing_handler.close()
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(desired_root_level)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True

    if log_file is not None:
        logger.info("Logging to %s and stdout.", log_file)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # set additional args
    for k, v in vars(test_args).items():
        if not hasattr(config, k):
            setattr(config, k, v)
            setattr(training_args, k, v)

    logger.info(f"Training/evaluation parameters {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    pad_token_added = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            pad_token_added = True

    # Enable list expansion to preserve nested batches
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        expand_list=True,
    )

    # Development mode: limit dataset sizes as early as possible to speed up
    dev_mode = getattr(test_args, "development_mode", False)
    if dev_mode:
        logger.info("Development mode enabled: applying early hold_out to datasets.")
    dev_test_hold_out = 5 if dev_mode else -1

    logger.info("Discovering prompt formats for deterministic splitting.")
    prompt_discovery_dataset = DatasetByPrompt(
        data_args,
        model_args.cache_dir,
        tokenizer,
        hold_out=dev_test_hold_out,
        random_hold_out=True,
        testdev_set=True,
    )
    all_prompt_formats = list(prompt_discovery_dataset.original_task_prompts)
    if not all_prompt_formats:
        raise ValueError("No prompt formats found via DatasetTemplates; cannot continue.")

    abl_setting = getattr(data_args, "abl_nprompts", -1)
    requested_k = abl_setting or -1
    if requested_k <= 0:
        train_prompt_formats = list(all_prompt_formats)
    else:
        k = min(requested_k, len(all_prompt_formats))
        train_prompt_formats = list(all_prompt_formats[:k])
    eval_prompt_formats = list(all_prompt_formats[len(train_prompt_formats) :])
    if not eval_prompt_formats:
        if requested_k > 0:
            logger.warning(
                "No remaining prompt formats beyond the first %s; evaluation will reuse training formats.",
                requested_k,
            )
        eval_prompt_formats = list(all_prompt_formats)
    val_prompt_formats = list(train_prompt_formats)

    logger.info(
        "All prompt formats (%d): %s",
        len(all_prompt_formats),
        ", ".join(all_prompt_formats),
    )
    logger.info(
        "Train/dev prompt formats (%d): %s",
        len(train_prompt_formats),
        ", ".join(train_prompt_formats),
    )
    logger.info(
        "Eval prompt formats (%d): %s",
        len(eval_prompt_formats),
        ", ".join(eval_prompt_formats),
    )

    study3_active = abl_setting != -1

    del prompt_discovery_dataset

    logger.info(f"Creating test DatasetByPrompt with hold_out {dev_test_hold_out}")
    test_data = DatasetByPrompt(
        data_args,
        model_args.cache_dir,
        tokenizer,
        hold_out=dev_test_hold_out,
        random_hold_out=True,
        testdev_set=False if test_args.self_train_option == "constrained" else True,
        prompt_names=eval_prompt_formats,
    )
    if test_args.train_random_n_prompts <= 0:
        default_prompt_count = len(train_prompt_formats) if len(train_prompt_formats) > 0 else len(all_prompt_formats)
        test_args.train_random_n_prompts = default_prompt_count

    config.num_choices = test_data.num_choices
    if test_args.metric_name == "none":
        metrics = load_metric(
            data_args.dataset_name,
            data_args.subset_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        metrics = load_metric(test_args.metric_name, cache_dir=model_args.cache_dir)

    logger.info(f"Model parameters {config}")

    def _model_init():
        # very slow
        model_cls = AutoModelForSeq2SeqLM if getattr(config, "is_encoder_decoder", False) else AutoModelForCausalLM
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            low_cpu_mem_usage=True,
        )
        if tokenizer.pad_token_id is not None and getattr(model.config, "pad_token_id", None) != tokenizer.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id
        if pad_token_added:
            model.resize_token_embeddings(len(tokenizer))
        if training_args.gradient_checkpointing:
            model.config.use_cache = False
        return model

    model = _model_init()

    is_decoder_only = not getattr(model.config, "is_encoder_decoder", False)
    if is_decoder_only and getattr(training_args, "do_train", False):
        model.config.use_cache = False

    if not getattr(model.config, "is_encoder_decoder", False):
        model.gradient_checkpointing_enable()

    peft_lora_applied = False
    if test_args.peft_option == "lora":
        if is_decoder_only:
            model, peft_lora_applied = apply_peft_lora_to_decoder_model(model, test_args, training_args)
        else:
            model, peft_lora_applied = apply_peft_lora_to_seq2seq_model(model, test_args, training_args)

    def _configure_trainable_parameters():
        peft_option = getattr(test_args, "peft_option", "none")

        if peft_option in {None, "", "none"}:
            logger.info("PEFT disabled; training all model parameters.")
            for _, p in model.named_parameters():
                p.requires_grad = True
            return

        if peft_option == "full":
            for _, p in model.named_parameters():
                p.requires_grad = True
            return

        if peft_option == "lora" and peft_lora_applied:
            logger.info("LoRA adapters managed by PEFT; keeping other parameters frozen.")
            return

        for n, p in model.named_parameters():
            if peft_option == "bitfit" and "bias" in n:
                logger.info("tune " + n)
                p.requires_grad = True
            elif peft_option == "lora" and "ef_" in n:
                logger.info("tune " + n)
                p.requires_grad = True
            elif peft_option == "prompt_tuning" and "ef_" in n:
                logger.info("tune " + n)
                p.requires_grad = True
            else:
                p.requires_grad = False

    _configure_trainable_parameters()
    logger.info(f"trainable parameters count: {sum(p.requires_grad for _, p in model.named_parameters())}")

    # Expose loss configuration knobs on the model config for patched forwards
    setattr(model.config, "loss_option", test_args.loss_option)
    setattr(model.config, "pseudo_train_loss_weight", getattr(test_args, "pseudo_train_loss_weight", 1.0))
    setattr(model.config, "cce_weight", getattr(test_args, "cce_weight", 1.0))

    data_collator.model = model
    logger.info(f"there are {test_data.num_prompts} variations in total in the test data and using {test_args.train_random_n_prompts} variations in the pipeline")

    train_split = data_args.testset_name.replace("dev", "train") if data_args.dataset_name == "anli" else "train"

    # Build a base dataset to derive stratified random dev indices and disjoint train indices
    sampling_split = train_split if test_args.train_data_source == "train" else None
    # In development mode, only take a small subset up front to avoid expensive processing
    if dev_mode:
        # Targets: ~20 train + ~5 dev; add a small margin to account for dedupe
        dev_base_hold_out = 25 + 5  # 30 examples total
    else:
        dev_base_hold_out = -1

    base_data_for_sampling = DatasetByPrompt(
        data_args,
        model_args.cache_dir,
        tokenizer,
        split=sampling_split,
        hold_out=dev_base_hold_out,
        random_hold_out=True,
        testdev_set=True,
        prompt_names=train_prompt_formats,
    )

    # Deduplicate by pre-tokenization rendered text per prompt: keep one base example per (prompt_name, input_text)
    def _dedupe_dbp(dbp, name, testdev_set_flag):
        # Deduplicate by rendered input text only (across all prompts), so that
        # each unique input appears at most once in the split. This guarantees
        # per-input cardinality equals K and eliminates cross-prompt identical texts.
        seen_inputs = set()  # input_text
        keep_indices = []
        for i in range(len(dbp)):
            duplicate = False
            rendered_prompts = dbp.get_rendered_prompts(i)
            current_inputs = []
            for prompt_view in rendered_prompts:
                inp = prompt_view.input_text
                current_inputs.append(inp)
                if inp in seen_inputs:
                    duplicate = True
            if duplicate:
                continue  # skip entire example to avoid introducing duplicate pairs
            # all keys are new; keep this base example and register its pairs
            keep_indices.append(i)
            for inp in current_inputs:
                seen_inputs.add(inp)
        removed = len(dbp) - len(keep_indices)
        if removed > 0:
            logger.warning(f"Deduplicated split '{name}': before={len(dbp)} after={len(keep_indices)} removed={removed}")
        # Map kept positions back to original split indices if this DBP is already subsetted
        if getattr(dbp, "subset_indices", None) is not None:
            new_subset_indices = [int(dbp.subset_indices[i]) for i in keep_indices]
        else:
            new_subset_indices = keep_indices

        # Rebuild a DatasetByPrompt limited to kept base indices
        return DatasetByPrompt(
            data_args,
            model_args.cache_dir,
            tokenizer,
            split=dbp.split,
            hold_out=-1,
            random_hold_out=True,
            testdev_set=testdev_set_flag,
            prompt_names=dbp.original_task_prompts,
            subset_indices=new_subset_indices,
        )

    base_data_for_sampling = _dedupe_dbp(base_data_for_sampling, "train_full", True)
    test_data = _dedupe_dbp(test_data, "eval", test_data.testdev_set)

    test_size_cap = getattr(test_args, "test_size", -1)
    if test_size_cap is not None and test_size_cap > -1 and len(test_data) > test_size_cap:
        logger.info(
            "Limiting test split to %d examples (was %d).",
            test_size_cap,
            len(test_data),
        )
        test_rng = np.random.RandomState(getattr(training_args, "seed", 42))
        selected_positions = sorted(int(i) for i in test_rng.choice(len(test_data), size=test_size_cap, replace=False))
        if getattr(test_data, "subset_indices", None) is not None:
            subset_indices = [int(test_data.subset_indices[i]) for i in selected_positions]
        else:
            subset_indices = selected_positions
        test_data = DatasetByPrompt(
            data_args,
            model_args.cache_dir,
            tokenizer,
            split=test_data.split,
            hold_out=-1,
            random_hold_out=test_data.random_hold_out,
            testdev_set=test_data.testdev_set,
            prompt_names=test_data.original_task_prompts,
            subset_indices=subset_indices,
        )

    # Compute labels for stratified sampling using the first original-task prompt
    labels = [base_data_for_sampling.get_label(idx) for idx in range(len(base_data_for_sampling))]

    import numpy as _np  # local alias to avoid shadowing

    rng = _np.random.RandomState(getattr(training_args, "seed", 42))
    N = len(labels)
    max_dev_cap = getattr(test_args, "max_dev_size", 0)
    test_len = len(test_data)
    # Dev size rule: prefer max_dev_size if available; otherwise match eval size
    if max_dev_cap > 0 and N >= max_dev_cap:
        desired_dev = max_dev_cap
    else:
        desired_dev = min(test_len, N)
        logger.warning(f"Dev-size fallback engaged for {data_args.dataset_name}/{data_args.subset_name}: " f"using desired_dev={desired_dev} (=min(test_size={test_len}, N)).")
    # If no dev requested, fall back to empty
    class_to_indices = {}
    for i, y in enumerate(labels):
        class_to_indices.setdefault(y, []).append(i)
    if desired_dev > 0:
        # Proportional quotas per class
        counts = {c: len(ixs) for c, ixs in class_to_indices.items()}
        total = float(sum(counts.values()))
        quotas = {c: int(_np.floor(desired_dev * cnt / total)) for c, cnt in counts.items()}
        allocated = sum(quotas.values())
        # Distribute remainder by largest fractional parts
        fracs = sorted(
            ((c, desired_dev * counts[c] / total - quotas[c]) for c in counts),
            key=lambda t: t[1],
            reverse=True,
        )
        for k in range(desired_dev - allocated):
            quotas[fracs[k % len(fracs)][0]] += 1
        dev_indices = []
        for c, q in quotas.items():
            if q <= 0:
                continue
            ixs = class_to_indices[c]
            pick = rng.choice(len(ixs), size=min(q, len(ixs)), replace=False)
            dev_indices.extend([ixs[j] for j in pick])
        dev_indices = set(dev_indices)
    else:
        dev_indices = set()

    all_indices = set(range(N))
    # Build disjoint train/dev index sets
    train_indices = list(all_indices - dev_indices)

    # Safety checks: ensure no overlap and full coverage of the sampled split
    assert dev_indices.isdisjoint(set(train_indices)), "Train and dev indices overlap; expected disjoint splits."
    assert dev_indices.union(set(train_indices)) == all_indices, "Train and dev indices do not cover the full sampled split."

    # Apply optional debug size for training subset after disjoint split
    if getattr(test_args, "debug_size", -1) and test_args.debug_size > -1:
        if test_args.debug_size < len(train_indices):
            pick = rng.choice(len(train_indices), size=test_args.debug_size, replace=False)
            train_indices = [train_indices[i] for i in pick]

    # Map deduped index space back to raw split indices for dataset construction
    def _map_to_raw_indices(dbp, idx_list):
        if getattr(dbp, "subset_indices", None) is not None:
            return [int(dbp.subset_indices[i]) for i in idx_list]
        return idx_list

    train_raw_indices = _map_to_raw_indices(base_data_for_sampling, train_indices)
    dev_raw_indices = _map_to_raw_indices(base_data_for_sampling, list(dev_indices))

    if getattr(test_args, "development_mode", False):
        if len(train_raw_indices) > 20:
            logger.info("Development mode: limiting train split to 20 examples (was %d).", len(train_raw_indices))
            train_raw_indices = train_raw_indices[:20]
        if len(dev_raw_indices) > 5:
            logger.info("Development mode: limiting dev split to 5 examples (was %d).", len(dev_raw_indices))
            dev_raw_indices = dev_raw_indices[:5]
        # test_data was already limited via hold_out above; skip redundant filtering here.

    # Build training and dev datasets from disjoint index sets (raw-indexed)
    data = DatasetByPrompt(
        data_args,
        model_args.cache_dir,
        tokenizer,
        split=sampling_split,
        hold_out=-1,
        random_hold_out=True,
        testdev_set=False,
        prompt_names=train_prompt_formats,
        subset_indices=train_raw_indices,
    )
    # Build dev data as a random, stratified, disjoint subset of the same split used for training
    dev_subset_indices = dev_raw_indices
    dev_data = DatasetByPrompt(
        data_args,
        model_args.cache_dir,
        tokenizer,
        split=sampling_split,
        hold_out=-1,
        random_hold_out=True,
        testdev_set=True,
        prompt_names=train_prompt_formats,
        subset_indices=dev_subset_indices,
    )

    # Optional cross-split safety: ensure we are not accidentally evaluating on the same split used for train/dev
    # (Only a coarse check using the configured split names; the content-level disjointness is enforced by indices above.)
    if sampling_split is not None:
        assert data_args.testset_name != sampling_split, f"Configured test split '{data_args.testset_name}' must differ from train/dev split '{sampling_split}'."

    # Cached dataset helpers -------------------------------------------------
    def _sanitize_identifier(value: str) -> str:
        value = value or "none"
        return re.sub(r"[^0-9A-Za-z._-]", "_", value)

    tokenizer_identifier = getattr(tokenizer, "name_or_path", None) or model_args.model_name_or_path
    model_identifier = _sanitize_identifier(model_args.model_name_or_path)
    if tokenizer_identifier != model_args.model_name_or_path:
        model_identifier = f"{model_identifier}__tok__{_sanitize_identifier(tokenizer_identifier)}"

    dataset_identifier_parts = [
        data_args.dataset_name,
        data_args.subset_name if data_args.subset_name != "none" else "none",
        data_args.testset_name,
        f"seed{getattr(training_args, 'seed', 42)}",
    ]
    dataset_identifier = "__".join(_sanitize_identifier(part) for part in dataset_identifier_parts)

    cache_root = Path("data") / model_identifier / dataset_identifier
    cache_enabled = test_args.train_data_source != "stream"

    if study3_active:
        study3_dir: Optional[Path] = "study_3" / cache_root / f"abl_np{abl_setting}"
        format_split_path: Optional[Path] = study3_dir / "format_splits.json"
        try:
            study3_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create study_3 directory %s (%s)", study3_dir, exc)
            study3_dir = None
            format_split_path = None
    else:
        study3_dir = None
        format_split_path = None

    if format_split_path is not None:
        try:
            with format_split_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "all_formats": all_prompt_formats,
                        "train_formats": train_prompt_formats,
                        "val_formats": val_prompt_formats,
                        "test_formats": eval_prompt_formats,
                    },
                    fh,
                    indent=2,
                    sort_keys=True,
                )
            logger.info("Persisted prompt format split to %s", format_split_path)
        except OSError as exc:
            logger.warning("Failed to persist prompt format splits to %s (%s)", format_split_path, exc)
    cache_config = {
        "model_name": model_args.model_name_or_path,
        "tokenizer_name": tokenizer_identifier,
        "seed": getattr(training_args, "seed", 42),
        "dataset_name": data_args.dataset_name,
        "subset_name": data_args.subset_name,
        "testset_name": data_args.testset_name,
        "prompt_set_name": data_args.prompt_set_name,
        "train_data_source": test_args.train_data_source,
        "train_random_n_prompts": test_args.train_random_n_prompts,
        "split_answer_groups": test_args.split_answer_groups,
        "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
        "development_mode": bool(getattr(test_args, "development_mode", False)),
        "train_prompt_formats": tuple(train_prompt_formats),
        "val_prompt_formats": tuple(val_prompt_formats),
        "test_prompt_formats": tuple(eval_prompt_formats),
        "version": 1,
    }

    def _normalize_indices(indices):
        if indices is None:
            return None
        return [int(i) for i in indices]

    def _load_cached_split(split_name, expected_indices):
        if not cache_enabled:
            return None
        split_path = cache_root / f"{split_name}.pt"
        if not split_path.exists():
            return None
        try:
            payload = torch.load(split_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - cache best effort
            logger.warning(
                "Failed to load cached %s split from %s (%s); rebuilding.",
                split_name,
                split_path,
                exc,
            )
            return None
        meta = payload.get("meta", {})
        if meta.get("version") != 1:
            logger.info("Cache version mismatch for %s split; rebuilding.", split_name)
            return None
        if meta.get("config") != cache_config:
            logger.info("Cache config mismatch for %s split; rebuilding.", split_name)
            return None
        cached_indices = meta.get("source_indices")
        if expected_indices is not None and cached_indices != expected_indices:
            logger.info("Cache indices mismatch for %s split; rebuilding.", split_name)
            return None
        return payload, split_path

    def _persist_split(split_name, dataset_obj, source_indices):
        serialized_cache = None
        if cache_enabled:
            split_path = cache_root / f"{split_name}.pt"
            try:
                cache_root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # pragma: no cover - cache best effort
                logger.warning(
                    "Unable to create cache directory %s (%s); skipping cache persistence.",
                    cache_root,
                    exc,
                )
            else:
                serialized_cache = dataset_obj.serialize()
                meta = serialized_cache.get("meta", {})
                meta["config"] = cache_config
                meta["split"] = split_name
                if source_indices is not None:
                    meta["source_indices"] = list(source_indices)
                serialized_cache["meta"] = meta
                try:
                    torch.save(serialized_cache, split_path)
                    logger.info("Cached %s split to %s", split_name, split_path)
                except Exception as exc:  # pragma: no cover - cache best effort
                    logger.warning(
                        "Failed to persist %s split to %s (%s)",
                        split_name,
                        split_path,
                        exc,
                    )

        if study3_active and study3_dir is not None:
            serialized_study3 = serialized_cache if serialized_cache is not None else dataset_obj.serialize()
            meta = serialized_study3.get("meta", {})
            meta["config"] = cache_config
            meta["split"] = split_name
            if source_indices is not None:
                meta["source_indices"] = list(source_indices)
            serialized_study3["meta"] = meta
            study3_path = study3_dir / f"{split_name}.pt"
            try:
                torch.save(serialized_study3, study3_path)
                logger.info("Persisted study_3 %s split to %s", split_name, study3_path)
            except Exception as exc:
                logger.warning(
                    "Failed to persist study_3 %s split to %s (%s)",
                    split_name,
                    study3_path,
                    exc,
                )

    def _build_split(split_name, dataset_by_prompt, source_indices):
        normalized = _normalize_indices(source_indices)
        cached = _load_cached_split(split_name, normalized)
        if cached is not None:
            payload, cache_path = cached
            logger.info("Loaded cached %s split from %s", split_name, cache_path)
            dataset_obj = FlipFlopDataset(
                dataset_by_prompt,
                test_args,
                test_args.train_random_n_prompts,
                training_args.per_device_eval_batch_size,
                serialized=payload,
            )
            _persist_split(split_name, dataset_obj, normalized)
            return dataset_obj
        dataset_obj = FlipFlopDataset(
            dataset_by_prompt,
            test_args,
            test_args.train_random_n_prompts,
            training_args.per_device_eval_batch_size,
        )
        _persist_split(split_name, dataset_obj, normalized)
        return dataset_obj

    train_data = _build_split("train", data, train_raw_indices)
    dev_set = _build_split("dev", dev_data, dev_subset_indices)
    test_cache_indices = getattr(test_data, "subset_indices", None)
    if test_cache_indices is None:
        test_cache_indices = list(range(len(test_data)))
    test_set = _build_split("test", test_data, test_cache_indices)

    logger.info(f"prompt groups {train_data.prompt_groups}")

    # ------------------------------------------------------------------
    # Post-variation duplicate checks based on pre-tokenization template text
    # Build variations directly from DatasetByPrompt via prompts.apply and
    # get_answer_choices_list, so we check the exact created text.
    # ------------------------------------------------------------------
    from collections import defaultdict as _dd

    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _display_text(text: str) -> str:
        if text is None:
            return "<none>"
        return text.replace("\n", "\\n")

    def _iter_variations(dbp):
        # Yields tuples: (base_idx, raw_id, prompt_name, input_text, label_text)
        for i in range(len(dbp)):
            raw_id = dbp.get_example_id(i)
            for prompt_view in dbp.get_rendered_prompts(i):
                inp = prompt_view.input_text
                for ans in prompt_view.choices:
                    yield i, raw_id, prompt_view.prompt_name, inp, ans

    def _build_text_variation_index(dbp, split_name):
        by_pair = _dd(list)
        by_input = _dd(list)
        for base_idx, raw_id, pname, inp, lab in _iter_variations(dbp):
            by_pair[(inp, lab)].append((base_idx, raw_id, pname))
            by_input[inp].append(lab)
        return by_pair, by_input

    def _assert_no_textual_duplicates_dbp(dbp, name):
        by_pair, _ = _build_text_variation_index(dbp, name)
        dup_groups = [((inp, lab), items) for (inp, lab), items in by_pair.items() if len(items) > 1]
        if dup_groups:
            logger.error(f"Found {len(dup_groups)} duplicate (input,label) groups within split '{name}'. Showing up to 5:")
            for (inp, lab), items in dup_groups[:5]:
                logger.error('- input="%s" | label="%s"', inp, lab)
                for base_idx, raw_id, pname in items[:5]:
                    logger.error(
                        "  base=%s id=%s prompt=%s",
                        str(base_idx),
                        str(raw_id),
                        str(pname),
                    )
            raise AssertionError(f"Duplicate (input,label) variations detected within split '{name}'.")

    def _assert_input_cardinality_dbp(dbp, name, expected_k: int):
        _, by_input = _build_text_variation_index(dbp, name)
        bad = []
        for inp, labs in by_input.items():
            uniq = set(labs)
            if len(labs) != expected_k or len(uniq) != expected_k:
                bad.append((inp, len(labs), len(uniq), list(uniq)[: expected_k + 2]))
        if bad:
            logger.error(f"Found {len(bad)} inputs in split '{name}' with incorrect cardinality (expected {expected_k}). Showing up to 5:")
            for inp, cnt, uniq_cnt, uniqs in bad[:5]:
                logger.error(
                    '- input="%s" | total_rows=%s distinct_labels=%s labels=%s',
                    inp,
                    str(cnt),
                    str(uniq_cnt),
                    str(uniqs),
                )
            raise AssertionError(f"Incorrect input-only cardinality detected in split '{name}'.")

    def _assert_texts_disjoint_dbp(dbp_a, name_a, dbp_b, name_b):
        set_a = set()
        for _, _, _, inp, _ in _iter_variations(dbp_a):
            set_a.add(inp)
        set_b = set()
        for _, _, _, inp, _ in _iter_variations(dbp_b):
            set_b.add(inp)
        inter = list(set_a.intersection(set_b))
        if inter:
            logger.error(f"Splits '{name_a}' and '{name_b}' share {len(inter)} input texts (ignoring labels). Showing up to 5:")
            for t in inter[:5]:
                logger.error('- input="%s"', t)
            raise AssertionError(f"Input-text overlap detected between '{name_a}' and '{name_b}'.")

    # Perform checks for the used splits: train, dev, eval (pre-tokenization, post-template)
    # Input+label uniqueness within each split
    _assert_no_textual_duplicates_dbp(data, "train")
    _assert_no_textual_duplicates_dbp(dev_data, "dev")
    _assert_no_textual_duplicates_dbp(test_data, "eval")
    # Input-only cardinality within each split
    _assert_input_cardinality_dbp(data, "train", expected_k=data.num_choices)
    _assert_input_cardinality_dbp(dev_data, "dev", expected_k=dev_data.num_choices)
    _assert_input_cardinality_dbp(test_data, "eval", expected_k=test_data.num_choices)
    # Cross-split input-text disjointness between used splits
    _assert_texts_disjoint_dbp(data, "train", dev_data, "dev")
    _assert_texts_disjoint_dbp(data, "train", test_data, "eval")
    _assert_texts_disjoint_dbp(dev_data, "dev", test_data, "eval")

    # ------------------------------------------------------------------
    # Post-duplication checks on FlipFlopDataset objects (final trainer inputs)
    # Ensure rendered prompt texts remain unique per split and disjoint across splits.
    # ------------------------------------------------------------------
    from collections import defaultdict as _ff_dd

    def _assert_flipflop_split_integrity(ff_dataset, name: str):
        raw_dataset = getattr(ff_dataset, "dataset", None)
        if raw_dataset is None:
            raise AssertionError(f"FlipFlopDataset for split '{name}' is missing the materialized dataset.")

        prompt_hash_to_labels = _ff_dd(set)
        prompt_hash_to_text = {}
        duplicate_pairs = []
        pair_hash_seen = {}

        for idx, example in enumerate(raw_dataset):
            prompt_text = example.get("prompt_text")
            label_text = example.get("label_text")
            if prompt_text is None or label_text is None:
                raise AssertionError(f"FlipFlopDataset split '{name}' contains an entry without prompt_text/label_text at index {idx}.")

            prompt_hash = _hash_text(prompt_text)
            pair_hash = _hash_text(f"{prompt_text}\u0000{label_text}")

            if pair_hash in pair_hash_seen:
                prev_idx = pair_hash_seen[pair_hash]
                duplicate_pairs.append((prompt_text, label_text, prev_idx, idx))
            else:
                pair_hash_seen[pair_hash] = idx

            prompt_hash_to_labels[prompt_hash].add(label_text)
            prompt_hash_to_text.setdefault(prompt_hash, prompt_text)

        if duplicate_pairs:
            logger.error(
                "Found %d duplicate (prompt_text,label_text) rows within FlipFlopDataset split '%s'. Showing up to 5:",
                len(duplicate_pairs),
                name,
            )
            for prompt_text, label_text, prev_idx, cur_idx in duplicate_pairs[:5]:
                logger.error(
                    '  text="%s" | label="%s" | first_idx=%s | duplicate_idx=%s',
                    _display_text(prompt_text),
                    label_text,
                    prev_idx,
                    cur_idx,
                )
            raise AssertionError(f"Duplicate (prompt_text,label_text) entries detected within FlipFlopDataset split '{name}'.")

        expected_labels = getattr(ff_dataset, "num_choices", None)
        overflow = []
        if expected_labels is not None and expected_labels > 0:
            for prompt_hash, labels in prompt_hash_to_labels.items():
                if len(labels) > expected_labels:
                    overflow.append((prompt_hash, len(labels)))
        if overflow:
            logger.error(
                "Found %d prompt texts in FlipFlopDataset split '%s' with more than %s label variants. Showing up to 5:",
                len(overflow),
                name,
                expected_labels,
            )
            for prompt_hash, label_count in overflow[:5]:
                logger.error(
                    '  text="%s" | label_count=%s',
                    _display_text(prompt_hash_to_text.get(prompt_hash, "<missing>")),
                    label_count,
                )
            raise AssertionError(f"FlipFlopDataset split '{name}' contains prompt texts duplicated more than num_choices ({expected_labels}).")

        return set(prompt_hash_to_text.keys()), prompt_hash_to_text

    def _assert_flipflop_texts_disjoint(a_name, a_hashes, a_text_map, b_name, b_hashes, b_text_map):
        shared = a_hashes.intersection(b_hashes)
        if shared:
            logger.error(
                "FlipFlopDataset splits '%s' and '%s' share %d rendered prompt texts. Showing up to 5:",
                a_name,
                b_name,
                len(shared),
            )
            for text_hash in list(shared)[:5]:
                text_a = a_text_map.get(text_hash, "<unknown>")
                text_b = b_text_map.get(text_hash, "<unknown>")
                logger.error(
                    '  hash=%s\n    %s text="%s"\n    %s text="%s"',
                    text_hash,
                    a_name,
                    _display_text(text_a),
                    b_name,
                    _display_text(text_b),
                )
            raise AssertionError(f"Rendered prompt texts overlap between FlipFlopDataset splits '{a_name}' and '{b_name}'.")

    train_prompt_hashes, train_prompt_map = _assert_flipflop_split_integrity(train_data, "train")
    dev_prompt_hashes, dev_prompt_map = _assert_flipflop_split_integrity(dev_set, "dev")
    test_prompt_hashes, test_prompt_map = _assert_flipflop_split_integrity(test_set, "eval")

    _assert_flipflop_texts_disjoint("train", train_prompt_hashes, train_prompt_map, "dev", dev_prompt_hashes, dev_prompt_map)
    _assert_flipflop_texts_disjoint("train", train_prompt_hashes, train_prompt_map, "eval", test_prompt_hashes, test_prompt_map)
    _assert_flipflop_texts_disjoint("dev", dev_prompt_hashes, dev_prompt_map, "eval", test_prompt_hashes, test_prompt_map)

    trainer = FlipFlopTrainer(
        model=model,
        train_dataset=train_data,
        dev_dataset=dev_set,
        eval_dataset=test_set,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        test_data_collator=data_collator,
        compute_metrics=compute_metrics,
        compute_unsupervised_metrics=compute_unsupervised_metrics,
        additional_metrics=metrics,
    )

    trainer.add_callback(
        DevLossEarlyStoppingCallback(
            patience=5,
            minimum_steps=test_args.min_train_steps,
            metric_key="unsupervised_dev_loss",
        )
    )

    base_dev_results = trainer.evaluate(
        eval_dataset=dev_set,
        metric_key_prefix="unsupervised_dev",
        force_output_suffix="0",
    )
    for k, v in base_dev_results.items():
        logger.info("base_dev_unsupervised_{} = {}".format(k, v))

    base_test_results = trainer.evaluate(
        eval_dataset=test_set,
        force_output_suffix="0",
    )
    for k, v in base_test_results.items():
        logger.info("base_test_{} = {}".format(k, v))

    if training_args.do_train:
        trainer.train()
    else:
        logger.info("Skipping training because --do_train was not provided.")

    eval_results = trainer.evaluate(eval_dataset=dev_set, metric_key_prefix="unsupervised_dev")
    for k, v in eval_results.items():
        logger.info("dev_unsupervised_{} = {}".format(k, v))

    # Evaluate on the test set explicitly (clarity for dataset selection in trainer.evaluate)
    eval_results = trainer.evaluate(eval_dataset=test_set)
    for k, v in eval_results.items():
        logger.info("{} = {}".format(k, v))

    compute_unsupervised_dev_best_results(training_args.output_dir, min_train_steps=test_args.min_train_steps)


if __name__ == "__main__":
    main()
