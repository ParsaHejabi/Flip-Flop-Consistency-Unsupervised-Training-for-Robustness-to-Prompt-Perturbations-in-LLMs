#!/usr/bin/env python3
"""Cross-dataset evaluation runner (study_2)."""

import argparse
import copy
import json
import logging
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from data_collator import DataCollatorForSeq2Seq
from dataloader import FlipFlopDataset
from flipflop_trainer import FlipFlopTrainer
from utils import compute_metrics, compute_unsupervised_metrics

try:
    from evaluate import load as load_metric
except ImportError as exc:  # pragma: no cover - dependency should already exist
    raise ImportError("study_2 requires the `evaluate` package. Install it before running.") from exc

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, set_seed

try:  # pragma: no cover - optional LoRA dependency
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None

DEFAULT_DATASETS: Tuple[str, ...] = (
    "wsc",
    "winogrande",
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "cb",
    "rte",
    "copa",
    "hellaswag",
    "story_cloze",
    "wic",
)

DATASET_SPECS: Dict[str, Dict[str, str]] = {
    "anli_r1": {
        "dataset_name": "anli",
        "subset_name": "none",
        "testset_name": "dev_r1",
        "prompt_set_name": "anli",
        "metric_name": "accuracy",
    },
    "anli_r2": {
        "dataset_name": "anli",
        "subset_name": "none",
        "testset_name": "dev_r2",
        "prompt_set_name": "anli",
        "metric_name": "accuracy",
    },
    "anli_r3": {
        "dataset_name": "anli",
        "subset_name": "none",
        "testset_name": "dev_r3",
        "prompt_set_name": "anli",
        "metric_name": "accuracy",
    },
    "cb": {
        "dataset_name": "super_glue",
        "subset_name": "cb",
        "testset_name": "validation",
        "prompt_set_name": "super_glue",
        "metric_name": "accuracy",
    },
    "rte": {
        "dataset_name": "super_glue",
        "subset_name": "rte",
        "testset_name": "validation",
        "prompt_set_name": "super_glue",
        "metric_name": "accuracy",
    },
    "copa": {
        "dataset_name": "super_glue",
        "subset_name": "copa",
        "testset_name": "validation",
        "prompt_set_name": "super_glue",
        "metric_name": "accuracy",
    },
    "wic": {
        "dataset_name": "super_glue",
        "subset_name": "wic",
        "testset_name": "validation",
        "prompt_set_name": "super_glue",
        "metric_name": "accuracy",
    },
    "wsc": {
        "dataset_name": "super_glue",
        "subset_name": "wsc.fixed",
        "testset_name": "validation",
        "prompt_set_name": "super_glue",
        "metric_name": "accuracy",
    },
    "winogrande": {
        "dataset_name": "winogrande",
        "subset_name": "winogrande_xl",
        "testset_name": "validation",
        "prompt_set_name": "winogrande",
        "metric_name": "accuracy",
    },
    "hellaswag": {
        "dataset_name": "hellaswag",
        "subset_name": "none",
        "testset_name": "validation",
        "prompt_set_name": "hellaswag",
        "metric_name": "accuracy",
    },
    "story_cloze": {
        "dataset_name": "story_cloze",
        "subset_name": "2016",
        "testset_name": "test",
        "prompt_set_name": "story_cloze",
        "metric_name": "accuracy",
    },
}

LOGGER = logging.getLogger("study_2")


def _sanitize_identifier(value: Optional[str]) -> str:
    if not value:
        value = "none"
    return re.sub(r"[^0-9A-Za-z._-]", "_", value)


def _dataset_identifier(spec: Dict[str, str], seed: int) -> str:
    parts = [
        spec["dataset_name"],
        spec.get("subset_name", "none") or "none",
        spec["testset_name"],
        f"seed{seed}",
    ]
    return "__".join(_sanitize_identifier(part) for part in parts)


def _find_best_checkpoint(ckpt_dir: Path) -> Path:
    root_state = ckpt_dir / "trainer_state.json"
    if root_state.exists():
        data = json.loads(root_state.read_text())
        best_path = Path(data["best_model_checkpoint"])
        if not best_path.is_absolute():
            best_path = (ckpt_dir / best_path).resolve()
        if best_path.exists():
            return best_path
    candidate_states = sorted(ckpt_dir.glob("checkpoint-*/trainer_state.json"), key=lambda p: p.parent.name)

    if not candidate_states:
        raise FileNotFoundError(f"No trainer_state.json found under {ckpt_dir}.")

    def _step_key(path: Path) -> int:
        name = path.parent.name
        if name.startswith("checkpoint-"):
            try:
                return int(name.split("-", 1)[1])
            except ValueError:
                return -1
        return -1

    candidate_states.sort(key=_step_key)
    data = json.loads(candidate_states[-1].read_text())
    best_path = Path(data["best_model_checkpoint"])
    if not best_path.is_absolute():
        best_path = (ckpt_dir / best_path).resolve()
    if not best_path.exists():
        raise FileNotFoundError(f"Resolved best checkpoint {best_path} does not exist.")
    return best_path


def _parse_run_config(run_sh: Path) -> Dict[str, str]:
    if not run_sh.exists():
        return {}
    text = run_sh.read_text()
    command_tokens: List[str] = []
    capturing = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not capturing and "main.py" in line and "python" in line:
            capturing = True
        if not capturing:
            continue
        if line.endswith("\\"):
            command_tokens.append(line[:-1].strip())
            continue
        command_tokens.append(line)
        break

    if not command_tokens:
        return {}
    command = " ".join(command_tokens)
    tokens = shlex.split(command)
    if "main.py" not in tokens:
        return {}
    idx = tokens.index("main.py")
    args_tokens = tokens[idx + 1 :]
    parsed: Dict[str, str] = {}
    i = 0
    while i < len(args_tokens):
        token = args_tokens[i]
        if token.startswith("--"):
            key = token[2:]
            value: Optional[str] = None
            if i + 1 < len(args_tokens) and not args_tokens[i + 1].startswith("--"):
                value = args_tokens[i + 1]
                i += 1
            parsed[key] = value if value is not None else "true"
        i += 1
    return parsed


def _infer_dataset_label(config: Dict[str, str]) -> Optional[str]:
    dataset_name = config.get("dataset_name")
    subset_name = config.get("subset_name") or "none"
    testset_name = config.get("testset_name")
    for label, spec in DATASET_SPECS.items():
        if spec["dataset_name"] == dataset_name and spec["subset_name"] == (subset_name or "none") and spec["testset_name"] == testset_name:
            return label
    return None


def _resolve_base_model(best_ckpt: Path, run_config: Dict[str, str]) -> str:
    candidate = run_config.get("model_name_or_path")
    if candidate:
        return candidate
    adapter_cfg = best_ckpt / "adapter_config.json"
    if adapter_cfg.exists():
        data = json.loads(adapter_cfg.read_text())
        base = data.get("base_model_name_or_path")
        if base:
            return base
    raise ValueError("Unable to resolve base model name. Ensure run.sh contains --model_name_or_path or adapter_config.json is present.")


def _resolve_tokenizer_name(run_config: Dict[str, str]) -> Optional[str]:
    tokenizer_name = run_config.get("tokenizer_name")
    if tokenizer_name:
        return tokenizer_name
    return None


def _make_model_identifier(base_model_name: str, tokenizer_name: Optional[str]) -> str:
    model_identifier = _sanitize_identifier(base_model_name)
    if tokenizer_name and tokenizer_name != base_model_name:
        model_identifier = f"{model_identifier}__tok__{_sanitize_identifier(tokenizer_name)}"
    return model_identifier


def _prepare_tokenizer(best_ckpt: Path, base_model_name: str, tokenizer_name: Optional[str], cache_dir: Optional[str]) -> Tuple[AutoTokenizer, bool]:
    sources: List[str] = []
    if tokenizer_name:
        sources.append(tokenizer_name)
    sources.append(str(best_ckpt))
    if base_model_name not in sources:
        sources.append(base_model_name)
    last_error: Optional[Exception] = None
    tokenizer: Optional[AutoTokenizer] = None
    for source in sources:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=cache_dir, use_fast=True)
            break
        except Exception as exc:  # pragma: no cover - best effort
            last_error = exc
    if tokenizer is None:
        raise RuntimeError(f"Failed to load tokenizer from any source ({sources}).") from last_error
    pad_added = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            pad_added = True
    return tokenizer, pad_added


def _load_model(
    best_ckpt: Path,
    base_model_name: str,
    tokenizer: AutoTokenizer,
    pad_added: bool,
    cache_dir: Optional[str],
) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(base_model_name, cache_dir=cache_dir)
    model_cls = AutoModelForSeq2SeqLM if getattr(config, "is_encoder_decoder", False) else AutoModelForCausalLM
    adapter_files = [best_ckpt / "adapter_model.safetensors", best_ckpt / "adapter_model.bin"]
    has_adapter = any(path.exists() for path in adapter_files)
    if has_adapter:
        if PeftModel is None:
            raise ImportError("LoRA checkpoint detected but `peft` is not installed.")
        base_model = model_cls.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
            config=config,
            low_cpu_mem_usage=True,
        )
        if tokenizer.pad_token_id is not None and getattr(base_model.config, "pad_token_id", None) != tokenizer.pad_token_id:
            base_model.config.pad_token_id = tokenizer.pad_token_id
        if pad_added:
            base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, str(best_ckpt))
    else:
        model = model_cls.from_pretrained(
            str(best_ckpt),
            cache_dir=cache_dir,
            config=config,
            low_cpu_mem_usage=True,
        )
        if tokenizer.pad_token_id is not None and getattr(model.config, "pad_token_id", None) != tokenizer.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id
        if pad_added:
            model.resize_token_embeddings(len(tokenizer))
    return model


def _prepare_eval_args(training_args: TrainingArguments, output_dir: Path, spec: Dict[str, str]) -> TrainingArguments:
    eval_args = copy.deepcopy(training_args)
    eval_args.output_dir = str(output_dir)
    eval_args.logging_dir = str(output_dir)
    eval_args.do_eval = True
    eval_args.do_train = False
    eval_args.report_to = []
    eval_args.disable_tqdm = True
    setattr(eval_args, "dataset_name", spec["dataset_name"])
    setattr(eval_args, "subset_name", spec["subset_name"])
    setattr(eval_args, "testset_name", spec["testset_name"])
    setattr(eval_args, "prompt_set_name", spec["prompt_set_name"])
    setattr(eval_args, "metric_name", spec["metric_name"])
    return eval_args


def _load_dataset_from_cache(
    spec: Dict[str, str],
    model_identifier: str,
    seed: int,
    eval_args: TrainingArguments,
    data_root: Path,
) -> Tuple[FlipFlopDataset, Path]:
    dataset_id = _dataset_identifier(spec, seed)
    cache_root = data_root / model_identifier / dataset_id
    serialized_path = cache_root / "test.pt"
    if not serialized_path.exists():
        raise FileNotFoundError(f"Expected cached dataset at {serialized_path}. Re-run main.py for this dataset to materialize the cache.")
    payload = torch.load(serialized_path, map_location="cpu")
    dataset = FlipFlopDataset(
        None,
        eval_args,
        getattr(eval_args, "train_random_n_prompts", -1),
        eval_args.per_device_eval_batch_size,
        serialized=payload,
    )
    meta = payload.get("meta", {})
    if "random_n_prompts" in meta:
        setattr(eval_args, "train_random_n_prompts", int(meta["random_n_prompts"]))
    if "split_answer_groups" in meta:
        setattr(eval_args, "split_answer_groups", int(meta["split_answer_groups"]))
    if "train_data_source" in meta:
        setattr(eval_args, "train_data_source", meta["train_data_source"])
    return dataset, serialized_path


def _collect_numeric(metrics: Dict[str, object]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            cleaned[key] = float(value)
        elif hasattr(value, "item"):
            try:
                cleaned[key] = float(value.item())
            except Exception:  # pragma: no cover
                continue
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run study_2 cross-dataset evaluation.")
    parser.add_argument("--ckpt_dir", required=True, help="Path to the training run directory containing checkpoints.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets to evaluate (default: all supported).",
    )
    parser.add_argument("--output_dir", required=True, help="Base output directory (will create results/<dataset_A>/<dataset>/).")
    parser.add_argument("--cache_dir", required=True, help="Cache directory for models/tokenizers/metrics.")
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root containing serialized dataset splits produced by main.py (default: data).",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip datasets without cached test.pt instead of failing.",
    )
    parser.add_argument("--best_ckpt", help="Optional explicit path to the best checkpoint directory.")
    parser.add_argument("--base_model_name", help="Optional base model name/path used during training.")
    parser.add_argument("--tokenizer_name", help="Optional tokenizer name/path to use during evaluation.")
    parser.add_argument("--dataset_a_label", help="Optional label for dataset_A (training dataset).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    handlers = [console_handler]

    root_logger = logging.getLogger()
    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)
        existing_handler.close()
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    LOGGER.setLevel(logging.NOTSET)
    LOGGER.propagate = True

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    LOGGER.info("Starting study_2 evaluation")
    LOGGER.info("Run directory: %s", ckpt_dir)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Cache directory: %s", cache_dir)
    cache_dir_str = str(cache_dir)
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset cache root {data_root} does not exist. Run Study 1 for each target dataset first."
        )
    LOGGER.info("Dataset cache root: %s", data_root)

    if args.best_ckpt:
        best_ckpt = Path(args.best_ckpt).expanduser().resolve()
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Provided best checkpoint {best_ckpt} does not exist.")
        LOGGER.info("Using user-provided best checkpoint: %s", best_ckpt)
    else:
        best_ckpt = _find_best_checkpoint(ckpt_dir)
        LOGGER.info("Best checkpoint resolved to: %s", best_ckpt)

    run_config = _parse_run_config(ckpt_dir / "run.sh")

    if args.base_model_name:
        base_model_name = args.base_model_name
        LOGGER.info("Using user-provided base model: %s", base_model_name)
    else:
        base_model_name = _resolve_base_model(best_ckpt, run_config)
        LOGGER.info("Base model inferred from checkpoint: %s", base_model_name)

    if args.tokenizer_name:
        tokenizer_name = args.tokenizer_name
        LOGGER.info("Using user-provided tokenizer: %s", tokenizer_name)
    else:
        tokenizer_name = _resolve_tokenizer_name(run_config)
        if tokenizer_name:
            LOGGER.info("Tokenizer inferred from run config: %s", tokenizer_name)
        else:
            LOGGER.info("Tokenizer not specified; using model defaults.")

    if args.datasets:
        dataset_labels_input = list(args.datasets)
        LOGGER.info("User-provided datasets: %s", ", ".join(dataset_labels_input))
    else:
        dataset_labels_input = list(DEFAULT_DATASETS)
        LOGGER.info("No datasets specified; defaulting to: %s", ", ".join(dataset_labels_input))

    for label in dataset_labels_input:
        if label not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset label '{label}'.")

    if args.dataset_a_label:
        dataset_a_label = args.dataset_a_label
        LOGGER.info("Using user-provided dataset_A label: %s", dataset_a_label)
    else:
        dataset_a_label = _infer_dataset_label(run_config) or _sanitize_identifier(ckpt_dir.name)
        LOGGER.info("Dataset_A inferred as: %s", dataset_a_label)

    dataset_order: List[str] = []

    def _append_unique(label: Optional[str]) -> None:
        if not label:
            return
        if label not in DATASET_SPECS:
            LOGGER.warning("Dataset label '%s' is not recognized; skipping it for evaluation order.", label)
            return
        if label not in dataset_order:
            dataset_order.append(label)

    _append_unique(dataset_a_label)
    for label in dataset_labels_input:
        _append_unique(label)

    if not dataset_order:
        raise ValueError("No valid datasets specified for evaluation.")

    LOGGER.info("Evaluation order: %s", ", ".join(dataset_order))

    training_args_path = best_ckpt / "training_args.bin"
    if not training_args_path.exists():
        raise FileNotFoundError(f"training_args.bin not found at {training_args_path}.")
    training_args = torch.load(training_args_path, weights_only=False)
    if not isinstance(training_args, TrainingArguments):
        raise TypeError("Loaded training_args.bin is not a TrainingArguments instance.")

    set_seed(getattr(training_args, "seed", 42))
    training_args.cache_dir = cache_dir_str

    tokenizer, pad_added = _prepare_tokenizer(best_ckpt, base_model_name, tokenizer_name, cache_dir_str)
    LOGGER.info("Tokenizer ready (pad_added=%s).", pad_added)
    model = _load_model(best_ckpt, base_model_name, tokenizer, pad_added, cache_dir_str)
    model.eval()
    LOGGER.info("Model loaded and switched to eval mode.")

    setattr(model.config, "loss_option", getattr(training_args, "loss_option", ""))
    setattr(model.config, "pseudo_train_loss_weight", getattr(training_args, "pseudo_train_loss_weight", 1.0))
    setattr(model.config, "cce_weight", getattr(training_args, "cce_weight", 1.0))

    pad_multiple = 8 if getattr(training_args, "fp16", False) else None
    data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=pad_multiple, expand_list=True)
    data_collator.model = model

    model_identifier = _make_model_identifier(base_model_name, tokenizer_name)
    seed = int(getattr(training_args, "seed", 42))

    output_root = Path(args.output_dir).expanduser().resolve()
    results_root = output_root / "results" / dataset_a_label
    os.makedirs(results_root, exist_ok=True)
    LOGGER.info("Results will be written under %s", results_root)

    summary: Dict[str, Dict[str, float]] = {}

    for dataset_label in dataset_order:
        spec = DATASET_SPECS[dataset_label]
        dataset_out_dir = results_root / dataset_label
        os.makedirs(dataset_out_dir, exist_ok=True)
        LOGGER.info("[%s] Output directory: %s", dataset_label, dataset_out_dir)
        eval_args = _prepare_eval_args(training_args, dataset_out_dir, spec)
        LOGGER.info("[%s] Training arguments: %s", dataset_label, training_args)
        LOGGER.info("[%s] Evaluation arguments: %s", dataset_label, eval_args)

        try:
            dataset, serialized_path = _load_dataset_from_cache(
                spec,
                model_identifier,
                seed,
                eval_args,
                data_root,
            )
            LOGGER.info("[%s] Loaded cached dataset from %s", dataset_label, serialized_path)
        except FileNotFoundError as exc:
            if args.skip_missing:
                LOGGER.warning("[%s] Skipping dataset due to missing cache: %s", dataset_label, exc)
                continue
            raise

        metric = load_metric(spec["metric_name"], cache_dir=cache_dir_str)
        LOGGER.info("[%s] Loaded metric '%s'", dataset_label, spec["metric_name"])

        trainer = FlipFlopTrainer(
            model=model,
            args=eval_args,
            train_dataset=None,
            eval_dataset=dataset,
            dev_dataset=None,
            processing_class=tokenizer,
            data_collator=data_collator,
            test_data_collator=data_collator,
            compute_metrics=compute_metrics,
            compute_unsupervised_metrics=compute_unsupervised_metrics,
            additional_metrics=metric,
        )

        LOGGER.info("[%s] Starting evaluation", dataset_label)
        metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix="eval", force_output_suffix=dataset_label)
        LOGGER.info("[%s] Evaluation complete", dataset_label)
        numeric_metrics = _collect_numeric(metrics)
        summary[dataset_label] = numeric_metrics
        LOGGER.info("[%s] Metrics: %s", dataset_label, numeric_metrics)

    summary_path = results_root / "study_2_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, sort_keys=True)
    LOGGER.info("Saved summary metrics to %s", summary_path)
    LOGGER.info("Processed %d dataset(s).", len(summary))


if __name__ == "__main__":
    main()
