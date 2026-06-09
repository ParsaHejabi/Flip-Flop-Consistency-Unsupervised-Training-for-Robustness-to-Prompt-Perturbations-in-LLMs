#!/usr/bin/env python3
"""Launch a paper-aligned Study 1 run without private machine paths."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "study1.json"
METHODS = {
    "base": "flip_flop_consensus_cross_entropy",
    "swarm": "consistency",
    "cce": "consensus_cross_entropy",
    "f2c": "flip_flop_consensus_cross_entropy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", choices=METHODS, default="f2c")
    parser.add_argument("--model", choices=["qwen", "llama"], default="qwen")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache_dir")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--gpu", help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    parser.add_argument("--train-formats", type=int, default=-1, help="Use the first K formats for Study 3.")
    parser.add_argument("--development-mode", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra arguments passed to main.py after '--'.")
    return parser.parse_args()


def f2c_config(dataset: dict, model_key: str) -> dict:
    config = dataset["f2c"]
    if "qwen" in config:
        return config[model_key]
    return config


def add(command: list, name: str, value) -> None:
    command.extend([f"--{name}", str(value)])


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.dataset not in config["datasets"]:
        choices = ", ".join(config["datasets"])
        raise SystemExit(f"Unknown dataset '{args.dataset}'. Choose one of: {choices}")

    dataset = config["datasets"][args.dataset]
    common = config["common"]
    model_name = config["models"][args.model]
    method_cce_weight = dataset.get("cce_weight", 1.0)
    ff = None
    if args.method == "f2c":
        ff = f2c_config(dataset, args.model)
        method_cce_weight = ff["cce_weight"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}.{args.model}.{args.method}.seed{common['seed']}.{timestamp}"
    output_dir = args.output_root.expanduser().resolve() / args.dataset / run_name
    cache_dir = args.cache_dir.expanduser().resolve()

    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "--dataset_name", dataset["dataset_name"],
        "--subset_name", dataset["subset_name"],
        "--prompt_set_name", dataset["dataset_name"],
        "--testset_name", dataset["testset_name"],
        "--model_name_or_path", model_name,
        "--cache_dir", str(cache_dir),
        "--metric_name", "accuracy",
        "--peft_option", "none" if args.method == "base" else "lora",
        "--loss_option", METHODS[args.method],
        "--ensemble_option", "avg_prob",
        "--pseudo_dist", "smooth",
        "--pseudo_target_mode", "pairwise",
        "--train_data_source", "train",
        "--split_answer_groups", "0",
        "--jsd", "0",
        "--detach_kl_left", "1",
        "--detach_kl_right", "0",
        "--disable_eval_mode", "0",
        "--seed", str(common["seed"]),
        "--debug_size", "10000",
        "--max_dev_size", str(dataset.get("max_dev_size", 1000)),
        "--test_size", str(dataset.get("test_size", -1)),
        "--train_random_n_prompts", str(dataset["prompt_count"]),
        "--abl_nprompts", str(args.train_formats),
        "--per_device_train_batch_size", str(common["per_device_train_batch_size"]),
        "--per_device_eval_batch_size", str(common["per_device_eval_batch_size"]),
        "--gradient_accumulation_steps", str(common["gradient_accumulation_steps"]),
        "--learning_rate", str(common["learning_rate"]),
        "--max_steps", str(common["max_steps"]),
        "--min_train_steps", str(common["min_train_steps"]),
        "--num_train_epochs", "50",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "5",
        "--eval_strategy", "steps",
        "--eval_steps", "5",
        "--save_strategy", "steps",
        "--save_steps", "5",
        "--save_total_limit", "35",
        "--metric_for_best_model", "unsupervised_dev_loss",
        "--greater_is_better", "True",
        "--lora_rank", str(common["lora_rank"]),
        "--lora_alpha", str(common["lora_alpha"]),
        "--lora_dropout", str(common["lora_dropout"]),
        "--cce_weight", str(method_cce_weight),
        "--output_dir", str(output_dir),
        "--run_name", run_name,
        "--report_to", args.report_to,
        "--overwrite_output_dir",
        "--gradient_checkpointing", "True",
        "--weight_decay", "0.01",
        "--warmup_steps", "0",
    ]

    if args.method != "base":
        command.append("--do_train")
    if args.bf16:
        command.extend(["--bf16", "True"])
    if args.development_mode:
        command.extend(["--development_mode", "True"])

    if args.method == "f2c":
        assert ff is not None
        add(command, "ff_top_k_teachers", ff["top_k"])
        add(command, "ff_unanimous_margin_tau", ff["tau"])
        add(command, "ff_flipflop_beta_jsd", ff["beta_jsd"])
        add(command, "ff_no_consensus_alpha_jsd", 0.0)
        add(command, "ff_weight_temp", ff["weight_temp"])
        add(command, "ff_weight_min", ff["weight_min"])
        add(command, "ff_weight_max", ff["weight_max"])

    extra_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    command.extend(extra_args)

    env = os.environ.copy()
    env["HF_HOME"] = str(cache_dir)
    env["HF_DATASETS_CACHE"] = str(cache_dir)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(shlex.join(command))
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
