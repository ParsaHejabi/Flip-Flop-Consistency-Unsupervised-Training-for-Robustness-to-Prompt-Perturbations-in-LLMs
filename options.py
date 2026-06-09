from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script " "with private models)."},
    )


@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "name of dataset, e.g. super_glue"})

    prompt_set_name: str = field(metadata={"help": ""})  # same as dataset name?

    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})

    subset_name: Optional[str] = field(default="none", metadata={"help": "name of dataset, e.g. "})

    task_type: Optional[str] = field(default="classification", metadata={"choices": ["generation", "classification"], "help": ""})

    testset_name: Optional[str] = field(default="test", metadata={"help": ""})

    cb_surgery: Optional[int] = field(default=0, metadata={"help": ""})

    abl_nprompts: Optional[int] = field(default=-1, metadata={"help": "ablation study on number of prompts"})


@dataclass
class TestArguments:
    train_data_source: Optional[str] = field(default="stream", metadata={"choices": ["stream", "train", "validation"], "help": "stream trains on one single test data"})

    train_duplicates: Optional[int] = field(default=1, metadata={"help": "> 1 to create larger batch size"})

    peft_option: str = field(default="none", metadata={"choices": ["prompt_tuning", "lora", "bitfit", "none", "full"], "help": ""})

    self_train_option: str = field(
        default="none",
        metadata={
            "choices": ["single", "multiple", "none", "constrained"],
            "help": "for constrained, the number of prompts used is the same for test and train," "otherwise, test always use all prompts",
        },
    )

    use_deepspeed: Optional[bool] = field(
        default=False,
    )

    debug_size: Optional[int] = field(
        default=-1,
        metadata={"help": ""},
    )

    development_mode: Optional[bool] = field(
        default=False,
        metadata={"help": "Limit dataset sizes for fast development iterations."},
    )

    test_size: Optional[int] = field(
        default=-1,
        metadata={"help": "Maximum number of examples to use from the test split."},
    )

    max_dev_size: Optional[int] = field(
        default=1000,
        metadata={"help": "maximum number of examples for unsupervised dev metric"},
    )

    metric_name: Optional[str] = field(
        default="none",
    )

    train_random_n_prompts: Optional[int] = field(default=-1, metadata={"help": "number of prompts for one single example when minimizing the entropy"})

    prob_temperature: Optional[float] = field(default=1.0, metadata={"help": "peakify the probability distribution"})

    loss_option: Optional[str] = field(
        default="entropy",
        metadata={
            "help": "loss type for test mode",
            "choices": [
                "token_level_divergence",
                "entropy",
                "token_level_entropy",
                "consistency",
                "pseudo_train",
                "consistency_pseudo_train",
                "flip_flop",
                "flip_flop_with_most_confident",
                "flip_flop_pseudo_train",
                "flip_flop_consensus_cross_entropy",
                "consensus_cross_entropy",
            ],
        },
    )

    pseudo_train_loss_weight: Optional[float] = field(default=1.0, metadata={"help": "used to"})

    cce_weight: Optional[float] = field(default=1.0, metadata={"help": "weight applied to consensus_cross_entropy loss"})

    pseudo_dist: Optional[str] = field(default="smooth", metadata={"help": "type of pseudo distribution", "choices": ["smooth", "argmax"]})

    # options for consistency loss
    jsd: Optional[int] = field(
        default=1,
        metadata={"help": "jsd"},
    )

    detach_kl_left: Optional[int] = field(
        default=0,
        metadata={"help": "detach the left side of KL"},
    )

    detach_kl_right: Optional[int] = field(
        default=0,
        metadata={"help": "detach the right side of KL"},
    )

    # parameter-efficient tuning specific options:
    lora_rank: Optional[int] = field(
        default=-1,
        metadata={"help": "rank of LoRA"},
    )

    # lora
    lora_alpha: Optional[float] = field(
        default=-1,
        metadata={"help": "alpha of LoRA"},
    )

    lora_layer_k: Optional[int] = field(default=24, metadata={"help": "legacy LoRA layer-selection option"})

    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": ""},
    )

    prune_prompt: Optional[str] = field(default=None, metadata={"help": "format is xx:yy, xx is the option, yy is the hyperparameter"})

    ensemble_option: Optional[str] = field(default="avg_prob", metadata={"choices": ["avg_prob", "majority_vote"]})

    split_answer_groups: Optional[int] = field(default=1, metadata={"help": "group prompts by fixed answer-choice vocabulary when enabled"})

    disable_eval_mode: Optional[int] = field(default=0, metadata={"help": "if 1, disable eval mode at inference time per train step"})

    write_eval_logits: Optional[bool] = field(
        default=False,
        metadata={"help": "if True, persist per-prompt logits from compute_metrics during evaluation"},
    )

    # random ensemble not implemented yet
    pseudo_target_mode: Optional[str] = field(default="pairwise", metadata={"help": "how to produce the pseudo target", "choices": ["pairwise", "full_ensemble", "random_ensemble"]})

    ensemble_subset_size: Optional[float] = field(default=-1.0, metadata={"help": "<1, > 0, set when pseudo_target_mode=random_ensemble, " "use this ratio of prompts to compute ensemble"})

    min_train_steps: Optional[int] = field(default=300, metadata={"help": "get best ckpt after this many steps with the unsupervised metric"})

    ff_top_k_teachers: Optional[int] = field(
        default=3,
        metadata={"help": "number of teachers to select inside the consensus group"},
    )

    ff_unanimous_margin_tau: Optional[float] = field(
        default=0.4,
        metadata={"help": "median-margin threshold for unanimous cases"},
    )

    ff_flipflop_beta_jsd: Optional[float] = field(
        default=0.1,
        metadata={"help": "always-on JSD weight when flip-flop is active"},
    )

    ff_no_consensus_alpha_jsd: Optional[float] = field(
        default=1.0,
        metadata={"help": "JSD weight when there is no consensus or flip-flop is not applied"},
    )

    ff_weight_temp: Optional[float] = field(
        default=0.5,
        metadata={"help": "temperature for mapping LL gap to flip-flop weight"},
    )

    ff_weight_min: Optional[float] = field(
        default=0.5,
        metadata={"help": "minimum flip-flop weight"},
    )

    ff_weight_max: Optional[float] = field(
        default=1.5,
        metadata={"help": "maximum flip-flop weight"},
    )
