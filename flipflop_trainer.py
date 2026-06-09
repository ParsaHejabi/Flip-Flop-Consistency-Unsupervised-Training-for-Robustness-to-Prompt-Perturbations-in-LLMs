# flipflop_trainer.py
import logging
import math
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

# HF's default rewrite_logs helper unconditionally prefixes custom metrics with
# "train/" unless they use the special "eval_"/"test_" prefixes. We want our
# validation metrics to live under "valid/" in W&B, so patch the helper once at
# import time to preserve that prefix instead of re-labeling.
try:
    from transformers.integrations import integration_utils as _integration_utils  # type: ignore

    _orig_rewrite_logs = getattr(_integration_utils, "rewrite_logs", None)

    if callable(_orig_rewrite_logs):

        def _rewrite_logs_valid_aware(d):
            new_d = {}
            for k, v in d.items():
                if k.startswith("valid/"):
                    new_d[k] = v
                elif k.startswith("valid_"):
                    new_d[f"valid/{k[6:]}"] = v
                elif k.startswith("eval_"):
                    new_d[f"eval/{k[5:]}"] = v
                elif k.startswith("test_"):
                    new_d[f"test/{k[5:]}"] = v
                else:
                    new_d[f"train/{k}"] = v
            return new_d

        _integration_utils.rewrite_logs = _rewrite_logs_valid_aware  # type: ignore[attr-defined]
except Exception:
    # Best-effort patching only; fall back to HF defaults if something goes wrong.
    pass
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalPrediction

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -----------------------
# Callbacks
# -----------------------


class DevLossEarlyStoppingCallback(TrainerCallback):
    """Early stopping on dev loss with a minimum step budget."""

    def __init__(
        self,
        patience: int = 5,
        minimum_steps: int = 0,
        metric_key: str = "unsupervised_dev_loss",
    ) -> None:
        self.patience = max(0, patience)
        self.minimum_steps = max(0, minimum_steps)
        self.metric_key = metric_key
        self.best_metric: Optional[float] = None
        self.best_step: Optional[int] = None
        self.num_bad_evals = 0

    def _extract_metric(self, metrics: Dict[str, Any]) -> Optional[float]:
        if self.metric_key in metrics:
            value = metrics[self.metric_key]
        else:
            value = next((metrics[k] for k in metrics if k.endswith(self.metric_key)), None)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> TrainerControl:
        if not metrics:
            return control

        metric_value = self._extract_metric(metrics)
        if metric_value is None:
            return control

        global_step = int(getattr(state, "global_step", 0))
        if self.best_metric is None or metric_value < self.best_metric:
            prev_best = self.best_metric
            self.best_metric = metric_value
            self.best_step = global_step
            self.num_bad_evals = 0
            if prev_best is None:
                logger.info(
                    "[early_stop] Initialize best %s=%.6f at step %s",
                    self.metric_key,
                    metric_value,
                    global_step,
                )
            else:
                logger.info(
                    "[early_stop] Improvement: %s %.6f -> %.6f at step %s",
                    self.metric_key,
                    prev_best,
                    metric_value,
                    global_step,
                )
            return control

        if global_step < self.minimum_steps:
            # Respect minimum training budget before counting patience.
            return control

        self.num_bad_evals += 1
        logger.info(
            "[early_stop] No improvement on %s (current=%.6f, best=%.6f at step %s). Patience %d/%d",
            self.metric_key,
            metric_value,
            self.best_metric,
            self.best_step,
            self.num_bad_evals,
            self.patience,
        )

        if self.num_bad_evals >= self.patience:
            logger.info(
                "[early_stop] Patience exhausted after %s evaluations without improvement. Stopping training.",
                self.num_bad_evals,
            )
            control.should_training_stop = True

        return control


# -----------------------
# Small helpers
# -----------------------
def per_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Mean token CE over non-ignored tokens per example.
    Assumes any causal alignment has already been applied upstream.
    """
    if logits.size(1) == 0:
        return torch.zeros(logits.size(0), device=logits.device, dtype=logits.dtype)

    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.size())
    m = labels.ne(-100)
    tok = m.sum(1).clamp_min(1).float()
    return (ce * m).sum(1) / tok  # [B]


def derive_consensus_label(prompt_preds: Optional[List[int]]) -> Optional[int]:
    """Return the majority-label consensus if it exceeds 50% of prompts."""
    if not prompt_preds:
        raise ValueError("prompt_preds is empty")

    label_counts = Counter(prompt_preds)
    top = label_counts.most_common(1)
    if not top:
        return None

    consensus_label, consensus_count = top[0]
    if consensus_count > len(prompt_preds) / 2.0:
        return int(consensus_label)

    return None


def symmetric_jsd_over_prompts(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute symmetric JSD across prompts after aligning label tokens."""

    aligned = align_prompt_label_logits(logits, labels)
    if aligned is None:
        raise AssertionError("[symmetric_jsd_over_prompts] Somehow the value returned from align_prompt_label_logits is None.")

    aligned_logits, aligned_mask, _, _ = aligned
    num_prompts = aligned_logits.size(0)
    if num_prompts < 2:
        raise AssertionError("[symmetric_jsd_over_prompts] There must be at least 2 prompts.")

    logp = F.log_softmax(aligned_logits, dim=-1)
    mask = aligned_mask
    valid_counts = mask.sum(dim=0)

    masked_logp = logp.masked_fill(~mask[:, :, None], float("-inf"))
    logsum = torch.logsumexp(masked_logp, dim=0)
    log_mix = logsum - valid_counts.clamp_min(1).float().log()[:, None]
    log_mix = torch.where(valid_counts[:, None] > 0, log_mix, torch.zeros_like(log_mix))
    log_mix_expanded = log_mix.unsqueeze(0).expand_as(logp)

    logp_filled = torch.where(mask[:, :, None], logp, log_mix_expanded)

    kl_pm = F.kl_div(logp_filled, log_mix_expanded, reduction="none", log_target=True).sum(-1)
    kl_mp = F.kl_div(log_mix_expanded, logp_filled, reduction="none", log_target=True).sum(-1)
    jsd = 0.5 * (kl_pm + kl_mp)

    joint = mask.float() * (valid_counts > 0).float()
    denom = joint.sum().clamp_min(1.0)
    return (jsd * joint).sum() / denom


def calculate_swarm_consistency_loss(args: Any, lm_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Consistency loss computed over aligned label tokens for decoder-only prompts."""

    if lm_logits.size(0) == 0:
        raise AssertionError("[calculate_swarm_consistency_loss] The logits are empty.")

    aligned = align_prompt_label_logits(lm_logits, labels)
    if aligned is None:
        raise AssertionError("[calculate_swarm_consistency_loss] Somehow the value returned from align_prompt_label_logits is None.")

    aligned_logits, aligned_mask, _, _ = aligned
    if aligned_logits.size(0) == 0:
        raise AssertionError("[calculate_swarm_consistency_loss] The aligned logits are empty.")

    logp = F.log_softmax(aligned_logits, dim=-1)
    mask = aligned_mask
    valid_counts = mask.sum(dim=0)

    masked_logp = logp.masked_fill(~mask[:, :, None], float("-inf"))
    logsum = torch.logsumexp(masked_logp, dim=0)
    log_mix = logsum - valid_counts.clamp_min(1).float().log()[:, None]
    log_mix = torch.where(valid_counts[:, None] > 0, log_mix, torch.zeros_like(log_mix))
    mix_expanded = log_mix.unsqueeze(0).expand_as(logp)

    use_jsd = bool(getattr(args, "jsd", 1))
    detach_left = bool(getattr(args, "detach_kl_left", 0))
    detach_right = bool(getattr(args, "detach_kl_right", 0))

    if use_jsd:
        left = logp
        right = mix_expanded
    else:
        left = mix_expanded
        right = logp

    if detach_left:
        left = left.detach()
    if detach_right:
        right = right.detach()

    kl = F.kl_div(right, left, reduction="none", log_target=True).sum(-1)
    mask_float = mask.float()
    denom = mask_float.sum().clamp_min(1.0)
    return (kl * mask_float).sum() / denom


def flipflop_student_to_teacher_kl(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Student-to-teacher KL using aligned label tokens for decoder-only prompts."""

    tmask = teacher_mask.to(torch.bool)
    if tmask.sum() == 0 or tmask.sum() == tmask.numel():
        raise AssertionError("[flipflop_student_to_teacher_kl] There must be at least one teacher and one student.", tmask.sum(), tmask.numel())

    aligned = align_prompt_label_logits(logits, labels)
    if aligned is None:
        raise AssertionError("[flipflop_student_to_teacher_kl] Somehow the value returned from align_prompt_label_logits is None.")

    aligned_logits, aligned_mask, _, _ = aligned
    teacher_logits = aligned_logits[tmask]
    student_logits = aligned_logits[~tmask]

    if teacher_logits.size(0) == 0 or student_logits.size(0) == 0:
        raise AssertionError("[flipflop_student_to_teacher_kl] The teacher or student logits are empty.", teacher_logits.size(0), student_logits.size(0))

    teacher_mask_tokens = aligned_mask[tmask]
    student_mask_tokens = aligned_mask[~tmask]

    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    teacher_mask_exp = teacher_mask_tokens[:, :, None]
    teacher_logp_masked = teacher_logp.masked_fill(~teacher_mask_exp, float("-inf"))
    teacher_counts = teacher_mask_tokens.sum(dim=0)

    logsum = torch.logsumexp(teacher_logp_masked, dim=0)
    teacher_mix = logsum - teacher_counts.clamp_min(1).float().log()[:, None]
    teacher_mix = torch.where(teacher_counts[:, None] > 0, teacher_mix, torch.zeros_like(teacher_mix)).detach()

    student_logp = F.log_softmax(student_logits, dim=-1)
    mix_expanded = teacher_mix.unsqueeze(0).expand_as(student_logp)

    teacher_valid = teacher_counts > 0
    joint_mask = student_mask_tokens & teacher_valid.unsqueeze(0)
    if not joint_mask.any():
        raise AssertionError("[flipflop_student_to_teacher_kl] There must be at least one joint mask.", joint_mask.any())

    kl = F.kl_div(student_logp, mix_expanded, reduction="none", log_target=True).sum(-1)
    mask_float = joint_mask.float()
    return (kl * mask_float).sum() / mask_float.sum()


def align_prompt_label_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """
    Align label-token logits for decoder-only prompt variants.

    - All prompts must produce at least one supervised token.
    - Label tokens may differ in text/length, but must remain contiguous per prompt.
    - The aligned slice is padded so index 0 corresponds to each prompt's first label token.
    """

    if logits.ndim != 3 or labels.ndim != 2:
        raise AssertionError("Expected logits [K, L, V] and labels [K, L].")
    if logits.size(0) != labels.size(0) or logits.size(1) != labels.size(1):
        raise AssertionError("Logits and labels must share prompt and sequence dimensions.")

    mask = labels.ne(-100)
    lengths = mask.sum(dim=1)
    zero_lengths = lengths == 0
    if zero_lengths.any():
        raise ValueError("All labels must have at least one non-ignored token.")

    device = logits.device
    mask_device = labels.device

    lengths_list = [int(l) for l in lengths.tolist()]
    if not lengths_list:
        return None

    max_len = max(lengths_list)

    aligned_logits: List[torch.Tensor] = []
    aligned_mask: List[torch.Tensor] = []

    vocab_size = logits.size(-1)

    for prompt_idx, length in enumerate(lengths_list):
        valid_positions = mask[prompt_idx].nonzero(as_tuple=False).view(-1)

        start_idx = int(valid_positions[0].item())
        expected = torch.arange(start_idx, start_idx + length, device=valid_positions.device, dtype=torch.long)
        if not torch.equal(valid_positions[:length], expected):
            raise AssertionError("Label tokens must be contiguous within each prompt sequence.")

        take = expected
        prompt_logits = logits[prompt_idx].index_select(0, take)

        padded_logits = torch.zeros(max_len, vocab_size, device=device, dtype=logits.dtype)
        padded_logits[:length] = prompt_logits
        aligned_logits.append(padded_logits)

        row_mask = torch.zeros(max_len, dtype=torch.bool, device=mask_device)
        row_mask[:length] = True
        aligned_mask.append(row_mask)

    aligned_logits_t = torch.stack(aligned_logits, dim=0)
    aligned_mask_t = torch.stack(aligned_mask, dim=0)

    return aligned_logits_t, aligned_mask_t, lengths, max_len


# -----------------------
# Flip-Flop control derivation
# -----------------------
def derive_flipflop_controls(
    args: Any,
    prompt_preds: List[int],
    per_answer_ll: List[float],  # flattened length == num_prompts * num_choices
    num_prompts: int,
    num_choices: int,
) -> Dict[str, Any]:
    """
    Compute teacher mask, weights and diagnostics for flip-flop logic based on
    per-prompt predictions and per-answer log-likelihoods.

    Returns keys:
      - flip_teacher_mask_all: Optional[List[int]] over prompts (1=teacher)
      - flip_weight: float in [ff_weight_min, ff_weight_max]
      - consensus_label: Optional[int]
      - branch: str diagnostic label
      - C, m_med, k, delta: diagnostics
      - teachers, students: prompt indices
      - ens_pred: ensemble predictions (avg or vote per args.ensemble_option)
    """
    from collections import Counter

    import numpy as _np

    K = int(num_prompts)
    C = int(num_choices)
    logprobs_arr = _np.array(per_answer_ll, dtype=_np.float32).reshape(K, C)

    flip_teacher_mask_all: Optional[List[int]] = None
    flip_weight: float = 0.0
    consensus_label: Optional[int] = None
    branch: str = "no_consensus"
    C_count: int = 0
    m_med: float = 0.0
    m_min: float = 0.0
    m_max: float = 0.0
    k: int = 0
    delta: float = 0.0
    teachers: List[int] = []
    students: List[int] = []

    loss_opt = str(getattr(args, "loss_option", ""))
    if "flip_flop" in loss_opt and len(prompt_preds) > 0:
        counts = Counter(prompt_preds)
        max_label, max_count = counts.most_common(1)[0]
        if max_count > K / 2.0:
            consensus_label = int(max_label)
            G = [i for i, p in enumerate(prompt_preds) if p == consensus_label]
            C_count = len(G)
            margins: Dict[int, float] = {}
            for i in G:
                row = logprobs_arr[i]
                others = _np.delete(row, consensus_label)
                margins[i] = float(row[consensus_label] - (others.max() if others.size > 0 else 0.0))
            if margins:
                margin_values = list(margins.values())
                m_med = float(_np.median(margin_values))
                m_min = float(min(margin_values))
                m_max = float(max(margin_values))

            tau = float(getattr(args, "ff_unanimous_margin_tau", 0.0))
            if C_count == K and m_med >= tau:
                branch = "unanimous_strong"
            elif C_count >= 2:
                k = min(int(getattr(args, "ff_top_k_teachers", 0)), C_count - 1)
                if k >= 2:
                    sorted_g = sorted(G, key=lambda i: margins[i], reverse=True)
                    teachers = list(sorted_g[:k])
                    students = [i for i in range(K) if i not in teachers]
                    if len(teachers) >= 2 and len(students) > 0:
                        flip_teacher_mask_all = [1 if i in teachers else 0 for i in range(K)]
                        mean_teach = float(logprobs_arr[teachers, consensus_label].mean())
                        mean_stud = float(logprobs_arr[students, consensus_label].mean())
                        delta = float(mean_teach - mean_stud)
                        temp = max(1e-6, float(getattr(args, "ff_weight_temp", 1.0)))
                        # use torch.sigmoid for parity with training-time behavior
                        w = float(torch.sigmoid(torch.tensor(delta / temp)).item())
                        fmin = float(getattr(args, "ff_weight_min", 0.0))
                        fmax = float(getattr(args, "ff_weight_max", 1e9))
                        flip_weight = float(fmin + (fmax - fmin) * w)
                        branch = "flip_flop: unanimous_weak" if C_count == K else "flip_flop: mixed"
                    else:
                        branch = "no_consensus: len(teachers) < 2 or len(students) == 0"
                else:
                    branch = "no_consensus: k < 2"
            else:
                branch = "no_consensus: C < 2"
        else:
            branch = "no_consensus: max_count <= K / 2"

    return {
        "flip_teacher_mask_all": flip_teacher_mask_all,
        "flip_weight": flip_weight,
        "consensus_label": consensus_label,
        "branch": branch,
        "C": C_count,
        "m_med": m_med,
        "m_min": m_min,
        "m_max": m_max,
        "k": k,
        "delta": delta,
        "teachers": teachers,
        "students": students,
    }


class FlipFlopTrainer(Trainer):
    """
    Trainer that understands your nested inner-batch structure from FlipFlopDataset:
      [ eval_inner_0 .. eval_inner_{E-1}, train_inner_0 .. train_inner_{T-1} ]
    """

    def __init__(
        self,
        *args,
        dev_dataset=None,  # kept for symmetry with your main.py
        test_data_collator=None,
        compute_unsupervised_metrics=None,  # optional extra hook
        additional_metrics=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dev_dataset = dev_dataset
        self.test_data_collator = test_data_collator
        self.compute_unsupervised_metrics = compute_unsupervised_metrics
        self.additional_metrics = additional_metrics
        self._component_names = ("nll_loss", "jsd_loss", "flip_loss", "swarm_loss", "consensus_cross_entropy")
        self._train_component_totals = {name: 0.0 for name in self._component_names}
        self._train_component_counts = {name: 0.0 for name in self._component_names}
        self._train_component_last_logged_step = 0

    # -----------------------
    # Dataloaders (override to preserve nested batches)
    # -----------------------
    def get_eval_dataloader(self, eval_dataset: Optional[Any] = None) -> DataLoader:  # type: ignore[override]
        """Force batch_size=1 for nested FlipFlopDataset to avoid collapsing multiple outers into one.

        This prevents the situation where only the first outer example contributes to metrics (n_outer=1) because
        a larger batch triggers the collator to pick the first element only.
        """
        # Defer to HF default if no dataset
        if eval_dataset is None and getattr(self, "eval_dataset", None) is None:
            return super().get_eval_dataloader(eval_dataset)

        ds = eval_dataset if eval_dataset is not None else self.eval_dataset

        # Detect our nested loop dataset type without importing globally to avoid circular deps.
        is_nested = False
        try:
            from dataloader import FlipFlopDataset  # local import

            is_nested = isinstance(ds, FlipFlopDataset)
        except Exception:
            # Heuristic fallback: presence of dev_size and tot_single_ds_size suggests nested FlipFlop dataset
            is_nested = hasattr(ds, "dev_size") and hasattr(ds, "tot_single_ds_size")

        if not is_nested:
            return super().get_eval_dataloader(eval_dataset)

        # For nested datasets, build a DataLoader with batch_size=1 and our test_data_collator
        eval_sampler = self._get_eval_sampler(ds)  # type: ignore[arg-type]
        return DataLoader(
            ds,
            sampler=eval_sampler,
            batch_size=1,
            collate_fn=self.test_data_collator if self.test_data_collator is not None else self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Any) -> DataLoader:  # type: ignore[override]
        """Matching behavior for test set dataloader: keep nested structure by forcing batch_size=1."""
        ds = test_dataset
        is_nested = False
        try:
            from dataloader import FlipFlopDataset  # local import

            is_nested = isinstance(ds, FlipFlopDataset)
        except Exception:
            is_nested = hasattr(ds, "dev_size") and hasattr(ds, "tot_single_ds_size")

        if not is_nested:
            return super().get_test_dataloader(test_dataset)

        test_sampler = self._get_eval_sampler(ds)  # type: ignore[arg-type]
        return DataLoader(
            ds,
            sampler=test_sampler,
            batch_size=1,
            collate_fn=self.test_data_collator if self.test_data_collator is not None else self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _write_loss_components_to_file(
        self,
        fout_dir: str,
        suffix: str,
        metric_key_prefix: str,
        component_avgs: Dict[str, float],
        total_loss: Optional[float],
    ) -> None:
        if not component_avgs and total_loss is None:
            return

        base_prefix = "unsupervised_dev" if metric_key_prefix.startswith("unsupervised_dev") else "accuracy"
        if fout_dir.startswith("results"):
            out_path = f"{fout_dir}.{base_prefix}_{suffix}"
        else:
            out_path = os.path.join(fout_dir, f"{base_prefix}_{suffix}")

        try:
            with open(out_path, "a") as fout:
                if total_loss is not None:
                    fout.write(f"loss={total_loss}\n")
                for name, value in component_avgs.items():
                    fout.write(f"{name}={value}\n")
        except OSError:
            # Best-effort logging; skip if we can't write.
            pass

    # -----------------------
    # Core loss (flat batch)
    # -----------------------
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,  # required by recent HF
    ):
        # Flip-flop knobs (from Training/TestArguments forwarded into self.args in main.py)
        loss_opt = str(getattr(self.args, "loss_option", "consistency"))
        beta_jsd = float(getattr(self.args, "ff_flipflop_beta_jsd", 0.0))
        noncons_a = float(getattr(self.args, "ff_no_consensus_alpha_jsd", 0.0))
        pseudo_w = float(getattr(self.args, "pseudo_train_loss_weight", 1.0))
        cce_weight = float(getattr(self.args, "cce_weight", 1.0))

        # Optional extra fields your dataset may provide (handle BatchEncoding/pop semantics)
        flip_teacher_mask = inputs.pop("flip_teacher_mask") if (hasattr(inputs, "__contains__") and "flip_teacher_mask" in inputs) else None  # [K] or None
        is_true_answer_state = inputs.pop("is_true_answer_state") if (hasattr(inputs, "__contains__") and "is_true_answer_state" in inputs) else -1  # scalar/list/tensor
        flip_loss_weight = float(inputs.pop("flip_loss_weight")) if (hasattr(inputs, "__contains__") and "flip_loss_weight" in inputs) else 0.0
        nonconsistency_alpha = float(inputs.pop("nonconsistency_alpha")) if (hasattr(inputs, "__contains__") and "nonconsistency_alpha" in inputs) else noncons_a
        consensus_is_active = inputs.pop("consensus_is_active") if (hasattr(inputs, "__contains__") and "consensus_is_active" in inputs) else 0.0

        if torch.is_tensor(consensus_is_active):
            consensus_is_active = float(consensus_is_active.detach().float().item())
        elif isinstance(consensus_is_active, (list, tuple)):
            consensus_is_active = float(consensus_is_active[0]) if consensus_is_active else 0.0
        else:
            try:
                consensus_is_active = float(consensus_is_active)
            except (TypeError, ValueError):
                consensus_is_active = 0.0

        # Vanilla forward (seq2seq or causal)
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits
        labels = inputs["labels"] if (hasattr(inputs, "__contains__") and "labels" in inputs) else None

        decoder_only = not getattr(getattr(model, "config", None), "is_encoder_decoder", False)
        if labels is not None and decoder_only and hasattr(logits, "ndim") and logits.ndim == 3 and torch.is_tensor(labels) and labels.ndim == 2:
            if logits.size(1) != labels.size(1):
                raise ValueError("Decoder-only alignment requires matching sequence lengths.")
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

        aligned_seq_len = logits.size(1) if torch.is_tensor(logits) and getattr(logits, "ndim", 0) >= 2 else 0

        # Per-variant NLL
        nll_vec = per_example_nll(logits, labels) if labels is not None else None
        if nll_vec is not None:
            if isinstance(is_true_answer_state, list):
                w = torch.tensor(is_true_answer_state, device=nll_vec.device, dtype=nll_vec.dtype)
                nll_scalar = (nll_vec * w).mean()
            elif torch.is_tensor(is_true_answer_state) and is_true_answer_state.ndim > 0:
                nll_scalar = (nll_vec * is_true_answer_state.to(nll_vec)).mean()
            elif isinstance(is_true_answer_state, (int, float)) and is_true_answer_state > 0:
                nll_scalar = nll_vec.mean() * float(is_true_answer_state)
            else:
                nll_scalar = nll_vec.mean()
        else:
            nll_scalar = torch.tensor(0.0, device=logits.device)

        # Component bookkeeping for logging
        comp = dict(
            # raw: true/normalized component value prior to any external weighting
            nll_loss_raw=(nll_scalar if nll_vec is not None else torch.tensor(0.0, device=logits.device)),
            nll_loss=torch.tensor(0.0, device=logits.device),
            jsd_loss_raw=torch.tensor(0.0, device=logits.device),
            jsd_loss=torch.tensor(0.0, device=logits.device),
            flip_loss_raw=torch.tensor(0.0, device=logits.device),
            flip_loss=torch.tensor(0.0, device=logits.device),
            swarm_loss_raw=torch.tensor(0.0, device=logits.device),
            swarm_loss=torch.tensor(0.0, device=logits.device),
            consensus_cross_entropy_raw=torch.tensor(0.0, device=logits.device),
            consensus_cross_entropy=torch.tensor(0.0, device=logits.device),
        )

        consensus_flag = torch.tensor(float(consensus_is_active), device=logits.device, dtype=logits.dtype)
        if nll_vec is not None:
            consensus_base = nll_vec.mean()
        else:
            consensus_base = logits[..., 0].mean() * 0.0
        consensus_ce_raw = consensus_base * consensus_flag

        # ------- choose loss -------
        if loss_opt == "consistency":
            # Use the modeling_t5 consistency objective; then normalize across valid tokens
            if aligned_seq_len == 0:
                raw = logits.new_zeros(())
            else:
                raw = calculate_swarm_consistency_loss(self.args, logits, labels)
            comp["swarm_loss_raw"] = raw
            comp["swarm_loss"] = raw
            loss = comp["swarm_loss"]
            logger.info(
                "[loss_components] mode=consistency swarm_raw=%.6f swarm=%.6f",
                float(comp["swarm_loss_raw"].detach().item()),
                float(comp["swarm_loss"].detach().item()),
            )

        elif loss_opt == "consistency_pseudo_train":
            if aligned_seq_len == 0:
                raw = logits.new_zeros(())
            else:
                raw = calculate_swarm_consistency_loss(self.args, logits, labels)
            comp["swarm_loss_raw"] = raw
            comp["swarm_loss"] = raw
            comp["nll_loss_raw"] = nll_scalar
            comp["nll_loss"] = nll_scalar * pseudo_w
            loss = comp["swarm_loss"] + comp["nll_loss"]
            logger.info(
                "[loss_components] mode=consistency_pseudo_train swarm_raw=%.6f swarm=%.6f nll_raw=%.6f nll=%.6f weight=%.4f",
                float(comp["swarm_loss_raw"].detach().item()),
                float(comp["swarm_loss"].detach().item()),
                float(comp["nll_loss_raw"].detach().item()),
                float(comp["nll_loss"].detach().item()),
                pseudo_w,
            )

        elif loss_opt == "pseudo_train":
            comp["nll_loss_raw"] = nll_scalar
            comp["nll_loss"] = nll_scalar
            loss = comp["nll_loss"]
            logger.info(
                "[loss_components] mode=pseudo_train nll_raw=%.6f nll=%.6f",
                float(comp["nll_loss_raw"].detach().item()),
                float(comp["nll_loss"].detach().item()),
            )

        elif loss_opt == "consensus_cross_entropy":
            comp["consensus_cross_entropy_raw"] = consensus_ce_raw
            comp["consensus_cross_entropy"] = consensus_ce_raw * cce_weight
            loss = comp["consensus_cross_entropy"]
            logger.info(
                "[loss_components] mode=consensus_cross_entropy cce_raw=%.6f cce=%.6f weight=%.4f",
                float(comp["consensus_cross_entropy_raw"].detach().item()),
                float(comp["consensus_cross_entropy"].detach().item()),
                cce_weight,
            )

        elif loss_opt in {"flip_flop", "flip_flop_pseudo_train", "flip_flop_consensus_cross_entropy"}:
            consensus_gate_required = loss_opt == "flip_flop_consensus_cross_entropy"
            consensus_gate_active = bool(consensus_flag.detach().item() >= 0.5)

            if consensus_gate_required and not consensus_gate_active:
                comp["_skip_component_logging"] = torch.tensor(1.0, device=logits.device)
                loss = logits.new_zeros(())
                logger.info("[flip_flop:loss] consensus inactive; skipping flip_flop_consensus_cross_entropy components.")
            else:
                tmask = None
                if flip_teacher_mask is not None:
                    if torch.is_tensor(flip_teacher_mask):
                        tmask = flip_teacher_mask.to(dtype=torch.bool, device=logits.device)
                    else:
                        tmask = torch.tensor(flip_teacher_mask, device=logits.device, dtype=torch.bool)

                valid_teacher_split = bool(tmask is not None and tmask.numel() > 0 and tmask.any() and (~tmask).any())
                mask_list: Optional[List[int]]
                if tmask is None:
                    mask_list = None
                else:
                    mask_list = [int(x) for x in tmask.detach().cpu().tolist()]
                logger.info(
                    "[flip_flop:loss] valid_teacher_split=%s beta_jsd=%.4f noncons_alpha=%.4f flip_loss_weight=%.4f mask=%s consensus_active=%.1f",
                    valid_teacher_split,
                    beta_jsd,
                    nonconsistency_alpha,
                    float(flip_loss_weight),
                    mask_list,
                    float(consensus_is_active),
                )

                if valid_teacher_split and beta_jsd > 0.0:
                    jsd_t = symmetric_jsd_over_prompts(logits[tmask], labels[tmask])
                    comp["jsd_loss_raw"] = jsd_t
                    comp["jsd_loss"] = beta_jsd * jsd_t
                    logger.info(
                        "[flip_flop:loss_components] teacher_split=True jsd_raw=%.6f jsd_weighted=%.6f beta_jsd=%.4f",
                        float(jsd_t.detach().item()),
                        float(comp["jsd_loss"].detach().item()),
                        beta_jsd,
                    )

                ff = flipflop_student_to_teacher_kl(logits, labels, tmask) if valid_teacher_split else None
                if ff is not None:
                    comp["flip_loss_raw"] = ff
                    comp["flip_loss"] = float(flip_loss_weight) * ff
                    logger.info(
                        "[flip_flop:loss_components] flip_loss_raw=%.6f flip_loss=%.6f flip_weight=%.4f",
                        float(ff.detach().item()),
                        float(comp["flip_loss"].detach().item()),
                        float(flip_loss_weight),
                    )

                if not valid_teacher_split:
                    jsd_all = symmetric_jsd_over_prompts(logits, labels)
                    comp["jsd_loss_raw"] = jsd_all
                    comp["jsd_loss"] = nonconsistency_alpha * jsd_all
                    logger.info(
                        "[flip_flop:loss_components] teacher_split=False jsd_all_raw=%.6f jsd_all_weighted=%.6f noncons_alpha=%.4f",
                        float(jsd_all.detach().item()),
                        float(comp["jsd_loss"].detach().item()),
                        nonconsistency_alpha,
                    )

                loss = comp["flip_loss"] + comp["jsd_loss"]
                if loss_opt == "flip_flop_pseudo_train":
                    comp["nll_loss"] = nll_scalar * pseudo_w
                    loss = loss + comp["nll_loss"]
                elif loss_opt == "flip_flop_consensus_cross_entropy":
                    comp["consensus_cross_entropy_raw"] = consensus_ce_raw
                    comp["consensus_cross_entropy"] = consensus_ce_raw * cce_weight
                    loss = loss + comp["consensus_cross_entropy"]
                    logger.info(
                        "[flip_flop:loss_components] consensus_ce_raw=%.6f consensus_ce=%.6f weight=%.4f",
                        float(consensus_ce_raw.detach().item()),
                        float(comp["consensus_cross_entropy"].detach().item()),
                        cce_weight,
                    )

        else:
            raise ValueError(f"Unknown loss_option={loss_opt}")

        # Component logging — routed to W&B when report_to includes "wandb". Also capture latest components for eval aggregation.
        comp_detached = {k: float(v.detach().item()) for k, v in comp.items()}
        self._last_loss_components = comp_detached
        return (loss, outputs) if return_outputs else loss

    # -----------------------
    # Cache per-example LL during eval inners
    # -----------------------
    def prediction_step(self, model: nn.Module, inputs: Dict[str, Any], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None):
        # Special path: eval_inners should not compute component losses; only get logits/labels and LL.
        skip_components = False
        if isinstance(inputs, Mapping) and ("_skip_loss_components" in inputs):
            skip_components = bool(inputs["_skip_loss_components"])  # type: ignore[index]
            # Avoid passing the marker to the model
            if hasattr(inputs, "pop"):
                inputs = dict(inputs)
                inputs.pop("_skip_loss_components", None)

        if skip_components:
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])  # type: ignore[attr-defined]
                else:
                    ignore_keys = []
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # When labels are present, also ignore the training-only "loss" key from outputs.
            effective_ignore = list(ignore_keys or []) + (["loss"] if has_labels else [])

            inputs_prep = self._prepare_inputs(inputs)
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    outputs = model(**inputs_prep)

            # Prefer the model's primary logits tensor when available.
            logits = None
            if hasattr(outputs, "logits"):
                logits = outputs.logits  # type: ignore[attr-defined]
            else:
                if isinstance(outputs, dict):
                    if "logits" in outputs:
                        logits = outputs["logits"]  # type: ignore[index]
                    else:
                        # Fallback: filter out ignored keys (including loss) and keep remaining values.
                        kept = tuple(v for k, v in outputs.items() if k not in effective_ignore)
                        logits = kept[0] if len(kept) == 1 else kept
                else:
                    logits = outputs

            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index - 1]  # type: ignore[index]

            # Extract labels without popping
            if has_labels:
                from transformers.trainer_pt_utils import nested_detach as _nested_detach  # local import to avoid top-level dependency

                labels = _nested_detach(tuple(inputs_prep.get(name) for name in self.label_names))  # type: ignore[arg-type]
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            # Standardize logits
            from transformers.trainer_pt_utils import nested_detach as _nested_detach2

            logits = _nested_detach2(logits)
            if isinstance(logits, (list, tuple)) and len(logits) == 1:
                logits = logits[0]

            loss = None
        else:
            loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # Cache per-example log-likelihoods for metrics when available (eval mode only)
        if (logits is not None) and (labels is not None) and hasattr(logits, "ndim") and getattr(logits, "ndim", 0) == 3 and not model.training:
            with torch.no_grad():
                logits_t = torch.as_tensor(logits)
                labels_t = torch.as_tensor(labels)
                decoder_only = not getattr(getattr(model, "config", None), "is_encoder_decoder", False)
                if decoder_only and logits_t.ndim == 3 and torch.is_tensor(labels_t) and labels_t.ndim == 2:
                    if logits_t.size(1) != labels_t.size(1):
                        raise ValueError("Decoder-only alignment requires matching sequence lengths.")
                    logits_t = logits_t[:, :-1, :].contiguous()
                    labels_t = labels_t[:, 1:].contiguous()

                ll = -per_example_nll(logits_t, labels_t)
            if not hasattr(self, "_eval_cache"):
                self._eval_cache = {}
            self._eval_cache.setdefault("per_example_ll", []).append(ll.detach().cpu())
        return loss, logits, labels

    # -----------------------
    # Nested "dev then train" micro-steps per outer example
    # -----------------------
    def training_step(
        self,
        model: nn.Module,
        inputs: Any,
        num_items_in_batch: Optional[torch.Tensor] = None,  # required by recent HF
    ) -> torch.Tensor:
        prev_training_state = getattr(self, "is_in_train", False)
        self.is_in_train = True
        step_component_totals = {name: 0.0 for name in self._component_names}
        step_component_count = 0

        # Our collator (expand_list=True) returns List[Dict] for a single outer example
        try:
            if isinstance(inputs, list) and inputs and isinstance(inputs[0], Mapping):
                # Derive how many eval inners: prefer dataset.dev_size, else fallback arg
                assert getattr(self.train_dataset, "dev_size", None) is not None, "dev_size is not set in the train_dataset"
                E = self.train_dataset.dev_size
                loss_opt = str(getattr(self.args, "loss_option", ""))
                eval_inners, train_inners = inputs[:E], inputs[E:]

            ens_pred = None
            ff_ctrl: Optional[Dict[str, Any]] = None
            prompt_preds_seq: Optional[List[int]] = None
            consensus_label: Optional[int] = None

            # 1) eval inners (no grad): collect LL and call compute_metrics if provided
            if len(eval_inners) > 0:
                model.eval()
                self._eval_cache = {}
                with torch.no_grad():
                    for inner in eval_inners:
                        marker = {"_skip_loss_components": True}
                        if isinstance(inner, Mapping):
                            inner_marked = dict(inner)
                            inner_marked.update(marker)
                        else:
                            inner_marked = inner
                            inner_marked["_skip_loss_components"] = True  # type: ignore[index]
                        _ = self.prediction_step(model, inner_marked, prediction_loss_only=False)
                if getattr(self, "_eval_cache", None):
                    # For inner-eval logging, utils.compute_metrics expects
                    # (logprobs, num_examples, num_targets, num_prompts, ...).
                    # We collected per-answer log-likelihoods in pll.
                    import torch as _t

                    pll = _t.cat(self._eval_cache.get("per_example_ll", [])).tolist() if "per_example_ll" in self._eval_cache else None
                    assert pll is not None, "per_example_ll is not set in the eval_cache"
                    assert self.compute_metrics is not None, "compute_metrics is not set in the trainer"
                    n_examples = 1
                    n_targets = getattr(self.train_dataset, "num_choices", None)
                    n_prompts = getattr(self.train_dataset, "num_prompts", None)
                    # Fallback-safe defaults
                    if n_targets is None or n_prompts is None:
                        raise ValueError("Missing num_choices/num_prompts on train_dataset for metrics computation.")

                    if self.args.pseudo_target_mode == "pairwise":  # TODO: Check the use of `pseudo_target_mode` and whether we need it or not.
                        prompt_preds, avg_ens_pred, vote_ens_pred, rand_indices = self.compute_metrics(
                            pll,
                            n_examples,
                            n_targets,
                            n_prompts,
                            pseudo_dist=self.args.pseudo_dist,
                            return_all_prompt_preds=True,
                            random_selection_ensemble=self.args.ensemble_subset_size,
                            self_train=self.args.self_train_option != "none",
                        )
                        # sort them back to the original order
                        inv_perm = np.argsort(rand_indices)
                        avg_ens_pred = [avg_ens_pred[i] for i in inv_perm]
                        vote_ens_pred = [vote_ens_pred[i] for i in inv_perm]
                    else:
                        prompt_preds, avg_ens_pred, vote_ens_pred, _ = self.compute_metrics(
                            pll,
                            n_examples,
                            n_targets,
                            n_prompts,
                            pseudo_dist=self.args.pseudo_dist,
                            return_all_prompt_preds=False,
                            random_selection_ensemble=self.args.ensemble_subset_size,
                            self_train=self.args.self_train_option != "none",
                        )

                    prompt_preds_seq = list(prompt_preds) if prompt_preds is not None else None

                    if self.args.ensemble_option == "avg_prob":
                        ens_pred = avg_ens_pred
                    elif self.args.ensemble_option == "majority_vote":
                        ens_pred = vote_ens_pred
                    else:
                        raise ValueError("unknown ensemble: {}".format(self.args.ensemble_option))

                    if "flip_flop" in loss_opt:
                        ff_ctrl = derive_flipflop_controls(
                            self.args,
                            prompt_preds,
                            pll,
                            int(n_prompts),
                            int(n_targets),
                        )
                        mask_all = ff_ctrl.get("flip_teacher_mask_all")
                        teacher_total = int(sum(mask_all)) if isinstance(mask_all, list) else 0
                        logger.info(
                            (
                                "[flip_flop:diagnostics] branch=%s, consensus=%s, K=%s, C=%s, "
                                "m_med=%.4f, m_min=%.4f, m_max=%.4f, delta=%.4f, flip_weight=%.4f, "
                                "top_k=%s, teacher_total=%s, teachers=%s, students=%s"
                            ),
                            ff_ctrl.get("branch"),
                            ff_ctrl.get("consensus_label"),
                            n_prompts,
                            ff_ctrl.get("C"),
                            float(ff_ctrl.get("m_med", 0.0)),
                            float(ff_ctrl.get("m_min", 0.0)),
                            float(ff_ctrl.get("m_max", 0.0)),
                            float(ff_ctrl.get("delta", 0.0)),
                            float(ff_ctrl.get("flip_weight", 0.0)),
                            ff_ctrl.get("k"),
                            teacher_total,
                            ff_ctrl.get("teachers"),
                            ff_ctrl.get("students"),
                        )

                    if loss_opt in {"consensus_cross_entropy", "flip_flop_consensus_cross_entropy"}:
                        consensus_label = derive_consensus_label(prompt_preds_seq)

            # 2) train inners (with optional normalization across micro-steps)
            model.train()
            n_train = max(1, len(train_inners))
            total = torch.tensor(0.0, device=self.args.device)
            seen_prompts = 0
            for idx_inner, inner in enumerate(train_inners):
                # Inject pseudo targets and flip-flop params for compute_loss
                num_choices = int(getattr(self.train_dataset, "num_choices", 1))
                ans_id = idx_inner % num_choices
                # current batch prompt count
                cur_prompt_num = int(inner["input_ids"].size(0)) if hasattr(inner.get("input_ids", None), "size") else int(getattr(self.train_dataset, "num_prompts", 1))

                last_prompt_batch = False
                # Pseudo targets
                if self.args.pseudo_target_mode == "pairwise":
                    last_prompt_batch = ans_id == (num_choices - 1)
                    if self.args.loss_option in ["consistency_pseudo_train", "pseudo_train", "flip_flop_pseudo_train"] and isinstance(ens_pred, list):
                        slice_end = seen_prompts + cur_prompt_num
                        if slice_end <= len(ens_pred):
                            inner["is_true_answer_state"] = [pred[ans_id] for pred in ens_pred[seen_prompts:slice_end]]
                        else:
                            inner["is_true_answer_state"] = -1
                    else:
                        inner["is_true_answer_state"] = -1
                else:
                    if self.args.loss_option in ["consistency_pseudo_train", "pseudo_train", "flip_flop_pseudo_train"] and isinstance(ens_pred, list) and ans_id < len(ens_pred):
                        inner["is_true_answer_state"] = ens_pred[ans_id]
                    else:
                        inner["is_true_answer_state"] = -1

                uses_consensus_gate = loss_opt in {"consensus_cross_entropy", "flip_flop_consensus_cross_entropy"} or "flip_flop" in loss_opt
                ff_branch = str(ff_ctrl.get("branch")) if ff_ctrl is not None else "n/a"
                if uses_consensus_gate:
                    gate_label: Optional[int] = None
                    if loss_opt in {"consensus_cross_entropy", "flip_flop_consensus_cross_entropy"}:
                        gate_label = consensus_label
                    if "flip_flop" in loss_opt and ff_ctrl is not None:
                        gate_label = ff_ctrl.get("consensus_label")

                    if gate_label is not None and ans_id == int(gate_label):
                        inner["consensus_is_active"] = 1.0
                    else:
                        inner["consensus_is_active"] = 0.0
                    logger.info(
                        "[flip_flop:gate] ans_id=%s branch=%s gate_label=%s consensus_active=%.1f",
                        ans_id,
                        ff_branch,
                        gate_label,
                        float(inner["consensus_is_active"]),
                    )

                # Flip-flop controls
                if "flip_flop" in loss_opt and ff_ctrl is not None:
                    ff_consensus_label = ff_ctrl.get("consensus_label")
                    branch = ff_branch
                    teacher_slice = None
                    flip_w = 0.0
                    jsd_alpha = float(getattr(self.args, "ff_no_consensus_alpha_jsd", 0.0))

                    if ff_consensus_label is not None and ans_id == int(ff_consensus_label):
                        if branch == "unanimous_strong":
                            jsd_alpha = float(getattr(self.args, "ff_flipflop_beta_jsd", 0.0))
                        elif branch in {"flip_flop: unanimous_weak", "flip_flop: mixed"}:
                            mask_all = ff_ctrl.get("flip_teacher_mask_all")
                            if isinstance(mask_all, list) and mask_all:
                                if self.args.pseudo_target_mode == "pairwise":
                                    slice_end = seen_prompts + cur_prompt_num
                                    if slice_end <= len(mask_all):
                                        teacher_slice = mask_all[seen_prompts:slice_end]
                                    else:
                                        logger.info(
                                            "[flip_flop:slice] ans_id=%s branch=%s slice [%s:%s) exceeds mask (len=%s); skipping",
                                            ans_id,
                                            branch,
                                            seen_prompts,
                                            slice_end,
                                            len(mask_all),
                                        )
                                else:
                                    teacher_slice = mask_all
                                if teacher_slice and 0 < sum(teacher_slice) < len(teacher_slice):
                                    flip_w = float(ff_ctrl.get("flip_weight", 0.0))
                                    jsd_alpha = float(getattr(self.args, "ff_flipflop_beta_jsd", 0.0))
                                    teacher_sum = int(sum(teacher_slice))
                                    slice_len = len(teacher_slice)
                                    logger.info(
                                        ("[flip_flop:slice] ans_id=%s branch=%s prompts=[%s:%s) slice_len=%s " "teacher_count=%s student_count=%s flip_w=%.4f jsd_alpha=%.4f teacher_slice=%s"),
                                        ans_id,
                                        branch,
                                        seen_prompts,
                                        seen_prompts + slice_len,
                                        slice_len,
                                        teacher_sum,
                                        slice_len - teacher_sum,
                                        flip_w,
                                        jsd_alpha,
                                        teacher_slice,
                                    )
                                else:
                                    if teacher_slice:
                                        logger.info(
                                            "[flip_flop:slice] ans_id=%s branch=%s prompts=[%s:%s) invalid teacher split %s; forcing None",
                                            ans_id,
                                            branch,
                                            seen_prompts,
                                            seen_prompts + (len(teacher_slice) if isinstance(teacher_slice, list) else 0),
                                            teacher_slice,
                                        )
                                    teacher_slice = None
                        elif "no_consensus" in branch:
                            teacher_slice = None

                    inner["flip_teacher_mask"] = teacher_slice
                    inner["flip_loss_weight"] = flip_w
                    inner["nonconsistency_alpha"] = jsd_alpha

                if self.args.pseudo_target_mode == "pairwise" and last_prompt_batch:
                    seen_prompts += cur_prompt_num

                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inner, return_outputs=False, num_items_in_batch=num_items_in_batch)

                comp_latest = getattr(self, "_last_loss_components", None)
                skip_components = bool(isinstance(comp_latest, dict) and comp_latest.get("_skip_component_logging"))

                if skip_components:
                    loss = loss.detach()
                else:
                    loss = loss / n_train
                    # Use Accelerate to handle backward pass (with grad scaling if enabled)
                    self.accelerator.backward(loss)
                    total = total + loss.detach()

                if isinstance(comp_latest, dict) and not skip_components:
                    if getattr(self, "is_in_train", False):
                        print(comp_latest, flush=True)
                    for name in self._component_names:
                        step_component_totals[name] += float(comp_latest.get(name, 0.0))
                    step_component_count += 1

            return total

            # Fallback to default path for flat batches
            return super().training_step(model, inputs, num_items_in_batch)
        finally:
            if getattr(self, "is_in_train", False):
                if not hasattr(self, "_train_component_totals"):
                    self._train_component_totals = {name: 0.0 for name in self._component_names}
                if not hasattr(self, "_train_component_counts"):
                    self._train_component_counts = {name: 0.0 for name in self._component_names}
                if step_component_count > 0:
                    count_delta = float(step_component_count)
                    for name in self._component_names:
                        self._train_component_totals[name] = self._train_component_totals.get(name, 0.0) + step_component_totals.get(name, 0.0)
                        self._train_component_counts[name] = self._train_component_counts.get(name, 0.0) + count_delta
            self.is_in_train = prev_training_state

    def train(self, *args, **kwargs):  # type: ignore[override]
        self._train_component_totals = {name: 0.0 for name in self._component_names}
        self._train_component_counts = {name: 0.0 for name in self._component_names}
        self._train_component_last_logged_step = self.state.global_step
        return super().train(*args, **kwargs)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:  # type: ignore[override]
        if "loss" in logs and "eval_loss" not in logs:
            step_delta = self.state.global_step - self._train_component_last_logged_step
            if step_delta <= 0:
                step_delta = 1
            if not hasattr(self, "_train_component_counts"):
                self._train_component_counts = {name: 0.0 for name in self._component_names}
            total_contrib = sum(self._train_component_totals.get(name, 0.0) for name in self._component_names)
            count_ref_name = self._component_names[0]
            count_ref = self._train_component_counts.get(count_ref_name, 0.0)
            scale = None
            if count_ref > 0 and total_contrib > 0.0:
                average_total = total_contrib / count_ref
                if average_total > 0.0:
                    scale = logs["loss"] / average_total
            for name in self._component_names:
                total = self._train_component_totals.get(name, 0.0)
                count = self._train_component_counts.get(name, 0.0)
                value = total / count if count > 0 else 0.0
                if scale is not None:
                    value = value * scale
                logs[name] = value
                self._train_component_totals[name] = 0.0
                self._train_component_counts[name] = 0.0
            self._train_component_last_logged_step = self.state.global_step
        super().log(logs, start_time)

    # -----------------------
    # Evaluate on nested eval datasets (same layout as train)
    # -----------------------
    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        force_output_suffix: Optional[str] = None,
    ):
        """
        Evaluate on a nested FlipFlopDataset-like structure.
        - Uses eval_inners to gather per-example log-likelihoods (for metrics/files).
        - Runs train_inners with injected controls to compute average loss (logged to W&B).
        - Chooses metrics function by dataset/prefix:
            dev/unsupervised_dev -> utils.compute_unsupervised_metrics
            otherwise           -> utils.compute_metrics
        """
        run_dev_eval_after = (
            metric_key_prefix == "eval" and getattr(self, "dev_dataset", None) is not None and not getattr(self, "_suppress_dev_eval", False) and bool(getattr(self, "is_in_train", False))
        )

        prev_training_state = getattr(self, "is_in_train", False)
        self.is_in_train = False

        dataloader = self.get_eval_dataloader(eval_dataset) if eval_dataset is not None else self.get_eval_dataloader()
        self._memory_tracker.start()
        model = self._wrap_model(self.model, training=False)
        model.eval()

        # For metrics accumulation
        all_logprobs: List[float] = []  # flattened over all examples
        total_loss = torch.tensor(0.0, device=self.args.device)
        total_loss_count = 0
        n_outer = 0
        component_sums: Dict[str, float] = {}
        component_counts = 0

        # Dataset introspection
        dataset_ref = getattr(dataloader, "dataset", None)
        E = getattr(dataset_ref, "dev_size", None)
        if E is None:
            raise ValueError("dev_size is not set")

        dev_metrics: Optional[Dict[str, float]] = None
        loss_opt = str(getattr(self.args, "loss_option", ""))

        for outer in dataloader:
            if not isinstance(outer, list):
                # Flat batch fallback (rare in this project)
                loss, preds, labels = self.prediction_step(model, outer, prediction_loss_only=False, ignore_keys=ignore_keys)
                comp_latest = getattr(self, "_last_loss_components", None)
                skip_components = bool(isinstance(comp_latest, dict) and comp_latest.get("_skip_component_logging"))

                if not skip_components and loss is not None:
                    total_loss = total_loss + loss.detach()
                    total_loss_count += 1
                    n_outer += 1

                if isinstance(comp_latest, dict) and not skip_components:
                    for name in self._component_names:
                        component_sums[name] = component_sums.get(name, 0.0) + float(comp_latest.get(name, 0.0))
                    component_counts += 1
                continue

            eval_inners = outer[:E]
            train_inners = outer[E:]

            # Phase 1: gather eval inners for metrics/logprobs; skip component losses
            self._eval_cache = {}
            with torch.no_grad():
                for inner in eval_inners:
                    marker = {"_skip_loss_components": True}
                    if isinstance(inner, Mapping):
                        inner_marked = dict(inner)
                        inner_marked.update(marker)
                    else:
                        raise ValueError("inner is not a mapping")
                    _ = self.prediction_step(model, inner_marked, prediction_loss_only=False, ignore_keys=ignore_keys)

            import torch as _t

            pll = _t.cat(self._eval_cache.get("per_example_ll", [])).tolist() if "per_example_ll" in self._eval_cache else None
            if pll is not None:
                all_logprobs.extend(pll)

            # For loss on train inners, we may need pseudo labels / flip-flop controls derived from PLL of this example
            ens_pred = None
            ff_ctrl: Optional[Dict[str, Any]] = None
            prompt_preds_seq: Optional[List[int]] = None
            consensus_label: Optional[int] = None
            if pll is not None and callable(self.compute_metrics):
                n_examples = 1
                n_targets = int(getattr(dataset_ref, "num_choices", getattr(self.train_dataset, "num_choices", 1)))
                n_prompts = int(getattr(dataset_ref, "num_prompts", getattr(self.train_dataset, "num_prompts", 1)))
                if self.args.pseudo_target_mode == "pairwise":
                    prompt_preds, avg_ens_pred, vote_ens_pred, rand_indices = self.compute_metrics(
                        pll,
                        n_examples,
                        n_targets,
                        n_prompts,
                        pseudo_dist=self.args.pseudo_dist,
                        return_all_prompt_preds=True,
                        random_selection_ensemble=self.args.ensemble_subset_size,
                        self_train=self.args.self_train_option != "none",
                    )
                    inv_perm = np.argsort(rand_indices)
                    avg_ens_pred = [avg_ens_pred[i] for i in inv_perm]
                    vote_ens_pred = [vote_ens_pred[i] for i in inv_perm]
                else:
                    prompt_preds, avg_ens_pred, vote_ens_pred, _ = self.compute_metrics(
                        pll,
                        n_examples,
                        n_targets,
                        n_prompts,
                        pseudo_dist=self.args.pseudo_dist,
                        return_all_prompt_preds=False,
                        random_selection_ensemble=self.args.ensemble_subset_size,
                        self_train=self.args.self_train_option != "none",
                    )

                prompt_preds_seq = list(prompt_preds) if prompt_preds is not None else None

                if self.args.ensemble_option == "avg_prob":
                    ens_pred = avg_ens_pred
                elif self.args.ensemble_option == "majority_vote":
                    ens_pred = vote_ens_pred
                else:
                    raise ValueError(f"unknown ensemble: {self.args.ensemble_option}")

                if "flip_flop" in loss_opt:
                    ff_ctrl = derive_flipflop_controls(
                        self.args,
                        prompt_preds,
                        pll,
                        int(n_prompts),
                        int(n_targets),
                    )

                if loss_opt in {"consensus_cross_entropy", "flip_flop_consensus_cross_entropy"}:
                    consensus_label = derive_consensus_label(prompt_preds_seq)

            # Phase 2: compute loss over train inners (normalized average per outer)
            n_train = max(1, len(train_inners))
            outer_loss = torch.tensor(0.0, device=self.args.device)
            outer_active = 0
            seen_prompts = 0
            for idx_inner, inner in enumerate(train_inners):
                inner_mod = dict(inner)
                num_choices = int(getattr(dataset_ref, "num_choices", getattr(self.train_dataset, "num_choices", 1)))
                ans_id = idx_inner % num_choices
                cur_prompt_num = (
                    int(inner_mod["input_ids"].size(0))
                    if hasattr(inner_mod.get("input_ids", None), "size")
                    else int(getattr(dataset_ref, "num_prompts", getattr(self.train_dataset, "num_prompts", 1)))
                )

                last_prompt_batch = False
                if self.args.pseudo_target_mode == "pairwise":
                    last_prompt_batch = ans_id == (num_choices - 1)
                    if self.args.loss_option in ["consistency_pseudo_train", "pseudo_train", "flip_flop_pseudo_train"] and isinstance(ens_pred, list):
                        slice_end = seen_prompts + cur_prompt_num
                        if slice_end <= len(ens_pred):
                            inner_mod["is_true_answer_state"] = [pred[ans_id] for pred in ens_pred[seen_prompts:slice_end]]
                        else:
                            inner_mod["is_true_answer_state"] = -1
                    else:
                        inner_mod["is_true_answer_state"] = -1
                else:
                    if self.args.loss_option in ["consistency_pseudo_train", "pseudo_train", "flip_flop_pseudo_train"] and isinstance(ens_pred, list) and ans_id < len(ens_pred):
                        inner_mod["is_true_answer_state"] = ens_pred[ans_id]
                    else:
                        inner_mod["is_true_answer_state"] = -1

                uses_consensus_gate = loss_opt in {"consensus_cross_entropy", "flip_flop_consensus_cross_entropy"} or "flip_flop" in loss_opt
                if uses_consensus_gate:
                    gate_label: Optional[int] = None
                    if loss_opt in {"consensus_cross_entropy", "flip_flop_consensus_cross_entropy"}:
                        gate_label = consensus_label
                    if "flip_flop" in loss_opt and ff_ctrl is not None:
                        gate_label = ff_ctrl.get("consensus_label")

                    if gate_label is not None and ans_id == int(gate_label):
                        inner_mod["consensus_is_active"] = 1.0
                    else:
                        inner_mod["consensus_is_active"] = 0.0

                if "flip_flop" in loss_opt and ff_ctrl is not None:
                    ff_consensus_label = ff_ctrl.get("consensus_label")
                    branch = str(ff_ctrl.get("branch"))
                    teacher_slice = None
                    flip_w = 0.0
                    jsd_alpha = float(getattr(self.args, "ff_no_consensus_alpha_jsd", 0.0))

                    if ff_consensus_label is not None and ans_id == int(ff_consensus_label):
                        if branch == "unanimous_strong":
                            jsd_alpha = float(getattr(self.args, "ff_flipflop_beta_jsd", 0.0))
                        elif branch in {"flip_flop: unanimous_weak", "flip_flop: mixed"}:
                            mask_all = ff_ctrl.get("flip_teacher_mask_all")
                            if isinstance(mask_all, list) and mask_all:
                                slice_end = seen_prompts + cur_prompt_num
                                if self.args.pseudo_target_mode == "pairwise" and slice_end <= len(mask_all):
                                    teacher_slice = mask_all[seen_prompts:slice_end]
                                elif self.args.pseudo_target_mode != "pairwise":
                                    teacher_slice = mask_all
                                if teacher_slice and 0 < sum(teacher_slice) < len(teacher_slice):
                                    flip_w = float(ff_ctrl.get("flip_weight", 0.0))
                                    jsd_alpha = float(getattr(self.args, "ff_flipflop_beta_jsd", 0.0))
                                else:
                                    teacher_slice = None

                    inner_mod["flip_teacher_mask"] = teacher_slice
                    inner_mod["flip_loss_weight"] = flip_w
                    inner_mod["nonconsistency_alpha"] = jsd_alpha

                loss, _, _ = self.prediction_step(model, inner_mod, prediction_loss_only=False, ignore_keys=ignore_keys)
                comp_latest = getattr(self, "_last_loss_components", None)
                skip_components = bool(isinstance(comp_latest, dict) and comp_latest.get("_skip_component_logging"))

                if not skip_components and loss is not None:
                    outer_loss = outer_loss + loss.detach()
                    outer_active += 1

                if isinstance(comp_latest, dict) and not skip_components:
                    for name in self._component_names:
                        component_sums[name] = component_sums.get(name, 0.0) + float(comp_latest.get(name, 0.0))
                    component_counts += 1

                if self.args.pseudo_target_mode == "pairwise" and last_prompt_batch:
                    seen_prompts += cur_prompt_num

            # Aggregate per outer example based on contributing batches only
            if outer_active > 0:
                total_loss = total_loss + outer_loss
                total_loss_count += outer_active
                n_outer += 1

        # Build metrics/logging
        metrics: Dict[str, float] = {}
        avg_loss_value: Optional[float] = None
        if total_loss_count > 0:
            avg_loss_value = (total_loss / float(total_loss_count)).detach().item()
            metrics[f"{metric_key_prefix}_loss"] = avg_loss_value

        component_avgs: Dict[str, float] = {}
        if component_counts > 0:
            component_avgs = {name: total / component_counts for name, total in component_sums.items()}
            for name, value in component_avgs.items():
                metrics[f"{metric_key_prefix}_{name}"] = float(value)

        # Compute and log task metrics to files and W&B
        n_examples_total = getattr(dataset_ref, "datasize", None) or len(dataset_ref)
        n_targets = int(getattr(dataset_ref, "num_choices", getattr(self.train_dataset, "num_choices", 1)))
        n_prompts = int(getattr(dataset_ref, "num_prompts", getattr(self.train_dataset, "num_prompts", 1)))
        fout_dir = str(getattr(self.args, "output_dir", "checkpoints"))
        if fout_dir:
            try:
                os.makedirs(fout_dir, exist_ok=True)
            except OSError:
                pass
        if force_output_suffix is not None:
            suffix = str(force_output_suffix)
        else:
            suffix = str(int(self.state.global_step)) if getattr(self, "state", None) is not None else "0"

        # Choose metric function: dev uses unsupervised; test uses supervised
        # Decide metric function: prefer explicit prefix, otherwise infer from the dataset identity.
        dataset_ref_for_check = getattr(dataloader, "dataset", None)
        use_unsup = (
            metric_key_prefix.startswith("unsupervised_dev")
            or (eval_dataset is not None and eval_dataset is getattr(self, "dev_dataset", None))
            or (dataset_ref_for_check is not None and dataset_ref_for_check is getattr(self, "dev_dataset", None))
        )
        if use_unsup and callable(self.compute_unsupervised_metrics):
            results, _ = self.compute_unsupervised_metrics(
                all_logprobs,
                n_examples_total,
                n_targets,
                n_prompts,
                golds=getattr(dataset_ref, "gold_labels", None),
                metrics=self.additional_metrics,
                fout_name=fout_dir,
                suffix=suffix,
            )
        elif callable(self.compute_metrics):
            results, _ = self.compute_metrics(
                all_logprobs,
                n_examples_total,
                n_targets,
                n_prompts,
                golds=getattr(dataset_ref, "gold_labels", None),
                metrics=self.additional_metrics,
                fout_name=fout_dir,
                suffix=suffix,
                pseudo_dist=getattr(self.args, "pseudo_dist", "smooth"),
                return_all_prompt_preds=False,
                random_selection_ensemble=getattr(self.args, "ensemble_subset_size", 0.0),
                self_train=self.args.self_train_option != "none",
            )
        else:
            results = {}

        self._write_loss_components_to_file(
            fout_dir,
            suffix,
            metric_key_prefix,
            component_avgs,
            avg_loss_value,
        )

        if isinstance(results, dict):
            for k, v in results.items():
                # Ensure scalars for W&B (skip arrays/lists unless scalar-valued)
                if isinstance(v, (list, tuple, dict)):
                    continue
                if isinstance(v, np.ndarray):
                    if v.size != 1:
                        continue
                    v = v.item()
                metrics[f"{metric_key_prefix}_{k}"] = float(v)

        if metric_key_prefix.startswith("unsupervised_dev"):
            logs_for_record: Dict[str, float] = {}
            for key, value in metrics.items():
                if key.startswith("unsupervised_dev_"):
                    stripped = key[len("unsupervised_dev_") :]
                    stripped = stripped or key
                    logs_for_record[f"valid/{stripped}"] = value
                else:
                    logs_for_record[key] = value
        else:
            logs_for_record = dict(metrics)

        self.log(logs_for_record)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        if run_dev_eval_after:
            try:
                self._suppress_dev_eval = True
                dev_metrics = self.evaluate(
                    eval_dataset=self.dev_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix="unsupervised_dev",
                )
            finally:
                self._suppress_dev_eval = False

        if dev_metrics and metrics is not None:
            # Surface dev metrics under the eval_ namespace so HF Trainer can track best checkpoints.
            for key, value in dev_metrics.items():
                metrics[f"eval_{key}"] = value

        self.is_in_train = prev_training_state
        return metrics
