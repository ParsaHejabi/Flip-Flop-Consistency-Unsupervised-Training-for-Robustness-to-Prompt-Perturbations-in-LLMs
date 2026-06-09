# Reproducibility Guide

This repository contains the executable code and configurations needed to rerun
the experiments. It intentionally excludes generated datasets, model
checkpoints, W&B logs, result dumps, and LaTeX sources.

## Environment

The paper used:

- Ubuntu 22.04.5 LTS
- Python 3.9
- PyTorch 2.8.0
- Transformers 4.56.1
- PEFT 0.17.1
- Accelerate 1.10.1
- Datasets 4.1.0
- Tokenizers 0.22.0
- SentencePiece 0.2.1
- bf16 training with LoRA rank 16, alpha 32, and dropout 0.05

Training took approximately 3 to 23 hours per dataset. Smaller runs used a pool
of seven NVIDIA RTX A6000 GPUs with 48 GB VRAM each; larger runs used one
NVIDIA A100 with 80 GB VRAM. Exact runtime and memory use depend on the dataset,
number of prompt formats, model, and installed CUDA build.

## Dataset Protocol

The code uses PromptSource original-task templates. The official validation
split is used as the test set because most official test labels are private.
The original training split is deduplicated, then partitioned into disjoint
train and stratified validation sets with seed 42.

| Dataset | Train | Validation | Test | Formats |
| --- | ---: | ---: | ---: | ---: |
| ANLI R1 | 10,000 | 1,000 | 1,000 | 15 |
| ANLI R2 | 10,000 | 1,000 | 1,000 | 15 |
| ANLI R3 | 10,000 | 1,000 | 1,200 | 15 |
| CB | 194 | 56 | 56 | 15 |
| RTE | 1,488 | 1,000 | 277 | 10 |
| COPA | 300 | 100 | 100 | 8 |
| HellaSwag | 10,000 | 600 | 600 | 4 |
| StoryCloze | 871 | 1,000 | 1,871 | 5 |
| WSC | 430 | 95 | 95 | 10 |
| WinoGrande | 10,000 | 1,000 | 1,267 | 5 |
| WiC | 4,428 | 1,000 | 638 | 10 |

StoryCloze has no public train split. Request the 2016 validation and test CSVs
from the [ROCStories website](https://www.cs.rochester.edu/nlp/rocstories/),
place them at `cache_dir/cloze_2016_val.csv` and
`cache_dir/cloze_2016_test.csv`, and use these columns:

```text
story_id,input_sentence_1,input_sentence_2,input_sentence_3,input_sentence_4,sentence_quiz1,sentence_quiz2,answer_right_ending
```

No dataset is redistributed by this repository.

## Study 1: Per-Dataset Training

The public launcher reads dataset definitions and the paper-reported F2C
hyperparameters from `configs/study1.json`.

```bash
python scripts/run_study1.py --dataset anli_r1 --method f2c --model qwen
python scripts/run_study1.py --dataset anli_r1 --method f2c --model llama
```

Supported methods are:

- `base`: evaluation only, with no parameter updates
- `swarm`: pairwise swarm-distillation consistency loss
- `cce`: Consensus Cross-Entropy only
- `f2c`: Consensus Cross-Entropy plus selective alignment

To run all Qwen Study 1 configurations:

```bash
for dataset in anli_r1 anli_r2 anli_r3 cb rte copa hellaswag story_cloze wsc winogrande wic; do
  for method in base swarm cce f2c; do
    python scripts/run_study1.py --dataset "$dataset" --method "$method" --model qwen
  done
done
```

The camera-ready paper reports dataset-specific F2C settings. These values were
not shared across datasets and were selected from a small set of manual
configurations rather than an exhaustive per-dataset search.

The paper's F2C definition skips examples without a strict majority. The public
launcher therefore sets the legacy `ff_no_consensus_alpha_jsd` option to zero;
this fallback is not part of the reported F2C objective or appendix
hyperparameter table.

Each run first writes the unmodified model evaluation to `accuracy_0`.
Subsequent evaluations are stored as `accuracy_<step>`. Select the checkpoint
with the highest validation F1 for paper-style reporting.

Aggregate completed runs with:

```bash
python analysis/aggregate_all_datasets.py mean_f1 raw_agreement std_f1 \
  --checkpoints-dir outputs

python analysis/aggregate_best_checkpoints_each_dataset.py outputs
```

## Study 2: Cross-Dataset Transfer

Study 2 evaluates a selected F2C checkpoint trained on one source dataset
against the other target datasets. First run Study 1 for each target dataset at
least once so `main.py` materializes compatible `data/.../test.pt` files.

```bash
python study_2/study_2_main.py \
  --ckpt_dir outputs/rte/<source-run> \
  --best_ckpt outputs/rte/<source-run>/checkpoint-<step> \
  --base_model_name Qwen/Qwen2.5-3B-Instruct \
  --dataset_a_label rte \
  --cache_dir cache_dir \
  --data_root data \
  --output_dir study_2
```

The paper uses eight source datasets: ANLI R1/R2/R3, RTE, COPA, HellaSwag,
StoryCloze, and WinoGrande. CB, WSC, and WiC are retained only as targets.

## Study 3: Held-Out Prompt Formats

Use `--train-formats K` to train on the first K prompt formats. The remaining
formats are used for evaluation, and the deterministic format split is saved
under `study_3/data/`.

```bash
python scripts/run_study1.py --dataset anli_r1 --method f2c --train-formats 5
python scripts/run_study1.py --dataset anli_r1 --method f2c --train-formats 10
```

The paper evaluates ANLI R1/R2/R3 with K=5 and K=10, holding out the final five
formats. RTE uses K=5 and holds out the remaining five formats.

To evaluate one trained checkpoint against a previously materialized held-out
format split:

```bash
python study_3/study_3_eval.py \
  --ckpt_dir outputs/anli_r1/<source-run> \
  --best_ckpt outputs/anli_r1/<source-run>/checkpoint-<step> \
  --datasets anli_r1 \
  --train_abl_np 5 \
  --eval_abl_np 10 \
  --data_root study_3/data \
  --cache_dir cache_dir \
  --output_dir study_3
```

## Reproducibility Notes

- The launcher prints the complete `main.py` command before execution.
- Seed 42 is used for dataset partitioning and training.
- `data/` contains rendered/tokenized split caches and source indices. Preserve
  these caches when comparing methods so every method uses the same examples.
- The code validates that rendered prompt texts are unique within each split
  and disjoint across train, validation, and evaluation splits.
- Exact floating-point equality is not guaranteed across GPU models, CUDA
  builds, or package builds.
- PromptSource and upstream dataset/model licenses apply independently.
