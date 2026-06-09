import argparse
import os
import re
import statistics
import sys
from datetime import datetime
from fnmatch import fnmatch
from html import escape
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore[assignment]


LossConfig = Tuple[str, Tuple[str, ...]]


LOSS_OPTION_CONFIG: Dict[str, LossConfig] = {
    "consistency": ("Swarm", ()),
    "consensus_cross_entropy": ("CCE", ("cce_weight",)),
    "flip_flop_consensus_cross_entropy": (
        "F$^2$C",
        (
            "cce_weight",
            "ff_top_k_teachers",
            "ff_unanimous_margin_tau",
            "ff_flipflop_beta_jsd",
            "ff_no_consensus_alpha_jsd",
            "ff_weight_temp",
            "ff_weight_min",
            "ff_weight_max",
        ),
    ),
    # "flip_flop_pseudo_train": (
    #     "Flip-Flop + entropy NLL",
    #     (
    #         "ff_top_k_teachers",
    #         "ff_unanimous_margin_tau",
    #         "ff_flipflop_beta_jsd",
    #         "ff_no_consensus_alpha_jsd",
    #         "ff_weight_temp",
    #         "ff_weight_min",
    #         "ff_weight_max",
    #         "pseudo_train_loss_weight",
    #     ),
    # ),
}


METHOD_ORDER: List[str] = ["Base"] + [cfg[0] for cfg in LOSS_OPTION_CONFIG.values()]


HYPERPARAM_DISPLAY: Dict[str, str] = {
    "cce_weight": "CCE Weight",
    "ff_top_k_teachers": "FF Top-k",
    "ff_unanimous_margin_tau": "FF Tau",
    "ff_flipflop_beta_jsd": "FF Beta JSD",
    "ff_no_consensus_alpha_jsd": "FF Alpha JSD",
    "ff_weight_temp": "FF Weight Temp",
    "ff_weight_min": "FF Weight Min",
    "ff_weight_max": "FF Weight Max",
    "pseudo_train_loss_weight": "Pseudo-Train Weight",
}

METRIC_LATEX_LABELS: Dict[str, str] = {
    "mean_f1": "$\\bar{F}_1$",
    "std_f1": "$\\sigma_{F_1}$",
    "raw_agreement": "$P_o$",
}

METRIC_DISPLAY_ORDER: Tuple[str, ...] = ("mean_f1", "std_f1", "raw_agreement")

METHOD_COL_WIDTH = "1.5cm"

LATEX_BEST_COLOR = "RoyalBlue"
LATEX_WORST_COLOR = "Orange"
HTML_BEST_BACKGROUND = "#dce9ff"
HTML_WORST_BACKGROUND = "#ffe6cc"
HTML_IMPROVEMENT_COLOR = "#1f5fbf"
HTML_REGRESSION_COLOR = "#c46a00"

DATASET_DISPLAY_NAMES: Dict[str, str] = {
    "anli_r1": "ANLI R1",
    "anli_r2": "ANLI R2",
    "anli_r3": "ANLI R3",
    "cb": "CB",
    "rte": "RTE",
    "copa": "COPA",
    "hellaswag": "HellaSwag",
    "story_cloze": "StoryCloze 2016",
    "wsc": "WSC",
    "winogrande": "Winogrande-XL",
    "wic": "WiC",
}

DATASET_SHORT_DISPLAY_NAMES: Dict[str, str] = {
    "anli_r1": "ANLI R1",
    "anli_r2": "ANLI R2",
    "anli_r3": "ANLI R3",
    "cb": "CB",
    "rte": "RTE",
    "copa": "COPA",
    "hellaswag": "HellaSwag",
    "story_cloze": "StoryCloze",
    "wsc": "WSC",
    "winogrande": "Winogrande",
    "wic": "WiC",
}

DATASET_DISPLAY_ORDER: Tuple[str, ...] = (
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "cb",
    "rte",
    "copa",
    "hellaswag",
    "story_cloze",
    "wsc",
    "winogrande",
    "wic",
)

DATASET_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("anli_r1", "anli_r2", "anli_r3", "cb", "rte", "copa"),
    ("hellaswag", "story_cloze", "wsc", "winogrande", "wic"),
)


def parse_metric_file(path: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    pattern = re.compile(r"^([^=]+)=([-+eE0-9\.]+)")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("gold=") or line.startswith("acc:") or line.startswith("ent:"):
                break
            match = pattern.match(line)
            if match:
                key = match.group(1).strip()
                try:
                    metrics[key] = float(match.group(2))
                except ValueError:
                    continue
    if "max_accuracy" in metrics and "min_accuracy" in metrics:
        metrics.setdefault("accuracy_spread", metrics["max_accuracy"] - metrics["min_accuracy"])
    if "raw_agreement" in metrics:
        metrics["raw_agreement"] *= 100.0
    return metrics


def better(metric: str, value: float, reference: float) -> bool:
    metric_l = metric.lower()
    if metric_l.endswith("_spread") or metric_l == "posix" or metric_l.startswith("std_"):
        return value < reference
    return value > reference


def extract_run_timestamp(raw_name: str) -> Optional[datetime]:
    parts = raw_name.split("_")
    if len(parts) < 2:
        return None
    candidate = "_".join(parts[-2:])
    try:
        return datetime.strptime(candidate, "%Y%m%d_%H:%M:%S.%f")
    except ValueError:
        return None


def parse_run_sh_vars(run_sh_path: str) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    if not os.path.isfile(run_sh_path):
        return assignments
    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*(?:#.*)?$")
    with open(run_sh_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            match = pattern.match(line)
            if not match:
                continue
            name, raw_value = match.group(1), match.group(2)
            if " python " in f" {raw_value} " or " --" in raw_value:
                continue
            if (raw_value.startswith("'") and raw_value.endswith("'")) or (raw_value.startswith('"') and raw_value.endswith('"')):
                value = raw_value[1:-1]
            else:
                value = raw_value
            assignments[name] = value
    return assignments


def select_checkpoint(
    run_dir: str,
    selection_metric: str,
    report_prefix: str,
    selection_prefix: str,
    max_step: Optional[int],
) -> Tuple[Optional[int], Dict[str, float], Optional[float]]:
    best_step: Optional[int] = None
    best_val: Optional[float] = None

    for entry in os.listdir(run_dir):
        if not entry.startswith(selection_prefix):
            continue
        try:
            step = int(entry.split("_")[-1])
        except ValueError:
            continue
        if step == 0:
            continue
        if max_step is not None and step > max_step:
            continue
        metrics = parse_metric_file(os.path.join(run_dir, entry))
        if selection_metric not in metrics:
            continue
        candidate = metrics[selection_metric]
        if best_val is None or better(selection_metric, candidate, best_val):
            best_val = candidate
            best_step = step

    if best_step is not None:
        report_path = os.path.join(run_dir, f"{report_prefix}{best_step}")
        if os.path.isfile(report_path):
            metrics = parse_metric_file(report_path)
            return best_step, metrics, best_val

    fallback_best_step: Optional[int] = None
    fallback_best_val: Optional[float] = None
    fallback_metrics: Optional[Dict[str, float]] = None
    for entry in os.listdir(run_dir):
        if not entry.startswith(report_prefix):
            continue
        try:
            step = int(entry.split("_")[-1])
        except ValueError:
            continue
        if step == 0:
            continue
        if max_step is not None and step > max_step:
            continue
        metrics = parse_metric_file(os.path.join(run_dir, entry))
        if selection_metric not in metrics:
            continue
        candidate = metrics[selection_metric]
        if fallback_best_val is None or better(selection_metric, candidate, fallback_best_val):
            fallback_best_val = candidate
            fallback_best_step = step
            fallback_metrics = metrics

    if fallback_metrics is not None:
        # No validation selection metric was available; report the checkpoint but
        # indicate selection value could not be computed so callers can decide.
        return fallback_best_step, fallback_metrics, None

    return None, {}, None


def _should_replace(existing_dt: Optional[datetime], new_dt: Optional[datetime]) -> bool:
    if existing_dt is None and new_dt is not None:
        return True
    if existing_dt is not None and new_dt is not None and new_dt >= existing_dt:
        return True
    return False


def collect_dataset_metrics(
    dataset_dir: str,
    metrics: Iterable[str],
    selection_metric: str,
    checkpoint_prefix: str,
    selection_prefix: str,
    max_step: Optional[int],
) -> Dict[str, Dict[str, Tuple[Optional[float], Optional[float], int]]]:
    metric_names = list(metrics)
    base_by_seed: Dict[str, Dict[str, object]] = {}
    method_groups: Dict[str, Dict[Tuple[str, ...], Dict[str, Dict[str, object]]]] = {}

    for entry in sorted(os.listdir(dataset_dir)):
        run_dir = os.path.join(dataset_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        run_vars = parse_run_sh_vars(os.path.join(run_dir, "run.sh"))
        loss_option = run_vars.get("loss_option")
        if loss_option is None:
            continue
        loss_option = loss_option.strip()
        config = LOSS_OPTION_CONFIG.get(loss_option)
        if config is None:
            continue
        method_label, param_keys = config

        timestamp = extract_run_timestamp(entry)
        seed_value = run_vars.get("seed")
        if seed_value is None:
            match_seed = re.search(r"\.seed(\d+)", entry)
            seed_value = match_seed.group(1) if match_seed else "-"
        seed_key = str(seed_value)

        base_file = os.path.join(run_dir, f"{checkpoint_prefix}0")
        if os.path.isfile(base_file):
            base_metrics_full = parse_metric_file(base_file)
            base_metrics = {m: base_metrics_full[m] for m in metric_names if m in base_metrics_full}
            record = {"metrics": base_metrics, "timestamp": timestamp}
            previous = base_by_seed.get(seed_key)
            prev_dt = previous.get("timestamp") if previous else None
            if previous is None or _should_replace(prev_dt, timestamp):
                base_by_seed[seed_key] = record

        selected_step, report_metrics, selection_value = select_checkpoint(
            run_dir,
            selection_metric,
            checkpoint_prefix,
            selection_prefix,
            max_step,
        )

        if not report_metrics or selection_value is None:
            continue

        filtered_metrics = {m: report_metrics[m] for m in metric_names if m in report_metrics}
        if not filtered_metrics:
            continue

        param_signature = tuple(str(run_vars.get(key, "-")) for key in param_keys)
        group_records = method_groups.setdefault(method_label, {})
        records_by_seed = group_records.setdefault(param_signature, {})
        record = {
            "metrics": filtered_metrics,
            "selection": selection_value,
            "seed": seed_key,
            "timestamp": timestamp,
            "step": selected_step,
        }
        previous = records_by_seed.get(seed_key)
        prev_dt = previous.get("timestamp") if previous else None
        if previous is None or _should_replace(prev_dt, timestamp):
            records_by_seed[seed_key] = record

    results: Dict[str, Dict[str, Tuple[Optional[float], Optional[float], int]]] = {}
    for method in METHOD_ORDER:
        results[method] = {m: (None, None, 0) for m in metric_names}

    # Aggregate base model metrics
    if base_by_seed:
        for metric in metric_names:
            vals = [rec["metrics"].get(metric) for rec in base_by_seed.values() if metric in rec["metrics"]]
            if not vals:
                continue
            mean_val = statistics.mean(vals)
            std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
            results["Base"][metric] = (mean_val, std_val, len(vals))

    # Aggregate method groups, picking the best hyperparameter configuration per method
    for method_label, groups in method_groups.items():
        if not groups:
            continue
        best_key: Optional[Tuple[str, ...]] = None
        best_mean_selection: Optional[float] = None
        for signature, seed_records in groups.items():
            values = [rec.get("selection") for rec in seed_records.values() if rec.get("selection") is not None]
            if not values:
                continue
            mean_sel = statistics.mean(values)
            if best_mean_selection is None or better(selection_metric, mean_sel, best_mean_selection):
                best_mean_selection = mean_sel
                best_key = signature
        if best_key is None:
            # Fallback to configuration with most seeds
            best_key = max(groups.keys(), key=lambda sig: len(groups[sig]))
        chosen_records = list(groups[best_key].values())
        for metric in metric_names:
            vals = [rec["metrics"].get(metric) for rec in chosen_records if metric in rec["metrics"]]
            if not vals:
                continue
            mean_val = statistics.mean(vals)
            std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
            results[method_label][metric] = (mean_val, std_val, len(vals))

    return results


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def dataset_latex_label(dataset: str) -> str:
    return latex_escape(DATASET_DISPLAY_NAMES.get(dataset, dataset))


def dataset_short_latex_label(dataset: str) -> str:
    return latex_escape(DATASET_SHORT_DISPLAY_NAMES.get(dataset, DATASET_DISPLAY_NAMES.get(dataset, dataset)))


def metric_latex_label(metric: str) -> str:
    return METRIC_LATEX_LABELS.get(metric, latex_escape(metric))


def order_metrics(metrics: Iterable[str]) -> List[str]:
    enumerated = list(enumerate(metrics))
    order_lookup = {name: idx for idx, name in enumerate(METRIC_DISPLAY_ORDER)}
    enumerated.sort(key=lambda item: (order_lookup.get(item[1], len(METRIC_DISPLAY_ORDER)), item[0]))
    return [item[1] for item in enumerated]


def order_datasets(datasets: Iterable[str]) -> List[str]:
    enumerated = list(enumerate(datasets))
    order_lookup = {name: idx for idx, name in enumerate(DATASET_DISPLAY_ORDER)}
    enumerated.sort(key=lambda item: (order_lookup.get(item[1], len(DATASET_DISPLAY_ORDER)), item[0]))
    return [item[1] for item in enumerated]


def scale_metric_for_display(metric: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value


def format_cell(metric: str, value: float, best_val: float, worst_val: float) -> str:
    formatted = f"{value:.3f}"
    best_fmt = f"{best_val:.3f}"
    worst_fmt = f"{worst_val:.3f}"
    if formatted == best_fmt:
        return f"\\textbf{{\\textcolor{{{LATEX_BEST_COLOR}}}{{{formatted}}}}}"
    if formatted == worst_fmt:
        return f"\\textcolor{{{LATEX_WORST_COLOR}}}{{{formatted}}}"
    return formatted


def format_transposed_cell(metric: str, value: float, best_val: Optional[float], worst_val: Optional[float]) -> str:
    formatted = f"{value:.2f}"
    if best_val is not None and abs(value - best_val) < 1e-12:
        return f"\\textbf{{\\textcolor{{{LATEX_BEST_COLOR}}}{{{formatted}}}}}"
    if worst_val is not None and abs(value - worst_val) < 1e-12:
        return f"\\textcolor{{{LATEX_WORST_COLOR}}}{{{formatted}}}"
    return formatted


def format_cell_mean_std(
    metric: str,
    mean_val: float,
    std_val: float,
    best_mean: float,
    worst_mean: float,
) -> str:
    base = f"{mean_val:.3f} ($\\pm$ {std_val:.3f})"
    best_fmt = f"{best_mean:.3f}"
    worst_fmt = f"{worst_mean:.3f}"
    if f"{mean_val:.3f}" == best_fmt:
        return f"\\textbf{{\\textcolor{{{LATEX_BEST_COLOR}}}{{{base}}}}}"
    if f"{mean_val:.3f}" == worst_fmt:
        return f"\\textcolor{{{LATEX_WORST_COLOR}}}{{{base}}}"
    return base


def format_value_for_display(mean_val: Optional[float], std_val: Optional[float], count: int) -> str:
    if mean_val is None:
        return "-"
    if count <= 1 or std_val is None:
        return f"{mean_val:.3f}"
    return f"{mean_val:.3f} +/- {std_val:.3f}"


def format_diff_latex(metric: str, base_val: Optional[float], new_val: Optional[float]) -> str:
    return ""


def format_diff_html(metric: str, base_val: Optional[float], new_val: Optional[float]) -> str:
    if base_val is None or new_val is None:
        return ""
    diff = new_val - base_val
    if abs(diff) < 1e-12:
        return '<span style="color:#666666;">→0.000</span>'
    improvement = better(metric, new_val, base_val)
    color = HTML_IMPROVEMENT_COLOR if improvement else HTML_REGRESSION_COLOR
    arrow = "▲" if diff > 0 else "▼"
    return f'<span style="color:{color};">{arrow}{abs(diff):.3f}</span>'


def compute_best_worst_maps(
    dataset_names: List[str],
    metrics: List[str],
    method_dataset_metrics: Dict[str, Dict[str, Dict[str, Tuple[Optional[float], Optional[float], int]]]],
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    best: Dict[Tuple[str, str], float] = {}
    worst: Dict[Tuple[str, str], float] = {}
    for dataset in dataset_names:
        for metric in metrics:
            candidate_values: List[float] = []
            for method in METHOD_ORDER:
                mean_val = method_dataset_metrics.get(method, {}).get(dataset, {}).get(metric, (None, None, 0))[0]
                if mean_val is not None:
                    candidate_values.append(mean_val)
            if not candidate_values:
                continue
            best_val = candidate_values[0]
            worst_val = candidate_values[0]
            for val in candidate_values[1:]:
                if better(metric, val, best_val):
                    best_val = val
                if better(metric, worst_val, val):
                    worst_val = val
            best[(dataset, metric)] = best_val
            worst[(dataset, metric)] = worst_val
    return best, worst


def generate_transposed_latex_table(
    dataset_names: List[str],
    metrics: List[str],
    method_dataset_metrics: Dict[str, Dict[str, Dict[str, Tuple[Optional[float], Optional[float], int]]]],
    output_path: str,
) -> None:
    metrics = order_metrics(metrics)
    dataset_names = order_datasets(dataset_names)

    best_map, worst_map = compute_best_worst_maps(dataset_names, metrics, method_dataset_metrics)
    metric_count = len(metrics)
    dataset_labels = [dataset_short_latex_label(ds) for ds in dataset_names]

    if dataset_names:
        column_spec = " ".join("c" for _ in dataset_names)
        table_spec = f"ll {column_spec}"
    else:
        table_spec = "ll"

    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\setlength{\\tabcolsep}{3.7pt}")
    lines.append("  \\resizebox{0.9\\linewidth}{!}{%")
    lines.append(f"  \\begin{{tabular}}{{{table_spec}}}")
    lines.append("    \\toprule")

    header_entries = ["", ""] + dataset_labels
    lines.append("    " + " & ".join(header_entries) + " \\\\")
    lines.append("    \\midrule")

    for method_index, method in enumerate(METHOD_ORDER):
        ds_map = method_dataset_metrics.get(method, {})
        method_label = latex_escape(method)
        for metric_index, metric in enumerate(metrics):
            row_entries: List[str] = []
            if metric_index == 0:
                row_entries.append(f"\\multirow{{{metric_count}}}{{*}}{{{method_label}}}")
            else:
                row_entries.append("")
            row_entries.append(metric_latex_label(metric))
            for dataset in dataset_names:
                metric_map = ds_map.get(dataset, {})
                mean_val = metric_map.get(metric, (None, None, 0))[0]
                scaled_mean = scale_metric_for_display(metric, mean_val)
                if scaled_mean is None:
                    row_entries.append("-")
                    continue
                best_val = best_map.get((dataset, metric))
                worst_val = worst_map.get((dataset, metric))
                row_entries.append(format_transposed_cell(metric, scaled_mean, best_val, worst_val))
            lines.append("    " + " & ".join(row_entries) + " \\\\")
        if method_index < len(METHOD_ORDER) - 1:
            lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}}")
    method_text = ", ".join(METHOD_ORDER)
    lines.append(
        "    \\caption{"
        + "Comparison across datasets for methods: "
        + f"{latex_escape(method_text)}. "
        + "Bold blue values mark the best metric per dataset column, orange values denote the worst."
        + "}"
    )
    lines.append("  \\label{aggregate_all_datasets_pivoted}")
    lines.append("\\end{table*}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Wrote transposed LaTeX table to {output_path}")


def generate_latex_table(
    dataset_names: List[str],
    metrics: List[str],
    method_dataset_metrics: Dict[str, Dict[str, Dict[str, Tuple[Optional[float], Optional[float], int]]]],
    selection_metric: str,
    output_path: str,
) -> None:
    metrics = order_metrics(metrics)
    dataset_names = order_datasets(dataset_names)

    best_map, worst_map = compute_best_worst_maps(dataset_names, metrics, method_dataset_metrics)
    base_map: Dict[Tuple[str, str], Optional[float]] = {}
    base_ds_map = method_dataset_metrics.get("Base", {})
    for dataset in dataset_names:
        metric_map = base_ds_map.get(dataset, {})
        for metric in metrics:
            entry = metric_map.get(metric)
            base_mean = entry[0] if entry else None
            base_map[(dataset, metric)] = base_mean
    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\large")
    caption_metrics = ", ".join(metric_latex_label(m) for m in metrics)
    selection_metric_label = metric_latex_label(selection_metric)
    method_text = ", ".join(METHOD_ORDER)
    lines.append(
        "  \\caption{"
        + f"Comparison across datasets for methods: {latex_escape(method_text)}. "
        + f"Selection metric: {selection_metric_label}. Metrics per dataset (Macro F1): {caption_metrics}. "
        + "Bold blue values mark the best metric per dataset column, orange values denote the worst."
        + "}"
    )
    lines.append("  \\setlength{\\tabcolsep}{3.7pt}")
    lines.append("  \\resizebox{\\linewidth}{!}{%")

    dataset_groups: List[List[str]] = []
    used_datasets: Set[str] = set()
    for group in DATASET_GROUPS:
        subset = [ds for ds in group if ds in dataset_names]
        if subset:
            dataset_groups.append(subset)
            used_datasets.update(subset)
    remaining = [ds for ds in dataset_names if ds not in used_datasets]
    if remaining:
        dataset_groups.append(remaining)

    def build_group_tabular(subset: List[str]) -> List[str]:
        if not subset:
            return []
        metric_count = len(metrics)
        col_blocks = ["c" * metric_count for _ in subset]
        if col_blocks:
            table_spec = f"p{{{METHOD_COL_WIDTH}}}{''.join(col_blocks)}"
        else:
            table_spec = f"p{{{METHOD_COL_WIDTH}}}"
        group_lines: List[str] = []
        group_lines.append(f"\\begin{{tabular}}{{{table_spec}}}")
        group_lines.append("  \\toprule")
        header_top = [""]
        for idx, dataset in enumerate(subset):
            span_fmt = "c"
            header_top.append(f"\\multicolumn{{{metric_count}}}{{{span_fmt}}}{{{dataset_latex_label(dataset)}}}")
        group_lines.append("  " + " & ".join(header_top) + " \\\\")
        header_bottom = ["Method"]
        for _ in subset:
            for metric in metrics:
                header_bottom.append(metric_latex_label(metric))
        group_lines.append("  " + " & ".join(header_bottom) + " \\\\")
        separator_rules: List[str] = ["\\cmidrule(lr){1-1}"]
        for idx in range(len(subset)):
            start = 2 + idx * metric_count
            end = start + metric_count - 1
            separator_rules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        group_lines.append("  " + " ".join(separator_rules))

        for method in METHOD_ORDER:
            row_cells = [latex_escape(method)]
            ds_map = method_dataset_metrics.get(method, {})
            for dataset in subset:
                metric_map = ds_map.get(dataset, {})
                for metric in metrics:
                    mean_val, std_val, count = metric_map.get(metric, (None, None, 0))
                    if mean_val is None:
                        row_cells.append("-")
                        continue
                    scaled_mean = scale_metric_for_display(metric, mean_val)
                    scaled_std = scale_metric_for_display(metric, std_val)
                    base_mean = base_map.get((dataset, metric))
                    scaled_base = scale_metric_for_display(metric, base_mean)
                    diff_text = ""
                    if scaled_base is not None and scaled_mean is not None:
                        diff_text = format_diff_latex(metric, scaled_base, scaled_mean)
                    best_val = best_map.get((dataset, metric))
                    worst_val = worst_map.get((dataset, metric))
                    scaled_best = scale_metric_for_display(metric, best_val)
                    scaled_worst = scale_metric_for_display(metric, worst_val)
                    if scaled_mean is None:
                        row_cells.append("-")
                        continue
                    if scaled_best is None or scaled_worst is None:
                        if count > 1 and scaled_std is not None:
                            value_text = f"{scaled_mean:.3f} ($\\pm$ {scaled_std:.3f})"
                        else:
                            value_text = f"{scaled_mean:.3f}"
                    elif count <= 1 or scaled_std is None:
                        value_text = format_cell(metric, scaled_mean, scaled_best, scaled_worst)
                    else:
                        value_text = format_cell_mean_std(
                            metric,
                            scaled_mean,
                            scaled_std,
                            scaled_best,
                            scaled_worst,
                        )
                    if diff_text and value_text != "-":
                        value_text = value_text + diff_text
                    row_cells.append(value_text)
            group_lines.append("  " + " & ".join(row_cells) + " \\\\")

        group_lines.append("  \\bottomrule")
        group_lines.append("\\end{tabular}")
        return group_lines

    if len(dataset_groups) <= 1:
        subset = dataset_groups[0] if dataset_groups else []
        for line in build_group_tabular(subset):
            lines.append("  " + line)
    else:
        lines.append("  \\begin{tabular}{@{}l@{}}")
        for idx, subset in enumerate(dataset_groups):
            for sub_line in build_group_tabular(subset):
                lines.append("    " + sub_line)
            if idx < len(dataset_groups) - 1:
                lines.append("    \\\\[60pt]")
        lines.append("  \\end{tabular}")

    lines.append("  }")
    lines.append("  \\label{aggregate_all_datasets}")
    lines.append("\\end{table*}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Wrote LaTeX table to {output_path}")


def generate_html_table(
    dataset_names: List[str],
    metrics: List[str],
    method_dataset_metrics: Dict[str, Dict[str, Dict[str, Tuple[Optional[float], Optional[float], int]]]],
    selection_metric: str,
    output_path: str,
) -> None:
    if pd is None:
        print("Skipping HTML output because pandas is not installed.")
        return

    metrics = order_metrics(metrics)
    dataset_names = order_datasets(dataset_names)
    dataset_display_map = {ds: DATASET_DISPLAY_NAMES.get(ds, ds) for ds in dataset_names}
    columns = pd.MultiIndex.from_product([[dataset_display_map[ds] for ds in dataset_names], metrics], names=["Dataset", "Metric"])
    df_display = pd.DataFrame("-", index=METHOD_ORDER, columns=columns)
    df_means = pd.DataFrame(float("nan"), index=METHOD_ORDER, columns=columns)

    best_map, worst_map = compute_best_worst_maps(dataset_names, metrics, method_dataset_metrics)
    base_map: Dict[Tuple[str, str], Optional[float]] = {}
    base_ds_map = method_dataset_metrics.get("Base", {})
    for dataset in dataset_names:
        metric_map = base_ds_map.get(dataset, {})
        for metric in metrics:
            entry = metric_map.get(metric)
            base_mean = entry[0] if entry else None
            base_map[(dataset, metric)] = base_mean

    for method in METHOD_ORDER:
        ds_map = method_dataset_metrics.get(method, {})
        for dataset in dataset_names:
            metric_map = ds_map.get(dataset, {})
            display_dataset = dataset_display_map[dataset]
            for metric in metrics:
                mean_val, std_val, count = metric_map.get(metric, (None, None, 0))
                scaled_mean = scale_metric_for_display(metric, mean_val)
                scaled_std = scale_metric_for_display(metric, std_val)
                df_means.loc[method, (display_dataset, metric)] = scaled_mean if scaled_mean is not None else float("nan")
                display_value = format_value_for_display(scaled_mean, scaled_std, count)
                base_mean = base_map.get((dataset, metric))
                scaled_base = scale_metric_for_display(metric, base_mean)
                diff_html = ""
                if scaled_mean is not None and scaled_base is not None:
                    diff_html = format_diff_html(metric, scaled_base, scaled_mean)
                if diff_html and display_value != "-":
                    display_value = f"{display_value}<br>{diff_html}"
                df_display.loc[method, (display_dataset, metric)] = display_value

    def highlight(_: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df_display.index, columns=df_display.columns)
        for dataset in dataset_names:
            display_dataset = dataset_display_map[dataset]
            for metric in metrics:
                best_val = scale_metric_for_display(metric, best_map.get((dataset, metric)))
                worst_val = scale_metric_for_display(metric, worst_map.get((dataset, metric)))
                for method in METHOD_ORDER:
                    mean_val = df_means.loc[method, (display_dataset, metric)]
                    if pd.isna(mean_val):
                        continue
                    if best_val is not None and abs(mean_val - best_val) < 1e-12:
                        styles.loc[method, (display_dataset, metric)] = (
                            f"background-color:{HTML_BEST_BACKGROUND};font-weight:bold;"
                        )
                    elif worst_val is not None and abs(mean_val - worst_val) < 1e-12:
                        styles.loc[method, (display_dataset, metric)] = (
                            f"background-color:{HTML_WORST_BACKGROUND};"
                        )
        return styles

    styler = df_display.style.apply(highlight, axis=None)
    styler = styler.set_table_styles(
        [
            {"selector": "table", "props": [("border-collapse", "collapse"), ("margin", "0 auto")]},
            {"selector": "th, td", "props": [("border", "1px solid #999"), ("padding", "0.35em 0.6em")]},
        ]
    )
    styler = styler.set_properties(**{"text-align": "center", "font-family": "monospace"})
    styler = styler.set_caption(f"Selection metric: {selection_metric}")

    html = styler.to_html()

    def compute_average_change(method: str, reference: str) -> Dict[str, Optional[float]]:
        deltas: Dict[str, Optional[float]] = {}
        for metric in metrics:
            per_dataset: List[float] = []
            for dataset in dataset_names:
                display_dataset = dataset_display_map[dataset]
                method_val = df_means.loc[method, (display_dataset, metric)]
                ref_val = df_means.loc[reference, (display_dataset, metric)]
                if pd.isna(method_val) or pd.isna(ref_val):
                    continue
                per_dataset.append(method_val - ref_val)
            deltas[metric] = statistics.mean(per_dataset) if per_dataset else None
        return deltas

    def format_change_value(value: Optional[float]) -> str:
        if value is None:
            return "-"
        adjusted = 0.0 if abs(value) < 1e-12 else value
        return f"{adjusted:+.3f}"

    def render_summary_table(title: str, data: Dict[str, Dict[str, Optional[float]]]) -> str:
        if not data:
            return ""
        header_cells = "".join(f"<th>{escape(metric)}</th>" for metric in metrics)
        rows: List[str] = []
        for method in METHOD_ORDER:
            if method not in data:
                continue
            cell_values = "".join(f"<td>{format_change_value(data[method].get(metric))}</td>" for metric in metrics)
            rows.append(f"<tr><td>{escape(method)}</td>{cell_values}</tr>")
        if not rows:
            return ""
        return (
            f"<h3>{escape(title)}</h3>"
            + '<table style="margin:0.5em auto;border-collapse:collapse;font-family:monospace;">'
            + f"<thead><tr><th>Method</th>{header_cells}</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    avg_change_vs_base: Dict[str, Dict[str, Optional[float]]] = {}
    for method in METHOD_ORDER[1:]:
        avg_change_vs_base[method] = compute_average_change(method, "Base")

    avg_change_vs_swarm: Dict[str, Dict[str, Optional[float]]] = {}
    if "Swarm" in METHOD_ORDER:
        for method in ("CCE", "F$^2$C"):
            if method in METHOD_ORDER:
                avg_change_vs_swarm[method] = compute_average_change(method, "Swarm")

    summary_sections: List[str] = []
    base_summary = render_summary_table("Average Δ vs Base", avg_change_vs_base)
    if base_summary:
        summary_sections.append(base_summary)
    swarm_summary = render_summary_table("Average Δ vs Swarm", avg_change_vs_swarm)
    if swarm_summary:
        summary_sections.append(swarm_summary)

    if summary_sections:
        summary_html = '<div class="aggregate-summary" style="max-width:85%;margin:2em auto;font-family:monospace;">' + "<h2>Average Change Statistics</h2>" + "".join(summary_sections) + "</div>"
        html += summary_html

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"Wrote HTML table to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics across datasets and output LaTeX and HTML tables.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "metrics",
        nargs="+",
        help="List of metric names to include; the first is used as the selection metric.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="checkpoints",
        help="Root directory containing dataset subdirectories with experiment runs.",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        default="accuracy_",
        help="Prefix of metrics computed on the evaluation split (e.g., accuracy_).",
    )
    parser.add_argument(
        "--selection-prefix",
        default="unsupervised_dev_",
        help="Prefix of metrics computed on the selection split (e.g., unsupervised_dev_).",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Maximum checkpoint step to consider when scanning metrics.",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=[],
        help=("Dataset names or glob patterns to exclude (case-insensitive). " "Supports wildcards and comma-separated entries."),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("results", "aggregate_all_datasets.tex"),
        help="Path to the LaTeX output file.",
    )
    parser.add_argument(
        "--transposed-output",
        default=os.path.join("results", "aggregate_all_datasets_transposed.tex"),
        help="Path to the transposed LaTeX output file.",
    )
    parser.add_argument(
        "--html-output",
        default=os.path.join("results", "aggregate_all_datasets.html"),
        help="Path to the HTML output file.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoints_dir):
        print(f"Error: {args.checkpoints_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    selection_metric = args.metrics[0]
    metrics = list(dict.fromkeys(args.metrics))

    raw_dirs = [os.path.join(args.checkpoints_dir, name) for name in sorted(os.listdir(args.checkpoints_dir)) if os.path.isdir(os.path.join(args.checkpoints_dir, name)) and name.lower() != "debug"]

    exclude_patterns: List[str] = []
    for token in args.exclude_datasets or []:
        if not token:
            continue
        for part in str(token).split(","):
            cleaned = part.strip()
            if cleaned:
                exclude_patterns.append(cleaned)

    dataset_dirs: List[str] = []
    dataset_names: List[str] = []
    for directory in raw_dirs:
        name = os.path.basename(directory)
        name_lower = name.lower()
        excluded = False
        for pattern in exclude_patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in name_lower or fnmatch(name_lower, pattern_lower):
                excluded = True
                break
        if not excluded:
            dataset_dirs.append(directory)
            dataset_names.append(name)

    method_dataset_metrics: Dict[str, Dict[str, Dict[str, Tuple[Optional[float], Optional[float], int]]]] = {}
    for method in METHOD_ORDER:
        method_dataset_metrics[method] = {}

    for dataset_name, dataset_dir in zip(dataset_names, dataset_dirs):
        dataset_max_step = args.max_step
        if dataset_name.lower() == "hellaswag" and (dataset_max_step is None or dataset_max_step > 70):
            dataset_max_step = 70
        per_method = collect_dataset_metrics(
            dataset_dir,
            metrics,
            selection_metric,
            args.checkpoint_prefix,
            args.selection_prefix,
            dataset_max_step,
        )
        for method in METHOD_ORDER:
            method_dataset_metrics.setdefault(method, {})[dataset_name] = per_method.get(method, {})

    generate_latex_table(dataset_names, metrics, method_dataset_metrics, selection_metric, args.output)
    generate_transposed_latex_table(dataset_names, metrics, method_dataset_metrics, args.transposed_output)
    generate_html_table(dataset_names, metrics, method_dataset_metrics, selection_metric, args.html_output)


if __name__ == "__main__":
    main()
