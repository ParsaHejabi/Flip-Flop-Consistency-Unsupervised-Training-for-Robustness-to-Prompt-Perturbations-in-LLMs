import argparse
import os
import re
import statistics
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


DEFAULT_METRICS = [
    "mean_f1",
    "std_f1",
    "raw_agreement",
    "avg_ensemble_f1",
    "vote_ensemble_f1",
]


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

    fallback_step: Optional[int] = None
    fallback_val: Optional[float] = None
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
        if fallback_val is None or better(selection_metric, candidate, fallback_val):
            fallback_val = candidate
            fallback_step = step
            fallback_metrics = metrics

    if fallback_metrics is not None:
        return fallback_step, fallback_metrics, None

    return None, {}, None


def _should_replace(existing_dt: Optional[datetime], new_dt: Optional[datetime]) -> bool:
    if existing_dt is None and new_dt is not None:
        return True
    if existing_dt is not None and new_dt is not None and new_dt >= existing_dt:
        return True
    return False


def collect_dataset_summary(
    dataset_dir: str,
    metrics: Iterable[str],
    selection_metric: str,
    checkpoint_prefix: str,
    selection_prefix: str,
    max_step: Optional[int],
) -> List[Dict[str, Any]]:
    metric_names = list(metrics)
    base_by_seed: Dict[str, Dict[str, object]] = {}
    method_groups: Dict[str, Dict[Tuple[str, ...], Dict[str, List[Dict[str, object]]]]] = {}
    method_params: Dict[str, Dict[Tuple[str, ...], Dict[str, str]]] = {}

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

        step, report_metrics, selection_value = select_checkpoint(
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
        param_display = {HYPERPARAM_DISPLAY.get(key, key): str(run_vars.get(key, "-")) for key in param_keys}

        method_groups.setdefault(method_label, {})
        method_params.setdefault(method_label, {})
        signature_records = method_groups[method_label].setdefault(param_signature, {})
        if param_signature not in method_params[method_label]:
            method_params[method_label][param_signature] = param_display

        record = {
            "metrics": filtered_metrics,
            "selection": selection_value,
            "seed": seed_key,
            "timestamp": timestamp,
            "step": step,
            "run_name": entry,
        }
        signature_records.setdefault(seed_key, [])
        signature_records[seed_key].append(record)

    entries: List[Dict[str, Any]] = []

    def _default_metrics_summary() -> Dict[str, Tuple[Optional[float], Optional[float], int]]:
        return {m: (None, None, 0) for m in metric_names}

    # Base model aggregation (single row)
    base_metrics_summary: Dict[str, Tuple[Optional[float], Optional[float], int]] = _default_metrics_summary()
    if base_by_seed:
        for metric in metric_names:
            vals = [rec["metrics"].get(metric) for rec in base_by_seed.values() if metric in rec["metrics"]]
            if not vals:
                continue
            mean_val = statistics.mean(vals)
            std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
            base_metrics_summary[metric] = (mean_val, std_val, len(vals))
    base_entry: Dict[str, Any] = {
        "method": "Base",
        "display": "Base",
        "metrics": base_metrics_summary,
        "params": {},
        "seeds": sorted(base_by_seed.keys(), key=lambda s: int(s) if s.isdigit() else s),
        "timestamp": max(
            (rec.get("timestamp") for rec in base_by_seed.values() if rec.get("timestamp") is not None),
            default=None,
        ),
        "mean_selection": None,
        "best_selection": None,
        "best_selection_seed": None,
        "best_selection_step": None,
        "best_checkpoint": None,
        "order": METHOD_ORDER.index("Base"),
        "order_serial": 0,
    }
    entries.append(base_entry)

    def _summarise_records(record_list: List[Dict[str, Any]]) -> Dict[str, Tuple[Optional[float], Optional[float], int]]:
        metrics_summary: Dict[str, Tuple[Optional[float], Optional[float], int]] = _default_metrics_summary()
        for metric in metric_names:
            vals = [rec["metrics"].get(metric) for rec in record_list if metric in rec["metrics"]]
            if not vals:
                continue
            mean_val = statistics.mean(vals)
            std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
            metrics_summary[metric] = (mean_val, std_val, len(vals))
        return metrics_summary

    for method_label in METHOD_ORDER:
        if method_label == "Base":
            continue
        groups = method_groups.get(method_label, {})
        if not groups:
            entries.append(
                {
                    "method": method_label,
                    "display": method_label,
                    "metrics": _default_metrics_summary(),
                    "params": {},
                    "seeds": [],
                    "timestamp": None,
                    "mean_selection": None,
                    "best_selection": None,
                    "best_selection_seed": None,
                    "best_selection_step": None,
                    "best_checkpoint": None,
                    "order": METHOD_ORDER.index(method_label),
                }
            )
            continue

        prepared_rows: List[Dict[str, Any]] = []
        for signature, seed_records in groups.items():
            flat_records: List[Dict[str, Any]] = []
            for records in seed_records.values():
                flat_records.extend(records)
            if not flat_records:
                continue

            signature_rows: List[Dict[str, Any]] = []
            for rec in flat_records:
                metrics_summary = _summarise_records([rec])
                seed_val = rec.get("seed")
                timestamp_val = rec.get("timestamp")
                selection_val = rec.get("selection")
                step_val = rec.get("step")
                run_name = rec.get("run_name") or ""
                best_checkpoint = None
                if step_val is not None:
                    checkpoint_name = f"{checkpoint_prefix}{step_val}"
                    best_checkpoint = os.path.join(run_name, checkpoint_name) if run_name else checkpoint_name
                signature_rows.append(
                    {
                        "method": method_label,
                        "metrics": metrics_summary,
                        "params": method_params.get(method_label, {}).get(signature, {}),
                        "seeds": [seed_val] if seed_val is not None else [],
                        "timestamp": timestamp_val,
                        "mean_selection": selection_val,
                        "best_selection": selection_val,
                        "best_selection_seed": seed_val,
                        "best_selection_step": step_val,
                        "best_checkpoint": best_checkpoint,
                        "order": METHOD_ORDER.index(method_label),
                        "run_name": run_name,
                        "_sort_selection": selection_val,
                        "order_serial": 0,
                    }
                )

            reverse_sort = better(selection_metric, 1.0, 0.0)

            def _signature_sort_key(row: Dict[str, Any]) -> float:
                val = row["_sort_selection"]
                if val is None:
                    return float("-inf") if reverse_sort else float("inf")
                return float(val)

            signature_rows.sort(key=_signature_sort_key, reverse=reverse_sort)

            needs_index = len(signature_rows) > 1 or len(groups) > 1
            for idx, row in enumerate(signature_rows, start=1):
                ts = row.get("timestamp")
                ts_label = ""
                if isinstance(ts, datetime):
                    ts_label = ts.strftime("%Y-%m-%d %H:%M:%S")
                elif row.get("run_name"):
                    ts_label = row["run_name"]
                label = method_label
                if needs_index:
                    label = f"{label} #{idx}"
                if ts_label:
                    label = f"{label} [{ts_label}]"
                row["display"] = label
                row["order_serial"] = len(prepared_rows)
                row.pop("_sort_selection", None)
                row.pop("run_name", None)
                prepared_rows.append(row)

        if not prepared_rows:
            entries.append(
                {
                    "method": method_label,
                    "display": method_label,
                    "metrics": _default_metrics_summary(),
                    "params": {},
                    "seeds": [],
                    "timestamp": None,
                    "mean_selection": None,
                    "best_selection": None,
                    "best_selection_seed": None,
                    "best_selection_step": None,
                    "best_checkpoint": None,
                    "order": METHOD_ORDER.index(method_label),
                    "order_serial": 0,
                }
            )
            continue

        entries.extend(prepared_rows)

    return entries


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def metric_latex_label(metric: str) -> str:
    return METRIC_LATEX_LABELS.get(metric, latex_escape(metric))


def order_metrics(metrics: Iterable[str]) -> List[str]:
    enumerated = list(enumerate(metrics))
    order_lookup = {name: idx for idx, name in enumerate(METRIC_DISPLAY_ORDER)}
    enumerated.sort(key=lambda item: (order_lookup.get(item[1], len(METRIC_DISPLAY_ORDER)), item[0]))
    return [item[1] for item in enumerated]


def scale_metric_for_display(metric: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value


def format_value(mean_val: Optional[float], std_val: Optional[float], count: int) -> str:
    if mean_val is None:
        return "-"
    if count <= 1 or std_val is None:
        return f"{mean_val:.3f}"
    return f"{mean_val:.3f} ($\\pm$ {std_val:.3f})"


def format_html_value(mean_val: Optional[float], std_val: Optional[float], count: int) -> str:
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
    color = "#228B22" if improvement else "#d62728"
    arrow = "▲" if diff > 0 else "▼"
    return f'<span style="color:{color};">{arrow}{abs(diff):.3f}</span>'


def compute_best_worst(
    rows: List[Dict[str, Any]],
    metrics: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    best_vals: Dict[str, float] = {}
    worst_vals: Dict[str, float] = {}
    for metric in metrics:
        best_val: Optional[float] = None
        worst_val: Optional[float] = None
        for row in rows:
            mean_val = row.get("metrics", {}).get(metric, (None, None, 0))[0]
            if mean_val is None:
                continue
            if best_val is None or better(metric, mean_val, best_val):
                best_val = mean_val
            if worst_val is None or better(metric, worst_val, mean_val):
                worst_val = mean_val
        if best_val is None or worst_val is None:
            continue
        best_vals[metric] = best_val
        worst_vals[metric] = worst_val
    return best_vals, worst_vals


def generate_latex_table(
    dataset_name: str,
    rows: List[Dict[str, Any]],
    metrics: List[str],
    selection_metric: str,
    output_path: str,
) -> None:
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row.get("order", 0),
            row.get("order_serial", 0),
            row.get("display", row.get("method", "")),
        ),
    )
    metrics = order_metrics(metrics)
    best_map, worst_map = compute_best_worst(ordered_rows, metrics)
    base_metrics_map = next((row.get("metrics", {}) for row in ordered_rows if row.get("method") == "Base"), {})

    all_params: List[str] = []
    for row in ordered_rows:
        params = row.get("params", {})
        for key in params.keys():
            if key not in all_params:
                all_params.append(key)
    all_params = [p for p in all_params if any(str(row.get("params", {}).get(p, "-")).strip() not in {"-", ""} for row in ordered_rows)]

    def _format_best_checkpoint(row: Dict[str, Any]) -> str:
        step = row.get("best_selection_step")
        if step is None:
            return "-"
        parts: List[str] = []
        checkpoint = row.get("best_checkpoint")
        if checkpoint:
            parts.append(latex_escape(str(checkpoint)))
        else:
            parts.append(f"step {step}")
        meta_bits: List[str] = []
        seed = row.get("best_selection_seed")
        if seed is not None:
            meta_bits.append(f"seed {latex_escape(str(seed))}")
        sel_val = row.get("best_selection")
        if sel_val is not None:
            meta_bits.append(f"{metric_latex_label(selection_metric)}={sel_val:.3f}")
        if meta_bits:
            parts.append("(" + ", ".join(meta_bits) + ")")
        return " ".join(parts)

    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    caption_metrics = ", ".join(metric_latex_label(m) for m in metrics)
    selection_metric_label = metric_latex_label(selection_metric)
    lines.append(
        "  \\caption{"
        + f"Best checkpoints for {latex_escape(dataset_name)}. "
        + f"Selection metric: {selection_metric_label}. Metrics (Macro F1): {caption_metrics}."
        + "}"
    )
    lines.append("  \\setlength{\\tabcolsep}{3pt}")
    lines.append("  \\resizebox{\\linewidth}{!}{%")
    column_count = 4 + len(all_params) + len(metrics)
    lines.append(f"  \\begin{{tabular}}{{p{{4.4cm}}|{'c' * (column_count - 1)}}}")
    lines.append("    \\toprule")
    header = ["Method", "Timestamp", "Seeds", "Best Checkpoint"] + [latex_escape(p) for p in all_params] + [metric_latex_label(m) for m in metrics]
    lines.append("    " + " & ".join(header) + " \\\\")
    lines.append("    \\midrule")

    for row_data in ordered_rows:
        timestamp = row_data.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp_str = "-"
        seeds = row_data.get("seeds", [])
        seeds_str = ", ".join(str(s) for s in seeds) if seeds else "-"
        params = row_data.get("params", {})
        metrics_map = row_data.get("metrics", {})
        line_cells = [
            latex_escape(str(row_data.get("display", row_data.get("method", "-")))),
            latex_escape(timestamp_str),
            latex_escape(seeds_str),
            _format_best_checkpoint(row_data),
        ]
        for param in all_params:
            line_cells.append(latex_escape(str(params.get(param, "-"))))
        for metric in metrics:
            mean_val, std_val, count = metrics_map.get(metric, (None, None, 0))
            scaled_mean = scale_metric_for_display(metric, mean_val)
            scaled_std = scale_metric_for_display(metric, std_val)
            best_val = best_map.get(metric)
            worst_val = worst_map.get(metric)
            scaled_best = scale_metric_for_display(metric, best_val)
            scaled_worst = scale_metric_for_display(metric, worst_val)
            base_entry = base_metrics_map.get(metric)
            base_mean = base_entry[0] if base_entry else None
            scaled_base = scale_metric_for_display(metric, base_mean)
            diff_text = ""
            if scaled_mean is not None and scaled_base is not None:
                diff_text = format_diff_latex(metric, scaled_base, scaled_mean)
            if scaled_mean is None:
                line_cells.append("-")
                continue
            if scaled_best is None or scaled_worst is None:
                value_text = format_value(scaled_mean, scaled_std, count)
            elif count <= 1 or scaled_std is None:
                formatted = format_value(scaled_mean, None, count)
                if abs(scaled_mean - scaled_best) < 1e-12:
                    value_text = f"\\textbf{{\\textcolor{{ForestGreen}}{{{formatted}}}}}"
                elif abs(scaled_mean - scaled_worst) < 1e-12:
                    value_text = f"\\textcolor{{red}}{{{formatted}}}"
                else:
                    value_text = formatted
            else:
                formatted = format_value(scaled_mean, scaled_std, count)
                if abs(scaled_mean - scaled_best) < 1e-12:
                    value_text = f"\\textbf{{\\textcolor{{ForestGreen}}{{{formatted}}}}}"
                elif abs(scaled_mean - scaled_worst) < 1e-12:
                    value_text = f"\\textcolor{{red}}{{{formatted}}}"
                else:
                    value_text = formatted
            if diff_text and value_text != "-":
                value_text += diff_text
            line_cells.append(value_text)
        lines.append("    " + " & ".join(line_cells) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  }")
    label_name = re.sub(r"[^A-Za-z0-9]+", "_", dataset_name.lower()).strip("_") or "dataset"
    lines.append(f"  \\label{{best_ckpt_{label_name}}}")
    lines.append("\\end{table*}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Wrote LaTeX table to {output_path}")


def generate_html_table(
    dataset_name: str,
    rows: List[Dict[str, Any]],
    metrics: List[str],
    selection_metric: str,
    output_path: str,
) -> None:
    if pd is None:
        print("Skipping HTML output because pandas is not installed.")
        return

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row.get("order", 0),
            row.get("order_serial", 0),
            row.get("display", row.get("method", "")),
        ),
    )
    metrics = order_metrics(metrics)
    all_params: List[str] = []
    for row in ordered_rows:
        params = row.get("params", {})
        for key in params.keys():
            if key not in all_params:
                all_params.append(key)
    all_params = [p for p in all_params if any(str(row.get("params", {}).get(p, "-")).strip() not in {"-", ""} for row in ordered_rows)]

    columns = ["Method", "Timestamp", "Seeds", "Best Checkpoint"] + all_params + metrics
    data: List[List[str]] = []
    mean_lookup: Dict[Tuple[str, str], Optional[float]] = {}
    best_map, worst_map = compute_best_worst(ordered_rows, metrics)
    base_metrics_map = next((row.get("metrics", {}) for row in ordered_rows if row.get("method") == "Base"), {})

    def _format_best_checkpoint_html(row: Dict[str, Any]) -> str:
        step = row.get("best_selection_step")
        if step is None:
            return "-"
        checkpoint = row.get("best_checkpoint")
        seed = row.get("best_selection_seed")
        sel_val = row.get("best_selection")
        primary = checkpoint or f"step {step}"
        details: List[str] = []
        if seed is not None:
            details.append(f"seed {seed}")
        if sel_val is not None:
            details.append(f"{selection_metric}={sel_val:.3f}")
        if details:
            return f"{primary}<br>({', '.join(details)})"
        return str(primary)

    for row_data in ordered_rows:
        timestamp = row_data.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp_str = "-"
        seeds = row_data.get("seeds", [])
        seeds_str = ", ".join(str(s) for s in seeds) if seeds else "-"
        params = row_data.get("params", {})
        metrics_map = row_data.get("metrics", {})
        method_label = str(row_data.get("display", row_data.get("method", "-")))
        row_values = [method_label, timestamp_str, seeds_str, _format_best_checkpoint_html(row_data)]
        for param in all_params:
            row_values.append(str(params.get(param, "-")))
        for metric in metrics:
            mean_val, std_val, count = metrics_map.get(metric, (None, None, 0))
            scaled_mean = scale_metric_for_display(metric, mean_val)
            scaled_std = scale_metric_for_display(metric, std_val)
            mean_lookup[(method_label, metric)] = scaled_mean
            value_text = format_html_value(scaled_mean, scaled_std, count)
            base_entry = base_metrics_map.get(metric)
            base_mean = base_entry[0] if base_entry else None
            scaled_base = scale_metric_for_display(metric, base_mean)
            diff_html = ""
            if scaled_mean is not None and scaled_base is not None:
                diff_html = format_diff_html(metric, scaled_base, scaled_mean)
            if diff_html and value_text != "-":
                value_text = f"{value_text}<br>{diff_html}"
            row_values.append(value_text)
        data.append(row_values)

    df = pd.DataFrame(data, columns=columns)

    def highlight(row: "pd.Series") -> List[str]:
        styles: List[str] = [""] * len(row)
        method_label = row.iloc[0]
        for idx, column in enumerate(columns):
            if column not in metrics:
                continue
            mean_val = mean_lookup.get((method_label, column))
            best_val = scale_metric_for_display(column, best_map.get(column))
            worst_val = scale_metric_for_display(column, worst_map.get(column))
            if mean_val is None:
                continue
            if best_val is not None and abs(mean_val - best_val) < 1e-12:
                styles[idx] = "background-color:#d5f5d5;font-weight:bold;"
            elif worst_val is not None and abs(mean_val - worst_val) < 1e-12:
                styles[idx] = "background-color:#f8dada;"
        return styles

    styler = df.style.apply(highlight, axis=1)
    styler = styler.set_table_styles(
        [
            {"selector": "table", "props": [("border-collapse", "collapse"), ("margin", "0 auto")]},
            {"selector": "th, td", "props": [("border", "1px solid #999"), ("padding", "0.35em 0.6em")]},
        ]
    )
    styler = styler.set_caption(f"{dataset_name}: selection metric {selection_metric}")
    styler = styler.set_properties(**{"text-align": "center", "font-family": "monospace"})

    html = styler.to_html()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"Wrote HTML table to {output_path}")


def write_summary_csv(
    dataset_name: str,
    rows: List[Dict[str, Any]],
    metrics: List[str],
    output_path: str,
) -> None:
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row.get("order", 0),
            row.get("order_serial", 0),
            row.get("display", row.get("method", "")),
        ),
    )
    header = [
        "method",
        "display",
        "metric",
        "mean",
        "std",
        "count",
        "seeds",
        "params",
        "best_checkpoint",
        "best_selection_seed",
        "best_selection_step",
        "best_selection",
    ]
    lines: List[str] = [",".join(header)]

    for row_data in ordered_rows:
        metrics_map = row_data.get("metrics", {})
        seeds = row_data.get("seeds", [])
        seeds_str = ";".join(str(s) for s in seeds)
        params = row_data.get("params", {})
        params_str = ";".join(f"{k}={v}" for k, v in sorted(params.items()))
        best_checkpoint = row_data.get("best_checkpoint")
        best_seed = row_data.get("best_selection_seed")
        best_step = row_data.get("best_selection_step")
        best_selection_val = row_data.get("best_selection")
        for metric in metrics:
            mean_val, std_val, count = metrics_map.get(metric, (None, None, 0))
            if mean_val is None:
                continue
            std_text = f"{std_val:.6f}" if std_val is not None else ""
            best_sel_text = f"{best_selection_val:.6f}" if best_selection_val is not None else ""
            line = [
                str(row_data.get("method", "")),
                str(row_data.get("display", "")),
                metric,
                f"{mean_val:.6f}",
                std_text,
                str(count),
                seeds_str,
                params_str,
                str(best_checkpoint or ""),
                str(best_seed or ""),
                str(best_step or ""),
                best_sel_text,
            ]
            lines.append(",".join(line))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Wrote CSV summary to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate best checkpoints for a dataset and output LaTeX/HTML tables.")
    parser.add_argument("parent_dir", help="Dataset directory under checkpoints (e.g., checkpoints/rte)")
    parser.add_argument(
        "target_metric",
        nargs="?",
        default=None,
        help="Metric to use when selecting checkpoints (defaults to the first metric in --metrics).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metrics to report in the tables.",
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
        "--latex-output",
        default="table.tex",
        help="Filename for the LaTeX table (written under aggregate_best_checkpoints).",
    )
    parser.add_argument(
        "--html-output",
        default="table.html",
        help="Filename for the HTML table (written under aggregate_best_checkpoints).",
    )
    parser.add_argument(
        "--csv-output",
        default="summary.csv",
        help="Filename for the CSV summary (written under aggregate_best_checkpoints).",
    )
    args = parser.parse_args()

    parent_dir = os.path.abspath(args.parent_dir)
    if not os.path.isdir(parent_dir):
        raise SystemExit(f"{parent_dir} is not a directory")

    requested_metrics = list(dict.fromkeys(args.metrics)) or DEFAULT_METRICS
    selection_metric = args.target_metric or requested_metrics[0]

    def is_dataset_directory(path: str) -> bool:
        try:
            entries = os.listdir(path)
        except OSError:
            return False
        for entry in entries:
            run_dir = os.path.join(path, entry)
            if not os.path.isdir(run_dir):
                continue
            if os.path.isfile(os.path.join(run_dir, "run.sh")):
                return True
        return False

    if is_dataset_directory(parent_dir):
        dataset_dirs = [parent_dir]
    else:
        dataset_dirs = []
        for entry in sorted(os.listdir(parent_dir)):
            if entry == "debug":
                continue
            candidate = os.path.join(parent_dir, entry)
            if not os.path.isdir(candidate):
                continue
            if is_dataset_directory(candidate):
                dataset_dirs.append(candidate)

    if not dataset_dirs:
        raise SystemExit(f"No dataset directories found under {parent_dir}")

    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(os.path.abspath(dataset_dir))
        dataset_max_step = args.max_step
        if dataset_name.lower() == "hellaswag" and (dataset_max_step is None or dataset_max_step > 70):
            dataset_max_step = 70
        aggregates = collect_dataset_summary(
            dataset_dir,
            requested_metrics,
            selection_metric,
            args.checkpoint_prefix,
            args.selection_prefix,
            dataset_max_step,
        )

        metrics_for_dataset: List[str] = []
        for metric in requested_metrics:
            has_value = any(row.get("metrics", {}).get(metric, (None, None, 0))[0] is not None for row in aggregates)
            if has_value:
                metrics_for_dataset.append(metric)
        if not metrics_for_dataset:
            metrics_for_dataset = list(requested_metrics)

        target_dir = os.path.join(
            "results",
            dataset_name,
            re.sub(r"\s+", "_", selection_metric) if selection_metric else "per-metric",
        )

        latex_path = os.path.join(target_dir, args.latex_output)
        html_path = os.path.join(target_dir, args.html_output)
        csv_path = os.path.join(target_dir, args.csv_output)

        print(f"[aggregate] dataset={dataset_name} output_dir={target_dir}")
        generate_latex_table(dataset_name, aggregates, metrics_for_dataset, selection_metric, latex_path)
        generate_html_table(dataset_name, aggregates, metrics_for_dataset, selection_metric, html_path)
        write_summary_csv(dataset_name, aggregates, metrics_for_dataset, csv_path)


if __name__ == "__main__":
    main()
