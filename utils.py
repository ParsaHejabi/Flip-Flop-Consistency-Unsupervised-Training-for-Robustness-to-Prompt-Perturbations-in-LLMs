import math
import os

import numpy as np
import scipy
from sklearn.metrics import precision_recall_fscore_support


def index_median(array):
    x = np.argsort(array)
    h = len(x) // 2
    # return (x[h] + x[h+1]) // 2 if len(x) % 2 == 0 else x[h]
    return h


def compute_precision_recall_f1(preds, golds, avg_type):
    p, r, f1, _ = precision_recall_fscore_support(golds, preds, average=avg_type, zero_division=0)
    return p, r, f1


def fleiss_kappa_with_agreement(predictions, num_targets):
    num_prompts = len(predictions)
    num_examples = len(predictions[0])
    M = []
    for i in range(num_examples):
        counts = [0] * num_targets
        for p in range(num_prompts):
            counts[predictions[p][i]] += 1
        M.append(counts)
    M = np.array(M)
    N, k = M.shape
    n_annotators = float(np.sum(M[0, :]))
    tot_annotations = N * n_annotators
    category_sum = np.sum(M, axis=0)
    p = category_sum / tot_annotations
    PbarE = np.sum(p * p)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    kappa = (Pbar - PbarE) / (1 - PbarE) if (1 - PbarE) != 0 else 0.0
    return float(kappa), float(Pbar)


def compute_posix(logit_matrix):
    """Compute POSIX for one example given logits per prompt."""
    N = len(logit_matrix)
    if N <= 1:
        return 0.0
    log_probs = [np.array(logits) - scipy.special.logsumexp(logits) for logits in logit_matrix]
    best = [np.argmax(lp) for lp in log_probs]
    total = 0.0
    for j in range(N):
        ref = log_probs[j][best[j]]
        for i in range(N):
            if i == j:
                continue
            total += abs(log_probs[i][best[j]] - ref)
    return total / (N * (N - 1))


def write_results_to_file(
    fout_name,
    suffix,
    all_prompt_metrics,
    all_prompt_predictions,
    avg_ensemble_metrics,
    avg_ensemble_preds,
    vote_ensemble_metrics,
    vote_ensemble_preds,
    golds,
    avg_entropy=None,
    fleiss_kappa=None,
    raw_agreement=None,
    posix=None,
    prefix="accuracy",
    extra_metrics=None,
):
    results = {}
    per_prompt = {}
    metric_keys = list(all_prompt_metrics[0].keys())
    for k in metric_keys:
        metrics_list = [m[k] * 100 for m in all_prompt_metrics]
        per_prompt[k] = metrics_list
        results[f"max_{k}"] = np.max(metrics_list)
        results[f"median_{k}"] = np.median(metrics_list)
        results[f"mean_{k}"] = np.mean(metrics_list)
        results[f"min_{k}"] = np.min(metrics_list)
        results[f"std_{k}"] = np.std(metrics_list)
        results[f"avg_ensemble_{k}"] = avg_ensemble_metrics[k] * 100
        results[f"vote_ensemble_{k}"] = vote_ensemble_metrics[k] * 100

    if "accuracy" in per_prompt:
        results["accuracy_spread"] = max(per_prompt["accuracy"]) - min(per_prompt["accuracy"])
    if "precision" in per_prompt:
        results["precision_spread"] = max(per_prompt["precision"]) - min(per_prompt["precision"])
    if "recall" in per_prompt:
        results["recall_spread"] = max(per_prompt["recall"]) - min(per_prompt["recall"])
    if "f1" in per_prompt:
        results["f1_spread"] = max(per_prompt["f1"]) - min(per_prompt["f1"])

    if fleiss_kappa is not None:
        results["fleiss_kappa"] = fleiss_kappa
    if raw_agreement is not None:
        results["raw_agreement"] = raw_agreement
    if posix is not None:
        results["posix"] = posix

    if extra_metrics is not None:
        results.update(extra_metrics)

    if fout_name.startswith("results"):
        nfout = f"{fout_name}.{prefix}_{suffix}"
    else:
        nfout = os.path.join(fout_name, f"{prefix}_{suffix}")

    acc_values = per_prompt.get("accuracy", [])
    median_prompt = all_prompt_predictions[index_median(acc_values)] if acc_values else all_prompt_predictions[0]
    max_prompt = all_prompt_predictions[np.argsort(acc_values)[-1]] if acc_values else all_prompt_predictions[0]

    with open(nfout, "w") as fout:
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                fout.write(f"{k}=" + " ".join([str(kk) for kk in v]) + "\n")
            else:
                fout.write(f"{k}={v}\n")
        if avg_entropy is not None:
            if acc_values:
                fout.write("acc: " + " ".join([str(v) for v in acc_values]) + "\n")
            fout.write("ent: " + " ".join([str(v) for v in avg_entropy]) + "\n")
            if "precision" in per_prompt:
                fout.write("precision: " + " ".join([str(v) for v in per_prompt["precision"]]) + "\n")
            if "recall" in per_prompt:
                fout.write("recall: " + " ".join([str(v) for v in per_prompt["recall"]]) + "\n")
            if "f1" in per_prompt:
                fout.write("F1: " + " ".join([str(v) for v in per_prompt["f1"]]) + "\n")

        for ii in range(len(all_prompt_predictions[0])):
            s = (
                ",".join(
                    [
                        f"gold={golds[ii]}",
                        f"median={median_prompt[ii]}",
                        f"max={max_prompt[ii]}",
                        f"avg_esemb={avg_ensemble_preds[ii]}",
                        f"vote_esemb={vote_ensemble_preds[ii]}",
                    ]
                )
                + ","
            )
            s += " ".join([str(all_prompt_predictions[jj][ii]) for jj in range(len(all_prompt_predictions))])
            fout.write(s + "\n")
    return results


def write_unsupervised_results_to_file(fout_name, results, all_prompt_predictions, golds=None):
    with open(fout_name, "w") as fout:
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                fout.write("{}={}".format(key, " ".join([str(kk) for kk in value])) + "\n")
            else:
                fout.write("{}={}".format(key, value) + "\n")

        # output predictions of prompts for each example
        for ii in range(len(all_prompt_predictions[0])):
            s = f"gold={golds[ii]}, " if golds is not None else ""
            s += " ".join([str(all_prompt_predictions[jj][ii]) for jj in range(len(all_prompt_predictions))])
            fout.write(s + "\n")


def compute_metrics(
    logprobs,
    num_examples,
    num_targets,
    num_prompts,
    golds=None,
    metrics=None,
    fout_name=None,
    suffix=None,
    pseudo_dist="smooth",
    return_all_prompt_preds=False,
    random_selection_ensemble=0.0,
    self_train=False,
    write_logits=False,
    **kwargs,
):
    predictions = [[] for _ in range(num_prompts)]
    entropies = [[] for _ in range(num_prompts)]
    avg_ensemble_predictions = []
    vote_ensemble_predictions = []
    all_avg_probs = []  # only used when num of examples=1
    idx = 0
    logits = [[] for _ in range(num_prompts)]
    for eidx in range(num_examples):
        avg_probs = np.zeros(num_targets)
        for pidx in range(num_prompts):
            max_ll, pred_label = -np.inf, -1
            # actually, the number of labels of each prompt should be the same
            normalized_probs = np.zeros(num_targets)
            logit = []
            for ii in range(num_targets):
                if logprobs[idx] > max_ll:
                    max_ll, pred_label = logprobs[idx], ii
                normalized_probs[ii] = math.exp(logprobs[idx])
                logit.append(logprobs[idx])
                idx += 1
            logits[pidx].append(logit)
            normalized_probs = normalized_probs / normalized_probs.sum()
            entropies[pidx].append(-(normalized_probs * np.log(normalized_probs)).sum())
            avg_probs += normalized_probs
            all_avg_probs.append(normalized_probs)
            predictions[pidx].append(pred_label)

        # import pdb; pdb.set_trace()
        if 0.0 < random_selection_ensemble < 1.0 and num_examples == 1:
            selected_prompts = np.random.permutation(num_prompts)[: int(num_prompts * random_selection_ensemble)]
            avg_probs = sum([all_avg_probs[jj] for jj in selected_prompts]) / len(selected_prompts)
            all_preds = [predictions[jj][-1] for jj in selected_prompts]
        else:
            avg_probs = avg_probs / num_prompts
            all_preds = [ppt[-1] for ppt in predictions]

        avg_label = np.argmax(avg_probs)
        counts = [all_preds.count(ii) for ii in range(num_targets)]
        vote_label = np.argmax(counts)
        total = float(sum(counts))
        vote_probs = [c / total for c in counts]

        if return_all_prompt_preds and num_examples == 1:
            if not self_train:
                random_indices = np.random.permutation(len(all_avg_probs))
            else:
                random_indices = np.arange(len(all_avg_probs))
            avg_probs = [all_avg_probs[ii] for ii in random_indices]
            vote_probs = [[1 if c == predictions[ii][-1] else 0 for c in range(num_targets)] for ii in random_indices]
            return [ppt[0] for ppt in predictions], avg_probs, vote_probs, random_indices

        if pseudo_dist == "argmax":
            avg_probs = [1 if c == avg_label else 0 for c in range(num_targets)]
            vote_probs = [1 if c == vote_label else 0 for c in range(num_targets)]

        if num_examples == 1:
            avg_ensemble_predictions.append(avg_probs)
            vote_ensemble_predictions.append(vote_probs)
        else:
            avg_ensemble_predictions.append(avg_label)
            vote_ensemble_predictions.append(vote_label)

    if num_examples == 1:
        return [ppt[0] for ppt in predictions], avg_ensemble_predictions[0], vote_ensemble_predictions[0]

    prompt_metrics = []
    avg_type = "binary" if num_targets == 2 else "macro"
    for ppred in predictions:
        m = metrics.compute(predictions=ppred, references=golds)
        p, r, f1 = compute_precision_recall_f1(ppred, golds, avg_type=avg_type)
        m["precision"] = p
        m["recall"] = r
        m["f1"] = f1
        prompt_metrics.append(m)
    avg_ensemble_metrics = metrics.compute(predictions=avg_ensemble_predictions, references=golds)
    p, r, f1 = compute_precision_recall_f1(avg_ensemble_predictions, golds, avg_type=avg_type)
    avg_ensemble_metrics["precision"] = p
    avg_ensemble_metrics["recall"] = r
    avg_ensemble_metrics["f1"] = f1
    avg_entropy = [np.mean(ents) for ents in entropies]
    vote_ensemble_metrics = metrics.compute(predictions=vote_ensemble_predictions, references=golds)
    p, r, f1 = compute_precision_recall_f1(vote_ensemble_predictions, golds, avg_type=avg_type)
    vote_ensemble_metrics["precision"] = p
    vote_ensemble_metrics["recall"] = r
    vote_ensemble_metrics["f1"] = f1

    kappa, agreement = fleiss_kappa_with_agreement(predictions, num_targets)

    # POSIX computation
    posix_vals = []
    for ex_idx in range(num_examples):
        logits_example = [logits[p][ex_idx] for p in range(num_prompts)]
        posix_vals.append(compute_posix(logits_example))
    posix_value = float(np.mean(posix_vals))

    if write_logits and fout_name is not None:
        if fout_name.startswith("results"):
            nfout = fout_name + ".logits.p"
        else:
            nfout = os.path.join(fout_name, f"logits.{suffix}.p")
        for pidx in range(num_prompts):
            with open(f"{nfout}{pidx}", "w") as fout:
                for logit in logits[pidx]:
                    fout.write(" ".join([str(l) for l in logit]) + "\n")

    results = write_results_to_file(
        fout_name,
        suffix,
        prompt_metrics,
        predictions,
        avg_ensemble_metrics,
        avg_ensemble_predictions,
        vote_ensemble_metrics,
        vote_ensemble_predictions,
        golds,
        avg_entropy=avg_entropy,
        fleiss_kappa=kappa,
        raw_agreement=agreement,
        posix=posix_value,
        prefix="accuracy",
    )
    print(results)
    return results, None


def print_dict(dd):
    for key, value in dd.items():
        if isinstance(value, np.ndarray):
            print("{}: {}".format(key, " ".join([str(kk) for kk in value])))
        else:
            print("{}: {}".format(key, value))


def compute_entropy(predictions, num_targets):
    all_entropy = []
    for prompt_p in predictions:
        # import pdb; pdb.set_trace()
        prob = np.bincount(prompt_p, minlength=num_targets)
        prob = prob / len(prompt_p)
        all_entropy.append(scipy.stats.entropy(prob))

    return np.array(all_entropy)


def compute_unsupervised_metrics(
    logprobs,
    num_examples,
    num_targets,
    num_prompts,
    golds=None,
    metrics=None,
    fout_name=None,
    suffix=None,
    return_all_prompt_preds=False,
    random_selection_ensemble=0.0,
    initial_predictions=None,
    **kwargs,
):

    # import pdb; pdb.set_trace()
    predictions = [[] for _ in range(num_prompts)]
    entropies = [[] for _ in range(num_prompts)]
    all_avg_probs = [[] for _ in range(num_prompts)]
    avg_ensemble_predictions = []
    vote_ensemble_predictions = []
    idx = 0
    for eidx in range(num_examples):
        avg_probs = np.zeros(num_targets)
        for pidx in range(num_prompts):
            max_ll, pred_label = -np.inf, -1
            normalized_probs = np.zeros(num_targets)
            for ii in range(num_targets):
                if logprobs[idx] > max_ll:
                    max_ll, pred_label = logprobs[idx], ii
                normalized_probs[ii] = math.exp(logprobs[idx])
                idx += 1
            normalized_probs = normalized_probs / normalized_probs.sum()
            entropies[pidx].append(-(normalized_probs * np.log(normalized_probs)).sum())
            all_avg_probs[pidx].append(normalized_probs)
            predictions[pidx].append(pred_label)
            avg_probs += normalized_probs

        avg_probs = avg_probs / num_prompts
        avg_label = np.argmax(avg_probs)
        all_preds = [predictions[j][-1] for j in range(num_prompts)]
        counts = [all_preds.count(ii) for ii in range(num_targets)]
        vote_label = np.argmax(counts)
        avg_ensemble_predictions.append(avg_label)
        vote_ensemble_predictions.append(vote_label)

    entropy = compute_entropy(predictions, num_targets)
    all_continuous_entropy = []
    for probs in all_avg_probs:
        all_continuous_entropy.append(scipy.stats.entropy(np.mean(probs, 0)))
    extra_metrics = {
        "all entropy": entropy,
        "avg entropy": entropy.mean(),
        "avg cont entropy": np.mean(all_continuous_entropy),
    }

    prompt_metrics = []
    avg_type = "binary" if num_targets == 2 else "macro"
    for ppred in predictions:
        m = metrics.compute(predictions=ppred, references=golds)
        p, r, f1 = compute_precision_recall_f1(ppred, golds, avg_type=avg_type)
        m["precision"] = p
        m["recall"] = r
        m["f1"] = f1
        prompt_metrics.append(m)
    avg_ensemble_metrics = metrics.compute(predictions=avg_ensemble_predictions, references=golds)
    p, r, f1 = compute_precision_recall_f1(avg_ensemble_predictions, golds, avg_type=avg_type)
    avg_ensemble_metrics["precision"] = p
    avg_ensemble_metrics["recall"] = r
    avg_ensemble_metrics["f1"] = f1
    vote_ensemble_metrics = metrics.compute(predictions=vote_ensemble_predictions, references=golds)
    p, r, f1 = compute_precision_recall_f1(vote_ensemble_predictions, golds, avg_type=avg_type)
    vote_ensemble_metrics["precision"] = p
    vote_ensemble_metrics["recall"] = r
    vote_ensemble_metrics["f1"] = f1

    kappa, agreement = fleiss_kappa_with_agreement(predictions, num_targets)

    # POSIX computation
    posix_vals = []
    for ex_idx in range(num_examples):
        logits_example = [all_avg_probs[p][ex_idx] for p in range(num_prompts)]
        logits_example = [np.log(lp) for lp in logits_example]
        posix_vals.append(compute_posix(logits_example))
    posix_value = float(np.mean(posix_vals))

    if initial_predictions is not None:
        initial_entropy = compute_entropy(initial_predictions, num_targets)
        extra_metrics["delta all entropy"] = entropy - initial_entropy
        extra_metrics["delta avg entropy"] = extra_metrics["delta all entropy"].mean()

    results = write_results_to_file(
        fout_name,
        suffix,
        prompt_metrics,
        predictions,
        avg_ensemble_metrics,
        avg_ensemble_predictions,
        vote_ensemble_metrics,
        vote_ensemble_predictions,
        golds,
        avg_entropy=entropy,
        fleiss_kappa=kappa,
        raw_agreement=agreement,
        posix=posix_value,
        prefix="unsupervised_dev",
        extra_metrics=extra_metrics,
    )
    print_dict(results)
    return results, (predictions if initial_predictions is None else None)


def summarize_metrics(predictions, avg_ensemble_predictions, vote_ensemble_predictions, golds, metrics, fout_name=None):
    prompt_metrics = []
    for ppred in predictions:
        prompt_metrics.append(metrics.compute(predictions=ppred, references=golds))
    avg_ensemble_metrics = metrics.compute(predictions=avg_ensemble_predictions, references=golds)
    vote_ensemble_metrics = metrics.compute(predictions=vote_ensemble_predictions, references=golds)

    results = {}
    for k, v in prompt_metrics[0].items():
        all_metrics = [pptm[k] * 100 for pptm in prompt_metrics]
        results["max_" + k] = np.max(all_metrics)
        results["median_" + k] = np.median(all_metrics)
        results["mean_" + k] = np.mean(all_metrics)
        results["min_" + k] = np.min(all_metrics)
        results["std_" + k] = np.std(all_metrics)

    for k, v in avg_ensemble_metrics.items():
        results["avg_ensemble_avg" + k] = v * 100

    for k, v in vote_ensemble_metrics.items():
        results["vote_ensemble_avg" + k] = v * 100

    if fout_name is not None:
        _ = write_results_to_file(
            fout_name,
            "final",
            prompt_metrics,
            predictions,
            avg_ensemble_metrics,
            avg_ensemble_predictions,
            vote_ensemble_metrics,
            vote_ensemble_predictions,
            golds,
        )
    return results


def compute_loss_scale(pred_labels, prompt_groups, group_id, answer_id):
    """
    compute how likely (unormalized) the prompts outside the current group supports the
    current answer
    """

    total = 0
    support = 0.0
    # for prompt_id, pred in enumerate(pred_labels):
    #     if prompt_id not in prompt_groups[group_id]:
    #         total += 1
    #         if pred == answer_id:
    #             support += 1.
    for prompt_id, pred in enumerate(pred_labels):
        total += 1
        if pred == answer_id:
            support += 1.0

    # only one group
    if total == 0:
        return 0

    return support


def compute_unsupervised_dev_best_results(dir_path, min_train_steps, metrics=["avg entropy", "avg cont entropy"]):
    unsup_dev_prefix = "unsupervised_dev_"
    eval_prefix = "accuracy_"
    all_checkpoints = []
    for fname in os.listdir(dir_path):
        if eval_prefix in fname:
            all_checkpoints.append(int(fname.split("_")[-1]))
    all_checkpoints.sort()

    best_ens_acc = 0.0
    best_ckpt = 0
    best_dev_results = {}
    all_results = {}
    for ckpt in all_checkpoints:
        acc_path = os.path.join(dir_path, eval_prefix + str(ckpt))
        with open(acc_path) as fin:
            result_dict = {}
            for line in fin:
                if line.startswith("acc:") or line.startswith("gold"):
                    break
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    try:
                        result_dict[k] = float(v)
                    except ValueError:
                        result_dict[k] = v
            all_results[ckpt] = result_dict
            for key in ["avg_ensemble_accuracy", "vote_ensemble_accuracy"]:
                if key in result_dict and result_dict[key] > best_ens_acc:
                    best_ckpt = ckpt
                    best_ens_acc = result_dict[key]

        if ckpt <= min_train_steps:
            continue
        if not os.path.exists(os.path.join(dir_path, unsup_dev_prefix + str(ckpt))):
            continue

        with open(os.path.join(dir_path, unsup_dev_prefix + str(ckpt))) as fin:
            for line in fin:
                if line.startswith("gold"):
                    break
                for metric in metrics:
                    if line.startswith(metric):
                        value = float(line.strip().split("=")[-1])
                        if metric in best_dev_results:
                            # larger metric is better: entropy
                            if value > best_dev_results[metric][-1]:
                                best_dev_results[metric] = (ckpt, value)
                        else:
                            best_dev_results[metric] = (ckpt, value)
    print("Best checkpoint at step {}: ".format(best_ckpt))
    print(all_results[best_ckpt])
    for k, v in best_dev_results.items():
        print("Best checkpoint selected by {} at step {} over {} files:".format(k, v[0], unsup_dev_prefix))
        print(all_results[v[0]])
