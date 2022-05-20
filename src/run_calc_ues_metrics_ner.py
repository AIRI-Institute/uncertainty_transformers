import yaml
import os
from yaml import Loader as Loader
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score

from analyze_results import (
    extract_result,
    aggregate_runs,
    from_model_outputs_calc_rcc_auc_ner,
)
from analyze_results import (
    format_results2,
    improvement_over_baseline,
    from_model_outputs_calc_pr_auc_ner,
    from_model_outputs_calc_rpp_ner,
    load_and_preprocess_ner,
    calc_metric_ner,
)

import yaml
from utils.utils_wandb import init_wandb, wandb
from ue4nlp.ue_scores import *

import logging

log = logging.getLogger()

import hydra


# block with fast metrics calculation for ner - mostly it's optimized version of similar functions in analyze_results.py
def extract_result_ner_multiple(time_dir, methods, metrics, level="token"):
    data = load_and_preprocess_ner(time_dir)
    results_dict = {}
    # check if data has mahalanobis distance or nuq ue scores in keys
    if (
        "mahalanobis_distance" in data[0].keys()
        or "stds" in data[0].keys()
        or "uncertainty_score" in data[0].keys()
    ):
        # then override methods
        if level == "token":
            maha_dist = lambda x: np.squeeze(x[:, 0], axis=-1)
        elif level == "seq":
            maha_dist = lambda x: np.squeeze(np.expand_dims(x[:, 0], axis=1), axis=-1)
        # by default, set methods for mahalanobis_distance and override for other cases
        methods = {"mahalanobis_distance": maha_dist}
        if "uncertainty_score" in data[0].keys():
            methods = {"mixup": maha_dist}
        if "stds" in data[0].keys():
            methods = {"stds": maha_dist}
        if "sampled_mahalanobis_distance" in data[0].keys():
            if level == "token":
                sm_maha_dist = lambda x: np.squeeze(x[:, 1:], axis=-1).max(1)
            elif level == "seq":
                sm_maha_dist = lambda x: np.squeeze(x[:, 1:], axis=-1).max(1)
            methods["sampled_mahalanobis_distance"] = sm_maha_dist
    elif "nuq_probabilities" in data[0].keys():
        # then override methods
        nuq_aleatoric = lambda x: np.squeeze(x[:, 0], axis=-1)
        nuq_epistemic = lambda x: np.squeeze(x[:, 1], axis=-1)
        nuq_total = lambda x: np.squeeze(x[:, 2], axis=-1)
        methods = {
            "nuq_aleatoric": nuq_aleatoric,
            "nuq_epistemic": nuq_epistemic,
            "nuq_total": nuq_total,
        }
    for metric in metrics:
        metric_func = choose_metric(metric)
        res = calc_metric_ner(*data, methods, metric_func, level)
        results_dict[metric] = res
    return results_dict


def aggregate_runs_ner_multiple(
    data_path, methods, metrics, task_type="ner-token", oos=False
):
    results = {key: [] for key in metrics}
    model_results = {key: [] for key in metrics}
    level = None
    for model_seed in os.listdir(data_path):
        try:
            model_seed_int = int(model_seed)
        except:
            if model_seed == "results":
                pass
            else:
                continue

        model_path = Path(data_path) / model_seed

        model_results = {key: [] for key in metrics}

        for run_seed in os.listdir(model_path):
            run_dir = model_path / run_seed

            try:
                level = task_type.split("-")[1]
                # now we append dict with all metrics for one log file
                buf_res = extract_result_ner_multiple(
                    run_dir, methods=methods, metrics=metrics, level=level
                )
                for key in buf_res.keys():
                    model_results[key].append(buf_res[key])
            except FileNotFoundError:
                pass

        for key, value in model_results.items():
            log.info(f"N runs: {len(value)}")
            model_avg_res = pd.DataFrame.from_dict(
                value, orient="columns"
            )  # .mean(axis=0)
            results[key].append(model_avg_res)

    df_results = {}
    for key in results.keys():
        df_results[key] = pd.concat(results[key], axis=0).reset_index(drop=True)
    return df_results


def get_model_type(model_path):
    model_type = model_path.split("/")[-2]
    return model_type


def collect_configs(dir_name):
    cfg_str = ""
    for model_seed in os.listdir(dir_name):
        model_path = Path(dir_name) / model_seed
        for run_seed in os.listdir(model_path):
            run_path = model_path / run_seed

            with open(run_path / ".hydra" / "config.yaml") as f:
                cfg = yaml.load(f, Loader=Loader)

            cfg_str_new = cfg["ue"]["ue_type"]
            if cfg["ue"]["ue_type"] == "mc-dpp":
                cfg_str_new += "_" + "_".join(
                    str(e)
                    for e in (
                        cfg["ue"]["dropout"]["dry_run_dataset"],
                        cfg["ue"]["dropout"]["mask_name"],
                        cfg["ue"]["dropout"]["max_frac"],
                        get_model_type(cfg["model"]["model_name_or_path"]),
                    )
                )

            if cfg_str:
                if cfg_str != cfg_str_new:
                    print("Error, different cfg_strs:", cfg_str, cfg_str_new)

            cfg_str = cfg_str_new

    return cfg_str


def choose_metric(metric_type):
    if metric_type in ["rejection-curve-auc", "roc-auc"]:
        return metric_type

    elif metric_type == "rcc-auc":
        return from_model_outputs_calc_rcc_auc_ner

    elif metric_type == "pr-auc":
        return from_model_outputs_calc_pr_auc_ner

    elif metric_type == "rpp":
        return from_model_outputs_calc_rpp_ner

    else:
        raise ValueError("Wrong metric type!")


@hydra.main(
    config_path=os.path.dirname(os.environ["HYDRA_CONFIG_PATH"]),
    config_name=os.path.basename(os.environ["HYDRA_CONFIG_PATH"]),
)
def main(config):
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    wandb_run = init_wandb(auto_generated_dir, config)

    default_methods = {
        "bald": bald,
        "var_ratio": var_ratio,
        "sampled_max_prob": sampled_max_prob,
        "variance": probability_variance,
        "sampled_entropy": mean_entropy,
    }

    agg_res_all = aggregate_runs_ner_multiple(
        config.runs_dir,
        methods=default_methods,
        metrics=config.metric_types,
        task_type="ner-token",
    )
    seq_agg_res_all = aggregate_runs_ner_multiple(
        config.runs_dir,
        methods=default_methods,
        metrics=config.metric_types,
        task_type="ner-seq",
    )
    for metric_type in agg_res_all.keys():
        log.info(f"Metric: {metric_type}")
        agg_res = agg_res_all[metric_type]
        seq_agg_res = seq_agg_res_all[metric_type]

        metric_path = Path(auto_generated_dir) / f"metrics_token_{metric_type}.json"
        with open(metric_path, "w") as f:
            f.write(agg_res.to_json())
        seq_metric_path = Path(auto_generated_dir) / f"metrics_seq_{metric_type}.json"
        with open(seq_metric_path, "w") as f:
            f.write(seq_agg_res.to_json())

        if wandb.run is not None:
            wandb.save(str(metric_path))
            wandb.save(str(seq_metric_path))

        if config.extract_config:
            log.info("Exp. config: " + collect_configs(config.runs_dir))

        if agg_res.empty:
            log.info("Broken\n")
            continue

        if metric_type == "rcc-auc":
            final_score = improvement_over_baseline(
                agg_res, baseline_col="max_prob", subtract=False, percents=False
            )
            final_score_seq = improvement_over_baseline(
                seq_agg_res, baseline_col="max_prob", subtract=False, percents=False
            )
        elif metric_type == "rpp":
            final_score = improvement_over_baseline(
                agg_res, baseline_col="max_prob", subtract=False, percents=True
            )
            final_score_seq = improvement_over_baseline(
                seq_agg_res, baseline_col="max_prob", subtract=False, percents=True
            )
        else:
            final_score = improvement_over_baseline(
                agg_res, baseline_col="max_prob", subtract=True, percents=True
            )
            final_score_seq = improvement_over_baseline(
                seq_agg_res, baseline_col="max_prob", subtract=True, percents=True
            )

        log.info("\n" + str(final_score))
        log.info("\n")
        log.info("\n" + str(final_score_seq))
        log.info("\n")


if __name__ == "__main__":
    main()
