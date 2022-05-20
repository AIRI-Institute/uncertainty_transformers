import yaml
import os
from yaml import Loader as Loader
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, f1_score
from functools import partial

from analyze_results import (
    extract_result,
    aggregate_runs,
    from_model_outputs_calc_rcc_auc,
    from_model_outptus_calc_rejection_table,
    format_arc_table_results,
)
from analyze_results import (
    format_results2,
    improvement_over_baseline,
    from_model_outputs_calc_pr_auc,
    from_model_outputs_calc_rpp,
    aggregate_runs_rejection_table,
)

from utils.utils_wandb import init_wandb, wandb
from ue4nlp.ue_scores import *

import logging

log = logging.getLogger()

import hydra


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
        return from_model_outputs_calc_rcc_auc

    elif metric_type == "pr-auc":
        return from_model_outputs_calc_pr_auc

    elif metric_type == "rpp":
        return from_model_outputs_calc_rpp

    elif metric_type == "table_accuracy":
        return from_model_outptus_calc_rejection_table

    elif metric_type == "table_f1_macro":
        return partial(from_model_outptus_calc_rejection_table,
                       metric=partial(f1_score, average="macro"))
    elif metric_type == "table_f1_micro":
        return partial(from_model_outptus_calc_rejection_table,
                       metric=partial(f1_score, average="micro"))
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
        "sampled_max_prob": sampled_max_prob,
        "variance": probability_variance,
        "sampled_entropy": mean_entropy,
        "var_ratio": var_ratio,
    }
    # TODO: switch if use maha
    if "maha" in config.runs_dir or "mixup" in config.runs_dir:
        maha_dist = lambda x: np.squeeze(np.squeeze(x, axis=-1), axis=-1)
        maha_dist = lambda x: np.squeeze(x[:, 0], axis=-1)
        default_methods = {"mahalanobis_distance": maha_dist}
        if "mixup" in config.runs_dir:
            default_methods = {"mixup": maha_dist}
        if "maha_mc" in config.runs_dir or "maha_sn_mc" in config.runs_dir:
            # Maha MC case
            sm_maha_dist = lambda x: np.squeeze(x[:, 1:], axis=-1).max(1)
            default_methods["sampled_mahalanobis_distance"] = sm_maha_dist
    if "sngp" in config.runs_dir:
        maha_dist = lambda x: np.squeeze(np.squeeze(x, axis=-1), axis=-1)
        maha_dist = lambda x: np.squeeze(x[:, 0], axis=-1)
        default_methods = {"stds": maha_dist}
    if "ddu" in config.runs_dir:
        maha_dist = lambda x: np.squeeze(x[:, 0], axis=-1)
        default_methods = {"ddu": maha_dist}
    # TODO: same for NUQ
    if "nuq" in config.runs_dir:
        nuq_aleatoric = lambda x: np.squeeze(x[0], axis=-1)
        nuq_epistemic = lambda x: np.squeeze(x[1], axis=-1)
        nuq_total = lambda x: np.squeeze(x[2], axis=-1)
        default_methods = {
            "nuq_aleatoric": nuq_aleatoric,
            "nuq_epistemic": nuq_epistemic,
            "nuq_total": nuq_total,
        }
    if 'sngp' in config.runs_dir:
        std = lambda x: np.squeeze(x[:, 0], axis=-1)
        default_methods = {"std": std}
    for metric_type in config.metric_types:
        log.info(f"Metric: {metric_type}")

        metric = choose_metric(metric_type=metric_type)

        agg_res = aggregate_runs(
            config.runs_dir, methods=default_methods, metric=metric
        )

        agg_res = agg_res.reset_index(drop=True)
        metric_path = Path(auto_generated_dir) / f"metrics_{metric_type}.json"
        with open(metric_path, "w") as f:
            f.write(agg_res.to_json())

        if wandb.run is not None:
            wandb.save(str(metric_path))

        if config.extract_config:
            log.info("Exp. config: " + collect_configs(config.runs_dir))

        if agg_res.empty:
            log.info("Broken\n")
            continue

        if metric_type == "rcc-auc":
            final_score = improvement_over_baseline(
                agg_res, baseline_col="max_prob", subtract=False, percents=False
            )
        elif metric_type == "rpp":
            final_score = improvement_over_baseline(
                agg_res, baseline_col="max_prob", subtract=False, percents=True
            )
        else:
            final_score = improvement_over_baseline(
                agg_res, baseline_col="max_prob", subtract=True, percents=True
            )

        log.info("\n" + str(final_score))
        log.info("\n")
    # Add special part for metrics in table format
    for metric_type in config.table_metrics:
        log.info(f"Table with: {metric_type}")

        metric = choose_metric(metric_type=metric_type)

        agg_res = aggregate_runs_rejection_table(
            config.runs_dir, methods=default_methods, metric=metric
        )

        final_score = format_arc_table_results(agg_res, baseline_col="max_prob",
                                               ndp=3)
        agg_res = agg_res.reset_index(drop=False)
        metric_path = Path(auto_generated_dir) / f"metrics_{metric_type}.json"
        with open(metric_path, "w") as f:
            f.write(agg_res.to_json())

        
        if wandb.run is not None:
            wandb.save(str(metric_path))

        if config.extract_config:
            log.info("Exp. config: " + collect_configs(config.runs_dir))

        if agg_res.empty:
            log.info("Broken\n")
            continue

        log.info("\n" + str(final_score))
        log.info("\n")


if __name__ == "__main__":
    main()
