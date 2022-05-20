import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import hydra
import yaml
from pathlib import Path
import json

import utils.utils_tasks as utils

import logging

log = logging.getLogger(__name__)


def accumulate_results(results_dir, final_dir):
    final_result = {
        "true_labels": [],
        "probabilities": [],
        "answers": [],
        "sampled_probabilities": [],
        "sampled_answers": [],
    }

    inf_time = 0
    eval_time = 0
    for seed in os.listdir(results_dir):
        results_file_path = Path(results_dir) / seed / "dev_inference.json"
        with open(results_file_path) as f:
            result = json.load(f)

        final_result["sampled_probabilities"].append(result["probabilities"])
        final_result["sampled_answers"].append(result["answers"])
        if 'ue_time' in result.keys():
            inf_time += result['ue_time']
        if 'eval_time' in result.keys():
            eval_time += result['eval_time']

    final_result["ue_time"] = inf_time
    final_result["eval_time"] = eval_time
    final_result["true_labels"] = result["true_labels"]
    final_result["answers"] = result["answers"]
    final_result["probabilities"] = result["probabilities"]

    with open(Path(final_dir) / "dev_inference.json", "w") as f:
        json.dump(final_result, f)


def run_glue_for_ensemble_series(config, work_dir):
    name = os.path.splitext(os.path.basename(config.config_path))[0]

    for ens_num in os.listdir(config.ensemble_series_dir):
        ensemble_path = Path(config.ensemble_series_dir) / ens_num / "models" / name

        output_dir = str(Path(work_dir) / "results" / str(ens_num))
        task = {
            "cuda_devices": str(config.cuda_devices),
            "config_path": config.config_path,
            "script": config.script,
            "args": config.args + " do_eval=True do_ue_estimate=False",
            "seeds": "[0]",
            "output_dir": output_dir,
            "hydra": {"run": {"dir": output_dir}},
            "model_series_dir": str(ensemble_path),
        }

        config_path = Path(work_dir) / f"config_{ens_num}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(task, f)

        command = f"HYDRA_CONFIG_PATH={config_path} python run_tasks_for_model_series.py"
        log.info(f"Command: {command}")
        ret = os.system(command)
        log.info(f"Return code: {ret}")

        final_dir = Path(work_dir) / "final_results" / "0" / str(ens_num)
        os.makedirs(final_dir, exist_ok=True)
        accumulate_results(
            results_dir=Path(output_dir) / "results", final_dir=final_dir
        )

    log.info("Done with all ensembles.")


if __name__ == "__main__":

    @hydra.main(
        config_path=os.path.dirname(os.environ["HYDRA_CONFIG_PATH"]),
        config_name=os.path.basename(os.environ["HYDRA_CONFIG_PATH"]),
    )
    def main(config):
        auto_generated_dir = os.getcwd()
        log.info(f"Work dir: {auto_generated_dir}")
        os.chdir(hydra.utils.get_original_cwd())

        run_glue_for_ensemble_series(config, auto_generated_dir)

    main()
