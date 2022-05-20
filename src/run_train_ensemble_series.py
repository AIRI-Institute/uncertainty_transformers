import os
import hydra
import yaml
from pathlib import Path
import copy
import itertools as it
import numpy as np

import utils.utils_tasks as utils

import logging

log = logging.getLogger(__name__)


def train_ensemble_series(config, work_dir):
    task_config_path = Path(config.config_dir) / config.task_configs

    task_cfg_name = os.path.splitext(config.task_configs)[0]
    # TODO: add support for data ratio

    for ens_num, seed in enumerate(config.seed_series):
        output_dir = str(Path(work_dir) / "ensembles" / str(ens_num))
        task = {
            "config_dir": config.config_dir,
            "task_configs": config.task_configs,
            "script": config.script,
            "args": config.args,
            "cuda_devices": list(config.cuda_devices),
            "seeds": list(seed),
            "output_dir": str(Path(work_dir) / "ensembles" / str(ens_num)),
            "hydra": {"run": {"dir": output_dir}},
        }

        config_path = Path(work_dir) / f"config_{ens_num}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(task, f)

        command = f"HYDRA_CONFIG_PATH={config_path} python run_train_models.py"
        log.info(f"Command: {command}")
        ret = os.system(command)
        log.info(f"Return code: {ret}")

    log.info("Finished with all ensembles.")


@hydra.main(
    config_path=os.path.dirname(os.environ["HYDRA_CONFIG_PATH"]),
    config_name=os.path.basename(os.environ["HYDRA_CONFIG_PATH"]),
)
def main(config):
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    train_ensemble_series(config, auto_generated_dir)


if __name__ == "__main__":
    main()
