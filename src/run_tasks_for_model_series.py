import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import hydra
import yaml
from pathlib import Path
from collections.abc import Iterable

import utils.utils_tasks as utils

import logging

log = logging.getLogger(__name__)


def run_tasks(config_path, cuda_devices):
    cuda_devices = (
        cuda_devices if isinstance(cuda_devices, Iterable) else [cuda_devices]
    )
    cuda_devices_str = "[" + ",".join(str(e) for e in cuda_devices) + "]"
    command = f"HYDRA_CONFIG_PATH={config_path} python run_tasks_on_multiple_gpus.py cuda_devices={cuda_devices_str}"
    log.info(f"Command: {command}")
    ret = os.system(command)
    if ret != 0:
        raise RuntimeError(ret)

    return ret

def run_glue_for_model_series_fast(config, work_dir):
    tasks = []
    for model_dir_name in os.listdir(config.model_series_dir):
        model_path = Path(config.model_series_dir) / model_dir_name
        model_args_str = config.args
        model_args_str += " "
        model_args_str += f"model.model_name_or_path={model_path}"

        if 'ensemble' in str(config.output_dir):
            seed = config.output_dir.split('/')[-1]
        else:
            seed = str(model_dir_name)
        args_str = model_args_str
        args_str += " "
        args_str += f"seed={seed}"
        args_str += " "
        output_dir = str(Path(work_dir) / "results" / model_dir_name)
        args_str += f"hydra.run.dir={output_dir}"
        args_str += " "
        args_str += f"output_dir={output_dir}"
        args_str += " "
        args_str += " do_train=False do_eval=True "

        task = {
            "config_path": config.config_path,
            "environ": "",
            "command": config.script,
            "name": f"model_{model_dir_name}_{seed}",
            "args": args_str,
        }

        tasks.append(task)

    config_path = Path(work_dir) / "config.yaml"
    config_structure = {}
    config_structure["cuda_devices"] = ""
    config_structure["tasks"] = tasks
    config_structure["hydra"] = {"run": {"dir": work_dir}}
    with open(config_path, "w") as f:
        yaml.dump(config_structure, f)

    run_tasks(config_path, config.cuda_devices)


if __name__ == "__main__":

    @hydra.main(
        config_path=os.path.dirname(os.environ["HYDRA_CONFIG_PATH"]),
        config_name=os.path.basename(os.environ["HYDRA_CONFIG_PATH"]),
    )
    def main(config):
        auto_generated_dir = os.getcwd()
        log.info(f"Work dir: {auto_generated_dir}")
        os.chdir(hydra.utils.get_original_cwd())

        run_glue_for_model_series_fast(config, auto_generated_dir)

    main()
