import os
import hydra
import yaml
from pathlib import Path

import utils.utils_tasks as utils
from collections.abc import Iterable

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


def train_models(config, work_dir):
    tasks = []

    task_configs = (
        config.task_configs
        if type(config.task_configs) is list
        else [config.task_configs]
    )

    for task_cfg in task_configs:
        task_config_path = Path(config.config_dir) / task_cfg

        task_cfg_name = os.path.splitext(task_cfg)[0]
        model_args_str = config.args

        for seed in config.seeds:
            args_str = model_args_str
            args_str += " do_train=True do_eval=False do_ue_estimate=False "
            args_str += f"seed={seed}"
            args_str += " "
            
            if task_cfg_name.split('_')[0] == 'sst2':
                task_cfg_name = 'sst2'
                
            output_dir = str(Path(work_dir) / "models" / task_cfg_name / str(seed))
            args_str += f"output_dir={output_dir}"
            args_str += " "
            args_str += f"hydra.run.dir={output_dir}"

            task = {
                "config_path": str(task_config_path),
                "environ": "",
                "command": config.script,
                "name": f"model_{task_cfg_name}_{seed}",
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


@hydra.main(
    config_path=os.path.dirname(os.environ["HYDRA_CONFIG_PATH"]),
    config_name=os.path.basename(os.environ["HYDRA_CONFIG_PATH"]),
)
def main(config):
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    train_models(config, auto_generated_dir)


if __name__ == "__main__":
    main()
