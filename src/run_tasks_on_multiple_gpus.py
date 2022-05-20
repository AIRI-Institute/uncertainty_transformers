import hydra
import os

import utils.utils_tasks as utils

import logging
from utils.utils_tasks import get_config

log = logging.getLogger(__name__)


def run_task(task):
    """This version probably works with run_average_results.py script"""
    log.info(f"Task name: {task.name}")

    task_args = task.args if "args" in task else ""
    task_args = task_args.replace("$\\", "\\$")

    # output_dir = task.output_dir
    # run_dir = "'${output_dir}/${data.task_name}/${ue.ue_type}/${training.seed}/${seed}'"

    command = f"CUDA_VISIBLE_DEVICES={utils.WORKER_CUDA_DEVICE} HYDRA_CONFIG_PATH={task.config_path} {task.environ} python {task.command} repeat={task.repeat} {task_args}"  # hydra.run.dir={run_dir} output_dir={output_dir}"

    log.info(f"Command: {command}")
    ret = os.system(command)
    ret = str(ret)
    log.info(f'Task "{task.name}" finished with return code: {ret}.')
    return ret


def run_task_old(task):
    """This version works with pipeline approach"""
    log.info(f"Task name: {task.name}")
    task_args = task.args if "args" in task else ""
    command = f"CUDA_VISIBLE_DEVICES={utils.WORKER_CUDA_DEVICE} HYDRA_CONFIG_PATH={task.config_path} {task.environ} python {task.command} {task_args} ++repeat={task.repeat}"
    log.info(f"Command: {command}")
    ret = os.system(command)
    ret = str(ret)
    log.info(f'Task "{task.name}" finished with return code: {ret}.')
    return ret


@hydra.main(
    config_path=get_config()[0],
    config_name=get_config()[1],
)
def main(configs):
    os.chdir(hydra.utils.get_original_cwd())
    try:
        utils.run_tasks(configs, run_task)
    except:
        utils.run_tasks(configs, run_task_old)


if __name__ == "__main__":
    main()
