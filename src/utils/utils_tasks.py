import multiprocessing as mp
from collections.abc import Iterable
import copy
import os

from omegaconf import DictConfig

import logging

log = logging.getLogger(__name__)


CUDA_DEVICES = mp.Queue()
WORKER_CUDA_DEVICE = None


def initialize_worker():
    global CUDA_DEVICES
    global WORKER_CUDA_DEVICE

    WORKER_CUDA_DEVICE = CUDA_DEVICES.get()
    log.info(f"Worker cuda device: {WORKER_CUDA_DEVICE}")


def repeat_tasks(tasks):
    rep_tasks = []
    for task in tasks:
        n_repeats = task.n_repeats if ("n_repeats" in task and task.n_repeats) else 1
        log.info(f"N repeats: {n_repeats}")
        for i in range(n_repeats):
            new_task = DictConfig(task)
            new_task.name = f"{new_task.name}_rep{i}"
            new_task.repeat = f"rep{i}"

            rep_tasks.append(new_task)

    return rep_tasks


def run_tasks(config, f_task):
    global CUDA_QUEUE

    if not isinstance(config.cuda_devices, Iterable):
        cuda_devices = [config.cuda_devices]
    else:
        cuda_devices = config.cuda_devices

    log.info(f"Cuda devices: {cuda_devices}")

    for cuda_device in cuda_devices:
        CUDA_DEVICES.put(cuda_device)

    log.info("All tasks: {}".format(str([t.name for t in config.tasks])))
    if "task_names" in config and config.task_names:
        task_names = config.task_names

        task_index = {t.name: t for t in config.tasks}

        tasks = []
        for task_name in task_names:
            task_name = task_name.split("@")
            if task_name[0] not in task_index:
                raise ValueError(f"Task name: {task_name[0]} is not in config file.")

            task = task_index[task_name[0]]
            if len(task_name) == 2:
                task.n_repeats = int(task_name[1])

            tasks.append(task)
    else:
        tasks = config.tasks

    log.info("Running tasks: {}".format(str([t.name for t in tasks])))

    tasks = repeat_tasks(tasks)

    pool = mp.Pool(len(cuda_devices), initializer=initialize_worker)
    try:
        pool.map(f_task, tasks)

        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()


def init_random_seed_for_repeat(task_config):
    import time
    from .utils_exps import initialize_seeds

    log.info(f"Repeat: {task_config.repeat} ====================================")
    base = (
        (task_config.random_seed + 1)
        if ("fixed_seed" in task_config) and task_config.fixed_seed
        else time.time_ns() // 100000
    )
    seed = base * (int(task_config.repeat[3:]) + 1) % 1000000000
    log.info(f"Random seed: {seed}")
    initialize_seeds(seed)
    task_config.random_seed = seed


def get_config():
    # so there are different ways to set PATH and NAME, use function that works with both
    try:
        path = os.environ["HYDRA_CONFIG_PATH"]
        name = os.environ["HYDRA_CONFIG_NAME"]
    except:
        path = os.path.dirname(os.environ["HYDRA_CONFIG_PATH"])
        name = os.path.basename(os.environ["HYDRA_CONFIG_PATH"])
    return path, name


# def run_tasks(config_path, cuda_devices):
#     command = f"HYDRA_CONFIG_PATH={config_path} python run_tasks_on_multiple_gpus.py cuda_devices={cuda_devices}"
#     log.info(f"Command: {command}")
#     ret = os.system(command)
#     if ret != 0:
#         raise RuntimeError(ret)

#     return ret
