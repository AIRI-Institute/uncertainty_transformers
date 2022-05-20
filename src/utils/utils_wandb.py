import logging
import os

import wandb

log = logging.getLogger(__name__)


def _wandb_log(_dict):
    if wandb.run is not None:
        wandb.log(_dict)
    else:
        log.info(repr(_dict))


wandb.log_opt = _wandb_log


def init_wandb(directory, config):
    if "NO_WANDB" in os.environ and os.environ["NO_WANDB"] == "true":
        ## working without wandb :c
        log.info("== Working without wandb")
        return None

    # setting up env variables
    # os.environ["WANDB_ENTITY"] = "artem105"
    os.environ["WANDB_PROJECT"] = "uncertainty-estimation"
    os.environ["WANDB_DISABLED"] = "true"

    # generating group name and run name
    directory_contents = directory.split("/")
    run_name = directory_contents[-1]  # ${now:%H-%M-%S}-${repeat}
    date = directory_contents[-2]  # ${now:%Y-%m-%d}
    strat_name = directory_contents[-3]  # ${al.strat_name}
    model_name = directory_contents[
        -4
    ]  # ${model.model_type}_${model.classifier} for BERT
    task = directory_contents[-5]  # ${data.task}

    group_name = f"{task}|{model_name}|{strat_name}|{date}"
    run_name = f"{run_name}"

    return wandb.init(
        group=group_name,
        name=run_name,
        config=config,
        #mode="offline",
        job_type="train",
        force=True,
        tags=[strat_name, model_name, task],
        mode="offline",
    )
