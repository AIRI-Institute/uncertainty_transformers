from transformers import trainer
import logging
from optuna.samplers import TPESampler

log = logging.getLogger()


def hp_space_continuous(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 1e-2, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 15),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4, 8, 16, 32, 64]
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0, 0.01, 0.1]
        ),  # 0, 0.01, 0.1
        "lamb": trial.suggest_float("lamb", 2e-3, 1.0, log=True),
        "margin": trial.suggest_float("margin", 1e-2, 10, log=True),
        "lamb_intra": trial.suggest_categorical("lamb_intra", 2e-3, 1.0, log=True)
        # "margin": trial.suggest_float("margin", 5e-3, 10, log=True)
        # ue_args.lamb - used in both regs, ue_args.margin for metric loss
    }

def hp_space_discrete(trial):
    return {
        "learning_rate": trial.suggest_categorical(
            "learning_rate",
            [5e-6, 6e-6, 7e-6, 9e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4],
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 15),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4, 8, 16, 32, 64]
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0, 0.01, 0.1]
        ), 
        "lamb": trial.suggest_categorical(
            "lamb",
            [1e-3, 2e-3, 3e-3, 5e-3, 6e-3, 8e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 1.0],
        ),
        "margin": trial.suggest_categorical(
            "margin", [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        ),
        "lamb_intra": trial.suggest_categorical(
            "lamb_intra", [1e-3, 2e-3, 3e-3, 5e-3, 6e-3, 8e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 1.0]
        )
        # ue_args.lamb - used in both regs, ue_args.margin for metric loss
    }

def hp_space_discrete_sngp_ood(trial):
    return {
        "learning_rate": trial.suggest_categorical(
            "learning_rate",
            [5e-6, 6e-6, 7e-6, 9e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4],
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 60),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64, 128]
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0, 0.01, 0.1]
        ), 
    }


def hp_space_discrete_msd(trial):
    return {
        "learning_rate": trial.suggest_categorical(
            "learning_rate",
            [5e-6, 6e-6, 7e-6, 9e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4],
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 15),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4, 8, 16, 32, 64]
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0, 0.01, 0.1]
        ),
        "mixup": trial.suggest_categorical("mixup", [True]),
        "self_ensembling": trial.suggest_categorical("self_ensembling", [True]),
        "omega": trial.suggest_categorical(
            "omega",
            [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        ),
        "lam1": trial.suggest_categorical(
            "lam1",
            [1e-3, 2e-3, 3e-3, 5e-3, 6e-3, 8e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 1.0],
        ),
        "lam2": trial.suggest_categorical(
            "lam2",
            [1e-3, 2e-3, 3e-3, 5e-3, 6e-3, 8e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 1.0],
        )
        # ue_args.lamb - used in both regs, ue_args.margin for metric loss
    }


def hp_space_discrete_sto(trial):
    return {
        "learning_rate": trial.suggest_categorical(
            "learning_rate",
            [5e-6, 6e-6, 7e-6, 9e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4],
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 15),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4, 8, 16, 32, 64]
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0, 0.01, 0.1]
        ),
        "hierarchial": trial.suggest_categorical("hierarchial", [False]),
        "num_centroids": trial.suggest_categorical("num_centroids", [16]),
        "tau_1": trial.suggest_categorical(
            "tau_1",
            [0.025, 0.1, 0.2, 1.0, 5.0, 10.0, 40.0],
        ),
        "tau_2": trial.suggest_categorical(
            "tau_2",
            [0.025, 0.1, 0.2, 1.0, 5.0, 10.0, 40.0],
        )
    }


def get_optimal_hyperparameters(
    trainer: "trainer",
    model_init,
    task: str = "cls",  # "cls" or "ner"
    use_continuous_distribution: bool = False,
    metric=None,
    use_sngp=False,
):
    # To avoid overriding
    trainer_hyp_opt = trainer
    trainer_hyp_opt.model_init = model_init

    if task == "cls" or task == "ood":
        if isinstance(metric, str):
            def compute_objective(metrics):
                return metrics["eval_max_prob"]
        else:
            def compute_objective(metrics):
                return metrics["eval_accuracy"]
    elif task == "ner":
        def compute_objective(metrics):
            return metrics["eval_f1"]
    else:
        raise NotImplementedError

    hp_space = hp_space_continuous if use_continuous_distribution else hp_space_discrete
    if not(use_continuous_distribution) and hasattr(trainer.model, "mixup"):
        hp_space = hp_space_discrete_msd
    if not(use_continuous_distribution) and hasattr(trainer.model.electra.encoder.layer[-1].attention.self, "hierarchial"):
        hp_space = hp_space_discrete_sto

    # set a sampler with the same seed as used in the trainer
    if use_sngp and task == "ood":
        hp_space = hp_space_discrete_sngp_ood
        
    seed = trainer.args.seed
    sampler = TPESampler(seed=seed)
    hyp_opt_result = trainer_hyp_opt.hyperparameter_search(
        direction="maximize" if metric!='rcc-auc' else 'minimize',
        hp_space=hp_space,
        backend="optuna",
        compute_objective=compute_objective,
        n_trials=20,
        sampler=sampler,
    )
    log.info(f"Optimal hyperparameters: {hyp_opt_result.hyperparameters}")
    log.info(f"Optimal metric value: {hyp_opt_result.objective}")

    result = hyp_opt_result.hyperparameters
    result.update({"objective": hyp_opt_result.objective})

    return result
