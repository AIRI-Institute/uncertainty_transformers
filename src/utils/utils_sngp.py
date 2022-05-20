from typing import Dict, Union, Any, Optional, Tuple, NamedTuple, List

from tqdm.auto import tqdm, trange
import numpy as np
from packaging import version
import warnings
import collections
import math
import time

import torch
from torch.utils.data.dataset import IterableDataset, Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import nn

from transformers import Trainer
from transformers.file_utils import is_torch_tpu_available

# from transformers.debug_utils import DebugOption
from transformers.trainer_callback import TrainerState
from transformers.utils import logging
from transformers.trainer_utils import (
    EvaluationStrategy,
    TrainOutput,
    EvalLoopOutput,
    EvalPrediction,
    speed_metrics,
    denumpify_detensorize,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.integrations import is_wandb_available


logger = logging.get_logger(__name__)


class EvalLoopOutputSNGP(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    stds: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutputSNGP(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    stds: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class SNGPTrainer(Trainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[float],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits, labels and cov_matrix (each being optional).
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            has_labels = "labels" in inputs.keys()

            if has_labels:
                # The .mean() is to reduce in case of distributed training
                loss = outputs[0].mean().item()
                logits = outputs[1]
                cov_matrix = outputs[2]
            else:
                loss = None
                logits = outputs[0]
                cov_matrix = outputs[1]
            if self.args.past_index >= 0:
                self._past = outputs[
                    self.args.past_index if has_labels else self.args.past_index - 1
                ]

        if prediction_loss_only:
            return (loss, None, None)

        logits = logits.detach()
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = inputs.get("labels").detach()
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels, cov_matrix)
    
    def _move_model_to_device(self, model, device):
        model = model.to(device)
            
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutputSNGP:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(
                dataloader, [self.args.device]
            ).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        stds_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_stds = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            loss, logits, labels, cov_matrix = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(torch.Tensor([loss]).repeat(batch_size))
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )
            if cov_matrix is not None:
                cov_matrix = self._pad_across_processes(cov_matrix)
                cov_matrix = self._nested_gather(cov_matrix)
                curr_stds = torch.zeros_like(logits)
                if len(logits.shape) == 2:
                    for i in range(curr_stds.shape[1]):
                        curr_stds[:, i] = torch.sqrt(torch.diag(cov_matrix))
                else:
                    #NER case
                    for i in range(curr_stds.shape[2]):
                        curr_stds[:, :, i] = torch.sqrt(torch.diag(cov_matrix)).reshape(logits.shape[0], logits.shape[1])
                stds_host = (
                    curr_stds
                    if stds_host is None
                    else nested_concat(stds_host, curr_stds, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                self.args.eval_accumulation_steps is not None
                and (step + 1) % self.args.eval_accumulation_steps == 0
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses
                        if all_losses is None
                        else np.concatenate((all_losses, losses), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = (
                        logits
                        if all_preds is None
                        else nested_concat(all_preds, logits, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels
                        if all_labels is None
                        else nested_concat(all_labels, labels, padding_index=-100)
                    )
                if stds_host is not None:
                    stds = nested_numpify(stds_host)
                    all_stds = (
                        stds
                        if all_stds is None
                        else nested_concat(all_stds, stds, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, stds_host = None, None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (
                losses
                if all_losses is None
                else np.concatenate((all_losses, losses), axis=0)
            )
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = (
                logits
                if all_preds is None
                else nested_concat(all_preds, logits, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels
                if all_labels is None
                else nested_concat(all_labels, labels, padding_index=-100)
            )
        if stds_host is not None:
            stds = nested_numpify(stds_host)
            all_stds = (
                stds
                if all_stds is None
                else nested_concat(all_stds, stds, padding_index=-100)
            )

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_stds is not None:
            all_stds = nested_truncate(all_stds, num_samples)

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
        ):
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels)
            )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutputSNGP(
            predictions=all_preds,
            stds=all_stds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutputSNGP:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        .. note::

            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        output.metrics.update(
            speed_metrics(metric_key_prefix, start_time, output.num_samples)
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutputSNGP(
            predictions=output.predictions,
            stds=output.stds,
            label_ids=output.label_ids,
            metrics=output.metrics,
        )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (:obj:`List[str]`, `optional`)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

                # release memory
                del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        #if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
        #    if self.args.n_gpu > 1:
        #        # nn.DataParallel(model) replicates the model, creating new variables and module
        #        # references registered here no longer work on other gpus, breaking the module
        #        raise ValueError(
        #            "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
        #        )
        #    else:
        #        debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        #if args.gradient_checkpointing:
        #    self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial
            #self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                    
                is_final_epoch = epoch == num_train_epochs - 1
                is_first_minibatch = step == 0
                is_last_minibatch = step == len(epoch_iterator) - 1
                inputs["is_final_epoch"] = is_final_epoch
                inputs["is_first_minibatch"] = is_first_minibatch
                inputs["is_last_minibatch"] = is_last_minibatch
                inputs["epoch"] = epoch

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    True
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            #if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            #    if is_torch_tpu_available():
            #        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            #        xm.master_print(met.metrics_report())
            #    else:
            #        logger.warning(
            #            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
            #            "configured. Check your training configuration if this is unexpected."
            #        )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples)#, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)



#     def train(
#         self,
#         resume_from_checkpoint: Optional[Union[str, bool]] = None,
#         trial: Union["optuna.Trial", Dict[str, Any]] = None,
#         **kwargs,
#     ):
#         """
#         Main training entry point.

#         Args:
#             resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
#                 If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
#                 :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
#                 `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
#                 training will resume from the model/optimizer/scheduler states loaded here.
#             trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
#                 The trial run or the hyperparameter dictionary for hyperparameter search.
#             kwargs:
#                 Additional keyword arguments used to hide deprecated arguments
#         """

#         # memory metrics - must set up as early as possible
#         self._memory_tracker.start()

#         args = self.args

#         self.is_in_train = True

#         # do_train is not a reliable argument, as it might not be set and .train() still called, so
#         # the following is a workaround:
#         if args.fp16_full_eval and not args.do_train:
#             self.model = self.model.to(args.device)

#         if "model_path" in kwargs:
#             resume_from_checkpoint = kwargs.pop("model_path")
#             warnings.warn(
#                 "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
#                 "instead.",
#                 FutureWarning,
#             )
#         if len(kwargs) > 0:
#             raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
#         # This might change the seed so needs to run first.
#         self._hp_search_setup(trial)

#         # Model re-init
#         model_reloaded = False
#         if self.model_init is not None:
#             # Seed must be set before instantiating the model when using model_init.
#             set_seed(args.seed)
#             self.model = self.call_model_init(trial)
#             model_reloaded = True
#             # Reinitializes optimizer and scheduler
#             self.optimizer, self.lr_scheduler = None, None

#         # Load potential model checkpoint
#         if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
#             resume_from_checkpoint = get_last_checkpoint(args.output_dir)
#             if resume_from_checkpoint is None:
#                 raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

#         if resume_from_checkpoint is not None:
#             if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
#                 raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

#             logger.info(f"Loading model from {resume_from_checkpoint}).")

#             if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
#                 config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
#                 checkpoint_version = config.transformers_version
#                 if checkpoint_version is not None and checkpoint_version != __version__:
#                     logger.warn(
#                         f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
#                         f"Transformers but your current version is {__version__}. This is not recommended and could "
#                         "yield to errors or unwanted behaviors."
#                     )

#             if args.deepspeed:
#                 # will be resumed in deepspeed_init
#                 pass
#             else:
#                 # We load the model state dict on the CPU to avoid an OOM error.
#                 state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
#                 # If the model is on the GPU, it still works!
#                 load_result = self.model.load_state_dict(state_dict, strict=False)
#                 if len(load_result.missing_keys) != 0:
#                     if load_result.missing_keys == self.model._keys_to_ignore_on_save:
#                         self.model.tie_weights()
#                     else:
#                         logger.warn(
#                             f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}."
#                         )
#                 if len(load_result.unexpected_keys) != 0:
#                     logger.warn(
#                         f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
#                     )

#         # If model was re-initialized, put it on the right device and update self.model_wrapped
#         if model_reloaded:
#             if self.place_model_on_device:
#                 self.model = self.model.to(args.device)
#             self.model_wrapped = self.model

#         # Keeping track whether we can can len() on the dataset or not
#         train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

#         # Data loader and number of training steps
#         train_dataloader = self.get_train_dataloader()

#         # Setting up training control variables:
#         # number of training epochs: num_train_epochs
#         # number of training steps per epoch: num_update_steps_per_epoch
#         # total number of training steps to execute: max_steps
#         if train_dataset_is_sized:
#             num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
#             num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
#             if args.max_steps > 0:
#                 max_steps = args.max_steps
#                 num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
#                     args.max_steps % num_update_steps_per_epoch > 0
#                 )
#             else:
#                 max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
#                 num_train_epochs = math.ceil(args.num_train_epochs)
#         else:
#             # see __init__. max_steps is set when the dataset has no __len__
#             max_steps = args.max_steps
#             num_train_epochs = int(args.num_train_epochs)
#             num_update_steps_per_epoch = max_steps

#         if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
#             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

#         delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
#         if args.deepspeed:
#             deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
#                 self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
#             )
#             self.model = deepspeed_engine.module
#             self.model_wrapped = deepspeed_engine
#             self.deepspeed = deepspeed_engine
#             self.optimizer = optimizer
#             self.lr_scheduler = lr_scheduler
#         elif not delay_optimizer_creation:
#             self.create_optimizer_and_scheduler(num_training_steps=max_steps)

#         self.state = TrainerState()
#         self.state.is_hyper_param_search = trial is not None

#         model = self._wrap_model(self.model_wrapped)

#         # for the rest of this function `model` is the outside model, whether it was wrapped or not
#         if model is not self.model:
#             self.model_wrapped = model

#         if delay_optimizer_creation:
#             self.create_optimizer_and_scheduler(num_training_steps=max_steps)

#         # Check if saved optimizer or scheduler states exist
#         self._load_optimizer_and_scheduler(resume_from_checkpoint)

#         # important: at this point:
#         # self.model         is the Transformers Model
#         # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

#         # Train!
#         if is_torch_tpu_available():
#             world_size = xm.xrt_world_size()
#         elif args.local_rank != -1:
#             world_size = dist.get_world_size()
#         else:
#             world_size = 1

#         total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * world_size
#         num_examples = (
#             self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
#         )

#         logger.info("***** Running training *****")
#         logger.info(f"  Num examples = {num_examples}")
#         logger.info(f"  Num Epochs = {num_train_epochs}")
#         logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
#         logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
#         logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
#         logger.info(f"  Total optimization steps = {max_steps}")

#         self.state.epoch = 0
#         start_time = time.time()
#         epochs_trained = 0
#         steps_trained_in_current_epoch = 0
#         steps_trained_progress_bar = None

#         # Check if continuing training from a checkpoint
#         if resume_from_checkpoint is not None and os.path.isfile(
#             os.path.join(resume_from_checkpoint, "trainer_state.json")
#         ):
#             self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
#             epochs_trained = self.state.global_step // num_update_steps_per_epoch
#             if not args.ignore_data_skip:
#                 steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
#                 steps_trained_in_current_epoch *= args.gradient_accumulation_steps
#             else:
#                 steps_trained_in_current_epoch = 0

#             logger.info("  Continuing training from checkpoint, will skip to saved global_step")
#             logger.info(f"  Continuing training from epoch {epochs_trained}")
#             logger.info(f"  Continuing training from global step {self.state.global_step}")
#             if not args.ignore_data_skip:
#                 logger.info(
#                     f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
#                     "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
#                     "flag to your launch command, but you will resume the training on data already seen by your model."
#                 )
#                 if self.is_local_process_zero() and not args.disable_tqdm:
#                     steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
#                     steps_trained_progress_bar.set_description("Skipping the first batches")

#         # Update the references
#         self.callback_handler.model = self.model
#         self.callback_handler.optimizer = self.optimizer
#         self.callback_handler.lr_scheduler = self.lr_scheduler
#         self.callback_handler.train_dataloader = train_dataloader
#         self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
#         self.state.trial_params = hp_params(trial) if trial is not None else None
#         # This should be the same if the state has been saved but in case the training arguments changed, it's safer
#         # to set this after the load.
#         self.state.max_steps = max_steps
#         self.state.num_train_epochs = num_train_epochs
#         self.state.is_local_process_zero = self.is_local_process_zero()
#         self.state.is_world_process_zero = self.is_world_process_zero()

#         # tr_loss is a tensor to avoid synchronization of TPUs through .item()
#         tr_loss = torch.tensor(0.0).to(args.device)
#         # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
#         self._total_loss_scalar = 0.0
#         self._globalstep_last_logged = self.state.global_step
#         model.zero_grad()

#         self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

#         # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
#         if not args.ignore_data_skip:
#             for epoch in range(epochs_trained):
#                 # We just need to begin an iteration to create the randomization of the sampler.
#                 for _ in train_dataloader:
#                     break

#         for epoch in range(epochs_trained, num_train_epochs):
#             if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
#                 train_dataloader.sampler.set_epoch(epoch)
#             elif isinstance(train_dataloader.dataset, IterableDatasetShard):
#                 train_dataloader.dataset.set_epoch(epoch)

#             if is_torch_tpu_available():
#                 parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
#                 epoch_iterator = parallel_loader
#             else:
#                 epoch_iterator = train_dataloader

#             # Reset the past mems state at the beginning of each epoch if necessary.
#             if args.past_index >= 0:
#                 self._past = None

#             steps_in_epoch = (
#                 len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
#             )
#             self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

#             for step, inputs in enumerate(epoch_iterator):

#                 # Skip past any already trained steps if resuming training
#                 if steps_trained_in_current_epoch > 0:
#                     steps_trained_in_current_epoch -= 1
#                     if steps_trained_progress_bar is not None:
#                         steps_trained_progress_bar.update(1)
#                     if steps_trained_in_current_epoch == 0:
#                         self._load_rng_state(resume_from_checkpoint)
#                     continue
#                 elif steps_trained_progress_bar is not None:
#                     steps_trained_progress_bar.close()
#                     steps_trained_progress_bar = None

#                 if step % args.gradient_accumulation_steps == 0:
#                     self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

#                 is_final_epoch = epoch == num_train_epochs - 1
#                 is_first_minibatch = step == 0
#                 is_last_minibatch = step == len(epoch_iterator) - 1
#                 inputs['is_final_epoch'] = is_final_epoch
#                 inputs['is_first_minibatch'] = is_first_minibatch
#                 inputs['is_last_minibatch'] = is_last_minibatch

#                 if (
#                     ((step + 1) % args.gradient_accumulation_steps != 0)
#                     and args.local_rank != -1
#                     and args._no_sync_in_gradient_accumulation
#                 ):
#                     # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
#                     with model.no_sync():
#                         tr_loss += self.training_step(model, inputs)
#                 else:
#                     tr_loss += self.training_step(model, inputs)
#                 self.current_flos += float(self.floating_point_ops(inputs))

#                 # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
#                 if self.deepspeed:
#                     self.deepspeed.step()

#                 if (step + 1) % args.gradient_accumulation_steps == 0 or (
#                     # last step in epoch but step is always smaller than gradient_accumulation_steps
#                     steps_in_epoch <= args.gradient_accumulation_steps
#                     and (step + 1) == steps_in_epoch
#                 ):
#                     # Gradient clipping
#                     if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
#                         # deepspeed does its own clipping

#                         if self.use_amp:
#                             # AMP: gradients need unscaling
#                             self.scaler.unscale_(self.optimizer)

#                         if hasattr(self.optimizer, "clip_grad_norm"):
#                             # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
#                             self.optimizer.clip_grad_norm(args.max_grad_norm)
#                         elif hasattr(model, "clip_grad_norm_"):
#                             # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
#                             model.clip_grad_norm_(args.max_grad_norm)
#                         else:
#                             # Revert to normal clipping otherwise, handling Apex or full precision
#                             torch.nn.utils.clip_grad_norm_(
#                                 amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
#                                 args.max_grad_norm,
#                             )

#                     # Optimizer step
#                     optimizer_was_run = True
#                     if self.deepspeed:
#                         pass  # called outside the loop
#                     elif is_torch_tpu_available():
#                         xm.optimizer_step(self.optimizer)
#                     elif self.use_amp:
#                         scale_before = self.scaler.get_scale()
#                         self.scaler.step(self.optimizer)
#                         self.scaler.update()
#                         scale_after = self.scaler.get_scale()
#                         optimizer_was_run = scale_before <= scale_after
#                     else:
#                         self.optimizer.step()

#                     if optimizer_was_run and not self.deepspeed:
#                         self.lr_scheduler.step()

#                     model.zero_grad()
#                     self.state.global_step += 1
#                     self.state.epoch = epoch + (step + 1) / steps_in_epoch
#                     self.control = self.callback_handler.on_step_end(args, self.state, self.control)

#                     self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

#                 if self.control.should_epoch_stop or self.control.should_training_stop:
#                     break

#             self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
#             self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

#             if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
#                 if is_torch_tpu_available():
#                     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
#                     xm.master_print(met.metrics_report())
#                 else:
#                     logger.warning(
#                         "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
#                         "configured. Check your training configuration if this is unexpected."
#                     )
#             if self.control.should_training_stop:
#                 break

#         if args.past_index and hasattr(self, "_past"):
#             # Clean the state at the end of training
#             delattr(self, "_past")

#         logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
#         if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
#             # Wait for everyone to get here so we are sur the model has been saved by process 0.
#             if is_torch_tpu_available():
#                 xm.rendezvous("load_best_model_at_end")
#             elif args.local_rank != -1:
#                 dist.barrier()

#             logger.info(
#                 f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
#             )
#             # We load the model state dict on the CPU to avoid an OOM error.
#             state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME), map_location="cpu")
#             # If the model is on the GPU, it still works!
#             self.model.load_state_dict(state_dict)

#             if self.deepspeed:
#                 self.deepspeed.load_checkpoint(
#                     self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
#                 )

#         metrics = speed_metrics("train", start_time, self.state.max_steps)
#         self.store_flos()
#         metrics["total_flos"] = self.state.total_flos
#         self.log(metrics)

#         self.control = self.callback_handler.on_train_end(args, self.state, self.control)
#         # add remaining tr_loss
#         self._total_loss_scalar += tr_loss.item()

#         self.is_in_train = False

#         self._memory_tracker.stop_and_update_metrics(metrics)

#         return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)
