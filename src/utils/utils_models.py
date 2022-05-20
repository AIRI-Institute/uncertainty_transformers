from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from torch.nn.utils import spectral_norm
import torch
from utils.classification_models import (
    create_bert,
    create_xlnet,
    create_deberta,
    create_electra,
    create_roberta,
    create_distilbert,
    build_model,
)
from utils.ner_models import (
    create_deberta_ner,
    create_electra_ner,
    create_distilbert_ner,
)
import logging

log = logging.getLogger(__name__)

def create_tokenizer(model_args, config):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=config.cache_dir,
    )
    return tokenizer

def create_model(num_labels, model_args, data_args, ue_args, config):
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=config.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=config.cache_dir,
    )

    use_sngp = ue_args.ue_type == "sngp"
    use_duq = ue_args.ue_type == "duq"
    use_spectralnorm = "use_spectralnorm" in ue_args.keys() and ue_args.use_spectralnorm
    use_mixup = "mixup" in config.keys() and config.mixup.use_mixup
    use_selective = (
        "use_selective" in ue_args.keys() and ue_args.use_selective
    )
    model_path_or_name = model_args.model_name_or_path

    models_constructors = {
        "electra": create_electra,
        "roberta": create_roberta,
        "deberta": create_deberta,
        "distilbert": create_distilbert,
        "xlnet": create_xlnet,
        "bert": create_bert,
    }
    for key, value in models_constructors.items():
        if key in model_path_or_name:
            return models_constructors[key](model_config, tokenizer, use_sngp, use_duq,
                                            use_spectralnorm, use_mixup, use_selective, ue_args,
                                            model_path_or_name, config), tokenizer
    raise ValueError(
        f"Cannot find model with this name or path: {model_path_or_name}"
    )


def create_model_ner(num_labels, model_args, data_args, ue_args, config):
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=config.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=config.cache_dir,
        use_fast=True,
        add_prefix_space=True
    )

    use_spectralnorm = "use_spectralnorm" in ue_args.keys() and ue_args.use_spectralnorm
    use_mixup = "mixup" in config.keys() and config.mixup.use_mixup
    use_sngp = ue_args.ue_type == "sngp"
    use_selective = (
        "use_selective" in ue_args.keys() and ue_args.use_selective
    )
    model_path_or_name = model_args.model_name_or_path

    models_constructors = {
        "electra": create_electra_ner,
        "deberta": create_deberta_ner,
        "distilbert": create_distilbert_ner,
    }
    for key, value in models_constructors.items():
        if key in model_path_or_name:
            return models_constructors[key](model_config, tokenizer, use_sngp,
                                            use_spectralnorm, use_mixup, use_selective, ue_args,
                                            model_path_or_name, config), tokenizer
    raise ValueError(
        f"Cannot find model with this name or path: {model_name_or_path}"
    )
    return model, tokenizer
