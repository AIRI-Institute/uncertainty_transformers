from ue4nlp.transformers_cached import (
    ElectraForTokenClassificationCached,
    DebertaForTokenClassificationCached,
    DistilBertForTokenClassificationCached
)


from utils.utils_heads import (
    spectral_normalized_model,
    SpectralNormalizedPooler,
    ElectraNERHeadCustom,
    ElectraNERHeadSN,
    ElectraSelfAttentionStochastic,
    replace_attention,
)

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    ElectraForTokenClassification,
    DistilBertForTokenClassification,
    DebertaForTokenClassification,
)
from ue4nlp.transformers_mixup import (
    ElectraForTokenClassificationMSD,
    DistilBertForTokenClassificationMSD,
    DebertaForTokenClassificationMSD,
)
from ue4nlp.bert_sngp_model import (
    SNGPElectraForTokenClassificationCached,
)
from torch.nn.utils import spectral_norm
import torch
import logging
from utils.classification_models import build_model, load_electra_sn_head

log = logging.getLogger(__name__)


def create_electra_ner(model_config, tokenizer, use_sngp,
                       use_spectralnorm, use_mixup, use_selective, ue_args,
                       model_path_or_name, config):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_mixup:
        model = build_model(ElectraForTokenClassificationMSD,
                            model_path_or_name, **model_kwargs)
        # set MSD params
        log.info("Created mixup model")
        model.post_init(config.mixup)
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                model.classifier = ElectraNERHeadSN(model)
                if model.self_ensembling:
                    model.model_2.classifier = ElectraNERHeadSN(model.model_2)
                log.info("Replaced ELECTRA's head with SN")
            elif config.spectralnorm_layer == "all":
                # TODO:
                # Doesn't work with loading
                model.classifier = ElectraNERHeadCustom(model)
                if model.self_ensembling:
                    model.model_2.classifier = ElectraNERHeadCustom(model.model_2)
                spectral_normalized_model(model)
                log.info("Replaced ELECTRA's encoder with SN")
        else:
            model.classifier = ElectraNERHeadCustom(model)
            if model.self_ensembling:
                model.model_2.classifier = ElectraNERHeadCustom(model.model_2)
        if config.do_eval and not (config.do_train):
            load_electra_sn_head(model_path_or_name, model, "ELECTRA")
        log.info("Replaced ELECTRA's head")
    elif use_sngp:
        if ue_args.use_cache:
            model_kwargs.update(dict(ue_config=ue_args.sngp))
            model = build_model(SNGPElectraForTokenClassificationCached,
                                model_path_or_name, **model_kwargs)
            log.info("Loaded ELECTRA with SNGP")
        else:
            raise ValueError(
                f"{model_path_or_name} does not work without cache."
            )
    else:
        model = build_model(ElectraForTokenClassificationCached,
                            model_path_or_name, **model_kwargs)
        model.use_cache = True if ue_args.use_cache else False
        if use_spectralnorm:
            # TODO: change to another heads
            if config.spectralnorm_layer == "last":
                sn_value = None if 'sn_value' not in ue_args.keys() else ue_args.sn_value
                model.classifier = ElectraNERHeadSN(model, sn_value)
                log.info("Replaced ELECTRA's head with SN")
            elif config.spectralnorm_layer == "all":
                # TODO:
                # Doesn't work with loading
                model.classifier = ElectraNERHeadCustom(model)
                spectral_normalized_model(model)
                log.info("Replaced ELECTRA's encoder with SN")
            # TODO: add loading head
            if config.do_eval and not (config.do_train):
                # here we load full model, because we couldn't load CustomHead into base model
                load_electra_sn_head(model_path_or_name, model, "ELECTRA SN")
        else:
            model.classifier = ElectraNERHeadCustom(model)
            log.info("Replaced ELECTRA's head")
            if config.do_eval and not (config.do_train):
                # here we load full model, because we couldn't load CustomHead into base model
                load_electra_sn_head(model_path_or_name, model, "ELECTRA")
        if ue_args.get("use_sto", False):
            # replace attention by stochastic version
            if ue_args.sto_layer == "last":
                model = replace_attention(model, ue_args, -1)
            elif ue_args.sto_layer == "all":
                for idx, _ in enumerate(model.electra.encoder.layer):
                    model = replace_attention(model, ue_args, idx)
            log.info("Replaced ELECTRA's attention with Stochastic Attention")
    return model


def create_deberta_ner(model_config, tokenizer, use_sngp,
                       use_spectralnorm, use_mixup, use_selective, ue_args,
                       model_path_or_name, config):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_mixup:
        model = build_model(DebertaForTokenClassificationMSD,
                            model_path_or_name, **model_kwargs)
        # set MSD params
        log.info("Created mixup model")
        model.post_init(config.mixup)
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                model.classifier = ElectraNERHeadSN(model)
                if model.self_ensembling:
                    model.model_2.classifier = ElectraNERHeadSN(model.model_2)
                log.info("Replaced DeBERTA's head with SN")
            elif config.spectralnorm_layer == "all":
                # TODO:
                # Doesn't work with loading
                #model.classifier = ElectraNERHeadCustom(model)
                spectral_normalized_model(model)
                log.info("Replaced DeBERTA's encoder with SN")
            # TODO: add loading head
        else:
            model.classifier = ElectraNERHeadCustom(model)
            if model.self_ensembling:
                model.model_2.classifier = ElectraNERHeadCustom(model.model_2)
            log.info("Replaced DeBERTA's head")
        if config.do_eval and not (config.do_train):
            # here we load full model, because we couldn't load CustomHead into base model
            load_electra_sn_head(model_path_or_name, model, "DeBERTA")
    else:
        model = build_model(DebertaForTokenClassificationCached,
                            model_path_or_name, **model_kwargs)
        model.use_cache = True if ue_args.use_cache else False
        # TODO: change to another heads
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                sn_value = None if 'sn_value' not in ue_args.keys() else ue_args.sn_value
                model.classifier = ElectraNERHeadSN(model, sn_value)
                log.info("Replaced DeBERTA's head with SN")
            elif config.spectralnorm_layer == "all":
                # TODO:
                # Doesn't work with loading
                #model.classifier = ElectraNERHeadCustom(model)
                spectral_normalized_model(model)
                log.info("Replaced DeBERTA's encoder with SN")
            # TODO: add loading head
            if config.do_eval and not (config.do_train):
                # here we load full model, because we couldn't load CustomHead into base model
                load_electra_sn_head(model_path_or_name, model, "DeBERTA SN")
        else:
            model.classifier = ElectraNERHeadCustom(model)
            log.info("Replaced DeBERTA's head")
            if config.do_eval and not (config.do_train):
                # here we load full model, because we couldn't load CustomHead into base model
                load_electra_sn_head(model_path_or_name, model, "DeBERTA")
    return model


def create_distilbert_ner(model_config, tokenizer, use_sngp,
                          use_spectralnorm, use_mixup, use_selective, ue_args,
                          model_path_or_name, config):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_mixup:
        model = build_model(DistilBertForTokenClassificationMSD,
                            model_path_or_name, **model_kwargs)
        # set MSD params
        log.info("Created mixup model")
        model.post_init(config.mixup)
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                model.classifier = ElectraNERHeadSN(model)
                if model.self_ensembling:
                    model.model_2.classifier = ElectraNERHeadSN(model.model_2)
                log.info("Replaced DistilBERT's head with SN")
            elif config.spectralnorm_layer == "all":
                model.classifier = ElectraNERHeadCustom(model)
                if model.self_ensembling:
                    model.model_2.classifier = ElectraNERHeadCustom(model.model_2)
                spectral_normalized_model(model)
                log.info("Replaced DistilBERT's encoder with SN")
        else:
            model.classifier = ElectraNERHeadCustom(model)
            if model.self_ensembling:
                model.model_2.classifier = ElectraNERHeadCustom(model.model_2)
        if config.do_eval and not (config.do_train):
            log.info("Replaced DistilBERT's head")
        if config.do_eval and not (config.do_train):
            load_electra_sn_head(model_path_or_name, model, "DistilBERT")
    else:
        model = build_model(DistilBertForTokenClassificationCached,
                            model_path_or_name, **model_kwargs)
        model.use_cache = True if ue_args.use_cache else False
        # TODO: change to another heads
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                model.classifier = ElectraNERHeadSN(model)
                log.info("Replaced DistilBERT's head with SN")
            elif config.spectralnorm_layer == "all":
                # TODO:
                # Doesn't work with loading
                model.classifier = ElectraNERHeadCustom(model)
                spectral_normalized_model(model)
                log.info("Replaced DistilBERT's encoder with SN")
            # TODO: add loading head
            if config.do_eval and not (config.do_train):
                # here we load full model, because we couldn't load CustomHead into base model
                load_electra_sn_head(model_path_or_name, model, "DistilBERT SN")
        else:
            model.classifier = ElectraNERHeadCustom(model)
            log.info("Replaced DistilBERT's head")
            if config.do_eval and not (config.do_train):
                load_electra_sn_head(model_path_or_name, model, "DistilBERT")
    return model
