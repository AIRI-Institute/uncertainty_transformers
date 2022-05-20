from ue4nlp.transformers_cached import (
    ElectraForSequenceClassificationCached,
    ElectraForSequenceClassificationAllLayers,
    BertForSequenceClassificationCached,
    RobertaForSequenceClassificationCached,
    DebertaForSequenceClassificationCached,
    DistilBertForSequenceClassificationCached
)

from ue4nlp.electra_duq_model import ElectraForSequenceClassificationDUQ
from ue4nlp.bert_sngp_model import (
    SNGPBertForSequenceClassificationCached,
    SNGPElectraForSequenceClassificationCached,
)

from utils.utils_heads import (
    ElectraClassificationHeadCustom,
    ElectraClassificationHeadIdentityPooler,
    BertClassificationHeadIdentityPooler,
    ElectraClassificationHeadSN,
    spectral_normalized_model,
    SpectralNormalizedPooler,
    ElectraSelfAttentionStochastic,
    replace_attention,
    ElectraClassificationHS,
    BERTClassificationHS
)

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ElectraForSequenceClassification,
    DebertaForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
)
from ue4nlp.transformers_mixup import (
    ElectraForSequenceClassificationMSD,
    XLNetForSequenceClassificationMSD,
    DistilBertForSequenceClassificationMSD,
    DebertaForSequenceClassificationMSD,
)

from torch.nn.utils import spectral_norm
import torch
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def build_model(model_class, model_path_or_name, **kwargs):
    return model_class.from_pretrained(model_path_or_name, **kwargs)


def load_electra_sn_encoder(model_path_or_name, model):
    model_full = torch.load(
        model_path_or_name + "/pytorch_model.bin"
    )

    for i, electralayer in enumerate(model.electra.encoder.layer): 
        electralayer_name = f'electra.encoder.layer.{i}.output.dense'
        electralayer.output.dense.weight_orig.data = model_full[f"{electralayer_name}.weight_orig"].data
        electralayer.output.dense.weight_u.data = model_full[f"{electralayer_name}.weight_u"].data
        electralayer.output.dense.weight_v.data = model_full[f"{electralayer_name}.weight_v"].data
        electralayer.output.dense.bias.data = model_full[f"{electralayer_name}.bias"].data
        
    del model_full
    torch.cuda.empty_cache()
    log.info("Loaded Electra's SN encoder")

def load_electra_sn_head(model_path_or_name, model, name):
    model_full = torch.load(
        model_path_or_name + "/pytorch_model.bin"
    )
    model.classifier.eval_init(model_full)
    del model_full
    torch.cuda.empty_cache()
    log.info(f"Loaded {name}'s head")


def load_distilbert_sn_head(model_path_or_name, model):
    model_full = torch.load(
        model_path_or_name + "/pytorch_model.bin"
    )
    model.pre_classifier.weight_orig.data = model_full["pre_classifier.weight_orig"].data
    model.pre_classifier.weight_u.data = model_full["pre_classifier.weight_u"].data
    model.pre_classifier.weight_v.data = model_full["pre_classifier.weight_v"].data
    model.pre_classifier.bias.data = model_full["pre_classifier.bias"].data
    del model_full
    torch.cuda.empty_cache()
    log.info("Loaded DistilBERT's head")


def create_bert(model_config, tokenizer, use_sngp, use_duq, use_spectralnorm,
                use_mixup, use_selective, ue_args, model_path_or_name, config):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if ue_args.use_cache:
        if use_sngp:
            model_kwargs.update(dict(ue_config=ue_args.sngp))
            model = build_model(SNGPBertForSequenceClassificationCached,
                                model_path_or_name, **model_kwargs)
        elif use_spectralnorm:
            model = build_model(BertForSequenceClassificationCached,
                                model_path_or_name, **model_kwargs)
            model.use_cache = True
            model.bert.pooler = SpectralNormalizedPooler(model.bert.pooler)
            log.info("Replaced BERT Pooler with SN")
        else:
            # common BERT case
            model = build_model(BertForSequenceClassificationCached,
                                model_path_or_name, **model_kwargs)
            if ('use_hs_labels' in ue_args.keys()) and ue_args.use_hs_labels:
                model.classifier = BERTClassificationHS(model.classifier, n_labels=model_config.num_labels)
                log.info("Replaced BERT's head with hyperspherical labels")
        model.disable_cache()
    else:
        # without cache
        if use_spectralnorm and not (use_mixup):
            model = build_model(BertForSequenceClassification,
                                model_path_or_name, **model_kwargs)
            model.use_cache = False
            model.bert.pooler = SpectralNormalizedPooler(model.bert.pooler)
            log.info("Replaced BERT Pooler with SN")
        else:
            model = build_model(AutoModelForSequenceClassification,
                                model_path_or_name, **model_kwargs)
    return model


def create_electra(model_config, tokenizer, use_sngp, use_duq, use_spectralnorm,
                   use_mixup, use_selective, ue_args, model_path_or_name, config):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    # TODO: rearrange if
    if ue_args.ue_type == 'l-maha' or ue_args.ue_type == 'l-nuq':
        electra_classifier = ElectraForSequenceClassificationAllLayers
    else:
        electra_classifier = ElectraForSequenceClassification
    if ue_args.use_cache:
        if use_sngp:
            model_kwargs.update(dict(ue_config=ue_args.sngp))
            model = build_model(SNGPElectraForSequenceClassificationCached,
                                model_path_or_name, **model_kwargs)
            log.info("Loaded ELECTRA with SNGP")
        elif use_spectralnorm:
            model = build_model(ElectraForSequenceClassificationCached,
                                model_path_or_name, **model_kwargs)
            model.use_cache = True
            if 'last' in config.spectralnorm_layer:
                sn_value = None if 'sn_value' not in ue_args.keys() else ue_args.sn_value
                n_power_iterations = 1 if 'n_power_iterations' not in ue_args.keys() else ue_args.n_power_iterations
                model.classifier = ElectraClassificationHeadSN(model.classifier, sn_value, n_power_iterations)
                log.info("Replaced ELECTRA's head with SN")
                if config.do_eval and not (config.do_train):
                    load_electra_sn_head(model_path_or_name, model, "ELECTRA SN")
            elif config.spectralnorm_layer == "all":
                model.classifier = ElectraClassificationHeadCustom(model.classifier)
                spectral_normalized_model(model)
                log.info("Replaced ELECTRA's encoder with SN")            
        else:
            model = build_model(ElectraForSequenceClassificationCached,
                                model_path_or_name, **model_kwargs)
            model.use_cache = True
            if ('use_hs_labels' in ue_args.keys()) and ue_args.use_hs_labels:
                if not os.path.exists(Path(model_path_or_name) / 'hs_labels.pt'):
                    hs_labels = None
                else:
                    hs_labels = torch.load(Path(model_path_or_name) / 'hs_labels.pt')
                model.classifier = ElectraClassificationHS(model.classifier, n_labels=model_config.num_labels, hs_labels=hs_labels)
                torch.save(model.classifier.hs_labels, Path(config.output_dir) / 'hs_labels.pt')
                log.info("Replaced ELECTRA's head with hyperspherical labels")
            else:
                model.classifier = ElectraClassificationHeadCustom(model.classifier)
                log.info("Replaced ELECTRA's head")
        model.disable_cache()
        if ue_args.get("use_sto", False):
            # replace attention by stochastic version
            if ue_args.sto_layer == "last":
                model = replace_attention(model, ue_args, -1)
            elif ue_args.sto_layer == "all":
                for idx, _ in enumerate(model.electra.encoder.layer):
                    model = replace_attention(model, ue_args, idx)
            log.info("Replaced ELECTRA's attention with Stochastic Attention")
    else:
        if use_duq:
            log.info("Using ELECTRA DUQ model")
            model = build_model(ElectraForSequenceClassificationDUQ,
                                model_path_or_name, **model_kwargs)
            model.make_duq(
                output_dir=config.cache_dir,
                batch_size=config.training.per_device_train_batch_size,
                duq_params=config.ue.duq_params,
            )
        elif use_spectralnorm and not (use_mixup):
            model = build_model(electra_classifier,
                                model_path_or_name, **model_kwargs)
            if 'last' in config.spectralnorm_layer:
                sn_value = None if 'sn_value' not in ue_args.keys() else ue_args.sn_value
                n_power_iterations = 1 if 'n_power_iterations' not in ue_args.keys() else ue_args.n_power_iterations
                model.classifier = ElectraClassificationHeadSN(model.classifier, sn_value, n_power_iterations)
                log.info("Replaced ELECTRA's head with SN")
                if config.do_eval and not (config.do_train):
                    load_electra_sn_head(model_path_or_name, model, "ELECTRA SN")
            elif config.spectralnorm_layer == "all":
                model.classifier = ElectraClassificationHeadCustom(model.classifier)
                spectral_normalized_model(model)
                log.info("Replaced ELECTRA's encoder with SN")
        elif use_mixup:
            model = build_model(ElectraForSequenceClassificationMSD,
                                model_path_or_name, **model_kwargs)
            # set MSD params
            log.info("Created mixup model")
            model.post_init(config.mixup)
            if use_spectralnorm:
                if config.spectralnorm_layer == "last":
                    model.classifier = ElectraClassificationHeadSN(model.classifier)
                    if model.self_ensembling:
                        model.model_2.classifier = ElectraClassificationHeadSN(model.model_2.classifier)
                    log.info("Replaced ELECTRA's head with SN")
                elif config.spectralnorm_layer == "all":
                    model.classifier = ElectraClassificationHeadCustom(model.classifier)
                    if model.self_ensembling:
                        model.model_2.classifier = ElectraClassificationHeadCustom(model.model_2.classifier)
                    spectral_normalized_model(model)
                    log.info("Replaced ELECTRA's encoder with SN")
                if config.do_eval and not (config.do_train):
                    load_electra_sn_head(model_path_or_name, model, "ELECTRA SN")
            else:
                # TODO: Check how this works if we replaced both classifiers
                model.classifier = ElectraClassificationHeadCustom(model.classifier)
                if model.self_ensembling:
                    model.model_2.classifier = ElectraClassificationHeadCustom(model.model_2.classifier)
                log.info("Replaced ELECTRA's head")
        else:
            model = build_model(electra_classifier,
                                model_path_or_name, **model_kwargs)
            #model.classifier = ElectraClassificationHeadCustom(model.classifier)
            if ('use_hs_labels' in ue_args.keys()) and ue_args.use_hs_labels:
                if not os.path.exists(Path(model_path_or_name) / 'hs_labels.pt'):
                    hs_labels = None
                else:
                    hs_labels = torch.load(Path(model_path_or_name) / 'hs_labels.pt')
                model.classifier = ElectraClassificationHS(model.classifier, n_labels=model_config.num_labels, hs_labels=hs_labels)
                torch.save(model.classifier.hs_labels, Path(config.output_dir) / 'hs_labels.pt')
                log.info("Replaced ELECTRA's head with hyperspherical labels")
            else:
                model.classifier = ElectraClassificationHeadCustom(model.classifier)
                log.info("Replaced ELECTRA's head")
            log.info("Replaced ELECTRA's head")
        if ue_args.get("use_sto", False):
            # replace attention by stochastic version
            if ue_args.sto_layer == "last":
                model = replace_attention(model, ue_args, -1)
            elif ue_args.sto_layer == "all":
                for idx, _ in enumerate(model.electra.encoder.layer):
                    model = replace_attention(model, ue_args, idx)
            log.info("Replaced ELECTRA's attention with Stochastic Attention")
        
    if use_spectralnorm and 'resid' in config.spectralnorm_layer:
        if 'last' not in config.spectralnorm_layer:
            model.classifier = ElectraClassificationHeadCustom(model.classifier)
            log.info("Replaced ELECTRA's head")
        for electralayer in model.electra.encoder.layer:
            electralayer.output.dense = torch.nn.utils.spectral_norm(electralayer.output.dense)
        log.info("Replaced residual connections after attention in ELECTRA's encoder")
        if config.do_eval and not (config.do_train):
            load_electra_sn_encoder(model_path_or_name, model)
            
    return model


def create_roberta(model_config, tokenizer, use_sngp, use_duq, use_spectralnorm,
                   use_mixup, use_selective, ue_args, model_path_or_name, config):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_spectralnorm:
        if ue_args.use_cache:
            model = build_model(RobertaForSequenceClassificationCached,
                                model_path_or_name, **model_kwargs)
            model.use_cache = True
            model.disable_cache()
        else:
            model = build_model(RobertaForSequenceClassification,
                                model_path_or_name, **model_kwargs)
            model.use_cache = False
        model.classifier = ElectraClassificationHeadSN(model.classifier)
        log.info("Replaced RoBERTA's head with SN")
    else:
        if ue_args.use_cache:
            model = build_model(RobertaForSequenceClassificationCached,
                                model_path_or_name, **model_kwargs)
            model.use_cache = True
            model.disable_cache()
        else:
            model = build_model(RobertaForSequenceClassification,
                                model_path_or_name, **model_kwargs)
            model.use_cache = False
        model.classifier = ElectraClassificationHeadCustom(model.classifier)
        log.info("Replaced RoBERTA's head")
    return model


def create_deberta(model_config, tokenizer, use_sngp, use_duq, use_spectralnorm,
                   use_mixup, use_selective, ue_args, model_path_or_name, config):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_mixup:
        model = build_model(DebertaForSequenceClassificationMSD,
                            model_path_or_name, **model_kwargs)
        # set MSD params
        log.info("Created mixup model")
        model.post_init(config.mixup)
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                model.pooler = SpectralNormalizedPooler(model.pooler)
                if model.self_ensembling:
                    model.model_2.pooler = SpectralNormalizedPooler(model.model_2.pooler)
                log.info("Replaced DeBERTA's pooler with SN")
                if config.do_eval and not (config.do_train):
                    model_full = torch.load(model_path_or_name + "/pytorch_model.bin")
                    model.pooler.eval_init(model_full)
                    del model_full
                    log.info("Loaded DeBERTA's pooler with SN")
            elif config.spectralnorm_layer == "all":
                spectral_normalized_model(model)
                log.info("Replaced DeBERTA's encoder with SN")
    else:
        # build cached model with or without cache, and add spectralnorm case
        model = build_model(DebertaForSequenceClassificationCached,
                            model_path_or_name, **model_kwargs)
        model.use_cache = True if ue_args.use_cache else False
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                sn_value = None if 'sn_value' not in ue_args.keys() else ue_args.sn_value
                model.pooler = SpectralNormalizedPooler(model.pooler, sn_value)
                log.info("Replaced DeBERTA's pooler with SN")
                if config.do_eval and not (config.do_train):
                    model_full = torch.load(model_path_or_name + "/pytorch_model.bin")
                    model.pooler.eval_init(model_full)
                    del model_full
                    log.info("Loaded DeBERTA's pooler with SN")
            elif config.spectralnorm_layer == "all":
                spectral_normalized_model(model)
                log.info("Replaced DeBERTA's encoder with SN")
        elif not ue_args.use_cache:
            model = build_model(AutoModelForSequenceClassification,
                                model_path_or_name, **model_kwargs)
        if ue_args.use_cache:
            model.disable_cache()
    return model


def create_distilbert(model_config, tokenizer, use_sngp, use_duq, use_spectralnorm,
                      use_mixup, use_selective, ue_args, model_path_or_name, config):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_mixup:
        model = build_model(DistilBertForSequenceClassificationMSD,
                            model_path_or_name, **model_kwargs)
        # set MSD params
        log.info("Created mixup model")
        model.post_init(config.mixup)
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                model.pre_classifier = spectral_norm(model.pre_classifier)
                if model.self_ensembling:
                    model.model_2.pre_classifier = spectral_norm(model.model_2.pre_classifier)
                log.info("Replaced DistilBERT's head with SN")
            elif config.spectralnorm_layer == "all":
                spectral_normalized_model(model)
                log.info("Replaced DistilBERT's encoder with SN")
            if config.do_eval and not (config.do_train):
                load_distilbert_sn_head(model_path_or_name, model)
    else:
        model = build_model(DistilBertForSequenceClassificationCached,
                            model_path_or_name, **model_kwargs)
        model.use_cache = True if ue_args.use_cache else False
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                model.pre_classifier = spectral_norm(model.pre_classifier)
                log.info("Replaced DistilBERT's head with SN")
            elif config.spectralnorm_layer == "all":
                spectral_normalized_model(model)
                log.info("Replaced DistilBERT's encoder with SN")
            if config.do_eval and not (config.do_train):
                load_distilbert_sn_head(model_path_or_name, model)
        elif not ue_args.use_cache:
            model = build_model(AutoModelForSequenceClassification,
                                model_path_or_name, **model_kwargs)
        if ue_args.use_cache:
            model.disable_cache()
    return model


def create_xlnet(model_config, tokenizer, use_sngp, use_duq, use_spectralnorm,
                 use_mixup, use_selective, ue_args, model_path_or_name, config):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_mixup:
        model = build_model(XLNetForSequenceClassificationMSD,
                            model_path_or_name, **model_kwargs)
        # set MSD params
        log.info("Created mixup model")
        model.post_init(config.mixup)
        # model.classifier = ElectraClassificationHeadCustom(model.classifier)
        log.info("Don't replaced XLNet's head")
    else:
        model = build_model(AutoModelForSequenceClassification,
                            model_path_or_name, **model_kwargs)
    return model