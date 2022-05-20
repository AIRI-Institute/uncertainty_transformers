from ue4nlp.ue_estimator_mc import UeEstimatorMc
from ue4nlp.ue_estimator_mcddpp import UeEstimatorMcDdpp
from ue4nlp.ue_estimator_sngp import UeEstimatorSngp
from ue4nlp.ue_estimator_nuq import UeEstimatorNUQ, UeEstimatorNUQNer
from ue4nlp.ue_estimator_l_nuq import UeEstimatorLNUQ
from ue4nlp.ue_estimator_mahalanobis import UeEstimatorMahalanobis, UeEstimatorMahalanobisNer
from ue4nlp.ue_estimator_hybrid import UeEstimatorHybrid
from ue4nlp.ue_estimator_mc_mahalanobis import UeEstimatorMcMahalanobis, UeEstimatorMcMahalanobisNer
from ue4nlp.ue_estimator_l_mahalanobis import UeEstimatorLMahalanobis
from ue4nlp.ue_estimator_msd import UeEstimatorMSD, UeEstimatorMSDNer
from ue4nlp.ue_estimator_ddu import UeEstimatorDDU, UeEstimatorDDUNer
from ue4nlp.ue_estimator_decomposing import UeEstimatorDecomposing, UeEstimatorDecomposingNer
from ue4nlp.ue_estimator_sto import UeEstimatorSTO
from utils.utils_data import load_ood_dataset, load_ood_dataset_ner
import numpy as np

import logging

log = logging.getLogger(__name__)


def create_ue_estimator(
    model,
    ue_args,
    eval_metric,
    calibration_dataset,
    train_dataset,
    cache_dir,
    config=None,
):
    if ue_args.ue_type == "sngp":
        return UeEstimatorSngp(model, ue_args, eval_metric)

    elif ue_args.ue_type == "mc" or ue_args.ue_type == "mc-dc":
        return UeEstimatorMc(
            model, ue_args, eval_metric, calibration_dataset, train_dataset
        )

    elif ue_args.ue_type == "mc-dpp":
        if ue_args.dropout.dry_run_dataset == "eval":
            dry_run_dataset = "eval"
        elif ue_args.dropout.dry_run_dataset == "train":
            dry_run_dataset = train_dataset
        elif ue_args.dropout.dry_run_dataset == "val":
            dry_run_dataset = calibration_dataset
        else:
            raise ValueError()

        ood_dataset = None
        if ue_args.dropout.use_ood_sampling:
            ood_dataset = load_ood_dataset(
                ue_args.dropout.ood_sampling.dataset_name,
                model._max_len, 
                model.tokenizer, 
                cache_dir,
                config
            ) 

        return UeEstimatorMcDdpp(
            model,
            ue_args,
            eval_metric,
            calibration_dataset,
            dry_run_dataset,
            ood_dataset=ood_dataset,
        )
    
    elif ue_args.ue_type == "nuq" or ue_args.ue_type == "l-nuq":
        
        if hasattr(ue_args.nuq, 'n_points'):
            #the case when we use hidden params for tune_bandwidth
            if len(train_dataset) < ue_args.nuq.n_neighbors:
                ue_args.nuq.n_neighbors = int(len(train_dataset)//2)
            
            try:
                max_elements_label = np.bincount(train_dataset['label']).max()
            except:
                max_elements_label = np.bincount(train_dataset.dataset['label']).max()
                
            if ue_args.nuq.n_folds > max_elements_label:
                ue_args.nuq.n_folds = int(max_elements_label)

            ue_args.nuq.tune_bandwidth = f'{ue_args.nuq.tune_bandwidth}:n_points={ue_args.nuq.n_points};n_folds={ue_args.nuq.n_folds};n_samples={ue_args.nuq.n_samples}'
        log.info(f'\n\nFull NUQ tune_bandwidth parameter value {ue_args.nuq.tune_bandwidth}\n\n')
        
        UeEstimator = UeEstimatorLNUQ if ue_args.ue_type == "l-nuq" else UeEstimatorNUQ
        return UeEstimator(
            model, ue_args, config, train_dataset, calibration_dataset
        )
    elif ue_args.ue_type == "maha":
        return UeEstimatorMahalanobis(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "hybrid":
        return UeEstimatorHybrid(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "l-maha":
        return UeEstimatorLMahalanobis(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "mc_maha":
        return UeEstimatorMcMahalanobis(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "msd":
        return UeEstimatorMSD(
            model, config, ue_args, eval_metric, calibration_dataset, train_dataset
        )
    elif ue_args.ue_type == "ddu":
        return UeEstimatorDDU(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "decomposing_md":
        return UeEstimatorDecomposing(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "sto":
        return UeEstimatorSTO(
            model, ue_args, eval_metric, calibration_dataset, train_dataset
        )
    else:
        raise ValueError()
        
        
def create_ue_estimator_ner(
    model,
    ue_args,
    eval_metric,
    calibration_dataset,
    train_dataset,
    cache_dir,
    config=None,
    data_args=None,
):
    if ue_args.ue_type == "sngp":
        return UeEstimatorSngp(model, ue_args, eval_metric)

    elif ue_args.ue_type == "mc" or ue_args.ue_type == "mc-dc":
        return UeEstimatorMc(
            model, ue_args, eval_metric, calibration_dataset, train_dataset
        )

    elif ue_args.ue_type == "mc-dpp":
        if ue_args.dropout.dry_run_dataset == "eval":
            dry_run_dataset = "eval"
        elif ue_args.dropout.dry_run_dataset == "train":
            dry_run_dataset = train_dataset
        elif ue_args.dropout.dry_run_dataset == "val":
            dry_run_dataset = calibration_dataset
        else:
            raise ValueError()

        ood_dataset = None
        if ue_args.dropout.use_ood_sampling:
            ood_dataset = load_ood_dataset_ner(
                ue_args.dropout.ood_sampling.dataset_name,
                data_args, 
                model._bpe_tokenizer, 
                cache_dir,
                config
            )

        return UeEstimatorMcDdpp(
            model,
            ue_args,
            eval_metric,
            calibration_dataset,
            dry_run_dataset,
            ood_dataset=ood_dataset,
            ner=True,
        )
    elif ue_args.ue_type == "nuq":
        return UeEstimatorNUQNer(
            model, ue_args, config, train_dataset, calibration_dataset
        )
    elif ue_args.ue_type == "maha":
        return UeEstimatorMahalanobisNer(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "mc_maha":
        return UeEstimatorMcMahalanobisNer(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "msd":
        return UeEstimatorMSDNer(
            model, config, ue_args, eval_metric, calibration_dataset, train_dataset
        )
    elif ue_args.ue_type == "ddu":
        return UeEstimatorDDUNer(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "decomposing_md":
        return UeEstimatorDecomposingNer(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "sto":
        return UeEstimatorSTO(
            model, ue_args, eval_metric, calibration_dataset, train_dataset
        )
    else:
        raise ValueError()
