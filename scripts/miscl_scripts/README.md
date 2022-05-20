# How to reproduce results for the ELECTRA with metric loss
1. metric_hypopt.sh - run to obtain optimal hyperparameters
2. metric_electra_opt_params.sh - run to obtain results for the ELECTRA with the metric loss and with the paper params

# How to reproduce results for the DistilBERT model
1. hypopt_distilbert.sh/hypopt_distilbert_cer.sh/hypopt_distilbert_metric.sh/hypopt_distilbert_dpp.sh - scripts for hyperparameters optimization for the DistilBERT for raw/CER/metric regularizers and for DPP params.
2. distilbert_raw_cer.sh - script for training, estimation and metrics calculation for the DistilBERT model with a raw/CER regularizers.
3. distilbert_metric.sh - script for training, estimation and metrics calculation for the DistilBERT model with the metric regularizer.
4. distilbert_ensemble.sh - script for training, estimation and metrics calculation for the DistilBERT model with DeepEnsemble.

# How to reproduce results for the DeBERTa model
1. deberta_hp_search.sh - scripts for hyperparameters optimization for the DeBERTa for raw/CER/metric regularizers
2. run_deberta_glue_dpp_hp.sh - scripts for optimization DPP params for the DeBERTa
3. run_train_deberta_models_glue.sh - script for training all models with all regularizers.
4. run_eval_deberta_all.sh - script for evaluation all methods for all regularizers.

# How to reproduce results for the models with MSD
1. hypopt_mixup.sh - scripts for hyperparameters optimization for the ELECTRA model with the MSD
2. mixup_opt_hyp.sh - script for training, estimation and metrics calculation for the ELECTRA model with the MSD.

# How to reproduce misclassification results
There are .sh files in this folder. Move these files to the directory with uncertainty-estimation repo directory, and run all scripts.

# How to reproduce results for the ELECTRA model (example of generation this scripts see here: uncertainty-estimation/src/generate_paper_scripts.ipynb)
1. run_train_models_electra.sh - script for training models with a raw/CER regularizers.
2. run_eval_raw_dpp_models.sh - script for evaluation DPP MC dropout without regularization.
3. run_eval_ddpp_models.sh - script for evaluation DDPP MC dropout with a raw/CER regularizers.
4. run_eval_mc_models.sh - script for evaluation MC dropout with a raw/CER regularizers.
5. run_eval_det_models.sh - script for evaluation Mahalanobis distance with a raw/CER regularizers.
6. run_train_ensemble_models.sh - script for train models for deep ensemble.
7. run_eval_ensemble_series.sh - script for evaluation models with deep ensemble.

# MD SN with new optimal HP
1. run_train_models_electra_with_new_sn.sh - scripts for training new SN models
2. run_eval_det_models_with_new_sn.sh - script for evaluation MD SN with new optimal HP.


# Old scripts and examples (before december 2021)

## How to reproduce misclassification results - new version (as in paper), 3 params
1. Results with metric loss for MRPC, COLA, SST-2, CONLL-2003 + DPP results for raw/CER CONLL - metric_opt_3hyp.sh

## How to run models with MSD
1. mixup_test.sh - exps with MSD for MC-last/MC-all for GLUE and CoNLL-2003

## How to run hyperparameter search for models with MSD
1. hypopt_mixup.sh - hyperparameter search for MSD models for GLUE and CoNLL-2003

## How to run models with metric loss - old version, with 2 params
1. metric_reg_full.sh - exps with metric loss for MC-last/MC-all/DPP on masks/DPP with OOD/NUQ for GLUE
2. metric_reg_smd.sh - exps with metric loss for sampled Mahalanobis distance for GLUE