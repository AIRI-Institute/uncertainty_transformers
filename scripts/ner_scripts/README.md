# NER experiments
To reproduce results from paper, run following scripts:
1. First run ner_hyp_subsampled_01.sh - experiments for 10 % of CoNLL-2003 for raw and CER model + SN models, MC-last/MC-all/Mahalanobis distance/SMD (all with optimal params)
2. After run scripts/miscl_scripts/metric_opt_3hyp.sh - experiments for 10 % of CoNLL-2003 for metric model with 3 params + metric SN model, MC-last/MC-all/Mahalanobis distance/SMD + DPP on masks/DPP with OOD for all models with optimal params

## Old scripts (could be used as examples)
1. ner_full_subsampled_to_01.sh - experiments for 10 % of CoNLL-2003 for model with and without regularization, MC-last/MC-all/DPP on masks/DPP with OOD
2. ner_maha_subsampled_to_01.sh - experiments for 10 % of CoNLL-2003 for model with and without regularization for Mahalanobis distance
3. ner_sn_maha_subsampled_to_01.sh - experiments for 10 % of CoNLL-2003 for model with and without regularization for Mahalanobis distance + spectral normalization for last layer
4. NerTable.ipynb - build tables for paper on NER results
5. ner_metric_maha.sh - experiments for 10 % of CoNLL-2003 for model with metric regularization for sampled Mahalanobis distance
6. ner_metric_subsampled_to_01.sh - experiments for 10 % of CoNLL-2003 for model with metric regularization, MC-last/MC-all/DPP on masks/DPP with OOD

##########

run_train_ensemble_models_ner.sh - train models for deep ensembles

run_train_models_ner.sh - train models for mc-dropout + mahalanobis

run_eval_ensemble_series_ner.sh - eval deep ensembles

run_eval_mc_fp_models_ner.sh - eval mc-dropout + mahalanobis
 