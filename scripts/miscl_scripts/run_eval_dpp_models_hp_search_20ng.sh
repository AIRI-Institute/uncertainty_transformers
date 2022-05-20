cd ../../src
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_20newsgroups_for_model_series.py cuda_devices=[2,7] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.3 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/20newsgroups.yaml output_dir=../workdir/run_glue_for_model_series/electra_raw_no_sn/20newsgroups/0.0/ddpp_dpp_0.3_50/ model_series_dir=/notebook/uncertainty-estimation/workdir/run_train_models_dpp_hp/electra_raw_no_sn/20newsgroups/0.0/models/20newsgroups
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_20newsgroups_for_model_series.py cuda_devices=[2,7] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.3 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/20newsgroups.yaml output_dir=../workdir/run_glue_for_model_series/electra_raw_no_sn/20newsgroups/0.0/ddpp_ood_0.3_50/ model_series_dir=/notebook/uncertainty-estimation/workdir/run_train_models_dpp_hp/electra_raw_no_sn/20newsgroups/0.0/models/20newsgroups
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_20newsgroups_for_model_series.py cuda_devices=[2,7] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.4 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/20newsgroups.yaml output_dir=../workdir/run_glue_for_model_series/electra_raw_no_sn/20newsgroups/0.0/ddpp_dpp_0.4_50/ model_series_dir=/notebook/uncertainty-estimation/workdir/run_train_models_dpp_hp/electra_raw_no_sn/20newsgroups/0.0/models/20newsgroups
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_20newsgroups_for_model_series.py cuda_devices=[2,7] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.4 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/20newsgroups.yaml output_dir=../workdir/run_glue_for_model_series/electra_raw_no_sn/20newsgroups/0.0/ddpp_ood_0.4_50/ model_series_dir=/notebook/uncertainty-estimation/workdir/run_train_models_dpp_hp/electra_raw_no_sn/20newsgroups/0.0/models/20newsgroups
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_20newsgroups_for_model_series.py cuda_devices=[2,7] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.5 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/20newsgroups.yaml output_dir=../workdir/run_glue_for_model_series/electra_raw_no_sn/20newsgroups/0.0/ddpp_dpp_0.5_50/ model_series_dir=/notebook/uncertainty-estimation/workdir/run_train_models_dpp_hp/electra_raw_no_sn/20newsgroups/0.0/models/20newsgroups
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_20newsgroups_for_model_series.py cuda_devices=[2,7] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.5 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/20newsgroups.yaml output_dir=../workdir/run_glue_for_model_series/electra_raw_no_sn/20newsgroups/0.0/ddpp_ood_0.5_50/ model_series_dir=/notebook/uncertainty-estimation/workdir/run_train_models_dpp_hp/electra_raw_no_sn/20newsgroups/0.0/models/20newsgroups
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_20newsgroups_for_model_series.py cuda_devices=[2,7] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.6 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/20newsgroups.yaml output_dir=../workdir/run_glue_for_model_series/electra_raw_no_sn/20newsgroups/0.0/ddpp_dpp_0.6_50/ model_series_dir=/notebook/uncertainty-estimation/workdir/run_train_models_dpp_hp/electra_raw_no_sn/20newsgroups/0.0/models/20newsgroups
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_20newsgroups_for_model_series.py cuda_devices=[2,7] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.6 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/20newsgroups.yaml output_dir=../workdir/run_glue_for_model_series/electra_raw_no_sn/20newsgroups/0.0/ddpp_ood_0.6_50/ model_series_dir=/notebook/uncertainty-estimation/workdir/run_train_models_dpp_hp/electra_raw_no_sn/20newsgroups/0.0/models/20newsgroups