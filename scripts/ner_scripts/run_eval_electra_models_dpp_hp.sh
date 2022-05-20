cd ../../src
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.3 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_dpp_rbf_0.3_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.3 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_ood_rbf_0.3_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.35 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_dpp_rbf_0.35_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.35 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_ood_rbf_0.35_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.4 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_dpp_rbf_0.4_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.4 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_ood_rbf_0.4_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.45 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_dpp_rbf_0.45_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.45 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_ood_rbf_0.45_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.5 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_dpp_rbf_0.5_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.5 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_ood_rbf_0.5_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.55 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_dpp_rbf_0.55_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.55 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_ood_rbf_0.55_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.6 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_dpp_rbf_0.6_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc-dpp do_ue_estimate\=True +ue.use_spectralnorm\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=False training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.6 ue.dropout.committee_size\=20 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/conll2003.yaml script=run_conll2003_dpp_hp.py output_dir=../workdir/run_tasks_for_model_series_dpp_hp/electra_raw_no_sn/conll2003/0.1/ddpp_ood_rbf_0.6_20/ model_series_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/conll2003/0.1/models/conll2003