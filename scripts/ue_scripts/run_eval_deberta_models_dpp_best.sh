cd ../../src
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_conll2003.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.subsample_perc_val\=0.1 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=20 ue.dropout.max_frac\=0.3 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=False training.num_train_epochs\=12 training.learning_rate\=5e-05 training.per_device_train_batch_size\=32 +training.weight_decay\=0 +ue.margin\=5.0 +ue.lamb_intra\=0.008 ue.lamb\=0.008 +ue.use_paper_version\=True' config_path=../configs/conll2003.yaml output_dir=../workdir/run_tasks_for_model_series/deberta_raw_no_sn/conll2003/0.1/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir=../workdir/run_tasks_for_model_series/deberta_raw_no_sn/conll2003/0.1/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/deberta_raw_no_sn/conll2003/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_conll2003.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.subsample_perc_val\=0.1 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=20 ue.dropout.max_frac\=0.45 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=True training.num_train_epochs\=12 training.learning_rate\=5e-05 training.per_device_train_batch_size\=32 +training.weight_decay\=0 +ue.margin\=5.0 +ue.lamb_intra\=0.008 ue.lamb\=0.008 +ue.use_paper_version\=True' config_path=../configs/conll2003.yaml output_dir=../workdir/run_tasks_for_model_series/deberta_raw_no_sn/conll2003/0.1/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_raw_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir=../workdir/run_tasks_for_model_series/deberta_raw_no_sn/conll2003/0.1/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/deberta_raw_no_sn/conll2003/ddpp_ood;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_conll2003.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.subsample_perc_val\=0.1 ue.use_selective\=True ue.reg_type\=reg-curr training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=20 ue.dropout.max_frac\=0.3 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=False training.num_train_epochs\=13 training.learning_rate\=2e-05 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1 +ue.margin\=0.025 +ue.lamb_intra\=0.1 ue.lamb\=0.2 +ue.use_paper_version\=True' config_path=../configs/conll2003.yaml output_dir=../workdir/run_tasks_for_model_series/deberta_reg_no_sn/conll2003/0.1/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_reg_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir=../workdir/run_tasks_for_model_series/deberta_reg_no_sn/conll2003/0.1/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/deberta_reg_no_sn/conll2003/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_conll2003.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.subsample_perc_val\=0.1 ue.use_selective\=True ue.reg_type\=reg-curr training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=20 ue.dropout.max_frac\=0.45 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=True training.num_train_epochs\=13 training.learning_rate\=2e-05 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1 +ue.margin\=0.025 +ue.lamb_intra\=0.1 ue.lamb\=0.2 +ue.use_paper_version\=True' config_path=../configs/conll2003.yaml output_dir=../workdir/run_tasks_for_model_series/deberta_reg_no_sn/conll2003/0.1/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_reg_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir=../workdir/run_tasks_for_model_series/deberta_reg_no_sn/conll2003/0.1/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/deberta_reg_no_sn/conll2003/ddpp_ood;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_conll2003.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.subsample_perc_val\=0.1 ue.use_selective\=True ue.reg_type\=metric training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=20 ue.dropout.max_frac\=0.3 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=False training.num_train_epochs\=7 training.learning_rate\=5e-05 training.per_device_train_batch_size\=8 +training.weight_decay\=0.01 +ue.margin\=0.5 +ue.lamb_intra\=0.01 ue.lamb\=0.002 +ue.use_paper_version\=True' config_path=../configs/conll2003.yaml output_dir=../workdir/run_tasks_for_model_series/deberta_metric_no_sn/conll2003/0.1/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_metric_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir=../workdir/run_tasks_for_model_series/deberta_metric_no_sn/conll2003/0.1/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/deberta_metric_no_sn/conll2003/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_conll2003.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.subsample_perc_val\=0.1 ue.use_selective\=True ue.reg_type\=metric training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=20 ue.dropout.max_frac\=0.45 ue.dropout.mask_name_for_mask\=rbf ue.committee_size\=100 ue.dropout.use_ood_sampling\=True training.num_train_epochs\=7 training.learning_rate\=5e-05 training.per_device_train_batch_size\=8 +training.weight_decay\=0.01 +ue.margin\=0.5 +ue.lamb_intra\=0.01 ue.lamb\=0.002 +ue.use_paper_version\=True' config_path=../configs/conll2003.yaml output_dir=../workdir/run_tasks_for_model_series/deberta_metric_no_sn/conll2003/0.1/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_metric_no_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir=../workdir/run_tasks_for_model_series/deberta_metric_no_sn/conll2003/0.1/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/deberta_metric_no_sn/conll2003/ddpp_ood;