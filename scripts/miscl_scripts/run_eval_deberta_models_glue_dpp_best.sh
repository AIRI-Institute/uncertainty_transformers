cd ../../src
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/deberta_raw_no_sn/mrpc/0.0/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_raw_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/deberta_raw_no_sn/mrpc/0.0/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_raw_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=reg-curr training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/deberta_reg_no_sn/mrpc/0.0/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_reg_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=reg-curr training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/deberta_reg_no_sn/mrpc/0.0/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_reg_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=metric training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/deberta_metric_no_sn/mrpc/0.0/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_metric_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=metric training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/deberta_metric_no_sn/mrpc/0.0/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_metric_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/deberta_raw_no_sn/cola/0.0/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_raw_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/deberta_raw_no_sn/cola/0.0/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_raw_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=reg-curr training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/deberta_reg_no_sn/cola/0.0/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_reg_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=reg-curr training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/deberta_reg_no_sn/cola/0.0/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_reg_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=metric training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/deberta_metric_no_sn/cola/0.0/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_metric_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=metric training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/deberta_metric_no_sn/cola/0.0/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_metric_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.3 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/deberta_raw_no_sn/sst2/0.0/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_raw_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=False ue.reg_type\=raw training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/deberta_raw_no_sn/sst2/0.0/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_raw_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=reg-curr training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.3 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/deberta_reg_no_sn/sst2/0.0/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_reg_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=reg-curr training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/deberta_reg_no_sn/sst2/0.0/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_reg_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=metric training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.3 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/deberta_metric_no_sn/sst2/0.0/ddpp_dpp/ model_series_dir=../workdir/run_train_models/deberta_metric_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 ue.use_selective\=True ue.reg_type\=metric training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.committee_size\=50 ue.dropout.max_frac\=0.6 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/deberta_metric_no_sn/sst2/0.0/ddpp_ood/ model_series_dir=../workdir/run_train_models/deberta_metric_no_sn/sst2/0.0/models/sst2