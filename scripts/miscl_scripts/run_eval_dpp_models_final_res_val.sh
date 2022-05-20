cd ../../src
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.1 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.5 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/mrpc.yaml output_dir=../workdir/final_res/run_glue_for_model_series/electra-raw/mrpc/0.1/ddpp_dpp/ model_series_dir=../../mlspace/run_train_models/electra-raw-False/mrpc/0.1/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.1 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.4 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/mrpc.yaml output_dir=../workdir/final_res/run_glue_for_model_series/electra-raw/mrpc/0.1/ddpp_ood/ model_series_dir=../../mlspace/run_train_models/electra-raw-False/mrpc/0.1/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.1 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.3 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/cola.yaml output_dir=../workdir/final_res/run_glue_for_model_series/electra-raw/cola/0.1/ddpp_dpp/ model_series_dir=../../mlspace/run_train_models/electra-raw-False/cola/0.1/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.1 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.6 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/cola.yaml output_dir=../workdir/final_res/run_glue_for_model_series/electra-raw/cola/0.1/ddpp_ood/ model_series_dir=../../mlspace/run_train_models/electra-raw-False/cola/0.1/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.1 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.5 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False' config_path=../configs/sst2.yaml output_dir=../workdir/final_res/run_glue_for_model_series/electra-raw/sst2/0.1/ddpp_dpp/ model_series_dir=../../mlspace/run_train_models/electra-raw-False/sst2/0.1/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2] args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.1 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.6 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True' config_path=../configs/sst2.yaml output_dir=../workdir/final_res/run_glue_for_model_series/electra-raw/sst2/0.1/ddpp_ood/ model_series_dir=../../mlspace/run_train_models/electra-raw-False/sst2/0.1/models/sst2