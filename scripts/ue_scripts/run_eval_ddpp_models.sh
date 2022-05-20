cd ../../src
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.55 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False +ue.use_paper_version\=True' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/mrpc/0.0/ddpp_dpp model_series_dir=../workdir/run_train_models/electra_raw_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/mrpc/0.0/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_no_sn/mrpc/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.4 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True +ue.use_paper_version\=True' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/mrpc/0.0/ddpp_ood model_series_dir=../workdir/run_train_models/electra_raw_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/mrpc/0.0/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_no_sn/mrpc/ddpp_ood;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.55 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False +ue.use_paper_version\=True ue.reg_type\=reg-curr ue.use_selective\=True' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/mrpc/0.0/ddpp_dpp model_series_dir=../workdir/run_train_models/electra_reg_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/mrpc/0.0/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_reg_no_sn/mrpc/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.4 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True +ue.use_paper_version\=True ue.reg_type\=reg-curr ue.use_selective\=True' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/mrpc/0.0/ddpp_ood model_series_dir=../workdir/run_train_models/electra_reg_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/mrpc/0.0/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_reg_no_sn/mrpc/ddpp_ood;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.55 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False +ue.use_paper_version\=True ue.reg_type\=metric ue.use_selective\=True' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/mrpc/0.0/ddpp_dpp model_series_dir=../workdir/run_train_models/electra_metric_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/mrpc/0.0/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_metric_no_sn/mrpc/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.4 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True +ue.use_paper_version\=True ue.reg_type\=metric ue.use_selective\=True' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/mrpc/0.0/ddpp_ood model_series_dir=../workdir/run_train_models/electra_metric_no_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/mrpc/0.0/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_metric_no_sn/mrpc/ddpp_ood;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.4 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False +ue.use_paper_version\=True' config_path=../configs/cola.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/cola/0.0/ddpp_dpp model_series_dir=../workdir/run_train_models/electra_raw_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/cola/0.0/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_no_sn/cola/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.45 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True +ue.use_paper_version\=True' config_path=../configs/cola.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/cola/0.0/ddpp_ood model_series_dir=../workdir/run_train_models/electra_raw_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/cola/0.0/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_no_sn/cola/ddpp_ood;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.4 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False +ue.use_paper_version\=True ue.reg_type\=reg-curr ue.use_selective\=True' config_path=../configs/cola.yaml output_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/cola/0.0/ddpp_dpp model_series_dir=../workdir/run_train_models/electra_reg_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/cola/0.0/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_reg_no_sn/cola/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.45 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True +ue.use_paper_version\=True ue.reg_type\=reg-curr ue.use_selective\=True' config_path=../configs/cola.yaml output_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/cola/0.0/ddpp_ood model_series_dir=../workdir/run_train_models/electra_reg_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/cola/0.0/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_reg_no_sn/cola/ddpp_ood;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.4 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False +ue.use_paper_version\=True ue.reg_type\=metric ue.use_selective\=True' config_path=../configs/cola.yaml output_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/cola/0.0/ddpp_dpp model_series_dir=../workdir/run_train_models/electra_metric_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/cola/0.0/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_metric_no_sn/cola/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.45 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True +ue.use_paper_version\=True ue.reg_type\=metric ue.use_selective\=True' config_path=../configs/cola.yaml output_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/cola/0.0/ddpp_ood model_series_dir=../workdir/run_train_models/electra_metric_no_sn/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/cola/0.0/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_metric_no_sn/cola/ddpp_ood;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.45 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False +ue.use_paper_version\=True' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/sst2/0.0/ddpp_dpp model_series_dir=../workdir/run_train_models/electra_raw_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/sst2/0.0/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_no_sn/sst2/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.35 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True +ue.use_paper_version\=True' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/sst2/0.0/ddpp_ood model_series_dir=../workdir/run_train_models/electra_raw_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_no_sn/sst2/0.0/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_no_sn/sst2/ddpp_ood;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.45 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False +ue.use_paper_version\=True ue.reg_type\=reg-curr ue.use_selective\=True' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/sst2/0.0/ddpp_dpp model_series_dir=../workdir/run_train_models/electra_reg_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/sst2/0.0/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_reg_no_sn/sst2/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.35 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True +ue.use_paper_version\=True ue.reg_type\=reg-curr ue.use_selective\=True' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/sst2/0.0/ddpp_ood model_series_dir=../workdir/run_train_models/electra_reg_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_reg_no_sn/sst2/0.0/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_reg_no_sn/sst2/ddpp_ood;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.45 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=False +ue.use_paper_version\=True ue.reg_type\=metric ue.use_selective\=True' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/sst2/0.0/ddpp_dpp model_series_dir=../workdir/run_train_models/electra_metric_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/sst2/0.0/ddpp_dpp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_metric_no_sn/sst2/ddpp_dpp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=mc-dpp do_ue_estimate\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout.is_reused_mask\=True ue.dropout.max_frac\=0.35 ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.use_ood_sampling\=True +ue.use_paper_version\=True ue.reg_type\=metric ue.use_selective\=True' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/sst2/0.0/ddpp_ood model_series_dir=../workdir/run_train_models/electra_metric_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_metric_no_sn/sst2/0.0/ddpp_ood extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_metric_no_sn/sst2/ddpp_ood;