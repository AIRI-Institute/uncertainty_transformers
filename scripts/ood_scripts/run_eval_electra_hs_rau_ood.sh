cd ../../src
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mahalanobis ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=imdb ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/imdb/mahalanobis model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mahalanobis ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=trec ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/trec/mahalanobis model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mahalanobis ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=wmt16 ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/wmt16/mahalanobis model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mahalanobis ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=amazon ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/amazon/mahalanobis model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mahalanobis ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=mnli ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/mnli/mahalanobis model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mahalanobis ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=rte ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/rte/mahalanobis model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mahalanobis ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=20newsgroups ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/20newsgroups/mahalanobis model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mc ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.dropout_subs\=all ue.use_cache\=False data.validation_subsample\=0.0 data.ood_data\=imdb ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/imdb/mc model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mc ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.dropout_subs\=all ue.use_cache\=False data.validation_subsample\=0.0 data.ood_data\=trec ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/trec/mc model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mc ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.dropout_subs\=all ue.use_cache\=False data.validation_subsample\=0.0 data.ood_data\=wmt16 ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/wmt16/mc model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mc ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.dropout_subs\=all ue.use_cache\=False data.validation_subsample\=0.0 data.ood_data\=amazon ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/amazon/mc model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mc ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.dropout_subs\=all ue.use_cache\=False data.validation_subsample\=0.0 data.ood_data\=mnli ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/mnli/mc model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mc ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.dropout_subs\=all ue.use_cache\=False data.validation_subsample\=0.0 data.ood_data\=rte ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/rte/mc model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=mc ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.dropout_subs\=all ue.use_cache\=False data.validation_subsample\=0.0 data.ood_data\=20newsgroups ue.calibrate\=False +ue.use_hs_labels\=True training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/20newsgroups/mc model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=nuq ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=imdb ue.calibrate\=False +ue.use_hs_labels\=True ue.nuq.n_neighbors\=60 ue.nuq.log_pN\=0 +ue.nuq.n_points\=50 +ue.nuq.n_folds\=20 +ue.nuq.n_samples\=5 training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/imdb/nuq model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=nuq ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=trec ue.calibrate\=False +ue.use_hs_labels\=True ue.nuq.n_neighbors\=60 ue.nuq.log_pN\=0 +ue.nuq.n_points\=50 +ue.nuq.n_folds\=20 +ue.nuq.n_samples\=5 training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/trec/nuq model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=nuq ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=wmt16 ue.calibrate\=False +ue.use_hs_labels\=True ue.nuq.n_neighbors\=60 ue.nuq.log_pN\=0 +ue.nuq.n_points\=50 +ue.nuq.n_folds\=20 +ue.nuq.n_samples\=5 training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/wmt16/nuq model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=nuq ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=amazon ue.calibrate\=False +ue.use_hs_labels\=True ue.nuq.n_neighbors\=60 ue.nuq.log_pN\=0 +ue.nuq.n_points\=50 +ue.nuq.n_folds\=20 +ue.nuq.n_samples\=5 training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/amazon/nuq model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=nuq ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=mnli ue.calibrate\=False +ue.use_hs_labels\=True ue.nuq.n_neighbors\=60 ue.nuq.log_pN\=0 +ue.nuq.n_points\=50 +ue.nuq.n_folds\=20 +ue.nuq.n_samples\=5 training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/mnli/nuq model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=nuq ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=rte ue.calibrate\=False +ue.use_hs_labels\=True ue.nuq.n_neighbors\=60 ue.nuq.log_pN\=0 +ue.nuq.n_points\=50 +ue.nuq.n_folds\=20 +ue.nuq.n_samples\=5 training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/rte/nuq model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py args='ue\=nuq ue.use_selective\=True ue.reg_type\=rau do_ue_estimate\=True ue.use_cache\=True data.validation_subsample\=0.0 data.ood_data\=20newsgroups ue.calibrate\=False +ue.use_hs_labels\=True ue.nuq.n_neighbors\=60 ue.nuq.log_pN\=0 +ue.nuq.n_points\=50 +ue.nuq.n_folds\=20 +ue.nuq.n_samples\=5 training\=electra_base' config_path=../configs/sst2_ood.yaml output_dir=../workdir/run_tasks_for_model_series/electra_hs_rau_no_sn/sst2/20newsgroups/nuq model_series_dir=../workdir/run_train_models/electra_hs_rau_no_sn/sst2/0.0/models/sst2