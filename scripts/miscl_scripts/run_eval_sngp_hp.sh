cd ../../src
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1e-05 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/1e-05_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/1e-05_0.999_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1e-05 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/1e-05_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/1e-05_0.99_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1e-05 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/1e-05_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/1e-05_0.9_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.0001 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.0001_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.0001_0.999_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.0001 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.0001_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.0001_0.99_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.0001 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.0001_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.0001_0.9_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.001 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.001_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.001_0.999_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.001 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.001_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.001_0.99_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.001 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.001_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.001_0.9_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.01 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.01_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.01_0.999_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.01 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.01_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.01_0.99_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.01 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.01_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.01_0.9_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.1 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.1_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.1_0.999_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.1 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.1_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.1_0.99_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.1 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/0.1_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/0.1_0.9_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/1_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/1_0.999_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/1_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/1_0.99_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/mrpc/1_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/mrpc/1_0.9_0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1e-05 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/1e-05_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/1e-05_0.999_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1e-05 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/1e-05_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/1e-05_0.99_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1e-05 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/1e-05_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/1e-05_0.9_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.0001 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.0001_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.0001_0.999_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.0001 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.0001_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.0001_0.99_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.0001 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.0001_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.0001_0.9_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.001 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.001_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.001_0.999_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.001 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.001_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.001_0.99_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.001 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.001_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.001_0.9_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.01 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.01_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.01_0.999_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.01 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.01_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.01_0.99_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.01 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.01_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.01_0.9_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.1 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.1_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.1_0.999_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.1 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.1_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.1_0.99_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.1 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/0.1_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/0.1_0.9_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/1_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/1_0.999_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/1_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/1_0.99_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/cola/1_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/cola/1_0.9_0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1e-05 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/1e-05_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/1e-05_0.999_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1e-05 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/1e-05_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/1e-05_0.99_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1e-05 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/1e-05_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/1e-05_0.9_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.0001 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.0001_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.0001_0.999_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.0001 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.0001_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.0001_0.99_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.0001 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.0001_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.0001_0.9_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.001 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.001_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.001_0.999_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.001 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.001_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.001_0.99_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.001 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.001_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.001_0.9_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.01 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.01_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.01_0.999_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.01 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.01_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.01_0.99_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.01 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.01_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.01_0.9_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.1 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.1_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.1_0.999_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.1 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.1_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.1_0.99_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=0.1 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/0.1_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/0.1_0.9_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1 ue.sngp.momentum\=0.999 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/1_0.999_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/1_0.999_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1 ue.sngp.momentum\=0.99 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/1_0.99_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/1_0.99_0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1,2,3] args='ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=True data.validation_subsample\=0.0 +data.eval_subsample\=0.2 ue.sngp.ridge_factor\=1 ue.sngp.momentum\=0.9 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series/electra-raw-sngp/sst2/1_0.9_0.0 model_series_dir=../workdir/run_train_models/electra-raw-sngp/sst2/1_0.9_0.0/models/sst2