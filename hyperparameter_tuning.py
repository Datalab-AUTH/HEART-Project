from lab.nilm_experiments import *
from constants.constants import *
from constants.enumerates import *
import timeit

start = timeit.default_timer()

experiment_parameters = {
    EPOCHS: 1,
    ITERATIONS: 1,
    INFERENCE_CPU: False,
    SAMPLE_PERIOD: 6,
    BATCH_SIZE: 1024,
    ITERABLE_DATASET: False,
    PREPROCESSING_METHOD: SupportedPreprocessingMethods.ROLLING_WINDOW,
    FIXED_WINDOW: 100,
    FILLNA_METHOD: None,
    SUBSEQ_WINDOW: None,
    TRAIN_TEST_SPLIT: 0.8,
    NOISE_FACTOR: 0,
}


devices = [
    ElectricalAppliances.MICROWAVE,
    ElectricalAppliances.FRIDGE,
    ElectricalAppliances.WASHING_MACHINE,
    ElectricalAppliances.DISH_WASHER,
    ElectricalAppliances.KETTLE
]

experiment_categories = [
    SupportedExperimentCategories.SINGLE_CATEGORY,
]

# DAE's input dim must be equal to input sequence length = window size
model_hparams = [
    {
        'model_name': 'S2P',
        'hparams': {'window_size': None},
    }
]


hparam_tuning = [

    {
        'model_name': 'S2P',
        'hparams': [
            {'window_size': None, 'dropout': 0.5},
            {'window_size': None, 'dropout': 0},
        ]
    },
]


model_hparams = ModelHyperModelParameters(model_hparams)

hparam_tuning = HyperParameterTuning(hparam_tuning)
experiment_parameters = ExperimentParameters(**experiment_parameters)

experiment = NILMExperiments(project_name='hparam_tuning', clean_project=False,
                             devices=devices, save_timeseries_results=False,
                             experiment_categories=experiment_categories,
                             experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                             experiment_parameters=experiment_parameters,
                             save_model=True, export_plots=True,
                             experiment_type=SupportedNilmExperiments.HYPERPARAM_TUNE_CV)

experiment.run_hyperparameter_tuning_cross_validation(hparam_tuning=hparam_tuning)

experiment.export_report(hparam_tuning=hparam_tuning, experiment_type=SupportedNilmExperiments.HYPERPARAM_TUNE_CV)

# experiment.run_cross_validation(model_hparams=model_hparams)
# experiment.run_hyperparameter_tuning_cross_validation(hparam_tuning=hparam_tuning)
# experiment.export_report(hparam_tuning=hparam_tuning, experiment_type=SupportedNilmExperiments.HYPERPARAM_TUNE_CV)

stop = timeit.default_timer()
print('\nExecution Time (minutes): ', (stop-start)/60)