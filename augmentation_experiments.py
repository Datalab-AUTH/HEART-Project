from lab.nilm_experiments import *
from constants.constants import *
from constants.enumerates import *
import timeit

start = timeit.default_timer()

experiment_parameters = {
    EPOCHS: 20,
    ITERATIONS: 5,
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
    ElectricalAppliances.WASHING_MACHINE,
    ElectricalAppliances.DISH_WASHER
]

experiment_categories = [
    SupportedExperimentCategories.SINGLE_CATEGORY,
]


# DAE's input dim must be equal to input sequence length = window size
model_hparams = [
    {
        'model_name': 'SimpleGru',
        'hparams': {},
    },
    {
        'model_name': 'WGRU',
        'hparams': {'dropout': 0.2},
    },
    {
        'model_name': 'S2P',
        'hparams': {'window_size': None},
    }

]

model_hparams = ModelHyperModelParameters(model_hparams)
experiment_parameters = ExperimentParameters(**experiment_parameters)

print('Training only on original data')

experiment = NILMExperiments(project_name='baseline', clean_project=True,
                             devices=devices, save_timeseries_results=True,
                             experiment_categories=experiment_categories,
                             experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                             experiment_parameters=experiment_parameters,
                             save_model=True, export_plots=True, experiment_type=SupportedNilmExperiments.BENCHMARK
                             )
experiment.run_benchmark(model_hparams=model_hparams)

experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)

""" 
print('Training on augmented data')

experiment = NILMExperiments(project_name='augmented', clean_project=True,
                             devices=devices, save_timeseries_results=True,
                             experiment_categories=experiment_categories,
                             experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                             experiment_parameters=experiment_parameters,
                             save_model=True, export_plots=True, experiment_type=SupportedNilmExperiments.BENCHMARK
                             )
experiment.run_benchmark(model_hparams=model_hparams)

experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)
"""
