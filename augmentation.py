"""
This module performs timeseries augmentation based on
the existing UK-DALE data and generates a new augmented dataset.
Author: George Dialektakis - DataLab AUTh
Date: September 2022
"""

import pandas as pd
from sdv.timeseries import PAR
from sdv.demo import load_timeseries_demo
import numpy as np
import nilmtk
import matplotlib.pyplot as plt
from nilmtk import DataSet, MeterGroup
from matplotlib import rcParams
import pickle


# plt.style.use('ggplot')
# rcParams['figure.figsize'] = (13, 10)

def compute_kWh(power_in_watts):
    power = (power_in_watts * 0.000277) / 1000
    return power


def compute_total_energy(mains_before, mains_after):
    mains_before.columns = mains_before.columns.get_level_values(0)
    total_energy_before = mains_before['power'].sum()
    print("Total Energy of mains: {}".format(compute_kWh(total_energy_before)))

    mains_after.columns = mains_after.columns.get_level_values(0)
    total_energy_after = mains_after['power'].sum()
    print("Total Energy of reduced mains: {}".format(compute_kWh(total_energy_after)))


def align_timeseries(mains_series: np.array, appliance_series: np.array):
    """
    performs alignment of mains and appliance time-series
    input:
            mains_series: np.array
            appliance_series: np.array
    output:
            mains_series: np.array
            appliance_series: np.array
    """
    mains_series = mains_series[~mains_series.index.duplicated()]
    appliance_series = appliance_series[~appliance_series.index.duplicated()]
    ix = mains_series.index.intersection(appliance_series.index)
    mains_series = mains_series[ix]
    appliance_series = appliance_series[ix]
    return mains_series, appliance_series


def extract_appliance_timeseries(elec, appliance: str):
    appliance_meter = elec[appliance]
    appliance_series = appliance_meter.power_series_all_data()
    return appliance_series


# Probably obsolete....just for testing...
def subtract_from_mains(mains_series: pd.Series, appliances_series: pd.Series):
    mains_wo_appliance = mains_series.subtract(appliances_series, fill_value=0)
    mains_wo_appliance_small = mains_wo_appliance[:1000]
    pickle.dump(mains_wo_appliance_small, open('series_sample', 'wb'))

    mains_wo_appliance_df = mains_wo_appliance.to_frame()
    mains_wo_appliance_df['timestamp'] = mains_wo_appliance_df.index
    # print(mains_wo_appliance_df.head())
    return mains_wo_appliance_df


def subtract_target_appliances_from_mains(mains_series: pd.Series, elec):
    appliances = ['microwave', 'fridge', 'washing machine', 'dish washer', 'kettle']

    mains_wo_appliances = mains_series

    for appliance in appliances:
        print(appliance)
        appliance_series = extract_appliance_timeseries(elec, appliance)
        mains_series_aligned, appliance_series_aligned = align_timeseries(mains_series, appliance_series)
        mains_wo_appliances = mains_wo_appliances.subtract(appliance_series_aligned, fill_value=0)

    mains_wo_appliances_df = mains_wo_appliances.to_frame()
    mains_wo_appliances_df['timestamp'] = mains_wo_appliances_df.index
    print(mains_wo_appliances_df.head())

    # Calculate total energy in kWh for mains before and after subtraction
    mains_series_df = mains_series.to_frame()
    compute_total_energy(mains_series_df, mains_wo_appliances_df)

    return mains_wo_appliances_df


def augment_timeseries(timeseries_df: pd.DataFrame, elec, mains_series):
    timeseries_df['timestamp'] = timeseries_df['timestamp'].dt.tz_localize(None)
    timeseries_df_small = timeseries_df.head(10000)

    timeseries_df_small.columns = timeseries_df_small.columns.get_level_values(0)
    print(timeseries_df_small.head())

    sequence_index = 'timestamp'
    model = PAR(
        sequence_index=sequence_index,
    )
    # Train the model
    model.fit(timeseries_df_small)
    # Generate new synthetic timeseries
    synthetic_mains = model.sample(1)
    print('New data')
    print(synthetic_mains.head())

    # TODO: Check the following steps

    # Generate synthetic timeseries for each appliance
    microwave_series, fridge_series, \
    washing_machine_series, washer_series, kettle_series = augment_appliances_timeseries(elec, mains_series)

    # Align synthetic mains with synthetic appliance series for each appliance
    synthetic_mains['power'], microwave_series['power'] = align_timeseries(synthetic_mains['power'],
                                                                           microwave_series['power'])

    synthetic_mains['power'], fridge_series['power'] = align_timeseries(synthetic_mains['power'],
                                                                           fridge_series['power'])

    synthetic_mains['power'], washing_machine_series['power'] = align_timeseries(synthetic_mains['power'],
                                                                           washing_machine_series['power'])

    synthetic_mains['power'], washer_series['power'] = align_timeseries(synthetic_mains['power'],
                                                                           washer_series['power'])

    synthetic_mains['power'], kettle_series['power'] = align_timeseries(synthetic_mains['power'],
                                                                           kettle_series['power'])

    # Add synthetic mains with each synthetic appliance series
    synthetic_mains['power'].add(microwave_series['power'])
    synthetic_mains['power'].add(fridge_series['power'])
    synthetic_mains['power'].add(washing_machine_series['power'])
    synthetic_mains['power'].add(washer_series['power'])
    synthetic_mains['power'].add(kettle_series['power'])

    # Append synthetic mains with synthetic appliance series to original mains
    # TODO: Make this operation for the whole mains df, not only for the small one.
    augmented_timeseries = timeseries_df_small.append(synthetic_mains)
    return augmented_timeseries


def augment_appliances_timeseries(elec, mains_series):
    microwave_series = generate_synthetic_appliance_series('microwave', elec, mains_series)
    fridge_series = generate_synthetic_appliance_series('fridge', elec, mains_series)
    washing_machine_series = generate_synthetic_appliance_series('washing machine', elec, mains_series)
    washer_series = generate_synthetic_appliance_series('dish washer', elec, mains_series)
    kettle_series = generate_synthetic_appliance_series('kettle', elec, mains_series)

    return microwave_series, fridge_series, washing_machine_series, washer_series, kettle_series


def generate_synthetic_appliance_series(appliance, elec, mains_series):
    sequence_index = 'timestamp'
    appliance_series = extract_appliance_timeseries(elec, appliance)
    mains_series_aligned, appliance_series = align_timeseries(mains_series, appliance_series)

    appliance_series_df = appliance_series.to_frame()

    appliance_series_df['timestamp'] = appliance_series_df.index
    appliance_series_df['timestamp'] = appliance_series_df['timestamp'].dt.tz_localize(None)

    # transform multi-index dataframe to normal dataframe
    appliance_series_df.columns = appliance_series_df.columns.get_level_values(0)

    # Create PAR model and generate synthetic data
    model = PAR(
        sequence_index=sequence_index,
    )
    # Train the model
    model.fit(appliance_series_df)
    # Generate new synthetic timeseries
    new_data = model.sample(1)

    return new_data


def write_data_to_hdf5(augmented_timeseries):
    # TODO: Check if we also need to store the synthetic appliance series to the file
    augmented_timeseries.to_hdf('/mnt/c/Users/gdialektakis/Desktop/torch-nilm-main/datasources/datasets'
                               '/ukdale_augmented.h5',
                               key='/building1/elec/meter54', format='table')
    return


if __name__ == "__main__":
    """
    The meters for our target appliances are the following:
        - microwave: meter13
        - fridge: meter12
        - washing machine: meter5
        - dish washer: meter6
        - kettle: meter10
        
        - mains: meter54
    """
    uk_dale = DataSet("/mnt/c/Users/gdialektakis/Desktop/torch-nilm-main/datasources/datasets/ukdale.h5")
    # uk_dale_reduced = DataSet("/mnt/c/Users/gdialektakis/Desktop/torch-nilm-main/datasources/datasets/ukdale_reduced.h5")

    # Mains meter is meter54
    uk_dale_df = pd.read_hdf('/mnt/c/Users/gdialektakis/Desktop/torch-nilm-main/datasources/datasets/ukdale.h5',
                             key='/building1/elec/meter54')

    uk_dale_df = uk_dale_df.head(100)
    # Select electricity data from House 1
    elec = uk_dale.buildings[1].elec

    # elec_reduced = uk_dale_reduced.buildings[1].elec

    # print(elec.mains())
    # Prints mains power times series
    # print(elec.mains().power_series_all_data().head())

    mains = elec.mains()
    mains_series = elec.mains().power_series_all_data()
    print('Total energy before:')
    print(mains.total_energy())

    # Subtract the power consumption of the 5 target appliances from mains
    mains_wo_appliances = subtract_target_appliances_from_mains(mains_series, elec)

    mains_wo_appliances.to_hdf('/mnt/c/Users/gdialektakis/Desktop/torch-nilm-main/datasources/datasets'
                               '/ukdale_reduced.h5',
                               key='/building1/elec/meter54', format='table')

    print('completed!')

    # reduced_mains = elec_reduced.mains()
    # reduced_mains_series = reduced_mains.power_series_all_data()
    # print(reduced_mains.total_energy())

    fridge_series = extract_appliance_timeseries(elec, 'fridge')

    mains_series_aligned, fridge_series_aligned = align_timeseries(mains_series, fridge_series)

    mains_wo_fridge_df = subtract_from_mains(mains_series, fridge_series_aligned)

    print(mains_wo_fridge_df.head())

    # Augmented mains timeseries without fridge
    # TODO: Augment the appliance timeseries as well before appending to mains
    augmented_mains_wo_fridge = augment_timeseries(mains_wo_fridge_df, elec, mains_series)

    augmented_timeseries = augment_timeseries(mains_wo_fridge_df, elec, mains_series)

    # Store augmented time series to .h5 file for later usage
    write_data_to_hdf5(augmented_timeseries)
