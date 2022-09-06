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
    #print(mains_wo_appliance_df.head())
    return mains_wo_appliance_df


def subtract_target_appliances_from_mains(mains_series: pd.Series):
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
    return mains_wo_appliances_df


def augment_timeseries(timeseries_df: pd.DataFrame):
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
    new_data = model.sample(1)
    print('New data')
    print(new_data.head())

    # TODO: Augment the appliance timeseries as well before appending to mains
    augmented_timeseries = timeseries_df_small.append(new_data)
    return augmented_timeseries


if __name__ == "__main__":
    """
    The meters for our target appliances are the following:
        - microwave: meter13
        - fridge: meter12
        - washing machine: meter5
        - dish washer: meter6
        - kettle: meter10
    """
    uk_dale = DataSet("/mnt/c/Users/gdialektakis/Desktop/torch-nilm-main/datasources/datasets/ukdale.h5")

    # Mains meter is meter54
    uk_dale_df = pd.read_hdf('/mnt/c/Users/gdialektakis/Desktop/torch-nilm-main/datasources/datasets/ukdale.h5',
                             key='/building1/elec/meter54')

    uk_dale_df = uk_dale_df.head(100)
    # Select electricity data from House 1
    elec = uk_dale.buildings[1].elec
    print('Submeters')
    print(elec)
    #print(elec.mains())
    # Prints mains power times series
    # print(elec.mains().power_series_all_data().head())
    mains = elec.mains()
    mains_series = elec.mains().power_series_all_data()

    # Subtract the power consumption of the 5 target appliances from mains
    #mains_wo_appliances = subtract_target_appliances_from_mains(mains_series)

    fridge_series = extract_appliance_timeseries(elec, 'washing machine')

    mains_series_aligned, fridge_series_aligned = align_timeseries(mains_series, fridge_series)

    print('Mains')
    # print(mains_series.head())
    # print(mains_series.tail())

    print('fridge')
    # print(fridge_series_aligned.head())
    # print(fridge_series_aligned.tail())

    mains_wo_fridge_df = subtract_from_mains(mains_series, fridge_series_aligned)

    print(mains_wo_fridge_df.head())

    # Augmented mains timeseries without fridge
    # TODO: Augment the appliance timeseries as well before appending to mains
    augmented_mains_wo_fridge = augment_timeseries(mains_wo_fridge_df)
