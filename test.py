import pandas as pd
from sdv.timeseries import PAR
from sdv.demo import load_timeseries_demo
import numpy as np
import nilmtk
import matplotlib.pyplot as plt
from nilmtk import DataSet, MeterGroup
from matplotlib import rcParams


# plt.style.use('ggplot')
# rcParams['figure.figsize'] = (13, 10)


def align_timeseries(mains_series: np.array, appliance_series: np.array):
    mains_series = mains_series[~mains_series.index.duplicated()]
    appliance_series = appliance_series[~appliance_series.index.duplicated()]
    ix = mains_series.index.intersection(appliance_series.index)
    mains_series = mains_series[ix]
    appliance_series = appliance_series[ix]
    return mains_series, appliance_series


def par_model():
    data = load_timeseries_demo()
    print(data.head())
    entity_columns = ['Symbol']
    context_columns = ['MarketCap', 'Sector', 'Industry']
    sequence_index = 'Date'

    model = PAR(
        entity_columns=entity_columns,
        context_columns=context_columns,
        sequence_index=sequence_index
    )

    model.fit(data)
    new_data = model.sample(1)
    print(new_data.head())


def extract_appliance_timeseries(elec, appliance: str):
    appliance_meter = elec[appliance]
    appliance_series = appliance_meter.power_series_all_data()
    return appliance_series


def subtract_from_mains(mains_series: pd.Series, appliances_series: pd.Series):
    mains_wo_appliance = mains_series.subtract(appliances_series, fill_value=0)
    mains_wo_appliance_df = mains_wo_appliance.to_frame()
    mains_wo_fridge_df['timestamp'] = mains_wo_appliance_df.index
    print(mains_wo_appliance_df.head())
    return mains_wo_appliance_df


def augment_timeseries(timeseries_df: pd.DataFrame):
    timeseries_df['timestamp'] = timeseries_df['timestamp'].dt.tz_localize(None)
    sequence_index = 'timestamp'
    model = PAR(
        sequence_index=sequence_index,
    )
    # Train the model
    model.fit(timeseries_df.head(10000))
    # Generate new synthetic timeseries
    new_data = model.sample(1)
    print(new_data.head())

    # TODO: Augment the appliance timeseries as well before appending to mains
    augmented_timeseries = timeseries_df.append(new_data)
    return augmented_timeseries


if __name__ == "__main__":
    # par_model()
    uk_dale = DataSet("/mnt/c/Users/gdialektakis/Desktop/torch-nilm-main/datasources/datasets/ukdale.h5")
    # Select electricity data from House 1
    elec = uk_dale.buildings[1].elec

    print(elec.mains())
    # Prints mains power times series
    # print(elec.mains().power_series_all_data().head())
    mains = elec.mains()

    mains_series = elec.mains().power_series_all_data()

    fridge_series = extract_appliance_timeseries(elec, 'fridge')

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
