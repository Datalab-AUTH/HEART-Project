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


def align_timeseries(mains_series: np.array, fridge_series: np.array):
    mains_series = mains_series[~mains_series.index.duplicated()]
    fridge_series = fridge_series[~fridge_series.index.duplicated()]
    ix = mains_series.index.intersection(fridge_series.index)
    mains_series = mains_series[ix]
    fridge_series = fridge_series[ix]
    return mains_series, fridge_series


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


if __name__ == "__main__":
    #par_model()
    uk_dale = DataSet("/mnt/c/Users/gdialektakis/Desktop/torch-nilm-main/datasources/Datasets/ukdale.h5")
    elec = uk_dale.buildings[1].elec

    print(elec.mains())
    # Prints mains power times series
    # print(elec.mains().power_series_all_data().head())
    mains = elec.mains()
    fridge_meter = elec['fridge']

    df = next(fridge_meter.load())

    # fridge_meter.plot()
    # elec.draw_wiring_graph()
    # plt.show()

    mains_series = elec.mains().power_series_all_data()
    fridge_series = fridge_meter.power_series_all_data()

    mains_series_aligned, fridge_series_aligned = align_timeseries(mains_series, fridge_series)

    print('Mains')
    # print(mains_series.head())
    # print(mains_series.tail())

    print('fridge')
    # print(fridge_series_aligned.head())
    # print(fridge_series_aligned.tail())

    mains_wo_fridge = mains_series.subtract(fridge_series_aligned, fill_value=0)
    mains_wo_fridge_df = mains_wo_fridge.to_frame()
    mains_wo_fridge_df['timestamp'] = mains_wo_fridge_df.index
    print(mains_wo_fridge_df.head())

    mains_wo_fridge_df['timestamp'] = mains_wo_fridge_df['timestamp'].dt.tz_localize(None)

    print(mains_wo_fridge_df.head())

    sequence_index = 'timestamp'
    model = PAR(
        sequence_index=sequence_index
    )

    model.fit(mains_wo_fridge_df.head(10000))
    new_data = model.sample(1)
    print(new_data.head())
