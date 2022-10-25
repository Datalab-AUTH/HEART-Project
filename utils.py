import pandas as pd
import pickle
import datetime


def series_to_df(series):
    df = series.to_frame()
    df['timestamp'] = df.index
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    # transform multi-index dataframe to normal dataframe
    df.columns = df.columns.get_level_values(0)
    return df


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


def add_datetime(df, initial_datetime):
    df['timestamp'][0] = initial_datetime
    for i in range(len(df)):
        df['timestamp'][i + 1] = initial_datetime + datetime.timedelta(seconds=i + 1)

    return df


def duplicate_df():
    for i in range(0, 30):
        mains_filename = './datasources/dataframes/mains/day_' + str(i) + '.pkl'
        mains_df = pd.read_pickle(mains_filename)
        mains_df.append(mains_df)

        mains_df.to_pickle(mains_filename)
        appliances = ['dish_washer', 'washing_machine']
        for appliance in appliances:
            appl_filename = './datasources/dataframes/appliances/' + str(appliance) + '/day' + str(i) + '.pkl'
            appl_df = pd.read_pickle(appl_filename)
            appl_df.append(appl_df)

            appl_df.to_pickle(appl_filename)

    return


def create_mains_dataframe():
    # Reads the pickle files of 30 days, assigns timestamp and appends to one final dataframe of mains
    mains_df = pd.DataFrame()
    initial_datetime = datetime.datetime(2015, 10, 5, 18, 00, 00, 100000)
    for i in range(0, 30):
        print('Dataframe {}'.format(i))
        filename = './datasources/dataframes/mains/day_' + str(i) + '.pkl'
        temp_df = pd.read_pickle(filename)

        # Assign datetime
        # Add one day = 24h for each separate file
        initial_datetime = initial_datetime + datetime.timedelta(hours=24)
        temp_df = add_datetime(temp_df, initial_datetime)
        mains_df.append(temp_df)

    mains_df.to_pickle("./datasources/dataframes/synthetic_data/synthetic_mains.pkl")
    return


def create_appliance_dataframe(appliance):
    # Reads the pickle files of 30 days, assigns timestamp and appends to one final dataframe of mains
    appliance_df = pd.DataFrame()
    initial_datetime = datetime.datetime(2015, 10, 5, 18, 00, 00, 100000)
    for i in range(0, 30):
        print('Dataframe {}'.format(i))
        filename = './datasources/dataframes/appliances/' + str(appliance) + '/day' + str(i) + '.pkl'
        temp_df = pd.read_pickle(filename)

        # Assign datetime
        # Add one day = 24h for each separate file
        initial_datetime = initial_datetime + datetime.timedelta(hours=24)
        temp_df = add_datetime(temp_df, initial_datetime)
        appliance_df.append(temp_df)

    write_path = './datasources/dataframes/synthetic_data/synthetic_' + str(appliance) + '.pkl'
    appliance_df.to_pickle(write_path)
    return


if __name__ == "__main__":
    create_mains_dataframe()
    create_appliance_dataframe(appliance='dish_washer')
    create_appliance_dataframe(appliance='washing_machine')
