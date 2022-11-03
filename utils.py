import pandas as pd
import pickle
import datetime
import pytz


def add_datetime(df, initial_datetime):
    local_tz = pytz.timezone('Europe/London')
    initial_datetime = local_tz.localize(initial_datetime)
    df['timestamp'][0] = initial_datetime
    for i in range(len(df)):
        df['timestamp'][i + 1] = initial_datetime + datetime.timedelta(seconds=i + 1)
        # df['timestamp'][i + 1] = local_tz.localize(df['timestamp'][i + 1])
        # df['timestamp'][i + 1] = df['timestamp'][i + 1].replace(tzinfo=pytz.utc).astimezone(local_tz)

    return df


def create_mains_dataframe():
    # Reads the pickle files of 30 days, assigns timestamp and appends to one final dataframe of mains
    mains_df = pd.DataFrame()
    # Last date in original mains
    # 2015-01-05  06:27:12 + 00:00
    initial_datetime = datetime.datetime(2015, 1, 5, 6, 27, 13, 0)
    for i in range(0, 30):
        print('Dataframe {}'.format(i))
        filename = './datasources/dataframes/duplicated/mains/day_' + str(i) + '.pkl'
        temp_df = pd.read_pickle(filename)

        # Assign datetime
        temp_df = add_datetime(temp_df, initial_datetime)
        mains_df = pd.concat([mains_df, temp_df], ignore_index=True)
        # Add one day = 24h for each separate file
        initial_datetime = initial_datetime + datetime.timedelta(hours=24)

    mains_df.to_pickle("./datasources/dataframes/synthetic_data/synthetic_mains.pkl")
    return


def create_appliance_dataframe(appliance):
    # Reads the pickle files of 30 days, assigns timestamp and appends to one final dataframe of mains
    appliance_df = pd.DataFrame()
    if appliance == 'dish_washer':
        initial_datetime = datetime.datetime(2015, 1, 5, 6, 2, 0, 0)
    else:
        initial_datetime = datetime.datetime(2015, 1, 5, 6, 2, 1, 0)

    for i in range(0, 30):
        print('Dataframe {}'.format(i))
        filename = './datasources/dataframes/appliances/' + str(appliance) + '/day' + str(i) + '.pkl'
        temp_df = pd.read_pickle(filename)
        if i > 10:
            a = 1
        # Assign datetime
        temp_df = add_datetime(temp_df, initial_datetime)
        appliance_df = pd.concat([appliance_df, temp_df], ignore_index=True)
        # Add one day = 24h for each separate file
        initial_datetime = initial_datetime + datetime.timedelta(hours=24)

    write_path = './datasources/dataframes/synthetic_data/synthetic_' + str(appliance) + '.pkl'
    appliance_df.to_pickle(write_path)
    return


def check_df():
    mains_filename = './datasources/dataframes/synthetic_data/synthetic_mains.pkl'
    synthetic_mains_df = pd.read_pickle(mains_filename)

    filename1 = './datasources/dataframes/synthetic_data/synthetic_dish_washer.pkl'
    synthetic_dish_washer = pd.read_pickle(filename1)

    filename2 = './datasources/dataframes/synthetic_data/synthetic_washing_machine.pkl'
    synthetic_washing_machine = pd.read_pickle(filename2)

    # Check if timestamp columns of synthetic mains and appliances are of the same shape and content
    assert synthetic_mains_df['timestamp'].equals(synthetic_dish_washer['timestamp'])
    assert synthetic_mains_df['timestamp'].equals(synthetic_washing_machine['timestamp'])

    print(synthetic_mains_df.shape[0])
    print('-------------------------')
    print(synthetic_mains_df.head(20))
    print('-------------------------')
    print(synthetic_mains_df.tail(20))


def localize_dt():
    # Transform datetime to London timezone
    mains_filename = './datasources/dataframes/synthetic_data/synthetic_mains.pkl'
    synthetic_mains_df = pd.read_pickle(mains_filename)

    local_tz = pytz.timezone('Europe/London')
    synthetic_mains_df['timestamp'] = synthetic_mains_df['timestamp'].apply(lambda x: local_tz.localize(x))
    print(synthetic_mains_df.head(10))

    synthetic_mains_df.to_pickle(mains_filename)

    print('Mains done')

    filename1 = './datasources/dataframes/synthetic_data/synthetic_dish_washer.pkl'
    synthetic_dish_washer = pd.read_pickle(filename1)
    synthetic_dish_washer['timestamp'] = synthetic_dish_washer['timestamp'].apply(lambda x: local_tz.localize(x))

    synthetic_dish_washer.to_pickle(filename1)

    print('Dish washer done')

    filename2 = './datasources/dataframes/synthetic_data/synthetic_washing_machine.pkl'
    synthetic_washing_machine = pd.read_pickle(filename2)
    synthetic_washing_machine['timestamp'] = synthetic_washing_machine['timestamp'].apply(
        lambda x: local_tz.localize(x))

    synthetic_washing_machine.to_pickle(filename2)
    print('Washing machine done')
    return


def assign_index():
    # Assigns the timestamp column to the index of the dataframes
    # Transform datetime to London timezone
    mains_filename = './datasources/dataframes/synthetic_data/synthetic_mains.pkl'
    synthetic_mains_df = pd.read_pickle(mains_filename)
    synthetic_mains_df.set_index('timestamp', inplace=True, drop=False)
    print(synthetic_mains_df.head(10))

    synthetic_mains_df.to_pickle(mains_filename)

    print('Mains done')

    filename1 = './datasources/dataframes/synthetic_data/synthetic_dish_washer.pkl'
    synthetic_dish_washer = pd.read_pickle(filename1)
    synthetic_dish_washer.set_index('timestamp', inplace=True, drop=False)

    synthetic_dish_washer.to_pickle(filename1)

    print('Dish washer done')

    filename2 = './datasources/dataframes/synthetic_data/synthetic_washing_machine.pkl'
    synthetic_washing_machine = pd.read_pickle(filename2)
    synthetic_washing_machine.set_index('timestamp', inplace=True, drop=False)

    synthetic_washing_machine.to_pickle(filename2)
    print('Washing machine done')
    return


def compare_original_to_synthetic_appliances():
    filename1 = './datasources/dataframes/appliances/original/washing_machine.pkl'
    filename2 = './datasources/dataframes/appliances/original/dish_washer.pkl'
    original_wash_machine = pd.read_pickle(filename1)
    original_dish_washer = pd.read_pickle(filename2)

    filename3 = './datasources/dataframes/synthetic_data/synthetic_washing_machine.pkl'
    filename4 = './datasources/dataframes/synthetic_data/synthetic_dish_washer.pkl'
    synthetic_wash_machine = pd.read_pickle(filename3)
    synthetic_dish_washer = pd.read_pickle(filename4)

    print('Original Washing Machine:')
    print(original_wash_machine.tail(10))

    print('Synthetic Washing Machine:')
    print(synthetic_wash_machine.head(10))

    print('\n------------------------------\n')

    print('Original Dish Washer:')
    print(original_dish_washer.tail(10))

    print('Synthetic Dish Washer:')
    print(synthetic_dish_washer.head(10))
    return


if __name__ == "__main__":
    # check_df()
    # localize_dt()
    # create_mains_dataframe()

    create_appliance_dataframe(appliance='dish_washer')
    print('---------------')
    print('Dish washer done!')
    print('---------------')
    create_appliance_dataframe(appliance='washing_machine')
    assign_index()
    compare_original_to_synthetic_appliances()
