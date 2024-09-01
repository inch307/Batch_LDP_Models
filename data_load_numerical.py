import pandas as pd
import numpy as np

def load_ny_datetime_2023():
    df = pd.read_parquet('data/yellow_tripdata_2023-01.parquet')
    dim = df.shape[1]
    df = df['tpep_pickup_datetime'].dt.hour * 3600 + df['tpep_pickup_datetime'].dt.minute * 60 + df['tpep_pickup_datetime'].dt.second
    df = df.dropna().reset_index(drop=True)
    
    m = df.min()
    M = df.max()
    df = 2 * ((df - m) / (M - m)) - 1

    data = df.to_numpy()
    data_shape = (len(data), dim)
    # dim 19
    return data, data.mean(), data.var(), data_shape

def load_hpc_voltage():
    df = pd.read_csv('data/hpc.csv')
    dim = df.shape[1]
    df = df['Voltage'].astype(float)
    df = df.dropna().reset_index(drop=True)

    m = df.min()
    M = df.max()
    df = 2 * ((df - m) / (M - m)) - 1


    data = df.to_numpy()
    data_shape = (len(data), dim)
    return data, data.mean(), data.var(), data_shape