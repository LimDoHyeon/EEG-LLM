import pandas as pd
import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Setting
n_channels = 9
sampling_freq = 250  # in Hertz
ch_names = [
    'FC3', 'FCz', 'FC4',
    'C3', 'Cz', 'C4',
    'CP3', 'CPz', 'CP4'
]
ch_types = ['eeg'] * n_channels
p_detrend = 0  # 0: OFF, 1: ON
p_normalization = 0  # 0: do not, 1: [0, 1] scaling, 2: standardization (x-mean)/var

def preprocess_data(filepaths, p_detrend, p_normalization):
    """
    Preprocess EEG data: Detrend and normalize if required.
    
    :param filepaths: List of file paths for CSV files
    :param p_detrend: Flag for detrending (0: OFF, 1: ON)
    :param p_normalization: Flag for normalization (0: OFF, 1: Min-Max, 2: Standardization)
    :return: Dictionary with preprocessed DataFrames
    """
    eeg_raw = {}
    
    def detrend_df(df):
        detrended_data = {col: signal.detrend(df[col]) for col in df.columns}
        return pd.DataFrame(detrended_data)
    
    for i in range(len(filepaths)):
        temp_pd = pd.read_csv(filepaths[i])
        temp_pd = temp_pd.drop(columns=['label'])
        
        if p_detrend == 1:
            temp_pd = detrend_df(temp_pd)
        
        if p_normalization == 1:
            scaler = MinMaxScaler()
            temp_pd = pd.DataFrame(scaler.fit_transform(temp_pd), columns=temp_pd.columns)
        elif p_normalization == 2:
            scaler = StandardScaler()
            temp_pd = pd.DataFrame(scaler.fit_transform(temp_pd), columns=temp_pd.columns)
        
        temp_pd = temp_pd.transpose()
        eeg_raw[i] = temp_pd
    
    return eeg_raw

def split_columns(df, chunk_size=1000):
    """
    Split DataFrame into chunks of specified column size.
    
    :param df: DataFrame to split
    :param chunk_size: Number of columns per chunk
    :return: List of DataFrames split into chunks
    """
    df_list = []
    num_cols = df.shape[1]
    for i in range(0, num_cols, chunk_size):
        df_chunk = df.iloc[:, i:i + chunk_size]
        df_list.append(df_chunk)
    return df_list

def split_data_into_chunks(eeg_raw, chunk_size=1000):
    """
    Split the preprocessed EEG data into chunks.
    
    :param eeg_raw: Dictionary with preprocessed DataFrames
    :param chunk_size: Number of columns per chunk
    :return: Dictionary with chunked DataFrames
    """
    eeg_raw_split = {}
    for i in range(len(eeg_raw)):
        temp_pd = eeg_raw[i]
        split_data = split_columns(temp_pd, chunk_size=chunk_size)
        eeg_raw_split[i] = split_data
    return eeg_raw_split