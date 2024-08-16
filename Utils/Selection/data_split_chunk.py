import pandas as pd

# Setting
n_channels = 9
sampling_freq = 250  # Original sampling frequency
ch_names = [
    'FC3', 'FCz', 'FC4',
    'C3', 'Cz', 'C4',
    'CP3', 'CPz', 'CP4'
]
ch_types = ['eeg'] * n_channels

def preprocess_data(filepaths, new_sampling_freq=100):
    """
    Preprocess EEG data by downsampling and then chunking.

    :param filepaths: List of file paths for CSV files
    :param new_sampling_freq: New sampling frequency after downsampling
    :return: Dictionary with preprocessed DataFrames
    """
    eeg_raw = {}

    for i in range(len(filepaths)):
        temp_pd = pd.read_csv(filepaths[i])
        temp_pd = temp_pd.drop(columns=['label'])
        
        # Transpose the data so channels are rows
        temp_pd = temp_pd.transpose()
        
        # Downsample
        num_samples = temp_pd.shape[1]
        downsample_factor = int(sampling_freq / new_sampling_freq)
        downsampled_pd = temp_pd.iloc[:, ::downsample_factor]
        
        eeg_raw[i] = downsampled_pd
    
    return eeg_raw

def split_columns(df, chunk_size=400):
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

def split_data_into_chunks(eeg_raw, chunk_size=400):
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
