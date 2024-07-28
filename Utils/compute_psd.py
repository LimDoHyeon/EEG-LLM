import os
import pandas as pd
import numpy as np
from scipy import signal
import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_and_preprocess_data(filenames, sampling_freq, p_detrend, p_normalization):
    eeg_raw = {}
    
    for i, filename in enumerate(filenames):
        temp_pd = pd.read_csv(filename)
        temp_pd = temp_pd.drop(columns=['label'])

        plt.figure()
        plt.plot(temp_pd['#  2'])
        plt.title(f"EEG Plot for label {i+1}")
        plt.xlabel("csv rows")
        plt.ylabel("EEG Signal Amplitude")
        plt.show()

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

def detrend_df(df):
    detrended_data = {col: signal.detrend(df[col]) for col in df.columns}
    detrended_df = pd.DataFrame(detrended_data)
    return detrended_df

def create_mne_raw_objects(eeg_raw, n_channels, sampling_freq, ch_names, ch_types):
    mne_raw = {}

    for i in range(len(eeg_raw)):
        info = mne.create_info(n_channels, sfreq=sampling_freq)
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
        info.set_montage('standard_1020')
        info['description'] = 'OpenBCI'
        info['bads'] = []

        raw = mne.io.RawArray(eeg_raw[i], info)
        mne_raw[i] = raw
    
    return mne_raw

def calculate_psd_for_all_channels(mne_raw_list, condition_names, fmin=4, fmax=36, freq_step=2):
    for raw_idx, (mne_raw, condition_name) in enumerate(zip(mne_raw_list, condition_names)):
        data_array = mne_raw.get_data(picks='all')
        sfreq = mne_raw.info['sfreq']
        channel_names = mne_raw.info['ch_names']

        print(f"Condition: {condition_name}")

        for channel_idx, channel_name in enumerate(channel_names):
            channel_data = data_array[channel_idx]
            psds, freqs = mne.time_frequency.psd_array_welch(channel_data, sfreq=sfreq, fmin=fmin, fmax=fmax)
            psds = 10. * np.log10(psds)
            mask = np.logical_and(freqs >= fmin, freqs <= fmax)
            filtered_freqs = freqs[mask]
            filtered_psds = psds[mask]

            freq_indices = np.arange(0, len(filtered_freqs), int(freq_step / (freqs[1] - freqs[0])))
            freq_indices = freq_indices[freq_indices < len(filtered_freqs)]
            selected_freqs = filtered_freqs[freq_indices]
            selected_psds = filtered_psds[freq_indices]

            print(f"  Channel: {channel_name}")
            print("  Frequencies:", selected_freqs)
            print("  PSD values:", selected_psds)
            print("\n")

def main():
    n_channels = 60
    sampling_freq = 250
    ch_names = [
        'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4',
        'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
        'FT7', 'FT8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz',
        'Cz', 'C1', 'C2','C3', 'C4', 'C5', 'C6', 'T7', 'T8',
        'CPz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
        'Pz', 'P1', 'P2','P3', 'P4', 'P5', 'P6', 'P7', 'P8',
        'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
        'Oz', 'O1', 'O2'
    ]
    ch_types = ['eeg'] * n_channels

    p_detrend = 0
    p_normalization = 0
    p_ica_flag = 0
    p_ts_psd_flag = 0

    filenames = [f"C:/Users/windows/Desktop/EEG-GPT-main/Dataset/eeg_data_label_{i}.csv" for i in range(1, 6)]
    
    eeg_raw = load_and_preprocess_data(filenames, sampling_freq, p_detrend, p_normalization)
    mne_raw = create_mne_raw_objects(eeg_raw, n_channels, sampling_freq, ch_names, ch_types)
    
    condition_names = ["right", "left", "tongue", "foot", "rest"]
    calculate_psd_for_all_channels([mne_raw[i] for i in range(len(filenames))], condition_names)

    mne_raw[0].plot_psd(fmax=50)
    plt.show()  # First PSD plot

    mne_raw[1].plot_psd(fmax=50)
    plt.show()  # Second PSD plot

    print('Done')

if __name__ == "__main__":
    main()