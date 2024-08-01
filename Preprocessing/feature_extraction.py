import mne
import numpy as np
import pandas as pd

"""
Additional filtering is not required as the data is already preprocessed.
"""


def load_eeg_data(file_path):
    data_src = pd.read_csv(file_path)
    data = data_src.iloc[:, :-1]  # Exclude the last column as it is a label
    label = data_src.iloc[:, -1]  # Use the last column as a label
    return data, label


def compute_band_power(raw, band):
    fmin, fmax = band  # Setting frequency band
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=256)  # Compute PSD
    # Compute power in the frequency band
    band_power = np.sum(psds, axis=-1)
    return band_power


def extract_features(data, selected_columns, sfreq=250):
    eeg_data = data.iloc[:, selected_columns].values  # Bring only the selected columns
    ch_names = data.columns[selected_columns].tolist()  # Use the selected column names as channel names
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')  # Create an Info object
    raw = mne.io.RawArray(eeg_data.T, info)  # RawArray 객체 생성

    # Frequency band definition
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 12)

    # Compute power in the frequency band
    delta_power = compute_band_power(raw, delta_band)
    theta_power = compute_band_power(raw, theta_band)
    alpha_power = compute_band_power(raw, alpha_band)

    # Power ratio calculation
    alpha_delta_ratio = alpha_power / delta_power
    theta_alpha_ratio = theta_power / alpha_power
    delta_theta_ratio = delta_power / theta_power

    # Create a feature dictionary
    features = pd.DataFrame({
        'Alpha:Delta Power Ratio': alpha_delta_ratio,
        'Theta:Alpha Power Ratio': theta_alpha_ratio,
        'Delta:Theta Power Ratio': delta_theta_ratio
    }, index=selected_columns)  # Use the names of the selected columns as the index

    return features