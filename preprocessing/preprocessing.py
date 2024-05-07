import numpy as np
from scipy.stats import kurtosis

CHANNELS = ['Cz', 'T5', 'T6', 'O1', 'O2']


def calculate_power_ratio(data, band1, band2):
    """
    Calculate the power ratio between two frequency bands.
    """
    # Implement power ratio calculation based on given frequency bands.
    # Placeholder return until function is fully implemented.
    return np.random.random()


def extract_features(eeg_data, channels=CHANNELS):
    """
    Extract features from given EEG data per channel.
    """
    features = {}
    for channel in channels:
        data = eeg_data[channel]
        features[channel] = {
            '90th_percentile': np.percentile(data, 90),
            'std_dev': np.std(data),
            'kurtosis': kurtosis(data),
            'alpha_delta_ratio': calculate_power_ratio(data, 'alpha', 'delta'),
            'theta_alpha_ratio': calculate_power_ratio(data, 'theta', 'alpha'),
            'delta_theta_ratio': calculate_power_ratio(data, 'delta', 'theta')
        }
    return features


def extract_all_epochs(eeg_data, epoch_length=20):
    """
    Divide EEG data into non-overlapping 20-second epochs and extract features.
    """
    num_samples = len(eeg_data[CHANNELS[0]])
    num_epochs = num_samples // (epoch_length * 250)

    all_features = []
    for i in range(num_epochs):
        epoch_data = {channel: eeg_data[channel][i * epoch_length * 250: (i + 1) * epoch_length * 250] for channel in
                      CHANNELS}
        features = extract_features(epoch_data)
        all_features.append(features)

    return all_features
