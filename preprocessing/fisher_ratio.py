import pandas as pd
import numpy as np
import mne


def rename_columns_to_numeric(data):
    new_columns = {col: str(i) for i, col in enumerate(data.columns)}
    new_columns[data.columns[-1]] = data.columns[-1]
    data.rename(columns=new_columns, inplace=True)
    return data


def calculate_fisher_ratio(data, bands, sfreq, channel_names):
    fisher_ratios = {band: {} for band in bands}

    labels = data['label'].values
    data_values = data.drop(columns=['label']).values

    for band_name, (low_freq, high_freq) in bands.items():
        band_data = mne.filter.filter_data(data_values.T, sfreq, l_freq=low_freq, h_freq=high_freq).T
        band_data = pd.DataFrame(band_data, columns=channel_names)
        band_data['label'] = labels

        features = []
        for col in band_data.drop(columns=['label']):
            mean_feature = band_data.groupby('label')[col].mean()
            std_feature = band_data.groupby('label')[col].std()
            features.append((col, mean_feature, std_feature))

        for col, mean_feature, std_feature in features:
            overall_mean = mean_feature.mean()
            S_B = sum((mean_feature - overall_mean) ** 2) * (len(mean_feature) - 1)
            S_W = sum(std_feature ** 2)
            fisher_ratio = S_B / S_W if S_W != 0 else 0
            fisher_ratios[band_name][col] = fisher_ratio

    return fisher_ratios


def sort_fisher_ratios(fisher_ratios):
    sorted_fisher_ratios = {}
    for band in fisher_ratios:
        sorted_fisher_ratios[band] = dict(sorted(fisher_ratios[band].items(), key=lambda item: item[1], reverse=True))
    return sorted_fisher_ratios
