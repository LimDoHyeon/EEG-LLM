import pandas as pd
import numpy as np
import mne

def rename_columns_to_numeric(data):
    new_columns = {col: str(i) for i, col in enumerate(data.columns)}
    new_columns[data.columns[-1]] = data.columns[-1]
    data.rename(columns=new_columns, inplace=True)
    return data

def calculate_fisher_ratio(data, bands, sfreq, channel_names):
    fisher_ratios = {label: {band: {} for band in bands} for label in data['label'].unique()}

    labels = data['label'].values
    data_values = data.drop(columns=['label']).values

    for band_name, (low_freq, high_freq) in bands.items():
        band_data = mne.filter.filter_data(data_values.T, sfreq, l_freq=low_freq, h_freq=high_freq).T
        band_data = pd.DataFrame(band_data, columns=channel_names)
        band_data['label'] = labels

        for label in band_data['label'].unique():
            label_data = band_data[band_data['label'] == label]
            features = []
            for col in label_data.drop(columns=['label']):
                mean_feature = label_data[col].mean()
                std_feature = label_data[col].std()
                features.append((col, mean_feature, std_feature))

            for col, mean_feature, std_feature in features:
                overall_mean = band_data[col].mean()
                S_B = (mean_feature - overall_mean) ** 2
                S_W = std_feature ** 2
                fisher_ratio = S_B / S_W if S_W != 0 else 0
                fisher_ratios[label][band_name][col] = fisher_ratio

    return fisher_ratios

def sort_fisher_ratios(fisher_ratios):
    sorted_fisher_ratios = {}
    for label in fisher_ratios:
        sorted_fisher_ratios[label] = {}
        for band in fisher_ratios[label]:
            sorted_fisher_ratios[label][band] = dict(sorted(fisher_ratios[label][band].items(), key=lambda item: item[1], reverse=True))
    return sorted_fisher_ratios

def print_fisher_ratios(fisher_ratios):
    for label in fisher_ratios:
        print(f"\nLabel {label}:")
        for band in fisher_ratios[label]:
            print(f"\n  {band} band:")
            sorted_items = sorted(fisher_ratios[label][band].items(), key=lambda x: x[1], reverse=True)
            print("  Channel | Fisher Ratio")
            print("  ------- | ------------")
            for channel, ratio in sorted_items:
                print(f"  {channel:7} | {ratio:.10f}")