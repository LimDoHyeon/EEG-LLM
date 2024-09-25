"""
Pipeline code for preprocessing EEG data
This code preprocesses the EEG(csv) data and converts it to jsonl format at once.
================================================
1. Load the csv file of EEG data
2. Preprocess the loaded data (refer to feature_extraction.py)
3. Convert the preprocessed data to json format (refer to csv_to_json_4o.py)
4. Save the converted data to the specified path
5. Convert the preprocessed json to jsonl format and save it
"""
from csv_to_json_4o import csv_to_json, json_to_jsonl
from feature_extraction import load_eeg_data
import pandas as pd
import numpy as np
import json


def pipeline(csv_path, csp_path, json_path, jsonl_path, window_size, selected_columns):
    """
    Load the EEG data csv file, convert the preprocessed data to json format, and convert the json to jsonl format and save it.
    :param csv_path:  EEG data csv file path
    :param json_path:  json file path to save the preprocessed data
    :param jsonl_path:  jsonl file path to save the preprocessed data
    :param window_size:  window size of EEG data
    :param selected_columns:  EEG channel to use
    """
    # EEG(csv) load
    data, label = load_eeg_data(csv_path)

    # Preprocess the loaded data and convert it to json format
    json_data = csv_to_json(data, csp_path, window_size, selected_columns, label)

    # Save the converted data to the specified path
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"Data has been successfully saved to {json_path}")

    # Convert the preprocessed json to jsonl format and save it
    json_to_jsonl(json_path, jsonl_path)


def main():
    base_path = 'your_path'

    train_csv_path = base_path + 'train.csv'
    val_csv_path = base_path + 'val.csv'

    train_json_path = base_path + 'json/train.json'
    train_jsonl_path = base_path + 'jsonl/train.jsonl'

    val_json_path = base_path + 'json/val.json'
    val_jsonl_path = base_path + 'jsonl/val.jsonl'

    train_csp_path = base_path + 'csp4/your_csp.csv'
    val_csp_path = base_path + 'csp4/your_csp.csv'

    train_csp, train_csp_label = load_eeg_data(train_csp_path)
    val_csp, val_csp_label = load_eeg_data(val_csp_path)
    train_csp = train_csp.to_numpy()
    val_csp = val_csp.to_numpy()

    window_size = 1000
    # FCz=0, C3=2, Cz=3, C4=4
    # selected_columns = [0, 2, 3, 4]  # EEG channels to use, selected by fisher ratio
    selected_columns = [
        [0, [(10, 12), (12, 14)]],  # FCz
        [2, [(20, 22), (22, 24)]],  # C3
        [3, [(8, 10)]],  # Cz
        [4, [(20, 22), (22, 24)]],  # C4
        [5, [(28, 30)]],  # CP3
    ]

    pipeline(train_csv_path, train_csp, train_json_path, train_jsonl_path, window_size, selected_columns)
    pipeline(val_csv_path, val_csp, val_json_path, val_jsonl_path, window_size, selected_columns)


if __name__ == '__main__':
    main()