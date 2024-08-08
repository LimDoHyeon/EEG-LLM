"""
Base functions for converting csv to json, json to jsonl.
"""

import pandas as pd
import numpy as np
import json
from Preprocessing.feature_extraction import extract_features


def csv_to_json(df, window_size, selected_columns, labels):
    """
    Convert a DataFrame of EEG data into a JSON format suitable for GPT-3 davinci.
    =================================
    1. You should pick selected_columns before run this function.
    2. It contains the process of feature extraction.
    =================================
    :param df: Data converted to pandas DataFrame from the original csv file
    :param window_size: Window size to divide EEG data
    :param selected_columns: EEG channel to use (provide a list)
    :param labels: Label for each window (provide a list, left, right, top, bottom)
    :return: List of data in JSON format
    """
    json_array = []

    for start in range(0, len(df) - window_size + 1, window_size):
        window_data = df.iloc[start:start + window_size, selected_columns]  # Pick a single window based on selected_columns
        label = int(labels[start])  # Assuming labels are provided for each window
        label = str(label)

        features = extract_features(window_data, list(range(len(selected_columns))))  # feature extraction
        features_dict = features.to_dict('index')  # DataFrame to dictionary

        # Generate features_dict_with_keys
        features_dict_with_keys = {
            f"at channel {selected_columns[i]}": [
                f"Alpha:Delta Power Ratio: {features_dict[i]['Alpha:Delta Power Ratio']}",
                f"Theta:Alpha Power Ratio: {features_dict[i]['Theta:Alpha Power Ratio']}",
                f"Delta:Theta Power Ratio: {features_dict[i]['Delta:Theta Power Ratio']}"
            ] for i in range(len(selected_columns))
        }

        # Set the GPT's role
        system_message = "Look at the feature values of a given EEG electrode and determine which label the data belongs to. The result should always provide only integer label values."

        # Prompt explaining the feature information
        prompt = f"Quantitative EEG: In a {window_size / 250} second period,"
        features_str = ""
        for key, value in features_dict_with_keys.items():
            features_str += f"{key}:\n"
            features_str += "\n".join([f"  {v}" for v in value])
            features_str += "\n"
        combined_prompt = f"{prompt}\n{features_str}"

        # Convert the data to JSON format
        json_entry = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": combined_prompt},
                {"role": "assistant", "content": label}
            ]
        }

        json_array.append(json_entry)

    return json_array


def csv_to_json_without_label(df, window_size, selected_columns):
    json_array = []

    for start in range(0, len(df) - window_size + 1, window_size):
        window_data = df.iloc[start:start + window_size, selected_columns]  # Pick a single window based on selected_columns

        features = extract_features(window_data, list(range(len(selected_columns))))  # feature extraction
        features_dict = features.to_dict('index')  # DataFrame to dictionary

        # Generate features_dict_with_keys
        features_dict_with_keys = {
            f"at channel {selected_columns[i]}": [
                f"Alpha:Delta Power Ratio: {features_dict[i]['Alpha:Delta Power Ratio']}",
                f"Theta:Alpha Power Ratio: {features_dict[i]['Theta:Alpha Power Ratio']}",
                f"Delta:Theta Power Ratio: {features_dict[i]['Delta:Theta Power Ratio']}"
            ] for i in range(len(selected_columns))
        }

        # Set the GPT's role
        system_message = "Look at the feature values of a given EEG electrode and determine which label the data belongs to. The result should always provide only integer label values."

        # Prompt explaining the feature information
        prompt = f"Quantitative EEG: In a {window_size / 250} second period,"
        features_str = ""
        for key, value in features_dict_with_keys.items():
            features_str += f"{key}:\n"
            features_str += "\n".join([f"  {v}" for v in value])
            features_str += "\n"
        combined_prompt = f"{prompt}\n{features_str}"

        # Convert the data to JSON format
        json_entry = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": combined_prompt},
            ]
        }

        json_array.append(json_entry)

    return json_array


def json_to_jsonl(json_dir, jsonl_dir):
    """
    Convert JSON file to JSONL file
    :param json_dir: JSON file path to load
    :param jsonl_dir: JSONL file path to save
    """
    json_data = load_json(json_dir)
    save_to_jsonl(json_data, jsonl_dir)

    print(f"Converted {json_dir} to {jsonl_dir}")


# Load JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# Function to save as JSONL file (convert completion value to string)
def save_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in data:
            jsonl_line = json.dumps(entry, ensure_ascii=False)
            jsonl_file.write(jsonl_line + '\n')

