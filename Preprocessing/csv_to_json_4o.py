"""
Base functions for converting csv to json, json to jsonl.
"""

import pandas as pd
import numpy as np
import json
from Preprocessing.feature_extraction import extract_features


def csv_to_json(df, csp, window_size, selected_columns, labels):
    """
    Convert a DataFrame of EEG data into a JSON format suitable for GPT-3 davinci.
    :param df: Data converted to pandas DataFrame from the original csv file
    :param window_size: Window size to divide EEG data
    :param selected_columns: EEG channel to use (provide a list with frequency bands)
    :param labels: Label for each window (provide a list, left, right, top, bottom)
    :return: List of data in JSON format
    """
    json_array = []

    # EEG 채널 이름을 selected_columns에 매핑합니다.
    channel_names = ['FCz', 'C3', 'Cz', 'C4', 'CP3']  # 각각 0, 1, 2, 3에 대응

    for start in range(0, len(df) - window_size + 1, window_size):
        window_data = df.iloc[start:start + window_size, :]  # 전체 데이터를 가져옴
        label = str(int(labels[start]))  # Assuming labels are provided for each window

        # Extract features using the updated extract_features function
        features = extract_features(window_data, selected_columns)  # feature extraction
        cspdata = pd.DataFrame(csp[int(start / 1000)]).T  # cspdata 가져옴
        # features와 cspdata 가로 방향으로 합침
        features = pd.concat([features, cspdata], axis=1)
        features_dict = features.to_dict('index')[0]  # DataFrame to dictionary

        # Generate features_dict_with_keys
        features_dict_with_keys = {}
        for i, (channel_idx, bands) in enumerate(selected_columns):
            key = f"at channel {channel_names[i]}"
            features_list = []
            for band in bands if isinstance(bands, list) else [bands]:
                band_key = f"{channel_names[i]}_{band[0]}-{band[1]}Hz"
                power_value = features_dict[band_key]

                # Flatten the power value if it's an array
                if isinstance(power_value, np.ndarray):
                    power_value = power_value.item()  # Convert array to scalar if it's 1D
                features_list.append(f"Power in {band[0]}-{band[1]} Hz: {power_value}")
            features_dict_with_keys[key] = features_list

        # Set the CSP 값을 라벨에 맞게 프롬프트에 추가
        csp_key = f"CSP values: 0: {cspdata.values[0][0]}, 1: {cspdata.values[0][1]}"

        # Set the GPT's role
        system_message = "Look at the feature values of a given EEG electrode and determine which label the data belongs to. The result should always provide only integer label values."

        # Prompt explaining the feature information
        prompt = f"Quantitative EEG: In a {window_size / 250} second period,"
        features_str = ""
        for key, value in features_dict_with_keys.items():
            features_str += f"{key}:\n"
            features_str += "\n".join([f"  {v}" for v in value])
            features_str += "\n"

        # CSP 값을 프롬프트에 포함
        combined_prompt = f"{prompt}\n{features_str}\n{csp_key}\n"

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


def csv_to_json_without_label(df, csp, window_size, selected_columns):
    """
    Convert a DataFrame of EEG data into a JSON format suitable for GPT-3 davinci.
    :param df: Data converted to pandas DataFrame from the original csv file
    :param window_size: Window size to divide EEG data
    :param selected_columns: EEG channel to use (provide a list with frequency bands)
    :param labels: Label for each window (provide a list, left, right, top, bottom)
    :return: List of data in JSON format
    """
    json_array = []

    # EEG 채널 이름을 selected_columns에 매핑합니다.
    channel_names = ['FCz', 'C3', 'Cz', 'C4', 'CP3']  # 각각 0, 1, 2, 3에 대응

    for start in range(0, len(df) - window_size + 1, window_size):
        window_data = df.iloc[start:start + window_size, :]  # 전체 데이터를 가져옴

        # Extract features using the updated extract_features function
        features = extract_features(window_data, selected_columns)  # feature extraction
        cspdata = pd.DataFrame(csp[int(start / 1000)]).T  # cspdata 가져옴
        # features와 cspdata 가로 방향으로 합침
        features = pd.concat([features, cspdata], axis=1)
        features_dict = features.to_dict('index')[0]  # DataFrame to dictionary

        # Generate features_dict_with_keys
        features_dict_with_keys = {}
        for i, (channel_idx, bands) in enumerate(selected_columns):
            key = f"at channel {channel_names[i]}"
            features_list = []
            for band in bands if isinstance(bands, list) else [bands]:
                band_key = f"{channel_names[i]}_{band[0]}-{band[1]}Hz"
                power_value = features_dict[band_key]

                # Flatten the power value if it's an array
                if isinstance(power_value, np.ndarray):
                    power_value = power_value.item()  # Convert array to scalar if it's 1D
                features_list.append(f"Power in {band[0]}-{band[1]} Hz: {power_value}")
            features_dict_with_keys[key] = features_list

        # Set the CSP 값을 라벨에 맞게 프롬프트에 추가
        csp_key = f"CSP values: 0: {cspdata.values[0][0]}, 1: {cspdata.values[0][1]}"

        # Set the GPT's role
        system_message = "Look at the feature values of a given EEG electrode and determine which label the data belongs to. The result should always provide only integer label values."

        # Prompt explaining the feature information
        prompt = f"Quantitative EEG: In a {window_size / 250} second period,"
        features_str = ""
        for key, value in features_dict_with_keys.items():
            features_str += f"{key}:\n"
            features_str += "\n".join([f"  {v}" for v in value])
            features_str += "\n"

        # CSP 값을 프롬프트에 포함
        combined_prompt = f"{prompt}\n{features_str}\n{csp_key}\n"

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

