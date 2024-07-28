"""
csv를 json으로, json을 jsonl로 변환하기 위한 베이스 함수.
"""

import pandas as pd
import numpy as np
import json
import Preprocessing.feature_extraction


def csv_to_json(df, window_size, selected_columns, labels):
    """
    Convert a DataFrame of EEG data into a JSON format suitable for GPT-3 davinci.
    =================================
    1. selected_columns 구체적인 선택 필요
    2. It contains the process of feature extraction.
    =================================
    :param df: 원본 csv 파일을 pandas DataFrame으로 변환한 데이터
    :param window_size: EEG 데이터를 나눌 윈도우 크기
    :param selected_columns: 사용할 EEG 채널 (리스트 제공)
    :param labels: 각 윈도우에 대한 라벨 (리스트 제공, left, right, top, bottom)
    :return: JSON 형식의 데이터 리스트
    """
    json_array = []

    for start in range(0, len(df) - window_size + 1, window_size):
        window_data = df.iloc[start:start + window_size, selected_columns]  # Generate window based on selected columns
        label = labels[start]  # Assuming labels are provided for each window

        features = feature_extraction.extract_features(window_data, list(range(len(selected_columns))))
        features_dict = features.to_dict('index')  # Convert DataFrame to dictionary

        # Generate features_dict_with_keys
        features_dict_with_keys = {
            f"at channel {selected_columns[i]}": [
                f"Alpha:Delta Power Ratio: {features_dict[i]['Alpha:Delta Power Ratio']}",
                f"Theta:Alpha Power Ratio: {features_dict[i]['Theta:Alpha Power Ratio']}",
                f"Delta:Theta Power Ratio: {features_dict[i]['Delta:Theta Power Ratio']}"
            ] for i in range(len(selected_columns))
        }

        prompt = f"Quantitative EEG: In a {window_size / 250} second period,"
        features_str = ""
        for key, value in features_dict_with_keys.items():
            features_str += f"{key}:\n"
            features_str += "\n".join([f"  {v}" for v in value])
            features_str += "\n"
        combined_prompt = f"{prompt}\n{features_str}"

        json_entry = {
            "prompt": combined_prompt,
            "completion": label
        }

        json_array.append(json_entry)

    return json_array


def csv_to_json_without_label(df, window_size, selected_columns):
    json_array = []

    for start in range(0, len(df) - window_size + 1, window_size):
        window_data = df.iloc[start:start + window_size, selected_columns]  # Generate window based on selected columns

        features = Preprocessing.feature_extraction.extract_features(window_data, list(range(len(selected_columns))))
        features_dict = features.to_dict('index')  # Convert DataFrame to dictionary

        # Generate features_dict_with_keys
        features_dict_with_keys = {
            f"at channel {selected_columns[i]}": [
                f"Alpha:Delta Power Ratio: {features_dict[i]['Alpha:Delta Power Ratio']}",
                f"Theta:Alpha Power Ratio: {features_dict[i]['Theta:Alpha Power Ratio']}",
                f"Delta:Theta Power Ratio: {features_dict[i]['Delta:Theta Power Ratio']}"
            ] for i in range(len(selected_columns))
        }

        prompt = f"Quantitative EEG: In a {window_size / 250} second period,"
        features_str = ""
        for key, value in features_dict_with_keys.items():
            features_str += f"{key}:\n"
            features_str += "\n".join([f"  {v}" for v in value])
            features_str += "\n"
        combined_prompt = f"{prompt}\n{features_str}"

        json_array.append(combined_prompt)

    return json_array


def json_to_jsonl(json_dir, jsonl_dir):
    """
    JSON 파일을 JSONL 파일로 변환
    :param json_dir: 불러올 JSON 파일 경로
    :param jsonl_dir: 저장할 JSONL 파일 경로
    """
    json_data = load_json(json_dir)
    save_to_jsonl(json_data, jsonl_dir)

    print(f"Converted {json_dir} to {jsonl_dir}")


# Function to load JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# Function to save data to JSONL file
def save_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in data:
            # completion 값을 문자열로 변환
            entry['completion'] = str(entry['completion'])
            json.dump(entry, jsonl_file)
            jsonl_file.write('\n')