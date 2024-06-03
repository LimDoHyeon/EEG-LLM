"""
csv를 json으로 변환하기 위한 베이스 함수.
"""

import pandas as pd
import numpy as np
import json
import feature_extraction

def csv_to_json(df, window_size, selected_columns, labels):
    """
    Convert a DataFrame of EEG data into a JSON format suitable for GPT-3 davinci.
    =================================
    1. selected_columns 구체적인 선택 필요
    =================================
    :param df: 원본 csv 파일을 pandas DataFrame으로 변환한 데이터
    :param window_size: EEG 데이터를 나눌 윈도우 크기
    :param selected_columns: 사용할 EEG 채널 (리스트 제공)
    :param labels: 각 윈도우에 대한 라벨 (리스트 제공, left, right, top, bottom)
    :return: JSON 형식의 데이터 리스트
    """
    json_array = []

    for start in range(0, len(df) - window_size + 1, window_size):
        window_data = df.iloc[start:start + window_size][selected_columns]  # DataFrame으로 윈도우 데이터 선택
        label = labels[start]  # Assuming labels are provided for each window

        prompt = "Quantitative EEG: In a {} second period,".format(window_size / 250)  # Assuming 250 Hz sampling rate
        features = feature_extraction.extract_features(window_data)  # 전체 DataFrame 전달

        json_entry = {
            "prompt": prompt,
            "features": features,
            "completion": label
        }
        json_array.append(json_entry)

    return json_array


"""
json_array = []

    for start in range(0, len(df) - window_size + 1, window_size):
    window_data = df.iloc[start:start + window_size][selected_columns]
    label = labels[start: start + window_size]

    prompt = "Quantitative EEG: In a {} second period,".format(window_size / 250)  # Assuming 250 Hz sampling rate
    features_dict = {}

    for column in selected_columns:
        window = window_data[column]  # Select column from DataFrame directly
        features = feature_extraction.extract_features(window)
        features_dict[column] = features

    json_entry = {
        "prompt": prompt,
        "features": features_dict,
        "completion": label
    }
    json_array.append(json_entry)

return json_array
"""
