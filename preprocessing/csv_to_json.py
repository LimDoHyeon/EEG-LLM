"""
csv를 json으로 변환하기 위한 베이스 함수.
"""

import pandas as pd
import numpy as np
import json

def extract_features(window):
    """
    Extract features from a given window of EEG data.
    ============================
    1. features는 아직 미정. 사용할 feature 확정 후 키 변경
    2. 각각의 feature에 대한 추출 메소드는 preprocessing.py에서 가져와서 사용할 것
    ============================
    :return: features 딕셔너리
    """
    features = {
        "alpha": np.mean(window),  # Example feature extraction
        "beta": np.var(window),  # Example feature extraction
        "theta": np.min(window),  # Example feature extraction
        "psd": np.max(window),  # Example feature extraction
        "fisher_ratio": np.sum(window)  # Example feature extraction
    }
    return features


def csv_to_json(df, window_size, selected_columns, labels):
    """
    Convert a DataFrame of EEG data into a JSON format suitable for GPT-3 davinci.
    ============================
    1. selected_columns 아직 정해지지 않음 - 정해지면 수정할 것
    ============================
    :param df: 원본 csv 파일을 pandas DataFrame으로 변환한 데이터
    :param window_size: EEG 데이터를 나눌 윈도우 크기
    :param selected_columns: 사용할 EEG 채널(리스트에 제공됨)
    :param labels: 각 윈도우에 대한 라벨(리스트에 제공됨, left, right, top, bottom)
    :return: JSON 형식의 데이터를 보관한 리스트
    """
    json_array = []

    for start in range(0, len(df) - window_size + 1, window_size):
        window_data = df.iloc[start:start + window_size]
        label = labels[start // window_size]  # Assuming labels are provided for each window

        prompt = "Quantitative EEG: In a {} second period,".format(window_size / 250)  # Assuming 250 Hz sampling rate
        features_dict = {}

        for column in selected_columns:
            window = window_data[column].values
            features = extract_features(window)
            features_dict[column] = features

        json_entry = {
            "prompt": prompt,
            "features": features_dict,
            "completion": label  # Add the appropriate completion label
        }

        json_array.append(json_entry)

    return json_array
