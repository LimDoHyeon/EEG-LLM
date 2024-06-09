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
        # window_data = df.iloc[start:start + window_size][selected_columns]  # DataFrame으로 윈도우 데이터 선택
        window_data = df.iloc[start:start + window_size, selected_columns]  # DataFrame으로 윈도우 데이터 선택
        label = labels[start]  # Assuming labels are provided for each window

        prompt = "Quantitative EEG: In a {} second period,".format(window_size / 250)  # Assuming 250 Hz sampling rate
        # features = feature_extraction.extract_features(window_data, selected_columns)  # 전체 DataFrame 전달
        features = feature_extraction.extract_features(window_data, list(range(len(selected_columns))))  # 인덱스를 사용하여 피처 추출

        features_dict = features.to_dict('index')  # DataFrame을 딕셔너리 형태로 변환
        features_dict_with_keys = {selected_columns[i]: features_dict[i] for i in range(len(selected_columns))}

        json_entry = {
            "prompt": prompt,
            "features": features_dict_with_keys,
            "completion": label
        }
        json_array.append(json_entry)

    return json_array
