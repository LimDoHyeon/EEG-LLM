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
        window_data = df.iloc[start:start + window_size, selected_columns]  # DataFrame으로 윈도우 데이터 선택
        label = labels[start]  # Assuming labels are provided for each window

        features = feature_extraction.extract_features(window_data, list(range(len(selected_columns))))  # 인덱스를 사용하여 피처 추출
        features_dict = features.to_dict('index')  # DataFrame을 딕셔너리 형태로 변환

        # 새로운 형식의 features_dict_with_keys 생성
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
        combined_prompt = f"{prompt}\n{features_str}"  # 실제 줄바꿈 적용

        json_entry = {
            "prompt": combined_prompt,  # 끝에 불필요한 줄바꿈 제거
            "completion": label
        }

        json_array.append(json_entry)

    return json_array