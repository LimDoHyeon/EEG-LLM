"""
파이프라인
=================
1. EEG 데이터의 csv 파일 로드
2. 로드된 데이터 전처리 (feature_extraction.py 참조)
3. 전처리된 데이터를 json 형식으로 변환 (csv_to_json.py 참조)
4. 변환된 데이터를 지정된 경로에 저장
"""
from csv_to_json import csv_to_json
from feature_extraction import load_eeg_data
import pandas as pd
import numpy as np
import json


def pipeline(csv_path, output_path, window_size, selected_columns):
    """
    EEG 데이터를 전처리하고 GPT-3 모델을 파인튜닝하는 파이프라인
    :param csv_path: EEG 데이터가 저장된 csv 파일 경로
    :param output_path: 전처리된 데이터를 저장할 json 파일 경로
    :param window_size: EEG 데이터를 나눌 윈도우 크기
    :param selected_columns: 사용할 EEG 채널(리스트에 제공됨)
    """
    # 1. EEG 데이터의 csv 파일 로드
    data, label = load_eeg_data(csv_path)

    # 2. 로드된 데이터 전처리 및 전처리된 데이터를 json 형식으로 변환
    json_data = csv_to_json(data, window_size, selected_columns, label)

    # 3. json 형식으로 변환된 데이터를 지정된 경로에 저장
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Data has been successfully saved to {output_path}")


csv_path = 'your_csv_path'
output_path = 'your_output_path'
window_size = 1000
selected_columns = [1, 4, 5, 7, 10]  # 사용할 EEG 채널 선택

pipeline(csv_path, output_path, window_size, selected_columns)