"""
파이프라인
=================
1. EEG 데이터의 csv 파일 로드
2. 로드된 데이터 전처리 (feature_extraction.py 참조)
3. 전처리된 데이터를 json 형식으로 변환 (csv_to_json.py 참조)
4. json 파일을 파인튜닝 모델에 전달(데이터로더는 필요없음 - subject가 3명이라서), train set을 전달하면 됨 (fine_tuning.py에서 수행, 여기선 return만)
5. 라벨이 없는 test set을 GPT에 전달 후 결과 확인(classification.py에서 수행 예정)
"""

import pandas as pd
import numpy as np
import json
from feature_extraction import extract_features
from csv_to_json import csv_to_json
from fine_tuning import fine_tune_model


def pipeline(csv_path, window_size, selected_columns, labels, api_key):
    """
    EEG 데이터를 전처리하고 GPT-3 모델을 파인튜닝하는 파이프라인
    :param csv_path: EEG 데이터가 저장된 csv 파일 경로
    :param window_size: EEG 데이터를 나눌 윈도우 크기
    :param selected_columns: 사용할 EEG 채널(리스트에 제공됨)
    :param labels: 각 윈도우에 대한 라벨(리스트에 제공됨, left, right, top, bottom)
    :param api_key: OpenAI API 키
    :return: 파인튜닝된 모델
    """
    # 1. EEG 데이터의 csv 파일 로드
    df = pd.read_csv(csv_path)

    # 2. 로드된 데이터 전처리
    preprocessed_df = df.copy()
    # TODO: 전처리 함수 추가 (feature_extraction.py)

    # 3. 전처리된 데이터를 json 형식으로 변환
    json_data = csv_to_json(preprocessed_df, window_size, selected_columns, labels)

    # 4. json 파일을 파인튜닝 모델에 전달
    fine_tune_model(json_data, api_key)