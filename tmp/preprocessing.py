"""
파이프라인
=================
1. EEG 데이터의 csv 파일 로드
2. 로드된 데이터 전처리 (feature_extraction.py 참조)
3. 전처리된 데이터를 json 형식으로 변환 (csv_to_json.py 참조)
4. 변환된 데이터를 지정된 경로에 저장
5. 전처리된 json을 jsonl 형식으로 변환 및 저장
"""
from csv_to_json import csv_to_json, json_to_jsonl
from feature_extraction import load_eeg_data
import pandas as pd
import numpy as np
import json


def pipeline(csv_path, json_path, jsonl_path, window_size, selected_columns):
    """
    csv 파일을 로드하고 전처리된 데이터를 json 형식으로 변환, json을 jsonl 형식으로 변환하여 저장하는 파이프라인

    :param csv_path:  EEG 데이터의 csv 파일 경로
    :param json_path:  전처리된 데이터를 저장할 json 파일 경로
    :param jsonl_path:  전처리된 데이터를 저장할 jsonl 파일 경로
    :param window_size:  EEG 데이터의 윈도우 사이즈
    :param selected_columns:  사용할 EEG 채널 선택
    """
    # EEG 데이터의 csv 파일 로드
    data, label = load_eeg_data(csv_path)

    # 로드된 데이터 전처리 및 전처리된 데이터를 json 형식으로 변환
    json_data = csv_to_json(data, window_size, selected_columns, label)

    # json 형식으로 변환된 데이터를 지정된 경로에 저장
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Data has been successfully saved to {json_path}")

    # 전처리된 json을 jsonl 형식으로 변환 및 저장
    json_to_jsonl(json_path, jsonl_path)


def main():
    train_csv_path = '/Users/imdohyeon/Documents/PythonWorkspace/EEG-GPT/Dataset/train.csv'
    test_csv_path = '/Users/imdohyeon/Documents/PythonWorkspace/EEG-GPT/Dataset/test.csv'

    train_json_path = '/Users/imdohyeon/Documents/PythonWorkspace/EEG-GPT/Dataset/json/train.json'
    train_jsonl_path = '/Users/imdohyeon/Documents/PythonWorkspace/EEG-GPT/Dataset/jsonl/train.jsonl'

    test_json_path = '/Users/imdohyeon/Documents/PythonWorkspace/EEG-GPT/Dataset/json/test.json'
    test_jsonl_path = '/Users/imdohyeon/Documents/PythonWorkspace/EEG-GPT/Dataset/jsonl/test.jsonl'

    window_size = 1000
    selected_columns = [0, 7, 8, 14, 15, 20, 30, 35, 37, 38, 43, 44, 45, 54, 58]  # 사용할 EEG 채널 선택

    pipeline(train_csv_path, train_json_path, train_jsonl_path, window_size, selected_columns)
    pipeline(test_csv_path, test_json_path, test_jsonl_path, window_size, selected_columns)


if __name__ == '__main__':
    main()