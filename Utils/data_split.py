import pandas as pd
import random


def data_split(data, train_dir, test_dir):
    """
    파이프라인 : 데이터를 1000개 단위로 묶어 배열에 저장 / 이 배열의 인덱스를 무작위로 섞음 / 섞인 인덱스의 80%를 훈련 데이터로, 20%를 테스트 데이터로 사용\
    / 묶어 둔 인덱스를 다시 데이터로 원상복구

    :param data: 전체 데이터
    :param train_dir: 훈련 데이터 저장 경로
    :param test_dir: 테스트 데이터 저장 경로
    """
    # Save the data in an array in units of 1000
    data_list = []
    for i in range(0, 180000, 1000):
        data_list.append(data.iloc[i:i + 1000])

    random.shuffle(data_list)

    # Use 80% of the shuffled indices as training data and 20% as test data
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for i in range(0, 144):
        train_data = pd.concat([train_data, data_list[i]])
    for i in range(144, 180):
        test_data = pd.concat([test_data, data_list[i]])

    # Save into csv files
    train_data.to_csv(train_dir, index=False)
    test_data.to_csv(test_dir, index=False)


def main():
    df = pd.read_csv('/Users/imdohyeon/Documents/PythonWorkspace/EEG-GPT/Dataset/eeg_data_180000.csv')
    train_dir = '/Users/imdohyeon/Documents/PythonWorkspace/EEG-GPT/Dataset/train_data.csv'
    test_dir = '/Users/imdohyeon/Documents/PythonWorkspace/EEG-GPT/Dataset/test_data.csv'
    data_split(df, train_dir, test_dir)