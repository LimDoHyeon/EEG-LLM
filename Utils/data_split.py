"""
Author: Do-Hyeon Lim
This code is for splitting data into train, validation, and test data.
"""

import pandas as pd
import random


def data_split(data, train_dir, val_dir, test_dir):
    """
    파이프라인 : 데이터를 1000개 단위로 묶어 배열에 저장 / 이 배열의 인덱스를 무작위로 섞음 / 섞인 인덱스의 80%를 훈련 데이터로, 20%를 테스트 데이터로 사용\
    / 묶어 둔 인덱스를 다시 데이터로 원상복구

    :param data: 전체 데이터
    :param train_dir: 훈련 데이터 저장 경로
    :param val_dir: 검증 데이터 저장 경로
    :param test_dir: 테스트 데이터 저장 경로
    """
    # Save the data in an array in units of 1000
    print('Scanned data length : ', len(data))

    data_list = []
    for i in range(0, len(data), 1000):
        data_list.append(data.iloc[i:i + 1000])

    random.shuffle(data_list)

    # Use 60% of the shuffled indices as training data, 20% as validation data and 20% as test data
    chunk_data = int(len(data) / 10000)  # 1000 * 10(0.6, 0.8, 1.0 -> 6, 8, 10 // 소수점 연산 미지원)
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    val_data = pd.DataFrame()
    for i in range(0, chunk_data*6):
        train_data = pd.concat([train_data, data_list[i]])
    for i in range(chunk_data*6, chunk_data*8):
        val_data = pd.concat([val_data, data_list[i]])
    for i in range(chunk_data*8, chunk_data*10):
        test_data = pd.concat([test_data, data_list[i]])

    # Drop the first column (index)
    train_data = train_data.iloc[:, 1:]
    val_data = val_data.iloc[:, 1:]
    test_data = test_data.iloc[:, 1:]

    # Save into csv files
    train_data.to_csv(train_dir, index=False)
    val_data.to_csv(val_dir, index=False)
    test_data.to_csv(test_dir, index=False)


def main():
    base_dir = '/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/공유 드라이브/4N_PKNU/Project/EEG-LLM/Dataset/subj(high+inter+low)/'
    df = pd.read_csv(base_dir + 'laf_eeg_data_9ch_840000.csv')
    train_dir = base_dir + 'train.csv'
    val_dir = base_dir + 'val.csv'
    test_dir = base_dir + 'test.csv'
    data_split(df, train_dir, val_dir, test_dir)


if __name__ == '__main__':
    main()