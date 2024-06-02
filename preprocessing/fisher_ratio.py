import pandas as pd
import numpy as np


def calculate_fisher_ratio(data):
    """
    각 특징에 대해 Fisher Ratio를 계산하는 함수.

    Parameters:
    data (pd.DataFrame): 마지막 열에 라벨이 포함된 데이터 프레임.

    Returns:
    list: 각 특징에 대한 Fisher Ratio 리스트.
    """
    features = []
    for col in data.columns[:-1]:  # 마지막 열은 라벨
        mean_feature = data.groupby('label')[col].mean()
        std_feature = data.groupby('label')[col].std()
        features.append((col, mean_feature, std_feature))

    fisher_ratios = {}
    for col, mean_feature, std_feature in features:
        overall_mean = mean_feature.mean()
        S_B = sum((mean_feature - overall_mean) ** 2) * (len(mean_feature) - 1)
        S_W = sum(std_feature ** 2)
        fisher_ratio = S_B / S_W
        fisher_ratios[col] = fisher_ratio

    return fisher_ratios


def sort_fisher_ratios(fisher_ratios):
    """
    Fisher Ratios를 오름차순으로 정렬하는 함수.

    Parameters:
    fisher_ratios (dict): 각 특징에 대한 Fisher Ratio 딕셔너리.

    Returns:
    dict: 오름차순으로 정렬된 Fisher Ratios 딕셔너리.
    """
    sorted_fisher_ratios = dict(sorted(fisher_ratios.items(), key=lambda item: item[1]))
    return sorted_fisher_ratios