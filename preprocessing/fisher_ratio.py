import pandas as pd
import numpy as np

def rename_columns_to_numeric(data):
    """
    열 이름을 숫자로 변환하는 함수.

    Parameters:
    data (pd.DataFrame): 열 이름을 변환할 데이터 프레임.

    Returns:
    pd.DataFrame: 열 이름이 숫자로 변환된 데이터 프레임.
    """
    new_columns = {col: str(i) for i, col in enumerate(data.columns)}
    new_columns[data.columns[-1]] = data.columns[-1]  # 마지막 열 이름 유지
    data.rename(columns=new_columns, inplace=True)
    return data


def calculate_fisher_ratio(data, bands):
    """
    주파수 대역별로 각 특징에 대해 Fisher Ratio를 계산하는 함수.

    Parameters:
    data (pd.DataFrame): 마지막 열에 라벨이 포함된 데이터 프레임.
    bands (dict): 주파수 대역의 이름과 해당 주파수 범위를 포함하는 딕셔너리.

    Returns:
    dict: 각 주파수 대역과 특징에 대한 Fisher Ratio 딕셔너리.
    """
    fisher_ratios = {band: {} for band in bands}

    for band_name, channels in bands.items():
        band_data = data[channels + ['label']]
        features = []
        for col in band_data.drop(columns=['label']):  # 라벨 제외
            mean_feature = band_data.groupby('label')[col].mean()
            std_feature = band_data.groupby('label')[col].std()
            features.append((col, mean_feature, std_feature))

        for col, mean_feature, std_feature in features:
            overall_mean = mean_feature.mean()
            S_B = sum((mean_feature - overall_mean) ** 2) * (len(mean_feature) - 1)
            S_W = sum(std_feature ** 2)
            fisher_ratio = S_B / S_W if S_W != 0 else 0  # zero division 방지
            fisher_ratios[band_name][col] = fisher_ratio

    return fisher_ratios

def sort_fisher_ratios(fisher_ratios):
    """
    Fisher Ratios를 내림차순으로 정렬하는 함수.

    Parameters:
    fisher_ratios (dict): 각 주파수 대역과 특징에 대한 Fisher Ratio 딕셔너리.

    Returns:
    dict: 내림차순으로 정렬된 Fisher Ratios 딕셔너리.
    """
    sorted_fisher_ratios = {}
    for band in fisher_ratios:
        sorted_fisher_ratios[band] = dict(sorted(fisher_ratios[band].items(), key=lambda item: item[1], reverse=True))
    return sorted_fisher_ratios


"""
Usage Example:

data = rename_columns_to_numeric(data)

bands = {
    'delta_band': [str(i) for i in range(0, 20)],
    'theta_band': [str(i) for i in range(20, 40)],
    'alpha_band': [str(i) for i in range(40, 60)]
}

# Calculate & sort Fisher Ratio
fisher_ratios = calculate_fisher_ratio(data, bands)
sorted_fisher_ratios = sort_fisher_ratios(fisher_ratios)

sorted_fisher_ratios

"""