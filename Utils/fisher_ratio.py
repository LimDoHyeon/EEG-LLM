import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 로드
df = pd.read_csv('psd_results_2Hz.csv')

# Fisher Ratio 계산 함수
def calculate_fisher_ratios(df):
    fisher_ratios = []

    target_labels = [1, 2, 3, 4]  # 움직임 상상 레이블
    rest_label = 5  # rest 상태 레이블

    for target_label in target_labels:
        # 특정 레이블의 데이터
        target_data = df[df['Label'] == target_label]
        # rest 레이블의 데이터
        rest_data = df[df['Label'] == rest_label]

        for channel in target_data['Channel'].unique():
            for freq_band in target_data['Frequency_Band'].unique():
                # 특정 레이블(target_label)의 평균 및 분산
                target_mean = target_data[(target_data['Channel'] == channel) & (target_data['Frequency_Band'] == freq_band)]['Mean_PSD'].values[0]
                target_variance = target_data[(target_data['Channel'] == channel) & (target_data['Frequency_Band'] == freq_band)]['Variance_PSD'].values[0]

                # rest 레이블의 평균 및 분산
                rest_mean = rest_data[(rest_data['Channel'] == channel) & (rest_data['Frequency_Band'] == freq_band)]['Mean_PSD']
                rest_variance = rest_data[(rest_data['Channel'] == channel) & (rest_data['Frequency_Band'] == freq_band)]['Variance_PSD']

                if not rest_mean.empty and not rest_variance.empty:
                    rest_mean = rest_mean.values[0]
                    rest_variance = rest_variance.values[0]

                    # Fisher Ratio 계산
                    fisher_ratio = (rest_mean - target_mean) ** 2 / ((rest_variance) ** 2 + (target_variance) ** 2)

                    fisher_ratios.append({
                        'Label': target_label,
                        'Channel': channel,
                        'Frequency_Band': freq_band,
                        'Fisher_Ratio': fisher_ratio
                    })

    return pd.DataFrame(fisher_ratios)

# Fisher Ratio 계산
fisher_df = calculate_fisher_ratios(df)

# Fisher Ratio 결과를 CSV 파일로 저장
fisher_df.to_csv('fisher_ratios.csv', index=False)
print("Fisher ratios saved to 'fisher_ratios.csv'")

# Fisher Ratio 히트맵 플로팅 함수
def plot_fisher_ratios(fisher_df):
    labels = fisher_df['Label'].unique()
    
    for label in labels:
        label_fisher_df = fisher_df[fisher_df['Label'] == label]
        pivot_table = label_fisher_df.pivot_table(index=['Channel'], columns='Frequency_Band', values='Fisher_Ratio')

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap='jet', annot=False)
        plt.title(f'Fisher Ratio Heatmap for Label {label} vs Rest')
        plt.xlabel('Frequency Band')
        plt.ylabel('Channel')
        plt.show()

# Plot Fisher Ratio heatmaps
plot_fisher_ratios(fisher_df)
