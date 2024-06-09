import mne
import numpy as np
import pandas as pd

"""
Additional filtering is not required as the data is already preprocessed.
"""


def load_eeg_data(file_path):
    data_src = pd.read_csv(file_path)
    data = data_src.iloc[:, :-1]  # 마지막 열은 라벨이므로 제외
    label = data_src.iloc[:, -1]  # 마지막 열은 라벨로 사용
    return data, label


def compute_band_power(raw, band):
    fmin, fmax = band  # 주파수 대역 설정
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    # data = data[:-1, :]  # 마지막 열은 라벨이므로 제외
    psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=256)  # 전력 스펙트럼 밀도 계산
    # 주파수 대역의 전력 계산
    band_power = np.sum(psds, axis=-1)
    return band_power


def extract_features(data, selected_columns, sfreq=250):
    # eeg_data = data.values  # 모든 열을 가져옴
    # ch_names = data.columns.tolist()  # 열 이름을 채널 이름으로 사용
    eeg_data = data.iloc[:, selected_columns].values  # 선택된 열만 가져옴
    ch_names = data.columns[selected_columns].tolist()  # 선택된 열의 이름을 채널 이름으로 사용
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')  # MNE의 Info 객체 생성
    raw = mne.io.RawArray(eeg_data.T, info)  # RawArray 객체 생성

    # 주파수 대역 정의
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 12)

    # 주파수 대역별 전력 계산
    delta_power = compute_band_power(raw, delta_band)
    theta_power = compute_band_power(raw, theta_band)
    alpha_power = compute_band_power(raw, alpha_band)

    # 전력 비율 계산
    alpha_delta_ratio = alpha_power / delta_power
    theta_alpha_ratio = theta_power / alpha_power
    delta_theta_ratio = delta_power / theta_power

    # 피처 딕셔너리 생성
    features = pd.DataFrame({
        'Alpha:Delta Power Ratio': alpha_delta_ratio,
        'Theta:Alpha Power Ratio': theta_alpha_ratio,
        'Delta:Theta Power Ratio': delta_theta_ratio
    }, index=selected_columns)  # 선택된 열의 이름을 인덱스로 사용

    return features


"""
    # 피처 딕셔너리 생성
    features = pd.DataFrame({
        'Alpha:Delta Power Ratio': alpha_delta_ratio,
        'Theta:Alpha Power Ratio': theta_alpha_ratio,
        'Delta:Theta Power Ratio': delta_theta_ratio
    })
"""