import mne
import numpy as np
import pandas as pd

"""
Additional filtering is not required as the data is already preprocessed.
"""


def load_eeg_data(file_path, sfreq=250):
    data = pd.read_csv(file_path)
    eeg_data = data.values  # 모든 열을 가져옴
    ch_names = data.columns.tolist()  # 열 이름을 채널 이름으로 사용

    # MNE의 Info 객체 생성
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    # RawArray 객체 생성
    raw = mne.io.RawArray(eeg_data.T, info)

    return raw


def compute_band_power(raw, band):
    fmin, fmax = band  # 주파수 대역 설정
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=fmin, fmax=fmax,
                                                     n_fft=2048)  # 전력 스펙트럼 밀도 계산
    # 주파수 대역의 전력 계산
    band_power = np.sum(psds, axis=-1)
    return band_power


def extract_features(raw):
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
    })

    return features.T