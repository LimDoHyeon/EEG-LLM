import mne
import numpy as np
import pandas as pd

"""
Additional filtering is not required as the data is already preprocessed.
"""


def load_eeg_data(file_path):
    """
    Load EEG data from a csv file and separate data and label.
    :param file_path: File path of the EEG data
    :return: EEG data (DataFrame), label
    """
    data_src = pd.read_csv(file_path)
    data = data_src.iloc[:, :-1]  # Exclude the last column as it is a label
    label = data_src.iloc[:, -1]  # Use the last column as a label
    return data, label


def compute_band_power(raw, band):
    """
    Compute the power in a specific frequency band.
    :param raw: MNE Raw object
    :param band: Frequency band of interest (tuple)
    :return: Power in the frequency band
    """
    fmin, fmax = band  # Setting frequency band
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=128)  # Compute PSD
    # Compute power in the frequency band
    band_power = np.sum(psds, axis=-1)
    return band_power


def extract_features(data, selected_columns, sfreq=250):
    """
    Extract features from EEG data. Furthermore, the data is downsampled to the target sampling frequency.
    :param data: EEG data (DataFrame)
    :param selected_columns: List of tuples containing channel index and frequency bands
    :param sfreq: Sampling frequency of the data
    :param target_sfreq: Target sampling frequency
    :return: Extracted features (DataFrame)
    """
    feature_dict = {}  # 결과를 저장할 딕셔너리

    for item in selected_columns:
        channel_idx = item[0]  # 채널 인덱스
        bands = item[1]  # 해당 채널에서 추출할 주파수 대역 리스트

        # 주파수 대역이 하나만 주어졌을 때도 리스트로 처리
        if isinstance(bands, tuple):
            bands = [bands]

        # 채널의 데이터 추출
        eeg_data = data.iloc[:, channel_idx].values  # 특정 채널의 데이터를 가져옴
        ch_name = data.columns[channel_idx]  # 채널 이름

        # mne RawArray 객체 생성
        info = mne.create_info(ch_names=[ch_name], sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data[np.newaxis, :], info)  # 2D array 필요

        # 주파수 대역별로 PSD 계산
        for band in bands:
            band_power = compute_band_power(raw, band)
            # 열 이름 생성 (예: Channel_1_10-12Hz)
            column_name = f'{ch_name}_{band[0]}-{band[1]}Hz'
            feature_dict[column_name] = band_power

    # 최종 데이터프레임 생성
    features = pd.DataFrame([feature_dict])

    return features
