import mne
import numpy as np
import pandas as pd
from data_split_chunk import preprocess_data, split_data_into_chunks, ch_names, ch_types, sampling_freq, p_detrend, p_normalization

def calculate_psd_for_labels(eeg_raw_split, ch_names, ch_types, sampling_freq, fmin=4, fmax=36, output_csv='psd_results_2Hz.csv'):
    """
    각 레이블의 데이터 조각에 대해 PSD를 계산하고 결과를 CSV 파일로 저장하는 함수

    :param eeg_raw_split: 레이블별 데이터 조각 리스트
    :param ch_names: 채널 이름 리스트
    :param ch_types: 채널 타입 리스트
    :param sampling_freq: 샘플링 주파수
    :param fmin: PSD 계산의 최소 주파수 (Hz)
    :param fmax: PSD 계산의 최대 주파수 (Hz)
    :param output_csv: 결과를 저장할 CSV 파일 이름
    """
    # 선택할 레이블의 데이터 조각
    label_indices = range(1, len(eeg_raw_split) + 1)  # 레이블 인덱스를 1부터 시작하도록 변경

    # 결과를 저장할 리스트
    results = []

    # 주파수 해상도를 2Hz로 설정하기 위한 n_fft 계산
    n_fft = int(2 ** (np.ceil(np.log2(sampling_freq / 2))))  # 대략적인 n_fft 값

    for label_index in label_indices:
        # 선택한 레이블의 데이터 조각 (라벨 인덱스를 0 기반 인덱스로 변환)
        chunks = eeg_raw_split[label_index - 1]
        
        # 각 채널과 chunk에 대한 PSD를 저장할 딕셔너리
        all_psds = {channel_name: [] for channel_name in ch_names}
        
        # 각 chunk에 대해 PSD를 계산하고 저장
        for chunk_index, chunk in enumerate(chunks):
            # MNE 객체 생성
            info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
            info.set_montage('standard_1020')
            
            info['description'] = 'OpenBCI'
            info['bads'] = []  # Names of bad channels
            
            # RawArray 객체 생성
            raw = mne.io.RawArray(chunk.to_numpy(), info)
            
            # 모든 채널에 대해 PSD 계산
            for channel_name in ch_names:
                data_array = raw.get_data(picks=[channel_name])
                sfreq = raw.info['sfreq']
                
                # PSD 계산
                psd, freqs = mne.time_frequency.psd_array_welch(data_array, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft)
                
                # psd를 로그 스케일로 변환
                psd = 10. * np.log10(psd)
                
                # PSD 결과 저장
                all_psds[channel_name].append(psd[0])  # Assuming one channel per data array
        
        # 각 채널의 PSD 평균과 분산 계산
        for channel_name in ch_names:
            psds_for_channel = np.array(all_psds[channel_name])
            mean_psd = np.mean(psds_for_channel, axis=0)
            variance_psd = np.var(psds_for_channel, axis=0)  # Calculate variance
            
            # 2Hz 간격으로 평균과 분산 계산
            for start_freq in range(4, 36, 2):
                end_freq = start_freq + 2
                mask = (freqs >= start_freq) & (freqs < end_freq)
                if np.any(mask):  # mask가 비어있지 않으면
                    mean = np.mean(mean_psd[mask])
                    variance = np.mean(variance_psd[mask])
                    
                    # 결과 리스트에 추가
                    results.append({
                        'Label': label_index,  # 라벨을 1부터 시작하도록 유지
                        'Channel': channel_name,
                        'Frequency_Band': f'{start_freq}-{end_freq}Hz',
                        'Mean_PSD': mean,
                        'Variance_PSD': variance
                    })

    # 결과를 pandas DataFrame으로 저장
    df = pd.DataFrame(results)

    # CSV 파일로 저장
    df.to_csv(output_csv, index=False)

    print(f"Results saved to {output_csv}")

# 사용
if __name__ == "__main__":
    filenames = [f"C:/Users/windows/Desktop/EEG-GPT-main/Dataset/laf_eeg_data_ch9_label{i}.csv" for i in range(1, 6)]
    
    # Preprocess data
    eeg_raw = preprocess_data(filenames, p_detrend, p_normalization)
    
    # Split data into chunks
    eeg_raw_split = split_data_into_chunks(eeg_raw, chunk_size=1000)
    
    # Calculate and save PSD
    calculate_psd_for_labels(eeg_raw_split, ch_names, ch_types, sampling_freq)
