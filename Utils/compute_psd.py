import mne
import numpy as np
import pandas as pd
from data_split_chunk import preprocess_data, split_data_into_chunks, ch_names, ch_types, sampling_freq, p_detrend, p_normalization

def calculate_and_save_psd(eeg_raw_split, filename, ch_names, ch_types, sampling_freq, fmin=4, fmax=36):
    """
    Calculate and save PSD for all channels of a given label.
    
    :param eeg_raw_split: Dictionary with chunked DataFrames
    :param filename: Name for the output filename
    :param ch_names: List of channel names
    :param ch_types: List of channel types
    :param sampling_freq: Sampling frequency
    :param fmin: Minimum frequency for PSD calculation (Hz)
    :param fmax: Maximum frequency for PSD calculation (Hz)
    """
    all_results = []

    # Process each label
    for label_index in eeg_raw_split.keys():
        chunks = eeg_raw_split[label_index]

        for chunk_index, chunk in enumerate(chunks):
            # Create MNE Info object
            info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
            info.set_montage('standard_1020')
            
            info['description'] = 'OpenBCI'
            info['bads'] = []  # Names of bad channels
            
            # Create RawArray object
            raw = mne.io.RawArray(chunk.to_numpy(), info)
            
            # Calculate PSD for each channel
            for channel_name in ch_names:
                data_array = raw.get_data(picks=[channel_name])
                sfreq = raw.info['sfreq']
                
                # Calculate PSD
                psd, freqs = mne.time_frequency.psd_array_welch(data_array, sfreq=sfreq, fmin=fmin, fmax=fmax)
                
                # Convert psd to log scale
                psd = 10. * np.log10(psd)
                
                # Save PSD result
                for i in range(len(freqs)):
                    all_results.append({
                        'Label': label_index + 1,
                        'Channel': channel_name,
                        'Frequency': freqs[i],
                        'PSD': psd[0][i]
                    })

    # Save results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(filename, index=False)

def main():
    filenames = [f"C:/Users/windows/Desktop/EEG-GPT-main/Dataset/laf_eeg_data_ch9_label{i}.csv" for i in range(1, 6)]
    
    # Preprocess data
    eeg_raw = preprocess_data(filenames, p_detrend, p_normalization)
    
    # Split data into chunks
    eeg_raw_split = split_data_into_chunks(eeg_raw, chunk_size=1000)
    
    # Calculate and save PSD
    calculate_and_save_psd(eeg_raw_split, 'psd_result.csv', ch_names, ch_types, sampling_freq)

if __name__ == "__main__":
    main()