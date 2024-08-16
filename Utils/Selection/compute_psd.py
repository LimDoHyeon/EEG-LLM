import mne
import numpy as np
import pandas as pd
from data_split_chunk import preprocess_data, split_data_into_chunks, ch_names, ch_types, sampling_freq

def calculate_psd_for_labels(eeg_raw_split, ch_names, ch_types, sampling_freq, fmin=4, fmax=36, output_csv='psd_results_2Hz.csv'):
    """
    Function to calculate PSD for chunks of data for each label and save the results to a CSV file.

    :param eeg_raw_split: List of data chunks split by labels
    :param ch_names: List of channel names
    :param ch_types: List of channel types
    :param sampling_freq: Sampling frequency
    :param fmin: Minimum frequency for PSD calculation (Hz)
    :param fmax: Maximum frequency for PSD calculation (Hz)
    :param output_csv: Name of the CSV file to save the results
    """
    # New sampling frequency
    new_sampling_freq = 100

    # Indices of labels to select
    label_indices = range(1, len(eeg_raw_split) + 1)  # Adjust label indices to start from 1

    # List to store results
    results = []

    # Calculate n_fft for setting frequency resolution to 2Hz
    n_fft = int(2 ** (np.ceil(np.log2(new_sampling_freq / 2))))  # Approximate n_fft value

    for label_index in label_indices:
        # Select data chunks for the specified label (convert label index to 0-based index)
        chunks = eeg_raw_split[label_index - 1]
        
        # Dictionary to store PSDs for each channel and chunk
        all_psds = {channel_name: [] for channel_name in ch_names}
        
        # Calculate and store PSD for each chunk
        for chunk_index, chunk in enumerate(chunks):
            # Create MNE info object
            info = mne.create_info(ch_names, ch_types=ch_types, sfreq=new_sampling_freq)
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
                psd, freqs = mne.time_frequency.psd_array_welch(data_array, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft)
                
                # Convert PSD to log scale
                psd = 10. * np.log10(psd)
                
                # Store PSD results
                all_psds[channel_name].append(psd[0])  # Assuming one channel per data array
        
        # Calculate mean and variance of PSD for each channel
        for channel_name in ch_names:
            psds_for_channel = np.array(all_psds[channel_name])
            mean_psd = np.mean(psds_for_channel, axis=0)
            variance_psd = np.var(psds_for_channel, axis=0)  # Calculate variance
            
            # Calculate mean and variance for each 2Hz interval
            for start_freq in range(4, 36, 2):
                end_freq = start_freq + 2
                mask = (freqs >= start_freq) & (freqs < end_freq)
                if np.any(mask):  # If mask is not empty
                    mean = np.mean(mean_psd[mask])
                    variance = np.mean(variance_psd[mask])
                    
                    # Add results to the list
                    results.append({
                        'Label': label_index,  # Keep label starting from 1
                        'Channel': channel_name,
                        'Frequency_Band': f'{start_freq}-{end_freq}Hz',
                        'Mean_PSD': mean,
                        'Variance_PSD': variance
                    })

    # Save results to a pandas DataFrame
    df = pd.DataFrame(results)

    # Save to CSV file
    df.to_csv(output_csv, index=False)

    print(f"Results saved to {output_csv}")

# Usage
if __name__ == "__main__":
    filenames = [f"C:/Users/windows/Desktop/EEG-GPT-main/Dataset/laf_eeg_data_ch9_label{i}.csv" for i in range(1, 6)]
    
    # Preprocess data
    eeg_raw = preprocess_data(filenames, new_sampling_freq=100)
    
    # Split data into chunks
    eeg_raw_split = split_data_into_chunks(eeg_raw, chunk_size=400)
    
    # Calculate and save PSD
    calculate_psd_for_labels(eeg_raw_split, ch_names, ch_types, sampling_freq)
