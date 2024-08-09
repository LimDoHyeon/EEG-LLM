"""
Author: Do-Hyeon Lim
This code is for concatenating label in data.
"""

import pandas as pd
import numpy as np

def main():
    data_dir = '/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/공유 드라이브/4N_PKNU/BXAI/EEG-LLM/Dataset/raw/eeg_no_label.csv'
    label_dir = '/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/공유 드라이브/4N_PKNU/BXAI/EEG-LLM/Dataset/raw/classlabel.csv'
    save_dir = '/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/공유 드라이브/4N_PKNU/BXAI/EEG-LLM/Dataset/eeg_data_180000.csv'
    label = pd.read_csv(label_dir, header=None)
    data = pd.read_csv(data_dir)

    label = label.dropna()
    label = label.reset_index(drop=True)
    label = label.reindex(np.repeat(label.index, 1000))
    label = label.reset_index(drop=True)

    data['Label'] = label[0]
    data.to_csv(save_dir)
    print("concatenation completed.")

if __name__ == '__main__':
    main()
