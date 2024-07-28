import pandas as pd
import numpy as np

def main():
    data_dir = '/Users/imdohyeon/Documents/PythonWorkspace/EEG-LLM/Dataset/raw/eeg_no_label.csv'
    label_dir = '/Users/imdohyeon/Documents/PythonWorkspace/EEG-LLM/Dataset/raw/classlabel.csv'
    label = pd.read_csv(label_dir, header=None)
    data = pd.read_csv(data_dir)

    label = label.dropna()
    label = label.reset_index(drop=True)
    label = label.reindex(np.repeat(label.index, 1000))
    label = label.reset_index(drop=True)

    data['Label'] = label[0]
    data.to_csv('/Users/imdohyeon/Documents/PythonWorkspace/EEG-LLM/Dataset/eeg_data_180000.csv')
    print("concatenation completed.")

if __name__ == '__main__':
    main()
