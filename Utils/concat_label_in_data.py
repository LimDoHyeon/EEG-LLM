"""
Author: Do-Hyeon Lim
This code is for concatenating label in data.
"""

import pandas as pd
import numpy as np

def main():
    data_dir = 'your_path'
    label_dir = 'your_path'
    save_dir = 'your_path'
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
