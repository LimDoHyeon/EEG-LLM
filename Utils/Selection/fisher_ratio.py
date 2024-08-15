import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('psd_results_2Hz.csv')

# Function to calculate Fisher Ratios
def calculate_fisher_ratios(df):
    fisher_ratios = []

    target_labels = [1, 2, 3, 4]  # Movement imagination labels
    rest_label = 5  # Rest state label

    for target_label in target_labels:
        # Data for the specific label
        target_data = df[df['Label'] == target_label]
        # Data for the rest label
        rest_data = df[df['Label'] == rest_label]

        for channel in target_data['Channel'].unique():
            for freq_band in target_data['Frequency_Band'].unique():
                # Mean and variance for the specific label (target_label)
                target_mean = target_data[(target_data['Channel'] == channel) & (target_data['Frequency_Band'] == freq_band)]['Mean_PSD'].values[0]
                target_variance = target_data[(target_data['Channel'] == channel) & (target_data['Frequency_Band'] == freq_band)]['Variance_PSD'].values[0]

                # Mean and variance for the rest label
                rest_mean = rest_data[(rest_data['Channel'] == channel) & (rest_data['Frequency_Band'] == freq_band)]['Mean_PSD']
                rest_variance = rest_data[(rest_data['Channel'] == channel) & (rest_data['Frequency_Band'] == freq_band)]['Variance_PSD']

                if not rest_mean.empty and not rest_variance.empty:
                    rest_mean = rest_mean.values[0]
                    rest_variance = rest_variance.values[0]

                    # Calculate Fisher Ratio
                    fisher_ratio = (rest_mean - target_mean) ** 2 / ((rest_variance) ** 2 + (target_variance) ** 2)

                    fisher_ratios.append({
                        'Label': target_label,
                        'Channel': channel,
                        'Frequency_Band': freq_band,
                        'Fisher_Ratio': fisher_ratio
                    })

    fisher_df = pd.DataFrame(fisher_ratios)
    
    # Sort the DataFrame by Fisher Ratio in descending order for each Label
    fisher_df = fisher_df.sort_values(by=['Label', 'Fisher_Ratio'], ascending=[True, False])
    
    return fisher_df

# Calculate Fisher Ratios
fisher_df = calculate_fisher_ratios(df)

# Save the Fisher Ratios result to a CSV file
fisher_df.to_csv('fisher_ratios.csv', index=False)
print("Fisher ratios saved to 'fisher_ratios.csv'")

# Function to plot Fisher Ratio heatmaps
def plot_fisher_ratios(fisher_df):
    labels = fisher_df['Label'].unique()
    
    # Define the desired frequency band order
    frequency_band_order = ['4-6Hz', '6-8Hz', '8-10Hz', '10-12Hz', '12-14Hz', '14-16Hz',
                            '16-18Hz', '18-20Hz', '20-22Hz', '22-24Hz', '24-26Hz', '26-28Hz', 
                            '28-30Hz', '30-32Hz', '32-34Hz', '34-36Hz']

    for label in labels:
        label_fisher_df = fisher_df[fisher_df['Label'] == label]

        # Pivot table with the specified frequency band order
        pivot_table = label_fisher_df.pivot_table(index='Channel', columns='Frequency_Band', values='Fisher_Ratio')
        pivot_table = pivot_table[frequency_band_order]  # Ensure the correct order of frequency bands

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap='jet', annot=False)
        plt.title(f'Fisher Ratio Heatmap for Label {label} vs Rest')
        plt.xlabel('Frequency Band')
        plt.ylabel('Channel')
        plt.show()

# Plot Fisher Ratio heatmaps
plot_fisher_ratios(fisher_df)

