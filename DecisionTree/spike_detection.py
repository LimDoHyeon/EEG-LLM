class SpikeDetector:
    def __init__(self, window_length, normalization_constant):
        self.window_length = window_length
        self.normalization_constant = normalization_constant

    def compute_line_length(self, data):
        """
        Compute the line length feature for the given data.
        """
        ll = sum(abs(data[i] - data[i - 1]) for i in range(1, len(data)))
        return ll / self.normalization_constant

    def detect_spike(self, eeg_epoch):
        """
        Detect spikes in the given EEG epoch using the line length feature.
        """
        ll_value = self.compute_line_length(eeg_epoch)
        return ll_value > self.threshold

    def set_threshold(self, training_data):
        """
        Set a detection threshold based on training data.
        """
        ll_values = [self.compute_line_length(epoch) for epoch in training_data]
        self.threshold = np.percentile(ll_values, 95)
