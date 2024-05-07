# In EEG-GPT project, Tree of Thought method is not used in actual.
# Instead, the Decision Tree which similar to Tree of Thought is used.
# So, after considerable thinking about which step GPT should take, DT will be implemented.

class TreeOfThought:
    def __init__(self, seizure_detector, spike_detector, qEEG_tool):
        self.seizure_detector = seizure_detector
        self.spike_detector = spike_detector
        self.qEEG_tool = qEEG_tool

    def analyze_eeg(self, eeg_data):
        """
        Analyze EEG data using a decision tree.
        """
        for epoch in eeg_data:
            if self.seizure_detector.detect(epoch):
                return 'abnormal'

            if self.spike_detector.detect_spike(epoch):
                return 'abnormal'

            if not self.qEEG_tool.is_similar_to_normal(epoch):
                return 'abnormal'

        return 'normal'
