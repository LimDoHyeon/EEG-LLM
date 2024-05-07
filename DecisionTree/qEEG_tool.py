from sklearn.metrics.pairwise import cosine_similarity

class qEEGFeatureComparisonTool:
    def __init__(self, normative_data):
        """
        Initialize with normative EEG feature data.
        """
        self.normative_data = normative_data

    def calculate_cosine_similarity(self, features):
        """
        Calculate the cosine similarity between given features and normative data.
        """
        similarities = [cosine_similarity([features], [norm]) for norm in self.normative_data]
        return max(similarities)

    def is_similar_to_normal(self, features, threshold=0.8):
        """
        Determine if the given features are similar to normal EEG data.
        """
        similarity = self.calculate_cosine_similarity(features)
        return similarity >= threshold
