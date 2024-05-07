from sklearn.metrics import roc_auc_score


def evaluate_model(eeg_data, labels, model):
    """
    Evaluate the performance of the decision tree model.
    """
    predictions = [model.analyze_eeg(epoch) for epoch in eeg_data]
    binary_labels = [1 if label == 'abnormal' else 0 for label in labels]
    binary_predictions = [1 if prediction == 'abnormal' else 0 for prediction in predictions]

    auroc = roc_auc_score(binary_labels, binary_predictions)
    return auroc
