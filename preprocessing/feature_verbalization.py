def verbalize_features(features):
    """
    Convert a feature dictionary into a verbal representation.
    """
    verbal_representation = "Quantitative EEG:"
    for channel, feature_set in features.items():
        verbal_representation += f"\n at channel {channel}:["
        for feature, value in feature_set.items():
            verbal_representation += f"{feature} = {value}, "
        verbal_representation = verbal_representation.rstrip(', ') + '];'

    return verbal_representation


def generate_prompts(eeg_data, labels):
    """
    Generate prompts for fine-tuning based on EEG data and labels.
    """
    all_features = extract_all_epochs(eeg_data)
    prompts = []
    for features, label in zip(all_features, labels):
        prompt = verbalize_features(features)
        completion = " normal" if label == 'normal' else " abnormal"
        prompts.append({"prompt": prompt, "completion": completion})

    return prompts
