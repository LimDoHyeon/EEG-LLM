import os
import openai
from openai import OpenAI
from Preprocessing.feature_extraction import load_eeg_data
from Preprocessing.csv_to_json_4o import csv_to_json_without_label
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def use_model(msg, model_id):
    completion = client.chat.completions.create(
        model=model_id,
        messages=msg
    )
    return completion.choices[0].message.content


def evaluate(data, csp, label, window_size, selected_columns, model_id):
    """
    Process :
    1. Receive test data (csv) as a parameter
    2. Preprocess and convert it into json format, input it into gpt one task at a time
    3. Save the completion of gpt to the buffer
    4. Collect the completions in the buffer and input them into F1 Score and Kappa Coefficient with the actual label
    5. Print the result
    """
    model_pred = []
    counted_label = [int(label[i]) for i in range(0, len(label), window_size)]

    # Get responses(prediction) from the model
    json_data = csv_to_json_without_label(data, csp, window_size, selected_columns)
    for i in range(len(json_data)):
        response = use_model(json_data[i]['messages'], model_id)
        print(i + 1, '/', str(len(json_data)), 'epochs completed : ', response, '/', counted_label[i])
        model_pred.append(response)

    model_pred = [int(pred) for pred in model_pred if pred]
    print('length of model_pred : ', len(model_pred))

    # Calculate Accuracy, F1 Score, Kappa Coefficient
    accuracy = accuracy_score(counted_label, model_pred)
    f1 = f1_score(counted_label, model_pred, average='weighted')
    rocauc = roc_auc_score(counted_label, model_pred)

    print('Accuracy : {0:.4f}'.format(accuracy))
    print('F1 Score : {0:.4f}'.format(f1))
    print('ROC-AUC Score : {0:.4f}'.format(rocauc))


def main():
    base_path = 'your_path'
    test_csv = base_path + 'test.csv'
    test_csp = base_path + 'csp/your_csp.csv'

    window_size = 1000
    selected_columns = [
        [0, [(10, 12), (12, 14)]],  # FCz
        [2, [(20, 22), (22, 24)]],  # C3
        [3, [(8, 10)]],  # Cz
        [4, [(20, 22), (22, 24)]],  # C4
        [5, [(28, 30)]],  # CP3
    ]

    # Evaluate the fine-tuned model
    model_id = 'ft:gpt-4o-2024-08-06:your_model'  # Fine-tuned model id (check it in the openai dashboard)

    data, label = load_eeg_data(test_csv)
    test_csp, test_csp_label = load_eeg_data(test_csp)
    test_csp = test_csp.to_numpy()
    evaluate(data, test_csp, label, window_size, selected_columns, model_id)


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    main()