"""
Load the fine-tuned GPT model and evaluate it with the test data.
metrics : Kappa's coefficient and F1 Score
"""

from openai import OpenAI
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from csv_to_json_4o import csv_to_json_without_label


def use_model(msg):
    completion = client.chat.completions.create(
        model="ft:gpt-4o-mini:my-org:custom_suffix:id",
        messages=msg
    )
    return completion.choices[0].message


def evaluate(data, label, window_size, selected_columns):
    """
    1. test 데이터(csv) 파라미터로 받음
    2. 전처리 후 json 형태로 변환하여, 한 task씩 gpt에 입력
    3. gpt의 completion을 버퍼에 저장
    4. 버퍼의 completion을 모아서 실제 라벨과 함께 F1 Score, Kappa Coefficient에 입력
    5. 결과 출력
    """
    model_pred = []

    # Get responses(prediction) from the model
    json_data = csv_to_json_without_label(data, window_size, selected_columns)
    for i in range(len(json_data)):
        response = use_model(json_data[i])
        model_pred.append(response)

    # Convert the model prediction and label to integer
    model_pred = [int(pred) for pred in model_pred]
    label = [int(lab) for lab in label]

    # Calculate Accuracy, F1 Score, Kappa Coefficient
    accuracy = accuracy_score(label, model_pred)
    f1 = f1_score(label, model_pred, average='weighted')
    kappa = cohen_kappa_score(label, model_pred)

    print('Accuracy : {0:.4f}'.format(accuracy))
    print('F1 Score : {0:.4f}'.format(f1))
    print('Kappa Coefficient : {0:.4f}'.format(kappa))
