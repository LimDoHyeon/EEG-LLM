"""
Process order
1. Load the jsonl file (train)
2. Load the GPT model to fine-tune
3. Put the jsonl input into the model and fine-tune it
4. Save the fine-tuned model
5. Define the evaluation function
6. Test the fine-tuned model with the test file and evaluate
"""

import os
from openai import OpenAI
import openai
import json
from Preprocessing.feature_extraction import load_eeg_data
from csv_to_json_4o import csv_to_json_without_label
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score


def create_file(train_jsonl_dir, val_jsonl_dir):
    """
    Create a file in the OpenAI API
    After creating the file, you should see the file ID(train/test) in the website(platform.openai.com).
    :param train_jsonl_dir: A directory of the train data in jsonl format
    :param val_jsonl_dir: A directory of the test data in jsonl format
    :return:
    """
    # load train data
    client.files.create(
        file=open(train_jsonl_dir, 'rb'),
        purpose='fine-tune'
    )
    print(f"Loaded {train_jsonl_dir}")

    # load validation data
    client.files.create(
        file=open(val_jsonl_dir, 'rb'),
        purpose='fine-tune'
    )
    print(f"Loaded {val_jsonl_dir}")


def train(training_file, val_file, model):
    """
    Fine-tuning the GPT model.
    After training, you should check the name of the model in the website(platform.openai.com).
    :param training_file: The ID of the training file
    :param val_file: The ID of the validation file
    :param model: anything you want(davinci-002 / gpt-3.5-turbo / and so on)
    """
    # start fine-tuning
    client.fine_tuning.jobs.create(
        training_file=training_file,
        validation_file=val_file,
        model=model
        # default=auto, thus it doesn't need to be specified
        # hyperparameters={
        #     'n_epochs':10,
        #     'batch_size':16,
        #     'learning_rate_multiplier':1e-4
        # }
    )

    print("Fine-tuning started. Please check the status in the website.")


def use_model(msg, model_id):
    completion = client.chat.completions.create(
        model=model_id,
        messages=msg
    )
    return completion.choices[0].message


def evaluate(data, label, window_size, selected_columns, model_id):
    """
    Process :
    1. Receive test data (csv) as a parameter
    2. Preprocess and convert it into json format, input it into gpt one task at a time
    3. Save the completion of gpt to the buffer
    4. Collect the completions in the buffer and input them into F1 Score and Kappa Coefficient with the actual label
    5. Print the result
    """
    model_pred = []

    # Get responses(prediction) from the model
    json_data = csv_to_json_without_label(data, window_size, selected_columns)
    for i in range(len(json_data)):
        response = use_model(json_data[i], model_id)
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