"""
프로세스 순서
1. jsonl 파일(train) 로드
2. 파인튜닝할 GPT 모델 로드
3. 모델에 jsonl 입력으로 넣고 파인튜닝
4. 파인튜닝된 모델 저장
5. 평가함수 정의
6. 테스트 파일로 파인튜닝된 모델 테스트 및 평가
"""

import os
from openai import OpenAI
import openai
import json


def create_file(train_jsonl_dir, test_jsonl_dir):
    """
    Create a file in the OpenAI API
    After creating the file, you should see the file ID(train/test) in the website(platform.openai.com).
    :param train_jsonl_dir: A directory of the train data in jsonl format
    :param test_jsonl_dir: A directory of the test data in jsonl format
    :return:
    """
    # load train data
    client.files.create(
        file=open(train_jsonl_dir, 'rb'),
        purpose='fine-tune'
    )
    print(f"Loaded {train_jsonl_dir}")

    # load test data
    client.files.create(
        file=open(test_jsonl_dir, 'rb'),
        purpose='fine-tune'
    )
    print(f"Loaded {test_jsonl_dir}")


def train(training_file, test_file, model):
    """
    Fine-tuning the GPT model.
    After training, you should check the name of the model in the website(platform.openai.com).
    :param training_file: The ID of the training file
    :param test_file: The ID of the test file
    :param model: anything you want(davinci-002 / gpt-3.5-turbo / and so on)
    """
    # start fine-tuning
    client.fine_tuning.jobs.create(
        training_file=training_file,
        validation_file=test_file,
        model=model
        # hyperparameters는 default=auto 이므로 설정하지 않음
        # hyperparameters={
        #     'n_epochs':10,
        #     'batch_size':16,
        #     'learning_rate_multiplier':1e-4
        # }
    )

    print("Fine-tuning ended.")


def main():
    train_jsonl_dir = ''
    test_jsonl_dir = ''
    training_file = ''
    test_file = ''
    model = 'davinci-002'

    create_file(train_jsonl_dir, test_jsonl_dir)
    train(training_file, test_file, model)



if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    main()