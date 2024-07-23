"""
Load the fine-tuned GPT model and evaluate it with the test data.
metrics : Kappa's coefficient and F1 Score
"""

from openai import OpenAI
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score


def use_model(msg):
    response = client.chat.completions.create(
        model="ft:davinci-002:personal::9nzVtmd3",
        messages=[],  # [] 자리에 msg 들어가야 함
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message

def eval():
    """
    1. test 데이터(csv) 로드
    2. 전처리 후 json 형태로 변환하여, 한 task씩 gpt에 입력
    3. gpt의 completion을 버퍼에 저장
    4. 버퍼의 completion을 모아서 실제 라벨과 함께 F1 Score, Kappa Coefficient에 입력
    5. 결과 출력
    """