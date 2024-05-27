import json
import pandas as pd
import numpy as np
import openai
import backoff
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to read JSON data
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Function to preprocess data for fine-tuning
def preprocess_data(data):
    fine_tune_data = []
    for item in data:
        prompt = item['prompt']  # Adjust according to your JSON structure
        completion = item['completion']  # Adjust according to your JSON structure
        fine_tune_data.append({"prompt": prompt, "completion": completion})
    return fine_tune_data

# Function to save preprocessed data to JSONL format
def save_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

# Function to perform fine-tuning
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def fine_tune_model(training_file):
    response = openai.File.create(
        file=open(training_file, 'rb'),
        purpose='fine-tune'
    )
    fine_tune_response = openai.FineTune.create(training_file=response['id'])
    return fine_tune_response

def main():
    # Read and preprocess the data
    input_json_path = 'data.json'  # Adjust to your JSON file path
    output_jsonl_path = 'fine_tune_data.jsonl'

    data = read_json(input_json_path)
    processed_data = preprocess_data(data)
    save_to_jsonl(processed_data, output_jsonl_path)

    # Perform fine-tuning
    fine_tune_response = fine_tune_model(output_jsonl_path)
    print(f"Fine-tuning job created: {fine_tune_response['id']}")

if __name__ == "__main__":
    main()
