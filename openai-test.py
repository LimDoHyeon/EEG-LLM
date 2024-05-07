import openai

def fine_tune_model(training_data, api_key):
    """
    Fine-tune the GPT-3 model using the OpenAI API.
    """
    openai.api_key = api_key
    openai.Completion.create(
        engine='gpt-3.5-turbo',
        training_data=training_data
    )
