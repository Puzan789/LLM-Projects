import requests
import json
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Load API keys
api_keys = [
    os.getenv("API_KEY_1"),
    os.getenv("API_KEY_2"),
    os.getenv("API_KEY_3"),
    os.getenv("API_KEY_4")
]

# Base URL for the API request
URL = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"

# Initial index for API keys
current_key_index = 0

def get_llm_response(generated_prompt: str, retries=3, backoff_factor=2):
    global current_key_index

    while retries > 0:
        # Set the current API key
        api_key = api_keys[current_key_index]
        
        # Define headers for the current API key
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': api_key
        }

        # Define payload
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": generated_prompt
                        }
                    ]
                }
            ]
        }

        # Make the request
        response = requests.post(URL, headers=headers, data=json.dumps(payload))

        # Check for 429 Resource Exhausted
        if response.status_code == 429:
            print(f"Quota exceeded for API key {api_key}. Switching API key...")

            # Switch to the next API key
            current_key_index = (current_key_index + 1) % len(api_keys)

            # Decrement the retries counter and wait before retrying
            retries -= 1
            wait_time = backoff_factor ** (3 - retries)  # Exponential backoff
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            # If the response is not 429, return the result
            response_json = response.json()
            try:
                llm_output = response_json['candidates'][0]['content']['parts'][0]['text']
                return llm_output
            except KeyError as e:
                print(f"Error processing response: {e}")
                return None

    print("All API keys exhausted or retries failed.")
    return None  # Return None if all retries and keys are exhausted



if __name__ == "__main__":
    print("Starting")
    generated_prompt = "What is the significance of the Declaration of Independence?"
    response = get_llm_response(generated_prompt)
    print("Response:", response)



