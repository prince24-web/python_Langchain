import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

# Get API key from .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set GEMINI_API_KEY in your .env file.")

# Configure Gemini
genai.configure(api_key=api_key)

def askGemini(prompt, model_name="gemini-1.5-flash"):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    print(askGemini("hey gemini, what cool stuff can i achive with ur api and python"))