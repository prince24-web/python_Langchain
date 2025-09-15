import os
from dotenv import load_dotenv
import google.generativeai as genai
from test import askGemini
import argparse

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found")

genai.configure(api_key=api_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', nargs='*', help='Prompt to send to Gemini')
    args = parser.parse_args()
    prompt = " ".join(args.prompt) if args.prompt else input("Enter your prompt: ")
    print(askGemini(prompt))