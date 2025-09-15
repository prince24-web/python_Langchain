import json
from test import askGemini
import argparse


def summarize_and_quiz(text):
    prompt = f"""
You are a helpful exam tutor. Read the TEXT below and return ONLY valid JSON with keys:
- summary: a short 2-3 sentence summary
- quiz: a list of 3 multiple-choice questions. Each question must be an object with:
  - question (string)
  - options (list of 4 strings)
  - answer_index (0-3) which is the index of the correct option

TEXT:
{text}

Return only JSON.
"""
    raw = askGemini(prompt).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[len("json"):].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw response:")
        print(raw)
        raise


def display(result):
    """Nicely print the summary and quiz from Gemini response"""
    print("\n📌 Summary:")
    print(result["summary"])
    print("\n📝 Quiz:")

    score = 0
    number_of_quiz = len(result["quiz"])

    for i, q in enumerate(result["quiz"], start=1):
        print(f"\n{i}. {q['question']}")
        for j, option in enumerate(q["options"], start=1):
            print(f"   {chr(96+j)}) {option}")  # a), b), c), d)

        user_input = input("Your answer (a/b/c/d): ").strip().lower()

        correct_letter = chr(97 + q["answer_index"])
        correct_option = q["options"][q["answer_index"]]
        

        if user_input == correct_letter:
            score+= 1
            print(f" correct! ✅, score current score is {score} / {number_of_quiz}")
        else:
            print(f" ❌ Wrong. correct anser: {correct_letter}) {correct_option}")

    print("\n Total score: ")
    print(f"you got {score} out of {number_of_quiz}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', nargs='*', help='Prompt to send to Gemini')
    args = parser.parse_args()
    prompt = " ".join(args.prompt) if args.prompt else input("Enter your prompt: ")

    result = summarize_and_quiz(prompt)
    display(result)
