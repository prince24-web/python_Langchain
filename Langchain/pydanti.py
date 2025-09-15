import os
import getpass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional
from pydantic import BaseModel, Field
# Load environment variables from .env
load_dotenv()

# Check if API key exists, otherwise ask
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")

# Initialize Gemini model using LangChain’s integration.
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

print("✅ Gemini model initialized successfully")

class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchling to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )

structured_llm = llm.with_structured_output(Joke)

result = structured_llm.invoke("Tell me a joke about cats")
print("\nHere’s your joke 😺:")
print(f" {result.setup}")
print(f" {result.punchline}")

if result.rating:
    print(f"(Funny rating: {result.rating}/10)")

