import os
import getpass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Union, Optional
from pydantic import BaseModel, Field

# Load .env file
load_dotenv()

# Check if API key exists, otherwise ask
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")

# Initialize Gemini model using LangChain’s integration
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

print("✅ Gemini model initialized successfully")

# --- Define Pydantic Schemas ---
class Joke(BaseModel):
    """Joke to tell user"""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )

class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""
    response: str = Field(description="A conversational response to the user's query")

class FinalResponse(BaseModel):
    """Final structured response: either a joke or a conversational answer"""
    final_output: Union[Joke, ConversationalResponse]

# --- Structured Output ---
structured_llm = llm.with_structured_output(FinalResponse)

# Invoke and store result
result = structured_llm.invoke("Tell me a joke about cats")

# Print nicely
print("\n🎭 Model Response:")
print(result)
