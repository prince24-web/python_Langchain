import os
import getpass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional
from typing_extensions import TypedDict, Annotated

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

#TypedDict
class Joke(TypedDict):
    """ Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "Punchling of the joke"]
    rating: Annotated[Optional[int], None, "how funny the joke is, from 1 to 10"]
structured_llm = llm.with_structured_output(Joke)

result = structured_llm.invoke("Tell me a joke about cats")
print("\nHere’s your joke 😺:")
print(result)