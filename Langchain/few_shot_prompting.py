from langchain_core.prompts import ChatPromptTemplate
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

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

print("✅ Gemini model initialized successfully")

# ----- Define schema for structured output -----
class Joke(TypedDict):
    """Joke to tell user"""
    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1–10"]

# ----- Few-shot examples in the system prompt -----
system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup (the response to "Who's there?") and the final punchline (the response to "<setup> who?").

Here are some examples of jokes:

example_user: Tell me a joke about planes
example_assistant: {{"setup": "Why don't planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2}}

example_user: Tell me another joke about planes
example_assistant: {{"setup": "Cargo", "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!", "rating": 10}}

example_user: Now about caterpillars
example_assistant: {{"setup": "Caterpillar", "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!", "rating": 5}}"""

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{input}")
])

# Apply schema for structured output
structured_llm = llm.with_structured_output(Joke)

# Combine prompt with structured output
few_shot_structured_llm = prompt | structured_llm

# Run
result = few_shot_structured_llm.invoke({"input": "what's something funny about woodpeckers"})
print(result)
