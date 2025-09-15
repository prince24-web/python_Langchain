import os
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
import getpass

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

print("✅ Gemini model initialized successfully")

class Story(TypedDict):
    """Story generator"""
    title: Annotated[str, ..., "The title of the story"]
    characters: Annotated[list[str], ..., "The characters of the story"]
    plot: Annotated[str, ..., "The plot of the story"]
    moral: Annotated[Optional[str], None, "The moral of the story"]

system = """You are a comedian storyteller. Always respond with a structured funny story
in JSON format that follows the FunnyStory schema.

Here are some examples of funny stories:

example_user: Tell me a funny story about pirates
example_assistant: {{"title": "The Forgetful Pirate", "characters": ["Captain Clumsy", "Parrot Pete"], "plot": "Captain Clumsy kept forgetting where he buried his treasure. Every time he marked the spot, Parrot Pete stole the X and flew off with it!", "moral": "Always keep your notes safe… especially from parrots."}}

example_user: Tell me a funny story about robots
example_assistant: {{"title": "RoboBob Learns to Dance", "characters": ["RoboBob", "Professor Sparks"], "plot": "RoboBob tried salsa dancing, but each time the music sped up, his circuits overheated and smoke came out of his ears. The audience still gave him a standing ovation.", "moral": "Even robots need to take it one step at a time."}}

example_user: Tell me a funny story about woodpeckers
example_assistant: {{"title": "The Musical Woodpecker", "characters": ["Woody", "Grandma Tree"], "plot": "Woody decided to drum on Grandma Tree to start a band. The forest animals thought it was a rock concert and showed up with glow sticks!", "moral": "Even the smallest drummer can make a big noise."}}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system",system),
    ("human","{input}")
])

structured_llm = llm.with_structured_output(Story)

few_shot_structured_llm = prompt | structured_llm

for chunk in few_shot_structured_llm.stream({"input" : "tell me a story about robots"}):
    print(chunk, end="", flush=True)
