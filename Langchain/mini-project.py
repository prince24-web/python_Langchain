import os, datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import getpass

# Load API key
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
print("✅ Gemini model initialized successfully")

# --- Define tools ---
def joke(setup: str, punchline: str, rating: int):
    return {"setup": setup, "punchline": punchline, "rating": rating}

def get_time():
    return {"time": datetime.datetime.now().strftime("%H:%M:%S")}

def add_numbers(a: int, b: int):
    return {"result": a + b}

# --- Few-shot examples ---
examples = [
    HumanMessage("Tell me a joke about cats", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{
            "name": "joke",
            "args": {
                "setup": "Why was the cat so good at video games?",
                "punchline": "Because it had nine lives!",
                "rating": 8,
            },
            "id": "1"
        }],
    ),
    ToolMessage("", tool_call_id="1"),

    HumanMessage("What time is it", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{
            "name": "get_time",
            "args": {},
            "id": "2"
        }],
    ),
    ToolMessage("", tool_call_id="2"),

    HumanMessage("Add 5 and 7", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{
            "name": "add_numbers",
            "args": {"a": 5, "b": 7},
            "id": "3"
        }],
    ),
    ToolMessage("", tool_call_id="3"),
]

# --- Prompt ---
system = """You are a funny but useful assistant. 
You can tell jokes, give the current time, or do simple math using the correct tool calls.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("placeholder", "{examples}"),
    ("human", "{input}")
])

structured_llm = llm
few_shot_tool_llm = prompt | structured_llm

# --- Run query ---
response = few_shot_tool_llm.invoke({
    "input": "Can you add 10 and 25?",
    "examples": examples
})

print("🤖 Model response:", response)

# --- Post-process tool call ---
if hasattr(response, "tool_calls") and response.tool_calls:
    for call in response.tool_calls:
        tool_name = call["name"]
        args = call["args"]

        if tool_name == "add_numbers":
            result = add_numbers(**args)
        elif tool_name == "joke":
            result = joke(**args)
        elif tool_name == "get_time":
            result = get_time()
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        print(f"✅ Tool executed: {tool_name} →", result)
else:
    print("⚠️ No tool calls found.")

