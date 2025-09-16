import os
import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import getpass

# Load API key
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")

# --- Define tools using @tool decorator ---
@tool
def joke(setup: str, punchline: str, rating: int):
    """Tell a joke with setup, punchline, and rating."""
    return {"setup": setup, "punchline": punchline, "rating": rating}

@tool
def get_time():
    """Get the current time."""
    return {"time": datetime.datetime.now().strftime("%H:%M:%S")}

@tool
def add_numbers(a: int, b: int):
    """Add two numbers together."""
    return {"result": a + b}

# Create tools list
tools = [joke, get_time, add_numbers]

# Initialize LLM with tools
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

print("✅ Gemini model initialized successfully with tools")

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
    ToolMessage('{"setup": "Why was the cat so good at video games?", "punchline": "Because it had nine lives!", "rating": 8}', tool_call_id="1"),

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
    ToolMessage('{"time": "14:30:25"}', tool_call_id="2"),

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
    ToolMessage('{"result": 12}', tool_call_id="3"),
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

# Create chain
chain = prompt | llm_with_tools

# --- Run query ---
response = chain.invoke({
    "input": "tell me a joke",
    "examples": examples
})

print("🤖 Model wants to use tools:", [call["name"] for call in response.tool_calls] if hasattr(response, "tool_calls") else "None")

# --- Post-process tool calls ---
if hasattr(response, "tool_calls") and response.tool_calls:
    for call in response.tool_calls:
        tool_name = call["name"]
        args = call["args"]

        # Find and execute the correct tool
        tool_function = None
        for tool in tools:
            if tool.name == tool_name:
                tool_function = tool
                break
        
        if tool_function:
            try:
                result = tool_function.invoke(args)
                print(f"✅ Tool executed: {tool_name} →", result)
            except Exception as e:
                print(f"❌ Error executing {tool_name}: {e}")
        else:
            print(f"❌ Unknown tool: {tool_name}")
else:
    print("⚠️ No tool calls found.")