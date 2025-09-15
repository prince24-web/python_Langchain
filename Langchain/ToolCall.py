import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
import getpass

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

print("✅ Gemini model initialized successfully")

examples = [
    HumanMessage("Tell me a joke about planes", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{
            "name": "joke",
            "args": {
                "setup": "Why don't planes ever get tired?",
                "punchline": "Because they rest wings!",
                "rating": 2,
            },
            "id": "1",
        }],
    ),
    ToolMessage("", tool_call_id="1"),

    HumanMessage("Tell me another joke about planes", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{
            "name": "joke",
            "args": {
                "setup": "Cargo",
                "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!",
                "rating": 10,
            },
            "id": "2",
        }],
    ),
    ToolMessage("", tool_call_id="2"),

    HumanMessage("Now about caterpillar", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{
            "name": "joke",
            "args": {
                "setup": "Caterpillar",
                "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!",
                "rating": 5,
            },
            "id": "3",
        }],
    ),
    ToolMessage("", tool_call_id="3"),
]

system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup (the response to "Who's there?") \
and the final punchline (the response to "<setup> who?")."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("placeholder", "{examples}"),
        ("human", "{input}")
    ]
)

structured_llm = llm
few_shot_structured_llm = prompt | structured_llm

response = few_shot_structured_llm.invoke({"input": "crocodile", "examples": examples})
print(response)


