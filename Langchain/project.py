import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from duckduckgo_search import DDGS

# --- Load API Key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found! Add it to your .env file.")

# --- Search Tool ---
@tool
def web_search(query: str, num_results: int = 5):
    """Search the web using DuckDuckGo"""
    results = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=num_results):
            results.append({
                "title": result.get("title", "No title"),
                "url": result.get("href", ""),
                "snippet": result.get("body", "")[:200] + "..."
            })
    return results

tools = [web_search]

# --- Initialize Gemini ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# --- Few-shot examples ---
examples = [
    HumanMessage("Search for Python tutorials", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{
            "name": "web_search",
            "args": {"query": "Python tutorials", "num_results": 3},
            "id": "1"
        }],
    ),
    ToolMessage(
        '{"query": "Python tutorials", "results": [{"title": "Learn Python", "url": "https://realpython.com", "snippet": "Python tutorials for developers..."}], "count": 1}',
        tool_call_id="1"
    ),
]

# --- System Instructions ---
system_prompt = """You are a helpful AI research assistant.
When users ask for information, always use the web_search tool.
Return results in a clear format with titles, URLs, and descriptions.
"""

# --- Prompt with examples ---
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{examples}"),  # <- few-shot goes here
    ("human", "{input}")
])

# --- Build chain ---
chain = prompt | llm_with_tools

# --- Run a query ---
print("🔍 Asking Gemini...")
response = chain.invoke({
    "input": "how old is donald trump",
    "examples": examples
})

# --- Handle tool calls ---
if hasattr(response, "tool_calls") and response.tool_calls:
    for call in response.tool_calls:
        tool_name = call["name"]
        args = call["args"]
        print(f"\n🤖 Gemini wants to use: {tool_name} with {args}")

        tool_dict = {t.name: t for t in tools}
        tool_function = tool_dict.get(tool_name)

        if tool_function:
            results = tool_function.invoke(args)
            print(f"\n✅ Found {len(results)} results:")
            for i, item in enumerate(results, 1):
                print(f"{i}. {item['title']}")
                print(f"   🔗 {item['url']}")
                print(f"   📄 {item['snippet']}\n")
        else:
            print(f"❌ Unknown tool: {tool_name}")
else:
    print("💬 Response:", response.content)
