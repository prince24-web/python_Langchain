import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("⚠️ Please set GEMINI_API_KEY in your .env file")

# Step 1: Define schema
class SearchResult(BaseModel):
    """Structured search result."""
    query: str = Field(description="The search query")
    findings: str = Field(description="Summary of findings")

# Step 2: Define a fake search tool
def fake_web_search(query: str):
    """Pretend to search the web."""
    if "AI" in query:
        return f"Found lots of AI research papers in 2025, including progress on RAG and SFT."
    else:
        return f"No major results for '{query}' — try another query."

search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
}

# Step 3: Setup Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
llm_with_search = llm.bind_tools([search_tool])

# Step 4: Enforce structured output with Pydantic
structured_search_llm = llm_with_search.with_structured_output(SearchResult)

# Step 5: Ask Gemini
result = structured_search_llm.invoke("Search for latest AI research and summarize")

# Step 6: Print results
print("✅ Structured Result:")
print(result)

# Simulate actually using the tool
print("\n🔎 Fake Web Search Output:")
print(fake_web_search(result.query))
