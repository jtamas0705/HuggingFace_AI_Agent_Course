import datasets
import asyncio
import httpx
import random
from dotenv import load_dotenv
from llama_index.core.schema import Document
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from huggingface_hub import list_models
from duckduckgo_search import DDGS

load_dotenv()

def get_websearch_tool(query: str):
    # Mimicking a real browser often bypasses the 202 error
    with DDGS(headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}) as ddgs:
        results = [r for r in ddgs.text(query, max_results=3)]
        return results

def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        text="\n".join([
            f"Name: {guest_dataset['name'][i]}",
            f"Relation: {guest_dataset['relation'][i]}",
            f"Description: {guest_dataset['description'][i]}",
            f"Email: {guest_dataset['email'][i]}"
        ]),
        metadata={"name": guest_dataset['name'][i]}
    )
    for i in range(len(guest_dataset))
]

bm25_retriever = BM25Retriever.from_defaults(nodes=docs)

def get_guest_info_retriever(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.retrieve(query)
    if results:
        return "\n\n".join([doc.text for doc in results[:3]])
    else:
        return "No matching guest information found."


async def main():
    # 1. Create a single client for the entire session
    async with httpx.AsyncClient() as client:
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=2048, http_client=client)

        websearch_tool = FunctionTool.from_defaults(get_websearch_tool)
        # Initialize the tool
        guest_info_tool = FunctionTool.from_defaults(get_guest_info_retriever)
        # Initialize the tool
        weather_info_tool = FunctionTool.from_defaults(get_weather_info)

        # Initialize the tool
        hub_stats_tool = FunctionTool.from_defaults(get_hub_stats)

        # Create Alfred, our gala agent, with the guest info tool
        alfred = AgentWorkflow.from_tools_or_functions(
            [guest_info_tool, websearch_tool, weather_info_tool, hub_stats_tool],
        )
        # Remembering state
        ctx = Context(alfred)
        # First interaction
        response1 = await alfred.run("Tell me about Lady Ada Lovelace.", ctx=ctx)
        print("🎩 Alfred's First Response:")
        print(response1)

        # Second interaction (referencing the first)
        response2 = await alfred.run("What projects is she currently working on?", ctx=ctx)
        print("🎩 Alfred's Second Response:")
        print(response2)

if __name__ == "__main__":
    asyncio.run(main())