from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    

llm = ChatAnthropic(model="gpt-5-nano")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)



agent = create_agent(
    model=llm,
    tools=[search_tool, wiki_tool, save_tool],
    system_prompt=(
        "You are a research assistant that helps generate structured research reports. "
        "Use the available tools when helpful. "
        "Always fill in topic, summary, sources, and tools_used."
    ),
    response_format=ResearchResponse,  # Pydantic structured output
)

query = input("What can I help you research? ")

result = agent.invoke({
    "messages": [
        {"role": "user", "content": query}
    ]
})

structured_response: ResearchResponse = result["structured_response"]

print(structured_response)
