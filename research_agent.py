from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
load_dotenv()


class ResearchResponse(BaseModel):
    topic : str
    summary : str
    sources : list[str]
    tools_used : list[str]


llm = ChatGroq(model="qwen-2.5-32b")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool , wiki_tool , save_tool]
agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(agent=agent,tools =tools,verbose= True)
query = input("what i can help in research?")

raw_response = agent_executor.invoke({"query":query})
try:
    print("Raw response:", raw_response)  # Debug print
    structured_response = parser.parse(raw_response["output"])
    print(structured_response)
except KeyError as e:
    print(f"Missing key in response: {e}")
except Exception as e:
    print(f"Error parsing the response: {e}")
    print(f"Response type: {type(raw_response)}")
