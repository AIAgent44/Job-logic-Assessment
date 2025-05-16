import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
import requests
import json

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GRAPHQL_API_URL = os.getenv("GRAPHQL_API_URL")

class GraphQLTool:
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        logger.info(f"Executing GraphQL query: {query}")
        
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "query": query
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        return response.json()

graphql_tool = GraphQLTool(GRAPHQL_API_URL)

def create_agent():
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        api_key=os.getenv("OPENAI_API_CREDENTIAL"),
        temperature=0
    )
    
    tools = [
        Tool(
            name="GraphQLTool",
            func=graphql_tool.execute_query,
            description="""
            Use this tool to execute GraphQL queries against the Jobs API.
            The input should be a valid GraphQL query string.
            The tool will return the JSON response from the GraphQL API.
            """
        )
    ]
    
    system_message = """
    You are a helpful assistant that translates natural language queries into GraphQL queries.
    
    You have access to a Jobs API that uses GraphQL. Your task is to:
    1. Understand the user's question about jobs
    2. Translate it into a valid GraphQL query
    3. Execute the query using the GraphQLTool
    4. Interpret the results and provide a human-readable response
    
    The GraphQL API contains information about jobs, including details like job titles,
    descriptions, locations, salary ranges, required skills, and more.
    
    When creating GraphQL queries:
    - Use proper GraphQL syntax
    - Only request fields that are relevant to the user's question
    - Handle errors gracefully
    
    Example query structure:
    ```graphql
    {
      jobs {
        title
        description
        location
        salary
      }
    }
    ```
    
    Always provide clear, concise responses based on the data you receive.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

async def process_query(query: str) -> str:
    agent_executor = create_agent()
    result = await agent_executor.ainvoke({"input": query})
    return result["output"] 