from langchain import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.utilities import GraphQLAPIWrapper
from langchain.chat_models import ChatOpenAI
import pydgraph
import json
import base64
import getpass
import pandas as pd
from python_graphql_client import GraphqlClient
import os
from datetime import datetime
from typing import Tuple
from langchain import LLMChain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


# Load configuration from user-specific file
config_file = "config-local.json"  # Name of the user-specific configuration file
with open(config_file, "r") as f:
    config = json.load(f)

# Extract values from the configuration
dgraph_cerebro = config["dgraph_cerebro"]
dgraph_graphql_endpoint = config["dgraph_graphql_endpoint"]
dgraph_grpc = config["dgraph_grpc"]
qdrnt_endpoint = config["qdrnt_endpoint"]
qdrnt_collection = config["qdrnt_collection"]
qdrnt_api_key = config["qdrnt_api_key"]
dgraph_cloud_user = config["dgraph_cloud_user"]
dgraph_cloud_passw = config["dgraph_cloud_passw"]
APIAdminKey = config["APIAdminKey"]

os.environ['LANGCHAIN_TRACING_V2'] = config["LANGCHAIN_TRACING_V2"]

os.environ['LANGCHAIN_ENDPOINT']=config["LANGCHAIN_ENDPOINT"]
os.environ['LANGCHAIN_API_KEY']=config["LANGCHAIN_API_KEY"]

OpenAIKey = config["OpenAIKey"]# the host or IP addr where your Dgraph alpha service is running
print(dgraph_cerebro)


# graph admin endpoint is /admin
dgraph_graphql_admin = dgraph_graphql_endpoint.replace("/graphql", "/admin")
print(dgraph_graphql_admin)

# DQL Client
client_stub = pydgraph.DgraphClientStub.from_cloud(dgraph_grpc,APIAdminKey )
client = pydgraph.DgraphClient(client_stub)


# GraphQL client and admin client
gql_client = GraphqlClient(endpoint=dgraph_graphql_endpoint)
headers = { "Dg-Auth": APIAdminKey }
gql_admin_client = GraphqlClient(endpoint=dgraph_graphql_admin, headers=headers)
gql_cloud_client = GraphqlClient(endpoint=dgraph_cerebro)

print("graphql client connections/objects established")

def get_dgraph_chain(
    system_prompt: str,
) -> Tuple[LLMChain, ConversationBufferMemory]:
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    """Return a basic LLMChain with memory."""
    prefix = "You are a chatbot tasked with helping an Oil Company explore data and identify and remediate issues" \
         "I've provided the graphql schema as well as an example query that shows you how to use all the filters with placeholder values $nameOfRig, $nameOfIssue, and $nameOfEquipment" \
         " Remember to only use a filter on its corresponding type. For example, dont try to filter for issues on Equipment, but on issues  " \
         "dont use any ordering in the graphql query" 
    graphql_fields_all_filters = """
query  {
  queryOilRig(filter: {name: {eq: "$nameOfRig"} })  {
    name
    issues(filter: {name: {anyoftext: "$nameOfIssue"} }) {
      id
      name
      description
      solution
      similarIssues {
        name
        description
        score
        solution
    }
    }
    equipment(filter: {name: {anyofterms: "$nameOfEquipment"}}) {
      name
    }
  }
}"""     
    suffix = """

    {chat_history}
    Question: {input}
    {agent_scratchpad}
    """
    gpt4LLM = ChatOpenAI(temperature=0,model_name='gpt-4', openai_api_key=OpenAIKey)

    tools = load_tools(
        ["graphql"],
        graphql_endpoint=dgraph_graphql_endpoint,
        gql_client=gql_client,
        llm=gpt4LLM,
    )
    zsaprompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt + "\nIt's currently {time}.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    ).partial(time=lambda: str(datetime.now()))


    chain = LLMChain(prompt=zsaprompt, llm=gpt4LLM)
    agent = ZeroShotAgent(llm_chain=chain, tools=tools, verbose=True)

    agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory)
    
    return agent_chain, memory


if __name__ == "__main__":
    chain, _ = get_dgraph_chain()
    print(chain.invoke({"input": "Hi there, I'm a human!"})["text"])
    print(chain.invoke({"input": "What's your name?"})["text"])