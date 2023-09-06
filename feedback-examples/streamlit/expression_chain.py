from datetime import datetime
from typing import Tuple
import json
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable, RunnableMap
config_file = "config-local.json"  # Name of the user-specific configuration file
with open(config_file, "r") as f:
    config = json.load(f)

OpenAIKey = config["OpenAIKey"]# the host or IP addr where your Dgraph alpha service is running


def get_expression_chain(
    system_prompt: str,
) -> Tuple[Runnable, ConversationBufferMemory]:
    """Return a chain defined primarily in LangChain Expression Language"""
    memory = ConversationBufferMemory(return_messages=True)
    ingress = RunnableMap(
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: memory.load_memory_variables(x)["history"],
            "time": lambda _: str(datetime.now()),
        }
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
    )
    llm = ChatOpenAI(temperature=0.7,openai_api_key=OpenAIKey)
    chain = ingress | prompt | llm
    return chain, memory


if __name__ == "__main__":
    chain, _ = get_expression_chain()
    in_ = "Hi there, I'm a human!"
    print(in_)
    for chunk in chain.stream({"input": in_}):
        print(chunk.content, end="", flush=True)
    in_ = "What's your name?"
    print()
    print(in_)
    for chunk in chain.stream({"input": in_}):
        print(chunk.content, end="", flush=True)
