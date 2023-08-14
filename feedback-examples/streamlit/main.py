"""Example Streamlit chat UI that exposes a Feedback button and link to LangSmith traces."""

import streamlit as st
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from expression_chain import get_expression_chain
from vanilla_chain import get_llm_chain
from text2GraphQLChain import get_dgraph_chain


client = Client()

st.set_page_config(
    page_title="Chat LangSmith",
    page_icon="🦜",
    layout="wide",
)
"# Chat🦜🛠️"
# Initialize State
if "messages" not in st.session_state:
    print("Initializing message history")
    st.session_state["messages"] = []
if "trace_link" not in st.session_state:
    st.session_state["trace_link"] = None
st.sidebar.markdown(
    """
# Menu
"""
)
if st.sidebar.button("Clear message history"):
    print("Clearing message history")
    st.session_state.messages = []

# Add a button to choose between llmchain and expression chain
_DEFAULT_SYSTEM_PROMPT = (""" 
"""
)

system_prompt = st.sidebar.text_area(
    "Custom Instructions",
    _DEFAULT_SYSTEM_PROMPT,
    help="Custom instructions to provide the language model to determine style, personality, etc.",
)
system_prompt = system_prompt.strip().replace("{", "{{").replace("}", "}}")

chain_type = st.sidebar.radio(
    "Choose a chain type",
    ("Expression Language Chain", "LLMChain"),
    help="Choose whether to use a vanilla LLMChain or an equivalent chain built using LangChain Expression Language.",
)

# Create Chain
if chain_type == "LLMChain":
    chain, memory = get_dgraph_chain(system_prompt)
else:
    chain, memory = get_expression_chain(system_prompt)


# Display chat messages from history on app rerun
def _get_openai_type(msg):
    if msg.type == "human":
        return "user"
    if msg.type == "ai":
        return "assistant"
    if msg.type == "chat":
        return msg.role
    return msg.type


# for msg in st.session_state.messages:
#     print("Message type:", type(msg))
#     print("Message content:", msg)

#     streamlit_type = _get_openai_type(msg)
#     avatar = "🦜" if streamlit_type == "assistant" else None
#     with st.chat_message(streamlit_type, avatar=avatar):
#         st.markdown(msg.content)
#     # Re-hydrate memory on app rerun
#     memory.chat_memory.add_message(msg)


def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)


run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)
if st.session_state.trace_link:
    st.sidebar.markdown(
        f'<a href="{st.session_state.trace_link}" target="_blank"><button>Latest Trace: 🛠️</button></a>',
        unsafe_allow_html=True,
    )

if prompt := st.chat_input(placeholder="Ask me a question!"):
    st.chat_message("user").write(prompt)
    prefix = "You are a chatbot tasked with helping an Oil Company explore data and identify and remediate issues" \
         "I've provided the graphql schema as well as an example query that shows you how to use all the filters with placeholder values $nameOfRig, $nameOfIssue, and $nameOfEquipment" \
         " Remember to only use a filter on its corresponding type. For example, dont try to filter for issues on Equipment, but on issues  " \
         "dont use any ordering in the graphql query" 
    graphql_fields_all_filters = """
query  {
  queryOilRig(filter: {name: {eq: "$nameOfRig"}})  {
    name
    issues(filter: {name: {anyoftext: "$nameOfIssue"}}) {
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
    augmented_prompt = prefix + " "+ prompt + " " + graphql_fields_all_filters + "Do not include ``` in the Action Input as this will cause an error. it should start with query {"

    with st.chat_message("assistant", avatar="🦜"):
        message_placeholder = st.empty()
        full_response = ""
        if chain_type == "LLMChain":
            scratchpad = "" # populated during agent execution
            message_placeholder.markdown("thinking...")
            print("******PROMPT")
            print(prompt)
            print("*******RunnableConfig*")
            print(runnable_config)
            full_response = chain.invoke(augmented_prompt, config=runnable_config)
        else:
            for chunk in chain.stream({"input": prompt}, config=runnable_config):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response['output'])
        print("************full_response")
        print(full_response)
        memory.save_context({"input": full_response['input']}, {"output": full_response['output']})
        st.session_state.messages = memory.buffer
        # The run collector will store all the runs in order. We'll just take the root and then
        # reset the list for next interaction.
        run = run_collector.traced_runs[0]
        run_collector.traced_runs = []
        col_blank, col_text, col1, col2, col3 = st.columns([10, 2, 1, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run.id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run.id, 0))
        # Requires langsmith >= 0.0.19
        url = client.share_run(run.id)
        # Or if you just want to use this internally
        # without sharing
        # url = client.read_run(run.id).url
        st.session_state.trace_link = url