"""Example Streamlit chat UI that exposes a Feedback button and link to LangSmith traces."""

import streamlit as st
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from expression_chain import get_expression_chain
from vanilla_chain import get_llm_chain
from text2GraphQLChain import get_dgraph_chain

from PIL import Image

image = Image.open("img/dgraph_bw_icon.png")

client = Client()

st.set_page_config(
    page_title="Chat LangSmith",
    page_icon=image,
    layout="wide",
)

st.markdown(
    """
    <style>
        .css-1avcm0n {
            background-color: #23212a;
        }
        
        .css-ve1keh.e1f1d6gn0 {
            display: flex;
        
        }
        
        .css-k7vsyb e1nzilvr1 {
            width: 50%;
        }
        
        #dgraph-chat {
            width: 100%;
        }
    
        div.css-1kyxreq.e115fcil2 {
            width: 50%; 
            background-color: #f7c022;
        }

        div.css-6qob1r.eczjsme3 {
            background-color: #100c19;
        }
        
        div.block-container.css-z5fcl4.ea3mdgi4 {
            background-color: #f4f4f4;
        }
        
        .element-container css-txwtdl e1f1d6gn2 {
            width: 50%;
        }
        
        .element-container css-txwtdl e1f1d6gn2 {
            width: 50%;
        }
        
        div.css-qcqlej.ea3mdgi3 {
            background-color: #f4f4f4;
        }
        
        div.stChatFloatingInputContainer.css-usj992.e1d2x3se2 {
            background-color: #f4f4f4;
        }
        
        .css-1yrzt5d {
            background-color: #f7c022;
        }
        
        span.css-10trblm.e1nzilvr0 {
            color: #100c19;
        }
        
        div.css-janbn0 {
           background-color: #ef255a;
           margin-right: 2rem;
        }
        
        div.stChatMessage.css-4oy321.eeusbqq4 {
            background-color: #23212a;
            border-radius: 6px;
            margin-left: 2rem;
        }
        
        .css-eb3jpk.eeusbqq2 {
            background-color: #85868a;
        }
        
        .css-4oy321 {
            padding: 1rem;
        }
        
        
        textarea.st-c0 {
            background-color: #23212a;
        }
        
        .css-hc3laj {
            background-color: #ef255a;
        }

        .st-d5 {
            background-color: #ef255a;
        }
        
        button.css-hc3laj.ef3psqc1:hover {
            border-color: #f7c022;
        }
        
        buttun.css-hc3laj.ef3psqc11:hover {
            border-color: #f7c022;
        }
        
        p {
            :hover {
                color: #f7c022;
            }
        }
        
        #titles-dgraph {
            display: flex;
            width: 100%;
            align-items: center;
        }
        button.css-19rxjzo.ef3psqc11 {
            border: 2px solid rgba(250, 250, 250, 0.2);
        }
        
        .css-hc3laj.ef3psqc11:hover {
            border-color: #f7c022;
        }
        
        .css-x78sv8.e1nzilvr4:hover {
            color: #f7c022;
        }
        
        #d_logo {
            margin-left: -2rem;
        }
        
    </style>
    <div id="titles-dgraph">
        <h1>Dgraph Chat </h1> 
        <img id="d_logo" src="http://localhost:8501/media/17389ca047525a2a325878f592e8602cbbda599ff9b885bfdd51414e.png" width=50 height=50>
    </div>
    """,
    unsafe_allow_html=True 
)

# "# Chat🦜🛠️"
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
    augmented_prompt = prefix + " "+ prompt + "Please do not include ``` in the Action Input as this will cause the query to error. " + graphql_fields_all_filters + "Do not include ``` in the Action Input as this will cause an error. it should start with query {"

    with st.chat_message("assistant", avatar="🤖"):
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