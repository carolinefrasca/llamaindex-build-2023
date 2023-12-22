from typing import Dict, Any
import asyncio

# Create a new event loop
loop = asyncio.new_event_loop()

# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    download_loader,
)
from llama_index.llama_pack.base import BaseLlamaPack
from llama_index.llms import OpenAI
import streamlit as st
from streamlit_pills import pills

st.set_page_config(
    page_title=f"Chat with {self.wikipedia_page}'s Wikipedia page, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

if "messages" not in st.session_state:  # Initialize the chat messages history
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Ask me a question about Snowflake!"}
    ]

st.title(
    f"Chat with {self.wikipedia_page}'s Wikipedia page, powered by LlamaIndex 💬🦙"
)
st.info(
    "This example is powered by the **[Llama Hub Wikipedia Loader](https://llamahub.ai/l/wikipedia)**. Use any of [Llama Hub's many loaders](https://llamahub.ai/) to retrieve and chat with your data via a Streamlit app.",
    icon="ℹ️",
)

def add_to_message_history(role, content):
    message = {"role": role, "content": str(content)}
    st.session_state["messages"].append(
        message
    )  # Add response to message history

@st.cache_resource
def load_index_data():
    WikipediaReader = download_loader(
        "WikipediaReader", custom_path="local_dir"
    )
    loader = WikipediaReader()
    docs = loader.load_data(pages=[self.wikipedia_page])
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5)
    )
    index = VectorStoreIndex.from_documents(
        docs, service_context=service_context
    )
    return index

index = load_index_data()

selected = pills(
    "Choose a question to get started or write your own below.",
    [
        "What is Snowflake?",
        "What company did Snowflake announce they would acquire in October 2023?",
        "What company did Snowflake acquire in March 2022?",
        "When did Snowflake IPO?",
    ],
    clearable=True,
    index=None,
)

if "chat_engine" not in st.session_state:  # Initialize the query engine
    st.session_state["chat_engine"] = index.as_chat_engine(
        chat_mode="context", verbose=True
    )

for message in st.session_state["messages"]:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# To avoid duplicated display of answered pill questions each rerun
if selected and selected not in st.session_state.get(
    "displayed_pill_questions", set()
):
    st.session_state.setdefault("displayed_pill_questions", set()).add(selected)
    with st.chat_message("user"):
        st.write(selected)
    with st.chat_message("assistant"):
        response = st.session_state["chat_engine"].stream_chat(selected)
        response_str = ""
        response_container = st.empty()
        for token in response.response_gen:
            response_str += token
            response_container.write(response_str)
        add_to_message_history("user", selected)
        add_to_message_history("assistant", response)

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    add_to_message_history("user", prompt)

    # Display the new question immediately after it is entered
    with st.chat_message("user"):
        st.write(prompt)

    # If last message is not from assistant, generate a new response
    # if st.session_state["messages"][-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = st.session_state["chat_engine"].stream_chat(prompt)
        response_str = ""
        response_container = st.empty()
        for token in response.response_gen:
            response_str += token
            response_container.write(response_str)
        # st.write(response.response)
        add_to_message_history("assistant", response.response)

    # Save the state of the generator
    st.session_state["response_gen"] = response.response_gen



# import streamlit as st
# from llama_index.llms import OpenAI
# import openai
# import logging
# import sys
# import os
# import llama_hub
# from streamlit_pills import pills
# from llama_index.tools.query_engine import QueryEngineTool
# from llama_index.objects import ObjectIndex, SimpleToolNodeMapping
# # from llama_index.query_engine import ToolRetrieverRouterQueryEngine
# from llama_index import (
#     VectorStoreIndex,
#     SummaryIndex,
#     ServiceContext,
#     StorageContext,
#     download_loader
# )
# from llama_index.query_engine.router_query_engine import RouterQueryEngine
# from llama_index.selectors.llm_selectors import (
#     LLMSingleSelector,
#     LLMMultiSelector,
# )
# from llama_index.selectors.pydantic_selectors import (
#     PydanticMultiSelector,
#     PydanticSingleSelector,
# )

# st.set_page_config(page_title="Chat with Snowflake's Wikipedia page, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
# st.title("Chat with Snowflake's Wikipedia page, powered by LlamaIndex 💬🦙")
# st.info("Because this chatbot is powered by **LlamaIndex's [router query engine](https://gpt-index.readthedocs.io/en/latest/examples/query_engine/RouterQueryEngine.html)**, it can answer both **summarization questions** and **context-specific questions** based on the contents of [Snowflake's Wikipedia page](https://en.wikipedia.org/wiki/Snowflake_Inc.).", icon="ℹ️")
# openai.api_key = st.secrets.openai_key

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# @st.cache_resource
# def load_index_data():
#     WikipediaReader = download_loader("WikipediaReader",custom_path="local_dir")
#     loader = WikipediaReader()
#     documents = loader.load_data(pages=['Snowflake Inc.'])

#     # initialize service context (set chunk size)
#     service_context = ServiceContext.from_defaults(chunk_size=1024)
#     nodes = service_context.node_parser.get_nodes_from_documents(documents)

#     # initialize storage context (by default it's in-memory)
#     storage_context = StorageContext.from_defaults()
#     storage_context.docstore.add_documents(nodes)

#     summary_index = SummaryIndex(nodes, storage_context=storage_context)
#     vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

#     return summary_index, vector_index

# summary_index, vector_index = load_index_data()

# if "list_query_engine" not in st.session_state.keys(): # Initialize the query engine
#     st.session_state.list_query_engine = summary_index.as_query_engine(response_mode="tree_summarize",use_async=True,)

# if "vector_query_engine" not in st.session_state.keys():
#     st.session_state.vector_query_engine = vector_index.as_query_engine()

# list_tool = QueryEngineTool.from_defaults(
#     query_engine=st.session_state.list_query_engine,
#     description=(
#         "Useful for questions summarizing Snowflake's Wikipedia page"
#     ),
# )

# vector_tool = QueryEngineTool.from_defaults(
#     query_engine=st.session_state.vector_query_engine,
#     description=(
#         "Useful for retrieving specific information about Snowflake"
#     ),
# )

# if "router_query_engine" not in st.session_state.keys(): # Initialize the query engine
#     st.session_state.router_query_engine = RouterQueryEngine(selector=PydanticSingleSelector.from_defaults(), query_engine_tools=[list_tool,vector_tool,],)

# selected = pills("Choose a question to get started or write your own below.", ["What is Snowflake?", "What company did Snowflake announce they would acquire in October 2023?", "What company did Snowflake acquire in March 2022?", "When did Snowflake IPO?"], clearable=True, index=None)

# if "messages" not in st.session_state.keys(): # Initialize the chat messages history
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Ask me a question about Snowflake!"}
#     ]

# def add_to_message_history(role, content):
#     message = {"role": role, "content": str(content)}
#     st.session_state.messages.append(message) # Add response to message history

# for message in st.session_state.messages: # Display the prior chat messages
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# query_engines=["list query engine","vector query engine",]

# if selected:
#     with st.chat_message("user"):
#         st.write(selected)
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = st.session_state.router_query_engine.query(selected)
#             st.write(str(response))
#             add_to_message_history("user",selected)
#             add_to_message_history("assistant",response)
#             selector_dict = response.metadata["selector_result"].dict()
#             query_engine_index = selector_dict["selections"][0]["index"]
#             query_engine_used = query_engines[query_engine_index]
#             reason = selector_dict["selections"][0]["reason"]
#             if reason[0:4]=="Snow":
#                 explanation = "Used the **" + query_engine_used + "** to answer this question because " + reason
#             else:
#                 explanation = "Used the **" + query_engine_used + "** to answer this question because " + reason[0:1].lower() + reason[1:]
#             st.success(explanation,icon="✅")

# if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

# # If last message is not from assistant, generate a new response
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = st.session_state.router_query_engine.query(prompt)
#             st.write(str(response))
#             add_to_message_history("assistant", response)
#             selector_dict = response.metadata["selector_result"].dict()
#             query_engine_index = selector_dict["selections"][0]["index"]
#             query_engine_used = query_engines[query_engine_index]
#             reason = selector_dict["selections"][0]["reason"]
#             if reason[0:4]=="Snow":
#                 explanation = "Used the **" + query_engine_used + "** to answer this question because " + reason
#             else:
#                 explanation = "Used the **" + query_engine_used + "** to answer this question because " + reason[0:1].lower() + reason[1:]
#             st.success(explanation,icon="✅")
