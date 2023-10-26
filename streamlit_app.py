import streamlit as st
from llama_index.llms import OpenAI
import openai
import logging
import sys
import os
import llama_hub
from streamlit_pills import pills
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.objects import ObjectIndex, SimpleToolNodeMapping
# from llama_index.query_engine import ToolRetrieverRouterQueryEngine
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    ServiceContext,
    StorageContext,
    download_loader
)
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import (
    LLMSingleSelector,
    LLMMultiSelector,
)
from llama_index.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)

st.set_page_config(page_title="Chat with Snowflake's Wikipedia page, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with Snowflake's Wikipedia page, powered by LlamaIndex 💬🦙")
st.info("Because this chatbot is powered by **LlamaIndex's [router query engine](https://gpt-index.readthedocs.io/en/latest/examples/query_engine/RouterQueryEngine.html)**, it can answer both **summarization questions** and **context-specific questions** based on the contents of [Snowflake's Wikipedia page](https://en.wikipedia.org/wiki/Snowflake_Inc.).", icon="ℹ️")
openai.api_key = st.secrets.openai_key

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

@st.cache_resource
def load_index_data():
    WikipediaReader = download_loader("WikipediaReader",custom_path="local_dir")
    loader = WikipediaReader()
    documents = loader.load_data(pages=['Snowflake Inc.'])

    # initialize service context (set chunk size)
    service_context = ServiceContext.from_defaults(chunk_size=1024)
    nodes = service_context.node_parser.get_nodes_from_documents(documents)

    # initialize storage context (by default it's in-memory)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    summary_index = SummaryIndex(nodes, storage_context=storage_context)
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

    return summary_index, vector_index

    # list_query_engine = summary_index.as_query_engine(
    #     response_mode="tree_summarize",
    #     use_async=True,
    # )
    # vector_query_engine = vector_index.as_query_engine()

    # list_tool = QueryEngineTool.from_defaults(
    #     query_engine=list_query_engine,
    #     description=(
    #         "Useful for questions summarizing Snowflake's Wikipedia page"
    #     ),
    # )

    # vector_tool = QueryEngineTool.from_defaults(
    #     query_engine=vector_query_engine,
    #     description=(
    #         "Useful for retrieving specific information about Snowflake"
    #     ),
    # )
    # return list_tool, vector_tool

    # if "query_engine" not in st.session_state.keys(): # Initialize the query engine
    #     st.session_state.query_engine = RouterQueryEngine(selector=PydanticSingleSelector.from_defaults(), query_engine_tools=[list_tool,vector_tool,],)

    # query_engine = RouterQueryEngine(
    #     selector=PydanticSingleSelector.from_defaults(),
    #     query_engine_tools=[
    #         list_tool,
    #         vector_tool,
    #     ],
    # )
    # return st.session_state.query_engine

     # initialize service context (set chunk size)
    # service_context = ServiceContext.from_defaults(chunk_size=1024)
    # nodes = service_context.node_parser.get_nodes_from_documents(documents)

    # # initialize storage context (by default it's in-memory)
    # storage_context = StorageContext.from_defaults()
    # storage_context.docstore.add_documents(nodes)

    # summary_index = SummaryIndex(nodes, storage_context=storage_context)
    # vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

    # list_query_engine = summary_index.as_query_engine(
    #     response_mode="tree_summarize", use_async=True
    # )
    # vector_query_engine = vector_index.as_query_engine(
    #     response_mode="tree_summarize", use_async=True
    # )

    # list_tool = QueryEngineTool.from_defaults(
    #     query_engine=list_query_engine,
    #     description="Useful for questions summarizing Snowflake's Wikipedia page",
    # )
    # vector_tool = QueryEngineTool.from_defaults(
    #     query_engine=vector_query_engine,
    #     description=(
    #         "Useful for retrieving specific information about Snowflake"
    #     ),
    # )

    # tool_mapping = SimpleToolNodeMapping.from_objects([list_tool, vector_tool])
    # obj_index = ObjectIndex.from_objects(
    #     [list_tool, vector_tool],
    #     tool_mapping,
    #     VectorStoreIndex,
    # )
    # return obj_index

# @st.cache_data
# def index_data(documents):
#     # initialize service context (set chunk size)
#     service_context = ServiceContext.from_defaults(chunk_size=1024)
#     nodes = service_context.node_parser.get_nodes_from_documents(documents)

#     # initialize storage context (by default it's in-memory)
#     storage_context = StorageContext.from_defaults()
#     storage_context.docstore.add_documents(nodes)

#     summary_index = SummaryIndex(nodes, storage_context=storage_context)
#     vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

#     list_query_engine = summary_index.as_query_engine(
#         response_mode="tree_summarize", use_async=True
#     )
#     vector_query_engine = vector_index.as_query_engine(
#         response_mode="tree_summarize", use_async=True
#     )

#     list_tool = QueryEngineTool.from_defaults(
#         query_engine=list_query_engine,
#         description="Useful for questions summarizing Snowflake's Wikipedia page",
#     )
#     vector_tool = QueryEngineTool.from_defaults(
#         query_engine=vector_query_engine,
#         description=(
#             "Useful for retrieving specific information about Snowflake"
#         ),
#     )

#     tool_mapping = SimpleToolNodeMapping.from_objects([list_tool, vector_tool])
#     obj_index = ObjectIndex.from_objects(
#         [list_tool, vector_tool],
#         tool_mapping,
#         VectorStoreIndex,
#     )
#     return obj_index

# documents = load_data()
# obj_index = load_index_data()
# query_engine = load_index_data()
# list_tool, vector_tool = load_index_data()
summary_index, vector_index = load_index_data()

if "list_query_engine" not in st.session_state.keys(): # Initialize the query engine
    st.session_state.list_query_engine = summary_index.as_query_engine(response_mode="tree_summarize",use_async=True,)

if "vector_query_engine" not in st.session_state.keys():
    st.session_state.vector_query_engine = vector_index.as_query_engine()

list_tool = QueryEngineTool.from_defaults(
    query_engine=st.session_state.list_query_engine,
    description=(
        "Useful for questions summarizing Snowflake's Wikipedia page"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=st.session_state.vector_query_engine,
    description=(
        "Useful for retrieving specific information about Snowflake"
    ),
)

if "router_query_engine" not in st.session_state.keys(): # Initialize the query engine
    st.session_state.router_query_engine = RouterQueryEngine(selector=PydanticSingleSelector.from_defaults(), query_engine_tools=[list_tool,vector_tool,],)

selected = pills("Choose a question to get started or write your own below.", ["What is Snowflake?", "What company did Snowflake announce they would acquire in October 2023?", "What company did Snowflake acquire in March 2022?", "When did Snowflake IPO?"], clearable=True, index=None)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Snowflake!"}
    ]

def add_to_message_history(role, content):
    message = {"role": role, "content": str(content)}
    st.session_state.messages.append(message) # Add response to message history

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# if "query_engine" not in st.session_state.keys(): # Initialize the query engine
#         st.session_state.query_engine = ToolRetrieverRouterQueryEngine(obj_index.as_retriever())

if selected:
    with st.chat_message("user"):
        st.write(selected)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.router_query_engine.query(selected)
            st.write(str(response))
            add_to_message_history("user",selected)
            add_to_message_history("assistant",response)
            # st.write(response.metadata["selector_result"])
            # st.write(response.metadata["selector_result"].ind)
            # st.write(type(response.metadata["selector_result"]))
            # st.write(type(response.metadata))
            result = str(response.metadata.get("selector_result"))
            st.write(result.split("index",1))
            # st.write(response.metadata.get("selector_result"))
            # st.write(type(response.metadata.get("selector_result")))
            # st.write(response.metadata["selector_result"].json)


if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.router_query_engine.query(prompt)
            st.write(str(response))
            add_to_message_history("assistant", response)
            # st.info()
            # st.write(response.metadata["selector_result"])
            # st.write(response.metadata["selector_result"].ind)
            # st.write(type(response.metadata["selector_result"]))

