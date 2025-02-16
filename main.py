from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.schema import IndexNode
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.llms import OpenAI
from llama_index.agent import OpenAIAgent
from llama_index import load_index_from_storage, StorageContext
from llama_index.node_parser import SentenceSplitter
from llama_index import VectorStoreIndex
from llama_index.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.agent import FnRetrieverOpenAIAgent

from pathlib import Path
import requests
import os 

os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

data_sources = [
    "root_cause_analysis",
    "product_interview_framework",
    "pm_interview",
    "planning_mvp"
]


def create_data_source_docs(data_sources):
    data_source_docs = {}
    
    for data_source in data_sources:
        data_source_docs[data_source] = SimpleDirectoryReader(
            input_files=[f"data/{data_source}.txt"]
        ).load_data()
    
    return data_source_docs

def create_query_engines_agents(data_sources, data_source_docs):
    node_parser = SentenceSplitter()
    agents = {}
    query_engines = {}
    all_nodes = []
    for idx, data_source in enumerate(data_sources):
        nodes = node_parser.get_nodes_from_documents(data_source_docs[data_source])
        all_nodes.extend(nodes)

        if not os.path.exists(f"data_source/{data_source}"):
            # build vector index
            vector_index = VectorStoreIndex(nodes, service_context=service_context)
            vector_index.storage_context.persist(
                persist_dir=f"data_source/{data_source}"
            )
        else:
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=f"data_source/{data_source}"),
                service_context=service_context,
            )

        # build summary index
        summary_index = SummaryIndex(nodes, service_context=service_context)
        # define query engines
        vector_query_engine = vector_index.as_query_engine()
        summary_query_engine = summary_index.as_query_engine()

        # define tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=(
                        "Useful for questions related to specific aspects of"
                        f" {data_source}."
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=(
                        "Useful for any requests that require a holistic summary"
                        f" of EVERYTHING about {data_source}. For questions about"
                        " more specific sections, please use the vector_tool."
                    ),
                ),
            ),
        ]

        # build agent
        function_llm = OpenAI(model="gpt-3.5-turbo")
        agent = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=function_llm,
            verbose=True,
            system_prompt=f"""\
    You are a specialized agent designed to answer queries about {data_source}.
    You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
    """,
        )

        agents[data_source] = agent
        query_engines[data_source] = vector_index.as_query_engine(
            similarity_top_k=2
        )
    
    return query_engines, agents, all_nodes

def create_obj_index(data_sources, agents):
    all_tools = []
    for data_source in data_sources:
        data_source_summary = (
            f"This content contains data about {data_source}. Use"
            f" this tool if you want to answer any questions about {data_source}.\n"
        )
        doc_tool = QueryEngineTool(
            query_engine=agents[data_source],
            metadata=ToolMetadata(
                name=f"tool_{data_source}",
                description=data_source_summary,
            ),
        )
        all_tools.append(doc_tool)

    tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
    obj_index = ObjectIndex.from_objects(
        all_tools,
        tool_mapping,
        VectorStoreIndex,
    )
    return obj_index


def create_top_agent(obj_index):

    top_agent = FnRetrieverOpenAIAgent.from_retriever(
        obj_index.as_retriever(similarity_top_k=3),
        system_prompt=""" \
    You are a personalized agent. you have all data related to my mail, my meetings, meeting notes, research journal, and my chat with my supervisor.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

    """,
        verbose=True,
    )
    return top_agent

def create_base_query_engine(all_nodes):
    base_index = VectorStoreIndex(all_nodes)
    base_query_engine = base_index.as_query_engine(similarity_top_k=4)
    return base_query_engine

def create_agent(data_sources):
    data_source_docs = create_data_source_docs(data_sources)
    query_engines, agents, all_nodes = create_query_engines_agents(data_sources, data_source_docs)
    obj_index = create_obj_index(data_sources, agents)
    top_agent = create_top_agent(obj_index)
    base_query_engine = create_base_query_engine(all_nodes)
    return top_agent, base_query_engine

def query_response(agent, query):
    response = agent.query(query)
    return response

def main(data_sources):
    top_agent, base_query_engine = create_agent(data_sources)
    query = "how do you perform root cause analysis to understand reduction in users pressing the add to cart bar button?"
    top_agent_response = query_response(top_agent, query)
    base_query_engine_response =  query_response(base_query_engine , query)
    print("top_agent_response: ", top_agent_response)
    print("base_query_engine_response: ", base_query_engine_response)

main(data_sources)