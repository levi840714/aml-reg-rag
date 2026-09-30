from typing import Any, cast

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aml_reg_rag.graph.nodes import AskNodes, route_after_verify
from aml_reg_rag.graph.state import AskState, initial_state
from aml_reg_rag.llm import structured_model
from aml_reg_rag.models import Answer
from aml_reg_rag.retrieval.pipeline import Retriever


def build_graph(retriever: Retriever, drafter: Runnable[Any, Any]) -> CompiledStateGraph:
    nodes = AskNodes(retriever, drafter)
    graph = StateGraph(AskState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify", nodes.verify)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify, ["generate", END])
    return graph.compile()


def make_drafter(model: str, timeout: float) -> Runnable[Any, Any]:
    return structured_model(model, Answer, timeout=timeout, include_raw=True)


def run_ask(graph: CompiledStateGraph, question: str, k: int) -> AskState:
    return cast(AskState, graph.invoke(initial_state(question, k)))
