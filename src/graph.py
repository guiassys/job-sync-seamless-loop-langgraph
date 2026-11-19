# src/graph.py

from langgraph.graph import StateGraph, START, END
from src.state import VideoState
from src.nodes import director_node, video_engineer_node, validator_node, editor_node


def create_workflow():
    """Constrói e compila o grafo de estados do pipeline Job Sync Seamless Loop."""
    # Inicializa o grafo tipado com o contrato de estado compartilhado
    workflow = StateGraph(VideoState)

    # Registra os nós (etapas do pipeline)
    workflow.add_node("seamless_loop_director_agent", director_node)
    workflow.add_node("video_engineer_agent", video_engineer_node)
    workflow.add_node("validator_agent", validator_node)
    workflow.add_node("editor_agent", editor_node)

    # Define o fluxo sequencial (arestas)
    workflow.add_edge(START, "seamless_loop_director_agent")
    workflow.add_edge("seamless_loop_director_agent", "video_engineer_agent")
    workflow.add_edge("video_engineer_agent", "validator_agent")
    workflow.add_edge("validator_agent", "editor_agent")
    workflow.add_edge("editor_agent", END)

    # Compila o grafo para execução
    return workflow.compile()