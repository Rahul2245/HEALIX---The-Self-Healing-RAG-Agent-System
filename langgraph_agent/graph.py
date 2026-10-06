"""LangGraph construction and optional local graph preview."""

from langgraph.graph import END, StateGraph

from langgraph_agent.document_loader import load_document
from langgraph_agent.nodes import (
    RAGState,
    generate_node,
    retrieve_node,
    retry_count_node,
    retry_node,
    score_node,
    should_retry,
)


def build_graph():
    """Build and compile the retrieval, generation, evaluation, and retry flow."""
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("score", score_node)
    builder.add_node("retry", retry_node)
    builder.add_node("increment_retry", retry_count_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "score")
    builder.add_conditional_edges(
        "score",
        should_retry,
        {"retry": "retry", "end": END},
    )
    builder.add_edge("retry", "increment_retry")
    builder.add_edge("increment_retry", "retrieve")
    return builder.compile()


def main() -> None:
    """Render the graph and run a sample question against a local PDF."""
    from IPython.display import Image, display

    graph = build_graph()
    display(Image(graph.get_graph().draw_mermaid_png()))

    documents = load_document("./Guide_AB_Testing.pdf")
    graph.invoke(
        {
            "text": documents,
            "query": "What is an A/B test?",
            "retrieved_docs": [],
            "retrieval_mode": "original",
            "retrieval_budget": 2,
            "failure_reason": "",
            "healing_trace": [],
            "answer": "",
            "score": 0.0,
            "retry_count": 0,
            "max_retries": 2,
        }
    )


if __name__ == "__main__":
    main()
