"""Streamlit interface for the self-healing RAG agent."""

import os

import streamlit as st

from langgraph_agent.document_loader import load_document
from langgraph_agent.graph import build_graph


def render_sidebar() -> tuple[object, str | None, str, int]:
    """Render controls and return the uploaded file, temporary path, and retry limit."""
    with st.sidebar:
        if st.button("Clear"):
            os.environ.pop("OPENAI_API_KEY", None)
            st.session_state.pop("openai_api_key", None)
            st.rerun()

        api_key = st.text_input(
            "OPENAI_API_KEY",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            key="openai_api_key",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        st.divider()
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        temp_file = None
        max_retries = 2

        if uploaded_file is not None:
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(uploaded_file.getvalue())
            st.caption("Document ready for indexing when you search.")
            max_retries = st.pills(
                "Maximum number of retries for the RAG self-healing",
                options=[1, 2, 3, 4, 5],
                default=2,
            ) or 2

        st.divider()
        st.write("Designed with :heart: by [Gustavo R. Santos](https://gustavorsantos.me)")

    return uploaded_file, temp_file, api_key, max_retries


def render_sources(documents: list[dict]) -> None:
    """Display retrieved chunks and their source locations."""
    if not documents:
        return

    with st.expander("Retrieved sources"):
        for document in documents:
            page = f" · page {document['page']}" if document.get("page") else ""
            st.markdown(
                f"**[{document['chunk_id']}]{page}** · "
                f"retrieval score {document['score']:.3f}"
            )
            st.caption(document["text"])


def run_search(
    uploaded_file: object,
    temp_file: str | None,
    question: str,
    api_key: str,
    max_retries: int,
) -> None:
    """Validate inputs, invoke the graph, and display the result."""
    if uploaded_file is None or temp_file is None:
        st.error("Upload a PDF before searching.")
        return
    if not question.strip():
        st.error("Enter a question before searching.")
        return
    if not api_key:
        st.error("Enter your OpenAI API key before searching.")
        return

    with st.spinner("Thinking...", show_time=True):
        st.write(":file_folder: | Log of execution:")
        try:
            documents = load_document(temp_file)
        except Exception as exc:
            st.error(f"Could not read the PDF: {exc}")
            return

        if not documents:
            st.error("The PDF did not contain any readable text.")
            return

        result = build_graph().invoke(
            {
                "text": documents,
                "query": question,
                "retrieval_mode": "original",
                "retrieval_budget": 2,
                "retry_count": 0,
                "max_retries": max_retries,
                "healing_trace": [],
                "retrieved_docs": [],
                "answer": "",
                "score": 0.0,
                "failure_reason": "",
            }
        )

    st.divider()
    st.subheader("📖 Answer")
    st.write(result["answer"])
    render_sources(result.get("retrieved_docs", []))

    if result.get("healing_trace"):
        st.subheader("🔧 Recovery attempts")
        for step in result["healing_trace"]:
            st.write(f"- {step}")


def main() -> None:
    """Render the page and handle a search submission."""
    st.set_page_config(
        page_title="Self-Healing RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    uploaded_file, temp_file, api_key, max_retries = render_sidebar()

    st.title("Self-Healing RAG Agent | 🤖")
    st.markdown("Ask questions about the contents of your uploaded PDF.")
    st.markdown(
        "The agent retrieves evidence, evaluates its answer, and attempts to recover "
        "by expanding retrieval and reranking when needed."
    )
    st.caption('Example: "Who is the author of this document?"')
    st.divider()

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")

    question = st.text_input(
        label="Ask me something from your document:",
        placeholder="e.g. What is the definition of A/B testing?",
    )
    if st.button("Search"):
        run_search(uploaded_file, temp_file, question, api_key, max_retries)


if __name__ == "__main__":
    main()
