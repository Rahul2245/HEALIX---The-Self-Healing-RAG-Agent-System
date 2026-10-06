"""Streamlit interface for the self-healing RAG agent."""

import os

import streamlit as st

from langgraph_agent.document_loader import load_document
from langgraph_agent.graph import build_graph


def render_sidebar() -> tuple[object, str | None, dict[str, str], int]:
    """Render provider controls and return upload, LLM settings, and retry limit."""
    with st.sidebar:
        if st.button("Clear"):
            for key in list(st.session_state):
                if key == "provider" or key.startswith(
                    ("api_key_", "base_url_", "generation_model_", "judge_model_")
                ):
                    st.session_state.pop(key, None)
            st.rerun()

        provider = st.selectbox(
            "Model provider",
            options=["OpenAI", "Ollama", "OpenAI-compatible API"],
            key="provider",
            help="Hosted providers use their API key and compatible endpoint. Ollama runs locally.",
        )
        defaults = {
            "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
            "Ollama": ("http://localhost:11434/v1", "llama3.2"),
            "OpenAI-compatible API": ("", ""),
        }
        default_url, default_model = defaults[provider]
        base_url = st.text_input(
            "API base URL",
            value=default_url,
            key=f"base_url_{provider}",
            help="For Ollama, use http://localhost:11434/v1. Use the API's OpenAI-compatible base URL otherwise.",
        )
        api_key = st.text_input(
            "API key (leave blank for local Ollama)",
            value=os.environ.get("OPENAI_API_KEY", "") if provider == "OpenAI" else "",
            type="password",
            key=f"api_key_{provider}",
        )
        generation_model = st.text_input(
            "Generation model", value=default_model, key=f"generation_model_{provider}"
        )
        judge_model = st.text_input(
            "Judge model (can match generation model)",
            value=default_model,
            key=f"judge_model_{provider}",
        )

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

    return uploaded_file, temp_file, {
        "base_url": base_url.strip(),
        "api_key": api_key.strip(),
        "provider": provider,
        "generation_model": generation_model.strip(),
        "judge_model": judge_model.strip(),
    }, max_retries


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
    llm_config: dict[str, str],
    max_retries: int,
) -> None:
    """Validate inputs, invoke the graph, and display the result."""
    if uploaded_file is None or temp_file is None:
        st.error("Upload a PDF before searching.")
        return
    if not question.strip():
        st.error("Enter a question before searching.")
        return
    if not llm_config["api_key"] and llm_config["provider"] != "Ollama":
        st.error("Enter an API key, or configure a local Ollama endpoint.")
        return
    if not llm_config["generation_model"] or not llm_config["judge_model"]:
        st.error("Enter both a generation model and a judge model.")
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
                "llm_config": llm_config,
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

    uploaded_file, temp_file, llm_config, max_retries = render_sidebar()

    st.title("Self-Healing RAG Agent | 🤖")
    st.markdown("Ask questions about the contents of your uploaded PDF.")
    st.markdown(
        "The agent retrieves evidence, evaluates its answer, and attempts to recover "
        "by expanding retrieval and reranking when needed."
    )
    st.caption('Example: "Who is the author of this document?"')
    st.divider()

    configured = bool(llm_config["api_key"] or llm_config["provider"] == "Ollama")
    if uploaded_file is not None and not configured:
        st.info(
            "PDF uploaded. Configure your provider and API key in the sidebar before asking "
            "a question about this document."
        )
    elif not configured:
        st.warning("Configure an API provider or local Ollama in the sidebar.")

    question = st.text_input(
        label="Ask me something from your document:",
        placeholder="e.g. What is the definition of A/B testing?",
        disabled=uploaded_file is not None and not configured,
    )
    if st.button("Search"):
        run_search(uploaded_file, temp_file, question, llm_config, max_retries)


if __name__ == "__main__":
    main()
