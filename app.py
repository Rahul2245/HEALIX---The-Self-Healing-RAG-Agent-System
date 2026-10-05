import os
import streamlit as st
from langgraph_agent.graph import build_graph
from langgraph_agent.document_loader import load_document


# Config page
st.set_page_config(page_title="Self-Healing RAG",
                   page_icon='🤖',
                   layout="wide",
                   initial_sidebar_state="expanded")


# Add a place to enter the API key
with st.sidebar:
    max_retries = 2
    api_key = st.text_input("OPENAI_API_KEY", type="password")

    # Save the API key to the environment variable
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Clear
    if st.button('Clear'):
        st.rerun()
    
    st.divider()
    # Load document to streamlit
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    # Print File Uploaded
    if uploaded_file is not None:
        st.caption("File uploaded")

    # If a file is uploaded, create the TextSplitter and vector database
    if uploaded_file :

        # Code to work around document loader from Streamlit and make it readable by langchain
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        # Maximum number of retries
        max_retries = st.pills("Maximum number of retries for the RAG Self-Healing",
                               options=[1, 2, 3, 4, 5],
                               default=2)

        st.caption("Document ready for indexing when you search.")

    st.divider()
    # About
    st.write("Designed with :heart: by [Gustavo R. Santos](https://gustavorsantos.me)")

# Title and Instructions
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    
st.title('Self-Healing RAG Agent | 🤖')
st.markdown('This AI Agent is trained to answer questions about the content from PDF File loaded.')
st.markdown("""
            The agent will then search the answer in the PDF file and try to generate an answer based on the content.   
            In case of failure, the agent tries to **:blue[self-heal] the RAG with the following strategr:**   
            * Increase the number of documents to retrieve from the PDF file
            * Reranking the results.
            """)
st.caption('Ask questions like: "Who is the author of this document?"')

st.divider()


# User question
question = st.text_input(label="Ask me something from your document:",
                         placeholder= "e.g. What is the definition of A/B testing?")


# Run the graph
if st.button('Search'):

    if not uploaded_file:
        st.error("Upload a PDF before searching.")
    elif not question.strip():
        st.error("Enter a question before searching.")
    elif not api_key:
        st.error("Enter your OpenAI API key before searching.")
    else:

        with st.spinner("Thinking..", show_time=True):
            st.write(":file_folder: | Log of execution:")
            try:
                documents = load_document(temp_file)
            except Exception as exc:
                st.error(f"Could not read the PDF: {exc}")
                documents = []

            if documents:
                graph = build_graph()
                result = graph.invoke({
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
                })

                st.divider()
                st.subheader("📖 Answer:")
                st.write(result["answer"])
                if result.get("retrieved_docs"):
                    with st.expander("Retrieved sources"):
                        for doc in result["retrieved_docs"]:
                            where = f" · page {doc['page']}" if doc.get("page") else ""
                            st.markdown(
                                f"**[{doc['chunk_id']}]{where}** · "
                                f"retrieval score {doc['score']:.3f}"
                            )
                            st.caption(doc["text"])
                if result.get("healing_trace"):
                    st.subheader("🔧 Recovery attempts")
                    for step in result["healing_trace"]:
                        st.write(f"- {step}")
            else:
                st.error("The PDF did not contain any readable text.")
