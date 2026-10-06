"""PDF loading and chunking utilities."""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_document(pdf_path: str) -> list[dict[str, str | int]]:
    """Load a PDF into text chunks with one-based page and source metadata."""
    pages = PyPDFLoader(pdf_path, mode="page").load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    return [
        {
            "text": chunk.page_content,
            "page": chunk.metadata.get("page", 0) + 1,
            "source": chunk.metadata.get("source", pdf_path),
        }
        for chunk in chunks
        if chunk.page_content.strip()
    ]


if __name__ == "__main__":
    pdf_file = "./Guide_AB_Testing.pdf"
    document_chunks = load_document(pdf_file)
    print(f"Generated {len(document_chunks)} chunks from the PDF.")
    print(document_chunks)
