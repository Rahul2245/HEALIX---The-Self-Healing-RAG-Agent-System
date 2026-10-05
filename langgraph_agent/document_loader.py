# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(pdf):
    # Load a PDF
    """
    Load a PDF and split it into chunks for efficient retrieval.

    :param pdf: PDF file to load
    :return: List of chunks of text
    """

    loader = PyPDFLoader(pdf, mode="page")
    docs = loader.load()
        
    # Instantiate Text Splitter with Chunk Size of 500 words and Overlap of 100 words so that context is not lost
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    # Split into chunks for efficient retrieval
    chunks = text_splitter.split_documents(docs)

    # Keep page metadata so generated citations can point back to the PDF.
    return [
        {
            "text": chunk.page_content,
            "page": chunk.metadata.get("page", 0) + 1,
            "source": chunk.metadata.get("source", pdf),
        }
        for chunk in chunks
        if chunk.page_content.strip()
    ]




# Test
if __name__ == "__main__":
    pdf = "./Guide_AB_Testing.pdf"
    chunks = load_document(pdf)

    print(f"Generated {len(chunks)} chunks from the PDF file.")
    print(chunks)
