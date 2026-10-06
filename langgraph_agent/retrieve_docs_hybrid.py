"""Standalone dense, sparse, and late-interaction retrieval prototype."""

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

COLLECTION_NAME = "hybrid_documents"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
LATE_INTERACTION_MODEL = "colbert-ir/colbertv2.0"


def initialize_models() -> tuple[TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding]:
    """Load dense, sparse, and late-interaction embedding models."""
    dense_model = TextEmbedding(DENSE_MODEL)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL)
    late_interaction_model = LateInteractionTextEmbedding(LATE_INTERACTION_MODEL)
    return dense_model, sparse_model, late_interaction_model


def embed_docs(documents: list[str]) -> QdrantClient:
    """Index documents with dense, sparse, and multi-vector representations."""
    if not documents:
        raise ValueError("At least one document is required to build the index.")

    dense_model, sparse_model, late_interaction_model = initialize_models()
    dense_vectors = list(dense_model.embed(documents))
    sparse_vectors = list(sparse_model.embed(documents))
    late_vectors = list(late_interaction_model.embed(documents))

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=len(dense_vectors[0]),
                distance=Distance.COSINE,
            ),
            "late": VectorParams(
                size=len(late_vectors[0][0]),
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=HnswConfigDiff(m=0),
            ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF),
        },
    )

    points = [
        PointStruct(
            id=index,
            vector={
                "dense": dense_vector,
                "sparse": sparse_vector.as_object(),
                "late": late_vector,
            },
            payload={"document": document},
        )
        for index, (document, dense_vector, sparse_vector, late_vector) in enumerate(
            zip(documents, dense_vectors, sparse_vectors, late_vectors, strict=True)
        )
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return client


def get_doc_answer(
    client: QdrantClient,
    query: str,
    k: int = 3,
):
    """Retrieve documents using dense and sparse prefetch with late reranking."""
    if not query.strip() or k <= 0:
        return []

    dense_model, sparse_model, late_interaction_model = initialize_models()
    dense_query = next(dense_model.query_embed(query))
    sparse_query = next(sparse_model.query_embed(query))
    late_query = next(late_interaction_model.query_embed(query))

    prefetch = [
        Prefetch(query=dense_query, using="dense", limit=k),
        Prefetch(
            query=SparseVector(**sparse_query.as_object()),
            using="sparse",
            limit=k,
        ),
    ]
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=late_query,
        using="late",
        with_payload=True,
        limit=k,
    )
    return result.points


if __name__ == "__main__":
    sample_documents = [
        "Transformers use self-attention to process sequences of tokens.",
        "The transformer architecture was introduced in the 2017 paper Attention Is All You Need.",
        "Transformer models are used in language, vision, audio, and other fields.",
    ]
    sample_query = "How do transformers process language?"
    document_index = embed_docs(sample_documents)
    matches = get_doc_answer(document_index, sample_query, k=3)
    for match in matches:
        print(match.payload["document"])
