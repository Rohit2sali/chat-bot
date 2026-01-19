import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


def store_message(text: str, role: str):
    embedding = embedding_model.encode(text).tolist()

    index.upsert(
        vectors=[
            {
                "id": f"{role}_{hash(text)}",
                "values": embedding,
                "metadata": {"role": role, "text": text}
            }
        ]
    )


def retrieve_context(query: str, k: int = 4):
    embedding = embedding_model.encode(query).tolist()

    results = index.query(
        vector=embedding,
        top_k=k,
        include_metadata=True
    )

    return [match["metadata"]["text"] for match in results["matches"]]
