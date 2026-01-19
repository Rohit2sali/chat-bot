import os
import requests
from pinecone import Pinecone

# --- CONFIGURATION ---
# Use Hugging Face's free API for embeddings
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_API_KEY = os.getenv("HF_API_KEY")  # You need to add this to Render Environment Variables

# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

def get_embedding(text: str):
    """
    Fetch embedding from Hugging Face API to save RAM.
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        # Fallback or error handling
        print(f"Error fetching embedding: {response.text}")
        return []
        
    # The API returns the vector directly
    return response.json()

def store_message(text: str, role: str):
    embedding = get_embedding(text)
    
    if not embedding: 
        return

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
    embedding = get_embedding(query)
    
    if not embedding:
        return []

    results = index.query(
        vector=embedding,
        top_k=k,
        include_metadata=True
    )

    return [match["metadata"]["text"] for match in results["matches"]]
