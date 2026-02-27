from backend.core.extensions import embedding_model

def get_embedding(text):
    """Generate embedding using SentenceTransformer"""
    try:
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()  # Convert to list for JSON serialization
    except Exception as e:
        print(f"Embedding error: {e}")
        return None