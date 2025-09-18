# ingest.py
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collection = "persona_clone"

# Create collection if not exists
if collection not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    {"id": "1", "text": "My name is Aisha Khan. I love coffee and gardening."},
    {"id": "2", "text": "I prefer short answers with a casual tone."},
    {"id": "3", "text": "I enjoy traveling to Goa and Kerala."}
]

points = []
for i, doc in enumerate(documents):
    embedding = model.encode(doc["text"]).tolist()
    points.append(
        {"id": i, "vector": embedding, "payload": {"text": doc["text"]}}
    )

client.upsert(collection_name=collection, points=points)
print("✅ Data uploaded to Qdrant")
