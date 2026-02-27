import os
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from backend.document_processor import LLMDocumentProcessor


# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


document_processor = LLMDocumentProcessor(
    api_key=os.getenv('LLAMA_CLOUD_API_KEY', ''),
    embedding_model=embedding_model
)

# Initialize OpenRouter client (uses OpenAI SDK)
openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)