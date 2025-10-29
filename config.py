import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    QDRANT_URL = os.environ.get('QDRANT_URL')
    QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
    UPLOAD_FOLDER = 'uploads'
    DOCUMENTS_FOLDER = 'documents'  # NEW: For permanent document storage
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    COLLECTION_NAME = 'chatbot_clones'
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
