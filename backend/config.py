import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

class Config:
    BASE_DIR = BASE_DIR
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    QDRANT_URL = os.environ.get('QDRANT_URL')
    QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    DOCUMENTS_FOLDER = os.path.join(BASE_DIR, 'documents')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    COLLECTION_NAME = 'chatbot_clones'
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
