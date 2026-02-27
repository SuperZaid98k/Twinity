import os
import uuid
import requests
from werkzeug.utils import secure_filename
from qdrant_client.models import PointStruct
from backend.core.extensions import qdrant_client
from backend.services.embedding_service import get_embedding
from backend.document_processor import LLMDocumentProcessor
from backend.core.extensions import document_processor
from flask import current_app


def process_uploaded_file(file_path: str, filename: str) -> tuple:
    """Process uploaded file and return chunks"""
    
    # LlamaParse handles ALL file types automatically - no need to check extension!
    chunks = document_processor.process_file(file_path)
    for i,c in enumerate(chunks):
        print(i,' : ',c['text'])
    # Convert to embeddings
    embedded_chunks = document_processor.chunk_to_embeddings(chunks)
    
    return embedded_chunks, len(embedded_chunks)


def store_chunks_in_qdrant(clone_id, embedded_chunks):
    """Store embedded chunks in Qdrant (expects pre-embedded chunks with metadata)"""
    points = []
    
    for idx, chunk_data in enumerate(embedded_chunks):
        # chunk_data is already a dict with 'embedding', 'text', and 'metadata'
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk_data['embedding'],  # Already embedded!
            payload={
                "clone_id": clone_id,
                "text": chunk_data['text'],
                "chunk_index": idx,
                **chunk_data.get('metadata', {})
            }
        )
        points.append(point)
    
    if points:
        try:
            qdrant_client.upsert(
                collection_name=current_app.config['COLLECTION_NAME'],
                points=points
            )
        except Exception as e:
            print(f"Error storing chunks: {e}")
            return 0
    
    return len(points)


def save_document(file, clone_id):
    """Save document permanently and return file info"""
    # Create clone-specific folder
    clone_doc_folder = os.path.join(
        current_app.config['DOCUMENTS_FOLDER'],
        clone_id
    )
    os.makedirs(clone_doc_folder, exist_ok=True)
    
    # Generate unique filename
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(clone_doc_folder, unique_filename)
    
    # Save file
    file.save(filepath)
    
    return {
        'filename': filename,
        'unique_filename': unique_filename,
        'filepath': filepath,
        'file_size': os.path.getsize(filepath)
    }


def store_document_reference(clone_id, description, file_info):
    """Store document description as vector with file metadata"""
    # Generate embedding for description
    embedding = get_embedding(description)
    
    if embedding is None:
        return False
    
    # Create point with document metadata
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={
            "clone_id": clone_id,
            "text": description,
            "type": "document_reference",
            "document_filename": file_info['filename'],
            "document_unique_filename": file_info['unique_filename'],
            "document_filepath": file_info['filepath'],
            "file_size": file_info['file_size'],
            "is_downloadable": True
        }
    )
    
    try:
        qdrant_client.upsert(
            collection_name=current_app.config['COLLECTION_NAME'],
            points=[point]
        )
        return True
    except Exception as e:
        print(f"Error storing document reference: {e}")
        return False


def download_media_file(media_url, content_type):
    ext = content_type.split('/')[-1]
    filename = f"whatsapp_{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join('uploads', filename)
    # Use Twilio credentials for HTTP basic auth
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        raise ValueError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set")
    
    try:
        with requests.get(
            media_url,
            stream=True,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        ) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        return filepath
    except Exception as e:
        print("Error downloading Twilio media file:", e)
        raise