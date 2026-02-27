from flask import current_app
from qdrant_client.models import Distance, VectorParams, KeywordIndexParams, KeywordIndexType
from backend.core.extensions import qdrant_client


def init_qdrant_collection():
    """Initialize Qdrant collection with correct vector size for all-MiniLM-L6-v2"""
    try:
        collections = qdrant_client.get_collections().collections
        collection_name = current_app.config['COLLECTION_NAME']

        if collection_name not in [col.name for col in collections]:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )

        # Ensure payload indexes exist even for pre-existing collections.
        # clone_id is used in most query filters.
        try:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="clone_id",
                field_schema=KeywordIndexParams(type = KeywordIndexType.KEYWORD)
            )
        except Exception:
            pass

        # is_downloadable is filtered when listing documents.
        try:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="is_downloadable",
                field_schema="bool"
            )
        except Exception:
            pass

        return True

    except Exception as e:
        print(f"Error initializing Qdrant collection: {e}")
        return False
