from flask import Blueprint, request
from datetime import datetime
import uuid

from twilio.twiml.messaging_response import MessagingResponse
from qdrant_client.models import PointStruct

from backend.repositories.clone_repo import load_clones, save_clones
from backend.services.embedding_service import get_embedding
from backend.services.document_service import download_media_file
from backend.services.whatsapp_service import detect_expiry_info_with_llm
from backend.utils.phone_utils import normalize_phone_number
from backend.core.extensions import qdrant_client, document_processor
from flask import current_app


whatsapp_bp = Blueprint('whatsapp', __name__)


@whatsapp_bp.route('/whatsapp_webhook', methods=['POST'])
def whatsapp_webhook():
    incoming_number = request.values.get('From', '')
    sender_phone = incoming_number.replace('whatsapp:', '').strip()
    
    sender_phone = normalize_phone_number(sender_phone)
    
    clones = load_clones()
    matched_clone = None
    matched_id = None
    
    for clone_id, clone in clones.items():
        stored_phone = clone.get('phone_number')
        if stored_phone:
            normalized_stored = normalize_phone_number(stored_phone)
            if sender_phone == normalized_stored:
                matched_clone = clone
                matched_id = clone_id
                break
    
    message_body = request.values.get('Body', '')
    num_media = int(request.values.get('NumMedia', 0))
    media_url = request.values.get('MediaUrl0', None)
    media_content_type = request.values.get('MediaContentType0', None)

    resp = MessagingResponse()

    if not matched_clone:
        resp.message("❌ This phone number is not registered to any chatbot.")
        return str(resp)

    text_chunks = []
    extracted_expiry_date = None
    expiry_results = None

    if message_body and num_media == 0:
        text = message_body.strip()
        extracted_expiry_date = detect_expiry_info_with_llm(text)
        text_chunks = [text]

    elif num_media > 0 and media_url:
        media_file = download_media_file(media_url, media_content_type)
        all_chunks = document_processor.process_file(media_file)
        texts = [chunk['text'] for chunk in all_chunks if chunk.get('text')]
        text_chunks.extend(texts)
        expiry_results = [
            detect_expiry_info_with_llm(chunk) for chunk in texts
        ]

    else:
        resp.message("❗️Empty message or unsupported media type.")
        return str(resp)

    expired_dates = []

    for i, chunk_text in enumerate(text_chunks):
        if not chunk_text.strip():
            continue
        
        expiry_date = (
            expiry_results[i]
            if expiry_results is not None
            else extracted_expiry_date
        )

        payload = {
            "clone_id": matched_id,
            "text": chunk_text,
            "source": "whatsapp",
            "phone_sender": sender_phone,
            "timestamp": datetime.now().isoformat(),
        }

        if expiry_date:
            payload['expiry_date'] = expiry_date
            expired_dates.append(expiry_date)

        embedding = get_embedding(chunk_text)

        if embedding is None:
            print ("Skipping chunk due to embedding failure.")
            continue

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=payload
        )

        qdrant_client.upsert(
            collection_name=current_app.config['COLLECTION_NAME'],
            points=[point]
        )

        matched_clone['chunk_count'] = (
            matched_clone.get('chunk_count', 0) + 1
        )
        matched_clone['last_updated'] = datetime.now().isoformat()

    clones[matched_id] = matched_clone
    save_clones(clones)

    confirm_msg = "✅ Your message has been added!"
    if expired_dates:
        confirm_msg += (
            f"\n(Info will auto-expire on: {', '.join(expired_dates)})"
        )

    resp.message(confirm_msg)
    return str(resp)