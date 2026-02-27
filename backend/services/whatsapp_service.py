import os
import uuid
import requests
import socket
import time
from datetime import datetime
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from qdrant_client.models import PointStruct

from backend.core.extensions import qdrant_client
from backend.services.embedding_service import get_embedding
from backend.services.document_service import download_media_file
from backend.repositories.clone_repo import load_clones, save_clones
from backend.utils.phone_utils import normalize_phone_number
from backend.core.extensions import openrouter_client
from backend.document_processor import LLMDocumentProcessor


def detect_expiry_info_with_llm(text):
    """
    Use LLM or OpenRouter to extract a date when this info expires,
    or return None for permanent info.
    Output format must be YYYY-MM-DD (or None)
    """
    from datetime import datetime
    now = datetime.now().date()
    print(now)
    instruction = """
    The following message/notice has been received:

    "{0}"
    
    If it is about an event or info that expires on a particular date,means that information will not be valuable after that time, extract that expiry date in format YYYY-MM-DD.
    If not time-bound or not expirable, just return "PERMANENT". Todays date is {1}
    Only output a single line: either a date (YYYY-MM-DD) or "PERMANENT". Do not explain.
    """.format(text, now)

    try:
        response = openrouter_client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract expiry dates for time-relevant messages."},
                {"role": "user", "content": instruction}
            ]
        )
        content = response.choices[0].message.content
        if content is None:
            return None
        result = content.strip()
        if "PERMANENT" in result.upper():
            return None
        try:
            dateval = datetime.strptime(result[:10], "%Y-%m-%d").date()
            return str(dateval)
        except Exception:
            pass
        return None
    except Exception as e:
        print("Expiry LLM error:", e)
        return None


def send_whatsapp_message(phone_number, msg):
    """Send WhatsApp message via Twilio with proper phone number formatting"""
    twilio_client = TwilioClient(
        os.getenv('TWILIO_ACCOUNT_SID'),
        os.getenv('TWILIO_AUTH_TOKEN')
    )
    
    from_whatsapp_number = 'whatsapp:+14155238886'
    
    cleaned_phone = ''.join(filter(str.isdigit, phone_number))
    
    if not phone_number.strip().startswith('+'):
        cleaned_phone = f"+91{cleaned_phone}"
    else:
        cleaned_phone = '+' + cleaned_phone
    
    to_whatsapp_number = f"whatsapp:{cleaned_phone}"
    
    try:
        twilio_client.messages.create(
            body=msg,
            from_=from_whatsapp_number,
            to=to_whatsapp_number
        )
        print(f"✅ WhatsApp message sent to {cleaned_phone}")
    except Exception as e:
        print(f"❌ Error sending WhatsApp to {to_whatsapp_number}: {e}")