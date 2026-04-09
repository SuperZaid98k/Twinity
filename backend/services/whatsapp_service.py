import os
import re
from datetime import datetime

from twilio.rest import Client as TwilioClient

from backend.core.extensions import openrouter_client
from backend.utils.phone_utils import normalize_phone_number


def detect_expiry_info_with_llm(text):
    """
    Extract a date when info expires.
    Returns `YYYY-MM-DD` or `None` for permanent/non-time-bound info.
    """
    if not text or not text.strip():
        return None

    now = datetime.now().date().isoformat()
    instruction = f"""
The following message/notice has been received:
"{text}"

If this information becomes invalid after a specific date, return only that date in YYYY-MM-DD format.
If it is not time-bound, return PERMANENT.
Today's date is {now}.
Output one line only: YYYY-MM-DD or PERMANENT.
""".strip()

    try:
        response = openrouter_client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Extract expiry dates for time-relevant messages.",
                },
                {"role": "user", "content": instruction},
            ],
        )
        content = response.choices[0].message.content or ""
        result = content.strip()

        if "PERMANENT" in result.upper():
            return None

        date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", result)
        if not date_match:
            return None

        parsed = datetime.strptime(date_match.group(0), "%Y-%m-%d").date()
        return parsed.isoformat()
    except Exception as err:
        print(f"Expiry LLM error: {err}")
        return None


def send_whatsapp_message(phone_number, msg):
    """Send WhatsApp message via Twilio with normalized E.164 phone number."""
    normalized_phone = normalize_phone_number(phone_number or "")
    if not normalized_phone:
        print(f"Error sending WhatsApp: invalid destination number: {phone_number}")
        return False

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        print("Error sending WhatsApp: missing TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN")
        return False

    twilio_client = TwilioClient(account_sid, auth_token)
    from_whatsapp_number = "whatsapp:+14155238886"
    to_whatsapp_number = f"whatsapp:{normalized_phone}"

    try:
        twilio_client.messages.create(
            body=msg,
            from_=from_whatsapp_number,
            to=to_whatsapp_number,
        )
        print(f"WhatsApp message sent to {normalized_phone}")
        return True
    except Exception as err:
        print(f"Error sending WhatsApp to {to_whatsapp_number}: {err}")
        return False
