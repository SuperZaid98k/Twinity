import re

def normalize_phone_number(phone):
    """Normalize phone number to E.164 format"""
    # Remove all non-digit characters except +
    cleaned = re.sub(r'[^\d+]', '', phone.strip())
    
    # If it doesn't start with +, add country code
    if not cleaned.startswith('+'):
        # Add +91 for India - change this for your country
        cleaned = f"+91{cleaned}"
    
    # Validate format: + followed by 10-15 digits
    if re.match(r'^\+\d{10,15}$', cleaned):
        return cleaned
    else:
        return None