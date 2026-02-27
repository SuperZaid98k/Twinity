import uuid
from collections import deque
from flask import session

chat_histories = {}

def get_session_id():
    """Get or create unique session ID for chat history"""
    if 'chat_session_id' not in session:
        session['chat_session_id'] = str(uuid.uuid4())
    return session['chat_session_id']

def get_chat_history(session_id, clone_id):
    """Get chat history for a specific session and clone"""
    key = f"{session_id}_{clone_id}"
    if key not in chat_histories:
        chat_histories[key] = deque(maxlen=10)  # Keep last 10 messages (5 exchanges)
    return chat_histories[key]

def add_to_chat_history(session_id, clone_id, role, content):
    """Add message to chat history"""
    history = get_chat_history(session_id, clone_id)
    history.append({'role': role, 'content': content})