from flask import Blueprint, request, jsonify, session

from backend.repositories.clone_repo import load_clones
from backend.utils.chat_memory import (
    get_session_id,
    get_chat_history,
    add_to_chat_history,
    chat_histories
)
from backend.services.rag_service import (
    search_relevant_chunks,
    generate_response
)

chat_bp = Blueprint('chat_api', __name__)


@chat_bp.route('/api/chat/<clone_id>', methods=['POST'])
def chat_api(clone_id):
    data = request.get_json()
    query = data.get('message')
    
    if not query:
        return jsonify({'error': 'No message provided'}), 400
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone:
        return jsonify({'error': 'Clone not found'}), 404
    
    if not clone.get('is_public') and (
        'username' not in session or clone['owner'] != session['username']
    ):
        return jsonify({'error': 'Access denied'}), 403
    
    session_id = get_session_id()
    chat_history = get_chat_history(session_id, clone_id)
    
    context_data = search_relevant_chunks(clone_id, query)
    
    if not context_data:
        if clone.get('chunk_count', 0) > 0:
            response_text = (
                f"I'm {clone['owner_full_name']}. I currently can't retrieve my stored knowledge. "
                f"Please check vector database connectivity and re-index if needed."
            )
        else:
            response_text = (
                f"I'm {clone['owner_full_name']}, but I don't have enough "
                f"information to answer that question yet. "
                f"Please add more data to my knowledge base."
            )
        documents = []
    else:
        response_text, documents = generate_response(
            query,
            context_data,
            clone['owner_full_name'],
            clone_id,
            chat_history
        )
    
    add_to_chat_history(session_id, clone_id, 'user', query)
    add_to_chat_history(session_id, clone_id, 'assistant', response_text)
    
    return jsonify({
        'response': response_text,
        'documents': documents,
        'clone_id': clone_id
    })


@chat_bp.route('/api/clear_chat/<clone_id>', methods=['POST'])
def clear_chat_history(clone_id):
    """Clear chat history for current session"""
    session_id = get_session_id()
    key = f"{session_id}_{clone_id}"
    if key in chat_histories:
        chat_histories[key].clear()
    return jsonify({'success': True, 'message': 'Chat history cleared'})
