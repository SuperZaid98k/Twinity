from flask import current_app
from qdrant_client.models import Filter, FieldCondition, MatchValue
from backend.core.extensions import qdrant_client, openrouter_client
from backend.services.embedding_service import get_embedding
from datetime import datetime
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam


def search_relevant_chunks(clone_id, query, limit=8):
    """Search for relevant chunks with metadata and relevance scores"""
    query_embedding = get_embedding(query)
    
    if query_embedding is None:
        return []
    
    try:
        # UPDATED: Use query_points instead of search
        search_result = qdrant_client.query_points(
            collection_name=current_app.config['COLLECTION_NAME'],
            query=query_embedding,  # UPDATED: 'query_vector' is now 'query'
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="clone_id",
                        match=MatchValue(value=clone_id)
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=limit
        )
        
        # UPDATED: Iterate over search_result.points
        return [
            {
                'text': hit.payload['text'],
                'score': hit.score,
                'is_downloadable': hit.payload.get('is_downloadable', False),
                'document_filename': hit.payload.get('document_filename'),
                'document_unique_filename': hit.payload.get('document_unique_filename'),
                'file_size': hit.payload.get('file_size'),
                'metadata': {
                    k: v for k, v in hit.payload.items() 
                    if k not in ['text', 'clone_id', 'chunk_index', 'is_downloadable', 
                                'document_filename', 'document_unique_filename', 'file_size']
                }
            }
            for hit in search_result.points
            if hit.payload is not None
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_response(query, context_data, clone_name, clone_id, chat_history=None):
    """Generate response with intelligent document filtering"""
    
    MIN_DOCUMENT_RELEVANCE = 0.6
    
    document_keywords = [
        'document', 'file', 'download', 'form', 'pdf', 'attachment',
        'send', 'share', 'give me', 'show me', 'need', 'want',
        'where is', 'can i get', 'do you have'
    ]
    query_lower = query.lower()
    is_asking_for_document = any(keyword in query_lower for keyword in document_keywords)
    
    greeting_patterns = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'what\'s up', 'whats up', 'sup', 'greetings',
        'nice to meet', 'thanks', 'thank you', 'bye', 'goodbye'
    ]
    is_greeting = any(pattern in query_lower for pattern in greeting_patterns)
    
    context_parts = []
    downloadable_docs = []
    
    for idx, item in enumerate(context_data[:3]):
        text = item['text']
        metadata = item.get('metadata', {})
        score = item.get('score', 0)
        
        if (item.get('is_downloadable') and 
            (is_asking_for_document or (score > MIN_DOCUMENT_RELEVANCE and not is_greeting))):
            downloadable_docs.append({
                'filename': item.get('document_filename'),
                'unique_filename': item.get('document_unique_filename'),
                'size': item.get('file_size'),
                'description': text,
                'relevance_score': score
            })
        
    for item in context_data:
        text = item['text']
        context_parts.append(text)

    context = "\n\n---\n\n".join(context_parts)
    
    history_text = ""
    if chat_history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in list(chat_history)[-4:]:
            role = "User" if msg['role'] == 'user' else clone_name
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are {clone_name}, having a conversation with a person. Use the context information and previous conversation to provide relevant, consistent answers.

These are the information about yourself:
{context}
{history_text}

Current question: {query}

Instructions:
- Answer based on the context provided and consider the conversation history only if required
- Be conversational and reference previous exchanges when relevant
- If the answer isn't in the context, say so politely
- DO NOT mention document availability - documents will be shown separately if relevant
- For greetings, respond naturally and warmly without referencing documents
- Keep your tone consistent with previous responses
- Just Answer the question Dont explain too much.
- Answer like you are talking in reality with the user and keep it precise
"""

    try:
        messages: list = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=f"You are {clone_name}, a helpful assistant with access to a specific knowledge base. Maintain conversation continuity."
            )
        ]
        
        if chat_history:
            for msg in list(chat_history)[-6:]:
                if msg['role'] == 'assistant':
                    messages.append(ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=msg['content']
                    ))
                else:
                    messages.append(ChatCompletionUserMessageParam(
                        role="user",
                        content=msg['content']
                    ))
        
        messages.append(ChatCompletionUserMessageParam(role="user", content=prompt))
        
        response = openrouter_client.chat.completions.create(
            model="qwen/qwen3-8b",
            messages=messages
        )
        
        ai_response = response.choices[0].message.content
        
        return ai_response, downloadable_docs
        
    except Exception as e:
        return f"Error generating response: {str(e)}", []
