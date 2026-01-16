from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file, abort
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import openai
import requests
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
import json
import os
import uuid
from datetime import datetime
import PyPDF2
from config import Config
from sentence_transformers import SentenceTransformer
from document_processor import LLMDocumentProcessor
from collections import deque
import threading
import time
from datetime import datetime
import socket
import re



os.makedirs('documents', exist_ok=True)
chat_histories = {}

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")


app = Flask(__name__)
app.config.from_object(Config)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=app.config['QDRANT_URL'],
    api_key=app.config['QDRANT_API_KEY']
)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


document_processor = LLMDocumentProcessor(
    api_key=os.environ.get('LLAMA_CLOUD_API_KEY'),
    embedding_model=embedding_model
)
# Initialize OpenRouter client (uses OpenAI SDK)
openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=app.config['OPENROUTER_API_KEY']
)

# Ensure data directories exist
os.makedirs('data', exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

# Initialize JSON storage files
def init_json_files():
    if not os.path.exists('data/users.json'):
        with open('data/users.json', 'w') as f:
            json.dump({}, f)
    if not os.path.exists('data/clones.json'):
        with open('data/clones.json', 'w') as f:
            json.dump({}, f)

init_json_files()

# Initialize Qdrant collection
def init_qdrant_collection():
    """Initialize Qdrant collection with correct vector size for all-MiniLM-L6-v2"""
    try:
        collections = qdrant_client.get_collections().collections
        if app.config['COLLECTION_NAME'] not in [col.name for col in collections]:
            qdrant_client.create_collection(
                collection_name=app.config['COLLECTION_NAME'],
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Changed to 384
            )
            # Create payload index for multi-tenancy
            qdrant_client.create_payload_index(
                collection_name=app.config['COLLECTION_NAME'],
                field_name="clone_id",
                field_schema="keyword"
            )
    except Exception as e:
        print(f"Error initializing Qdrant collection: {e}")


init_qdrant_collection()

# Helper functions
def load_users():
    with open('data/users.json', 'r') as f:
        return json.load(f)

def save_users(users):
    with open('data/users.json', 'w') as f:
        json.dump(users, f, indent=2)

def load_clones():
    with open('data/clones.json', 'r') as f:
        return json.load(f)

def save_clones(clones):
    with open('data/clones.json', 'w') as f:
        json.dump(clones, f, indent=2)


def process_uploaded_file(file_path: str, filename: str) -> tuple:
    """Process uploaded file and return chunks"""
    
    # LlamaParse handles ALL file types automatically - no need to check extension!
    chunks = document_processor.process_file(file_path)
    for i,c in enumerate(chunks):
        print(i,' : ',c['text'])
    # Convert to embeddings
    embedded_chunks = document_processor.chunk_to_embeddings(chunks)
    
    return embedded_chunks, len(embedded_chunks)


def get_embedding(text):
    """Generate embedding using SentenceTransformer"""
    try:
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()  # Convert to list for JSON serialization
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


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
                **chunk_data.get('metadata', {})  # Include all metadata from LlamaParse
            }
        )
        points.append(point)
    
    if points:
        try:
            qdrant_client.upsert(
                collection_name=app.config['COLLECTION_NAME'],
                points=points
            )
        except Exception as e:
            print(f"Error storing chunks: {e}")
            return 0
    
    return len(points)


def save_document(file, clone_id):
    """Save document permanently and return file info"""
    # Create clone-specific folder
    clone_doc_folder = os.path.join(app.config['DOCUMENTS_FOLDER'], clone_id)
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
            collection_name=app.config['COLLECTION_NAME'],
            points=[point]
        )
        return True
    except Exception as e:
        print(f"Error storing document reference: {e}")
        return False


from qdrant_client.models import Filter, FieldCondition, MatchValue

def search_relevant_chunks(clone_id, query, limit=8):
    """Search for relevant chunks with metadata and relevance scores"""
    query_embedding = get_embedding(query)
    
    if query_embedding is None:
        return []
    
    try:
        # UPDATED: Use query_points instead of search
        search_result = qdrant_client.query_points(
            collection_name=app.config['COLLECTION_NAME'],
            query=query_embedding,  # UPDATED: 'query_vector' is now 'query'
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="clone_id",
                        match=MatchValue(value=clone_id)
                    )
                ]
            ),
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
            for hit in search_result.points  # UPDATED: Access the list via .points
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []




def generate_response(query, context_data, clone_name, clone_id, chat_history=None):
    """Generate response with intelligent document filtering"""
    
    # Define minimum relevance score for documents (0.6 = 60% similarity)
    MIN_DOCUMENT_RELEVANCE = 0.6
    
    # Check if query is asking for documents/files
    document_keywords = [
        'document', 'file', 'download', 'form', 'pdf', 'attachment',
        'send', 'share', 'give me', 'show me', 'need', 'want',
        'where is', 'can i get', 'do you have'
    ]
    query_lower = query.lower()
    is_asking_for_document = any(keyword in query_lower for keyword in document_keywords)
    
    # Greeting/casual query detection
    greeting_patterns = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'what\'s up', 'whats up', 'sup', 'greetings',
        'nice to meet', 'thanks', 'thank you', 'bye', 'goodbye'
    ]
    is_greeting = any(pattern in query_lower for pattern in greeting_patterns)
    
    # Build context with section information
    context_parts = []
    downloadable_docs = []
    
    # Only check top 3 chunks for documents
    for idx, item in enumerate(context_data[:3]):
        text = item['text']
        metadata = item.get('metadata', {})
        score = item.get('score', 0)
        
        # Only include document if:
        # 1. User is asking for documents OR
        # 2. Document is highly relevant (score > threshold) AND not a greeting
        if (item.get('is_downloadable') and 
            (is_asking_for_document or (score > MIN_DOCUMENT_RELEVANCE and not is_greeting))):
            downloadable_docs.append({
                'filename': item.get('document_filename'),
                'unique_filename': item.get('document_unique_filename'),
                'size': item.get('file_size'),
                'description': text,
                'relevance_score': score
            })
        
        # Add section context if available
    for item in context_data:
        text = item['text']
        context_parts.append(text)

    
    context = "\n\n---\n\n".join(context_parts)
    
    # Build conversation history for context
    history_text = ""
    if chat_history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in list(chat_history)[-4:]:
            role = "User" if msg['role'] == 'user' else clone_name
            history_text += f"{role}: {msg['content']}\n"
    print(f'{context}\n\n{history_text}')
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
        messages = [
            {"role": "system", "content": f"You are {clone_name}, a helpful assistant with access to a specific knowledge base. Maintain conversation continuity."}
        ]
        
        if chat_history:
            for msg in list(chat_history)[-6:]:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        messages.append({"role": "user", "content": prompt})
        
        response = openrouter_client.chat.completions.create(
            model="qwen/qwen3-8b",
            messages=messages
        )
        
        ai_response = response.choices[0].message.content
        
        # Return response text and filtered documents
        return ai_response, downloadable_docs
        
    except Exception as e:
        return f"Error generating response: {str(e)}", []






def download_media_file(media_url, content_type):
    ext = content_type.split('/')[-1]
    filename = f"whatsapp_{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join('uploads', filename)
    # Use Twilio credentials for HTTP basic auth
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    try:
        with requests.get(media_url, stream=True, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        return filepath
    except Exception as e:
        print("Error downloading Twilio media file:", e)
        raise

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
    """.format(text,now)

    try:
        response = openrouter_client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract expiry dates for time-relevant messages."},
                {"role": "user", "content": instruction}
            ]
        )
        result = response.choices[0].message.content.strip()
        if "PERMANENT" in result.upper():
            return None
        # Else, try to parse date (basic check)
        from datetime import datetime
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
    
    # Twilio sandbox number
    from_whatsapp_number = 'whatsapp:+14155238886'
    
    # Clean and format phone number
    # Remove all spaces, dashes, parentheses, etc.
    cleaned_phone = ''.join(filter(str.isdigit, phone_number))
    
    # Add country code if not present
    if not phone_number.strip().startswith('+'):
        # Assume India (+91) if no country code - adjust for your region
        cleaned_phone = f"+91{cleaned_phone}"
    else:
        # Extract country code and number
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




def delete_expired_chunks_job():
    """Background job to delete expired chunks with proper error handling"""
    retry_count = 0
    max_retries = 3
    
    while True:
        try:
            # Check if we have internet connectivity
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=5)
            except OSError:
                print("⚠️  No internet connection. Skipping expired chunk cleanup.")
                time.sleep(900)  # Wait 15 min and retry
                continue
            
            now = datetime.now().date()
            
            # Scroll through all points
            scroll_result = qdrant_client.scroll(
                collection_name=app.config['COLLECTION_NAME'],
                scroll_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=200
            )
            points = scroll_result[0]
            
            expired_ids = []
            chunks_by_clone = {}
            
            for point in points:
                exp_date = point.payload.get('expiry_date')
                cid = point.payload.get('clone_id')
                
                if exp_date and exp_date != "PERMANENT":
                    try:
                        expiry_datetime = datetime.strptime(exp_date, "%Y-%m-%d").date()
                        if expiry_datetime < now:
                            expired_ids.append(point.id)
                            if cid not in chunks_by_clone:
                                chunks_by_clone[cid] = []
                            chunks_by_clone[cid].append(point.payload.get('text', '')[:80])
                    except ValueError as e:
                        print(f"Invalid date format in chunk: {exp_date}")
                        continue
            
            # Delete expired points
            if expired_ids:
                qdrant_client.delete(
                    collection_name=app.config['COLLECTION_NAME'],
                    points_selector=expired_ids
                )
                print(f"✅ Deleted {len(expired_ids)} expired chunks.")
                
                # Notify via WhatsApp
                clones = load_clones()
                for clone_id, chunks in chunks_by_clone.items():
                    phone_number = clones.get(clone_id, {}).get('phone_number')
                    if phone_number:
                        try:
                            msg = f"⏳ The following expired items were removed from your chatbot:\n"
                            for text in chunks[:5]:  # Limit to 5 to avoid long messages
                                msg += f"- {text}…\n"
                            if len(chunks) > 5:
                                msg += f"\n...and {len(chunks) - 5} more items."
                            send_whatsapp_message(phone_number, msg)
                        except Exception as e:
                            print(f"Error sending WhatsApp notification: {e}")
            else:
                print("✓ No expired chunks found.")
            
            # Reset retry counter on success
            retry_count = 0
            
        except socket.gaierror as e:
            retry_count += 1
            print(f"⚠️  Network error during expired chunk cleanup: {e}")
            print(f"   Retry {retry_count}/{max_retries} in 5 minutes...")
            if retry_count >= max_retries:
                print(f"❌ Failed after {max_retries} retries. Will try again in 15 minutes.")
                retry_count = 0
                time.sleep(900)  # Wait 15 min after max retries
            else:
                time.sleep(300)  # Wait 5 min before retry
            continue
            
        except Exception as e:
            print(f"❌ Expired chunk cleanup error: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait 15 minutes before next check
        time.sleep(900)



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



# Routes
@app.route('/')
def index():
    clones = load_clones()
    public_clones = {k: v for k, v in clones.items() if v.get('is_public', False)}
    return render_template('index.html', clones=public_clones)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        full_name = request.form.get('full_name')  # Added
        
        users = load_users()
        if username in users:
            flash('Username already exists')
            return redirect(url_for('register'))
        
        users[username] = {
            'password': generate_password_hash(password),
            'full_name': full_name,  # Added
            'created_at': datetime.now().isoformat(),
            'clones': []
        }
        save_users(users)
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        users = load_users()
        user = users.get(username)
        
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        
        flash('Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    users = load_users()
    user = users.get(session['username'])
    clones = load_clones()
    user_clones = {k: v for k, v in clones.items() if k in user.get('clones', [])}
    
    return render_template('dashboard.html', clones=user_clones)

@app.route('/create_clone', methods=['GET', 'POST'])
def create_clone():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        clone_name = request.form.get('clone_name')
        description = request.form.get('description')
        is_public = request.form.get('is_public') == 'on'
        
        clone_id = str(uuid.uuid4())
        
        # Get user's full name
        users = load_users()
        owner_full_name = users[session['username']].get('full_name', session['username'])
        
        clones = load_clones()
        clones[clone_id] = {
            'name': clone_name,
            'description': description,
            'is_public': is_public,
            'owner': session['username'],
            'owner_full_name': owner_full_name,  # Added
            'created_at': datetime.now().isoformat(),
            'chunk_count': 0
        }
        save_clones(clones)
        
        users[session['username']]['clones'].append(clone_id)
        save_users(users)
        
        flash(f'Clone "{clone_name}" created successfully!')
        return redirect(url_for('edit_clone', clone_id=clone_id))
    
    return render_template('create_clone.html')

@app.route('/edit_clone/<clone_id>', methods=['GET', 'POST'])
def edit_clone(clone_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        flash('Clone not found or access denied')
        return redirect(url_for('dashboard'))
    
    return render_template('edit_clone.html', clone=clone, clone_id=clone_id)

@app.route('/enable_whatsapp/<clone_id>', methods=['POST'])
def enable_whatsapp(clone_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        flash('Not allowed')
        return redirect(url_for('edit_clone', clone_id=clone_id))

    enable = bool(request.form.get('enable_whatsapp'))
    phone = request.form.get('phone_number', '').strip()
    
    if enable and phone:
        # Normalize phone number
        normalized_phone = normalize_phone_number(phone)
        
        if normalized_phone:
            clone['phone_number'] = normalized_phone
            save_clones(clones)
            flash(f'✅ WhatsApp enabled! Number saved as: {normalized_phone}')
        else:
            flash('❌ Invalid phone number format. Use format: +919876543210 or 9876543210')
            return redirect(url_for('edit_clone', clone_id=clone_id))
    else:
        clone.pop('phone_number', None)
        save_clones(clones)
        flash('WhatsApp integration disabled.')
    
    return redirect(url_for('edit_clone', clone_id=clone_id))

# Add after the edit_clone route (around line 360)

@app.route('/manage_data/<clone_id>')
def manage_data(clone_id):
    """View and manage clone's vector data"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        flash('Clone not found or access denied')
        return redirect(url_for('dashboard'))
    
    # Get all points for this clone from Qdrant
    try:
        # Scroll through all points for this clone
        points, _ = qdrant_client.scroll(
            collection_name=app.config['COLLECTION_NAME'],
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="clone_id",
                        match=MatchValue(value=clone_id)
                    )
                ]
            ),
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        # Format points for display
        chunks_data = []
        for point in points:
            chunks_data.append({
                'id': point.id,
                'text': point.payload.get('text', '')[:200] + '...',  # Preview
                'full_text': point.payload.get('text', ''),
                'section': point.payload.get('section', 'N/A'),
                'type': point.payload.get('type', 'general'),
                'chunk_index': point.payload.get('chunk_index', 0)
            })
        
        # Sort by chunk index
        chunks_data.sort(key=lambda x: x['chunk_index'])
        
    except Exception as e:
        print(f"Error fetching chunks: {e}")
        chunks_data = []
    
    return render_template('manage_data.html', 
                         clone=clone, 
                         clone_id=clone_id, 
                         chunks=chunks_data)

@app.route('/delete_chunks/<clone_id>', methods=['POST'])
def delete_chunks(clone_id):
    """Delete selected chunks from Qdrant"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    chunk_ids = data.get('chunk_ids', [])
    
    if not chunk_ids:
        return jsonify({'error': 'No chunks selected'}), 400
    
    try:
        # Delete points from Qdrant
        qdrant_client.delete(
            collection_name=app.config['COLLECTION_NAME'],
            points_selector=chunk_ids
        )
        
        # Update chunk count
        clone['chunk_count'] = max(0, clone.get('chunk_count', 0) - len(chunk_ids))
        save_clones(clones)
        
        return jsonify({
            'success': True,
            'deleted_count': len(chunk_ids),
            'remaining_chunks': clone['chunk_count']
        })
    except Exception as e:
        print(f"Error deleting chunks: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/preview_chunks/<clone_id>', methods=['POST'])
def preview_chunks(clone_id):
    """Preview chunks before uploading"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        return jsonify({'error': 'Access denied'}), 403
    
    chunks_preview = []
    chunks_data = []  # Store actual chunks
    
    # Check if text input is provided
    if 'text_content' in request.form and request.form['text_content'].strip():
        text_content = request.form['text_content'].strip()
        chunks = document_processor.process_text_input(text_content)
        
        for idx, chunk in enumerate(chunks):
            chunks_preview.append({
                'id': idx,
                'text': chunk['text'],
                'preview': chunk['text'][:150] + ('...' if len(chunk['text']) > 150 else ''),
                'metadata': chunk.get('metadata', {}),
                'selected': True  # Default selected
            })
        
        chunks_data = chunks  # Store for later
        
        # Store chunks in session
        session[f'preview_chunks_{clone_id}'] = {
            'chunks': chunks_data,
            'source': 'text'
        }
    
    # Check if file is provided
    elif 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        filename = secure_filename(file.filename)
        
        allowed_extensions = {
            'pdf', 'docx', 'doc', 'pptx', 'ppt',
            'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif',
            'txt', 'csv', 'xlsx', 'xls'
        }
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported file type. Supported: {", ".join(sorted(allowed_extensions))}'
            }), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'preview_{filename}')
        file.save(filepath)
        
        try:
            # Process file to get chunks
            chunks = document_processor.process_file(filepath)
            
            for idx, chunk in enumerate(chunks):
                chunks_preview.append({
                    'id': idx,
                    'text': chunk['text'],
                    'preview': chunk['text'][:150] + ('...' if len(chunk['text']) > 150 else ''),
                    'metadata': chunk.get('metadata', {}),
                    'selected': True,
                    'section': chunk.get('metadata', {}).get('section', 'N/A'),
                    'type': chunk.get('metadata', {}).get('type', 'general')
                })
            
            chunks_data = chunks
            
            # Store chunks temporarily in session for later upload
            session[f'preview_chunks_{clone_id}'] = {
                'chunks': chunks_data,
                'filename': filename,
                'source': 'file'
            }
            
        except Exception as e:
            # Clean up file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'No text or file provided'}), 400
    
    return jsonify({
        'success': True,
        'chunks': chunks_preview,
        'total_chunks': len(chunks_preview)
    })

@app.route('/upload_selected_chunks/<clone_id>', methods=['POST'])
def upload_selected_chunks(clone_id):
    """Upload only selected chunks to Qdrant"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    selected_indices = data.get('selected_indices', [])
    
    # Get preview chunks from session
    preview_data = session.get(f'preview_chunks_{clone_id}')
    
    if not preview_data:
        return jsonify({'error': 'No preview data found. Please upload file again.'}), 400
    
    chunks = preview_data.get('chunks', [])
    source = preview_data.get('source', 'file')
    filename = preview_data.get('filename', '')
    
    if not chunks:
        return jsonify({'error': 'No chunks data found.'}), 400
    
    # Filter to only selected chunks
    selected_chunks = [chunks[i] for i in selected_indices if i < len(chunks)]
    
    if not selected_chunks:
        return jsonify({'error': 'No chunks selected'}), 400
    
    # Convert to embeddings
    embedded_chunks = document_processor.chunk_to_embeddings(selected_chunks)
    
    # Store in Qdrant
    chunk_count = store_chunks_in_qdrant(clone_id, embedded_chunks)
    
    # Update clone metadata
    clone['chunk_count'] = clone.get('chunk_count', 0) + chunk_count
    clone['last_updated'] = datetime.now().isoformat()
    save_clones(clones)
    
    # Clean up session
    session.pop(f'preview_chunks_{clone_id}', None)
    
    # Clean up temp file if it was a file upload
    if source == 'file' and filename:
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'preview_{filename}')
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception as e:
                print(f"Error deleting temp file: {e}")
    
    return jsonify({
        'success': True,
        'chunks_added': chunk_count,
        'total_chunks': clone['chunk_count']
    })


@app.route('/upload_data/<clone_id>', methods=['POST'])
def upload_data(clone_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        return jsonify({'error': 'Access denied'}), 403
    
    embedded_chunks = []
    
    # Check if text input is provided
    if 'text_content' in request.form and request.form['text_content'].strip():
        text_content = request.form['text_content'].strip()
        
        # Process as text
        chunks = document_processor.process_text_input(text_content)
        embedded_chunks = document_processor.chunk_to_embeddings(chunks)
    
    # Check if file is provided
    elif 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        filename = secure_filename(file.filename)
        
        # SIMPLIFIED: LlamaParse supports almost any format!
        allowed_extensions = {
            'pdf', 'docx', 'doc', 'pptx', 'ppt',  # Documents
            'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif',  # Images
            'txt', 'csv', 'xlsx', 'xls'  # Text/Data files
        }
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported file type. Supported: {", ".join(sorted(allowed_extensions))}'
            }), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # ONE FUNCTION handles everything!
            embedded_chunks, _ = process_uploaded_file(filepath, filename)
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({'error': 'No text or file provided'}), 400
    
    # Store in Qdrant
    if embedded_chunks:
        chunk_count = store_chunks_in_qdrant(clone_id, embedded_chunks)
        
        # Update clone metadata
        clone['chunk_count'] = clone.get('chunk_count', 0) + chunk_count
        clone['last_updated'] = datetime.now().isoformat()
        save_clones(clones)
        
        return jsonify({
            'success': True,
            'chunks_added': chunk_count,
            'total_chunks': clone['chunk_count']
        })
    
    return jsonify({'error': 'Failed to process content'}), 500

@app.route('/upload_document/<clone_id>', methods=['POST'])
def upload_document(clone_id):
    """Upload document with description for later retrieval"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        return jsonify({'error': 'Access denied'}), 403
    
    # Check for file and description
    if 'document' not in request.files or not request.files['document'].filename:
        return jsonify({'error': 'No document provided'}), 400
    
    description = request.form.get('description', '').strip()
    if not description:
        return jsonify({'error': 'Description is required'}), 400
    
    file = request.files['document']
    
    try:
        # Save document permanently
        file_info = save_document(file, clone_id)
        
        # Store description as vector with document reference
        success = store_document_reference(clone_id, description, file_info)
        
        if success:
            # Update clone chunk count
            clone['chunk_count'] = clone.get('chunk_count', 0) + 1
            clone['last_updated'] = datetime.now().isoformat()
            save_clones(clones)
            
            return jsonify({
                'success': True,
                'message': 'Document uploaded successfully',
                'filename': file_info['filename'],
                'total_chunks': clone['chunk_count']
            })
        else:
            # Clean up file if vector storage failed
            if os.path.exists(file_info['filepath']):
                os.remove(file_info['filepath'])
            return jsonify({'error': 'Failed to store document reference'}), 500
            
    except Exception as e:
        print(f"Document upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_document/<clone_id>/<unique_filename>')
def download_document(clone_id, unique_filename):
    """Serve document for download"""
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone:
        return jsonify({'error': 'Clone not found'}), 404
    
    # Check if clone is public or user is owner
    if not clone.get('is_public') and (
        'username' not in session or clone['owner'] != session['username']
    ):
        return jsonify({'error': 'Access denied'}), 403
    
    # Construct file path
    clone_doc_folder = os.path.join(app.config['DOCUMENTS_FOLDER'], clone_id)
    filepath = os.path.join(clone_doc_folder, unique_filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Extract original filename
    original_filename = '_'.join(unique_filename.split('_')[1:])
    
    return send_file(
        filepath,
        as_attachment=True,
        download_name=original_filename
    )

@app.route('/chat/<clone_id>')
def chat_page(clone_id):
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone:
        flash('Clone not found')
        return redirect(url_for('index'))
    
    if not clone.get('is_public') and (
        'username' not in session or clone['owner'] != session['username']
    ):
        flash('This clone is private')
        return redirect(url_for('index'))
    
    return render_template('chat.html', clone=clone, clone_id=clone_id)

@app.route('/api/chat/<clone_id>', methods=['POST'])
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
        response_text = f"I'm {clone['owner_full_name']}, but I don't have enough information to answer that question yet. Please add more data to my knowledge base."
        documents = []
    else:
        # Generate response with intelligent document filtering
        response_text, documents = generate_response(query, context_data, clone['owner_full_name'], clone_id, chat_history)
        
        # Debug: Log relevance scores (remove in production)
        print(f"Query: {query}")
        print(f"Top chunk scores: {[item.get('score', 0) for item in context_data[:3]]}")
        print(f"Documents returned: {len(documents)}")
    
    # Add to chat history
    add_to_chat_history(session_id, clone_id, 'user', query)
    add_to_chat_history(session_id, clone_id, 'assistant', response_text)
    
    return jsonify({
        'response': response_text,
        'documents': documents,
        'clone_id': clone_id
    })


@app.route('/whatsapp_webhook', methods=['POST'])
def whatsapp_webhook():
    incoming_number = request.values.get('From', '')
    sender_phone = incoming_number.replace('whatsapp:', '').strip()
    
    # Normalize the incoming phone number
    sender_phone = normalize_phone_number(sender_phone)
    
    # ... rest of your webhook code ...
    
    # When matching, compare normalized numbers
    clones = load_clones()
    matched_clone = None
    matched_id = None
    
    for clone_id, clone in clones.items():
        stored_phone = clone.get('phone_number')
        if stored_phone:
            # Normalize stored phone too for comparison
            normalized_stored = normalize_phone_number(stored_phone)
            if sender_phone == normalized_stored:
                matched_clone = clone
                matched_id = clone_id
                break
    
    # ... rest of your webhook code ...
    message_body = request.values.get('Body', '')
    num_media = int(request.values.get('NumMedia', 0))
    media_url = request.values.get('MediaUrl0', None)
    media_content_type = request.values.get('MediaContentType0', None)


    from twilio.twiml.messaging_response import MessagingResponse
    resp = MessagingResponse()

    if not matched_clone:
        resp.message("❌ This phone number is not registered to any chatbot.")
        return str(resp)

    text_chunks = []
    extracted_expiry_date = None

    # 1. If message is text only
    if message_body and num_media == 0:
        text = message_body.strip()
        # Use LLM to check expiry and extract date
        extracted_expiry_date = detect_expiry_info_with_llm(text)
        text_chunks = [text]
    # 2. If file/media message (image/doc, possible notice, docx, etc.)
    elif num_media > 0 and media_url:
        media_file = download_media_file(media_url, media_content_type)
        # Use your multimodal document_processor to extract info
        all_chunks = document_processor.process_file(media_file)
        texts = [chunk['text'] for chunk in all_chunks if chunk.get('text')]
        text_chunks.extend(texts)
        # For each chunk, check expiry individually
        expiry_results = [detect_expiry_info_with_llm(chunk) for chunk in texts]
    else:
        resp.message("❗️Empty message or unsupported media type.")
        return str(resp)

    # 3. Store each extracted chunk
    expired_dates = []
    for i, chunk_text in enumerate(text_chunks):
        if not chunk_text.strip():
            continue
        # Use LLM logic to extract expiry for each chunk 
        expiry_date = expiry_results[i] if num_media > 0 and media_url else extracted_expiry_date

        # Compose payload, include expiry_date and raw date string if detected
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
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=payload
        )
        qdrant_client.upsert(
            collection_name=app.config['COLLECTION_NAME'],
            points=[point]
        )
        matched_clone['chunk_count'] = matched_clone.get('chunk_count', 0) + 1
        matched_clone['last_updated'] = datetime.now().isoformat()

    clones[matched_id] = matched_clone
    save_clones(clones)

    # Reply after successful ingestion
    confirm_msg = "✅ Your message has been added!"
    if expired_dates:
        confirm_msg += f"\n(Info will auto-expire on: {', '.join(expired_dates)})"
    resp.message(confirm_msg)
    return str(resp)





@app.route('/api/clear_chat/<clone_id>', methods=['POST'])
def clear_chat_history(clone_id):
    """Clear chat history for current session"""
    session_id = get_session_id()
    key = f"{session_id}_{clone_id}"
    if key in chat_histories:
        chat_histories[key].clear()
    return jsonify({'success': True, 'message': 'Chat history cleared'})



if __name__ == '__main__':
    # Start the expired chunk cleaner thread on app launch
    threading.Thread(target=delete_expired_chunks_job, daemon=True).start()
    app.run(debug=True, use_reloader=False)


