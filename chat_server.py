import os
import requests
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Connect to Qdrant Cloud
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
COLLECTION = "zaid_clone"

app = FastAPI()

# Allow local frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversation history (simple in-memory dict)
conversations = {}

def retrieve_context(user_message: str, top_k: int = 3):
    query_vec = embedder.encode(user_message).tolist()
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=top_k
    )
    return " ".join([r.payload["text"] for r in results])

def or_generate(user_message: str, session_id: str = "default"):
    # Initialize history for session if not exists
    if session_id not in conversations:
        conversations[session_id] = [
            {"role": "system", "content": "You are a friendly AI clone. Use personal info when relevant."}
        ]

    # 🔎 Get grounding info from Qdrant
    context = retrieve_context(user_message)
    conversations[session_id].append({"role": "system", "content": f"Relevant info: {context}"})
    conversations[session_id].append({"role": "user", "content": user_message})

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": conversations[session_id],
    }

    resp = requests.post(url, headers=headers, json=data)

    if resp.status_code != 200:
        return f"[Error] API returned {resp.status_code}: {resp.text}"

    try:
        result = resp.json()
        reply = result["choices"][0]["message"]["content"].strip()
        conversations[session_id].append({"role": "assistant", "content": reply})
        return reply
    except Exception:
        return f"[Error] Failed to parse response: {resp.text}"

@app.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))  # unique id per connection
    while True:
        try:
            msg = await websocket.receive_text()
            reply = or_generate(msg, session_id=session_id)
            await websocket.send_text(f"{reply}")
        except Exception:
            await websocket.close()
            break
