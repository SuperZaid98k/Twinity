# ========================
# Standard Library Imports
# ========================
import os
import threading
import sys

# Ensure absolute `backend.*` imports work even when run from `backend/`.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ========================
# Third-Party Imports
# ========================
from flask import Flask
from werkzeug.exceptions import RequestEntityTooLarge
from flask import jsonify

# ========================
# Internal App Imports
# ========================
from backend.config import Config

from backend.repositories.user_repo import init_json_files
from backend.core.qdrant_init import init_qdrant_collection
from backend.background.expiry_cleaner import delete_expired_chunks_job

from backend.routes.auth_routes import auth_bp
from backend.routes.page_routes import page_bp
from backend.routes.chat_routes import chat_bp
from backend.routes.clone_routes import clone_bp
from backend.routes.data_routes import data_bp
from backend.routes.whatsapp_routes import whatsapp_bp

# ========================
# Filesystem Preparation
# ========================
app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(app.config['DATA_DIR'], exist_ok=True)
os.makedirs(app.config['DOCUMENTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========================
# Blueprint Registration
# ========================
app.register_blueprint(auth_bp)
app.register_blueprint(page_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(clone_bp)
app.register_blueprint(data_bp)
app.register_blueprint(whatsapp_bp)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_e):
    return jsonify({'error': 'File too large. Max upload size is 16MB.'}), 413

# ========================
# One-Time Initializations
# ========================
init_json_files()
with app.app_context():
    qdrant_ready = init_qdrant_collection()

# ========================
# Application Entry Point
# ========================
if __name__ == '__main__':
    if qdrant_ready:
        threading.Thread(
            target=delete_expired_chunks_job,
            daemon=True
        ).start()
    else:
        print("Qdrant initialization failed; skipping expired chunk cleanup thread.")

    app.run(debug=True, use_reloader=False)
