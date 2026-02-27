import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
USERS_FILE = os.path.join(DATA_DIR, 'users.json')
CLONES_FILE = os.path.join(DATA_DIR, 'clones.json')

# Initialize JSON storage files
def init_json_files():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)
    if not os.path.exists(CLONES_FILE):
        with open(CLONES_FILE, 'w') as f:
            json.dump({}, f)

def load_users():
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)
