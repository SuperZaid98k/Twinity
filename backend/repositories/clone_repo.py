import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CLONES_FILE = os.path.join(DATA_DIR, 'clones.json')

def load_clones():
    with open(CLONES_FILE, 'r') as f:
        return json.load(f)

def save_clones(clones):
    with open(CLONES_FILE, 'w') as f:
        json.dump(clones, f, indent=2)
