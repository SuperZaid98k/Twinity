# ingest.py
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collection = "zaid_clone"

# Create collection if not exists
if collection not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    {"id": "1", "text": "Hello, I am Mohammad Zaid and I am Final Year Btech Student doing specialization in Artificial Intelligence and Data Science.My Objective is to leverage my skills in Python, Machine Learning and data analysis to contribute to innovative projects in a dynamic organization, while gaining valuable industry experience in Artificial Intelligence and Data Science."},
    {"id": "2", "text": """My skills includes  Programming Languages  : Python (Advance),  SQL (Intermediate) 
 Machine Learning And Deep Learning  : Scikit-learn, TensorFlow (Advance) 
 Data Collection  :  Selenium, BeautifulSoup, Requests (Intermediate) 
 Data Visualization  : Matplotlib, Seaborn (Intermediate) 
 Data Manipulation  : NumPy , Pandas (Advance) 
 Tools & Platforms  : Jupyter Notebook, Colab, VS Code, Excel 
 Statistics  :  Statistical analysis, Hypothesis Testing 
 Other Skills  : Web Dev, Exploratory Data Analysis (EDA) , NLP (Intermediate), AWS (EC2)."""},
    {"id": "3", "text": "I have worked on Projects like Automation and Data Collection, ML Model Deployment and Score App."},
    {"id": "4", "text": """In my Project of Automation and Data Collection i had   Built a Python tool to  automate downloading and 
 processing academic results. Extracted text from images using OCR and saved data in 
 Excel files. 
 The Tools which I Used were Python, Selenium ,Tesseract OCR, Pillow, OpenCV and Pandas ."""},
    {"id": "5", "text": """In my other Project of ML Model Deployment i had Developed a Flask-based web application to predict
student performance using linear regression.
 The Tools which I Used were Scikit-Learn, Pandas, Flask, VS Code."""},
    {"id": "6", "text": """In my other Project of Score App i had Created and hosted a web application for scoring projects at a
steam event for an organization.
 The Tools which I Used were Python, Excel, HTML, CSS, PythonAnywhere, Pandas."""}
]

points = []
for i, doc in enumerate(documents):
    embedding = model.encode(doc["text"]).tolist()
    points.append(
        {"id": i, "vector": embedding, "payload": {"text": doc["text"]}}
    )

client.upsert(collection_name=collection, points=points)
print("✅ Data uploaded to Qdrant")
