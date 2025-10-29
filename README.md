# 🤖 Flask RAG Multi-User Chatbot Platform

A production-ready Flask-based RAG (Retrieval-Augmented Generation) web application where users can create public AI chatbots ("clones") trained on their uploaded data, with automatic WhatsApp integration for real-time knowledge updates.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 🎯 Core Functionality
- **Multi-User System**: User registration, authentication, and personal dashboards
- **AI Chatbot Clones**: Create unlimited AI assistants trained on custom data
- **Public/Private Clones**: Share chatbots publicly or keep them private
- **RAG Architecture**: Intelligent retrieval-augmented generation for accurate responses
- **Conversation Context**: Chatbots remember previous exchanges for contextual responses

### 📤 Data Upload Methods
- **Text Input**: Direct text paste with semantic chunking
- **Document Upload**: AI-powered parsing of PDF, DOCX, PPTX, Excel, images
- **Chunk Selection**: Preview and select specific chunks before uploading
- **Document Attachments**: Upload files with descriptions for download in chat
- **WhatsApp Integration**: Automatic knowledge updates via WhatsApp messages

### 🤖 Advanced AI Features
- **Multimodal LLM Processing**: Intelligent extraction from documents and images using LlamaParse
- **Semantic Chunking**: Context-aware text splitting for better retrieval
- **Resume Parsing**: Automatic detection and section-based chunking (skills, experience, education)
- **Expiry Detection**: LLM-powered detection of time-sensitive information
- **Auto-Cleanup**: Background job automatically removes expired knowledge

### 📱 WhatsApp Automation
- **Real-time Updates**: Send messages/files to update chatbot knowledge instantly
- **Twilio Integration**: Secure WhatsApp Business API integration
- **Media Processing**: Automatic extraction from images and documents
- **Expiry Notifications**: Automatic WhatsApp alerts when knowledge expires
- **Easy Setup**: Step-by-step guide with QR code for quick onboarding

### 🎨 User Experience
- **Modern UI**: Gradient designs, smooth animations, responsive layout
- **Document Panel**: Separate downloadable files display (not inline in chat)
- **Chat History**: Context-aware responses based on conversation
- **Data Management**: View, edit, and delete knowledge chunks
- **Show/Hide Password**: User-friendly authentication forms

## 🏗️ Architecture

┌─────────────┐ ┌──────────────┐ ┌─────────────┐
│ Frontend │────▶│ Flask App │────▶│ Qdrant │
│ (HTML/CSS/JS)│ │ (Python) │ │ (Vectors) │
└─────────────┘ └──────────────┘ └─────────────┘
│
▼
┌──────────────┐
│ OpenRouter │
│ (LLM API) │
└──────────────┘
│
▼
┌──────────────┐
│ LlamaParse │
│ (Doc Parser) │
└──────────────┘
│
▼
┌──────────────┐
│ Twilio │
│ (WhatsApp) │
└──────────────┘

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Qdrant Cloud account
- OpenRouter API key
- LlamaParse API key
- Twilio account (optional, for WhatsApp)

### Installation

1. **Clone the repository**
git clone https://github.com/yourusername/flask-rag-chatbot.git
cd flask-rag-chatbot

2. **Create virtual environment**
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


3. **Install dependencies**
pip install -r requirements.txt


4. **Set up environment variables**

Create a `.env` file in the root directory:
SECRET_KEY=your-secret-key-here
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
LLAMA_CLOUD_API_KEY=your-llama-cloud-api-key
TWILIO_ACCOUNT_SID=your-twilio-sid # Optional
TWILIO_AUTH_TOKEN=your-twilio-token # Optional


5. **Run the application**
python app.py


6. **Access the app**
Navigate to `http://localhost:5000`

## 📁 Project Structure

flask-rag-chatbot/
├── app.py # Main Flask application
├── config.py # Configuration settings
├── document_processor.py # LLM-based document processing
├── requirements.txt # Python dependencies
├── .env # Environment variables (create this)
├── data/
│ ├── users.json # User accounts storage
│ └── clones.json # Chatbot clones metadata
├── documents/ # Permanent document storage
├── uploads/ # Temporary file uploads
├── templates/
│ ├── base.html # Base template
│ ├── index.html # Landing page
│ ├── login.html # User login
│ ├── register.html # User registration
│ ├── dashboard.html # User dashboard
│ ├── create_clone.html # Clone creation
│ ├── edit_clone.html # Clone editing & data upload
│ ├── manage_data.html # Data management
│ └── chat.html # Chat interface
└── static/
├── style.css # Main stylesheet
└── Screenshot-*.jpg # Twilio QR code


## 🔧 Configuration

### Qdrant Cloud Setup

1. Create a free account at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a new cluster
3. Copy the cluster URL and API key to `.env`

### OpenRouter Setup

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Generate an API key
3. Add to `.env` file

### LlamaParse Setup

1. Create account at [cloud.llamaindex.ai](https://cloud.llamaindex.ai)
2. Get API key (Free tier: 1000 pages/day)
3. Add to `.env` file

### WhatsApp Setup (Optional)

1. Create Twilio account at [twilio.com](https://www.twilio.com)
2. Set up WhatsApp sandbox
3. Configure webhook URL to `https://your-domain.com/whatsapp_webhook`
4. Add credentials to `.env`

**For local testing with ngrok:**

## 📖 Usage Guide

### Creating a Chatbot

1. **Register/Login**: Create an account or log in
2. **Create Clone**: Click "Create New Clone" and provide:
   - Clone name
   - Description
   - Public/Private visibility
3. **Add Data**: Upload training data via:
   - Text input
   - Document upload (PDF, DOCX, images, etc.)
   - WhatsApp messages

### WhatsApp Integration

1. **Enable WhatsApp**: Go to Edit Clone → WhatsApp Integration
2. **Enter Phone Number**: Add your WhatsApp number
3. **Join Sandbox**: 
   - Message `+1 415 523 8886` from WhatsApp
   - Send code: `join clearly-this`
   - Or scan the QR code displayed
4. **Start Uploading**: Send text/files to automatically update your chatbot

### Managing Data

- **View Chunks**: Click "Manage Data" to see all knowledge chunks
- **Delete Chunks**: Select and delete unwanted information
- **Preview Full Text**: View complete chunk content
- **Auto-Expiry**: Time-sensitive info automatically expires

### Chatting

1. Navigate to your chatbot or public chatbots
2. Ask questions based on uploaded knowledge
3. Download attached documents from the document panel
4. Clear chat history to start fresh

## 🛠️ Advanced Features

### Custom Chunking

The system intelligently chunks documents based on:
- **Resume Detection**: Separates contact, skills, experience, education
- **Section Headers**: Maintains document structure
- **Semantic Similarity**: Groups related content
- **Table Preservation**: Keeps tabular data intact

### Expiry Detection

LLM automatically determines if information is time-bound:
"Tomorrow is a holiday" → Expires next day
"Exam form deadline: 01/11/2025" → Expires on 02/11/2025
"My name is John" → Permanent (never expires)




### Background Jobs

- **Expired Chunk Cleanup**: Runs every 15 minutes
- **WhatsApp Notifications**: Alerts when knowledge expires
- **Automatic Vector Deletion**: Keeps database clean

## 🔒 Security

- Password hashing with Werkzeug
- Session-based authentication
- Access control for private clones
- Twilio webhook signature validation (production)
- File type and size validation
- SQL injection prevention via Qdrant

## 🚀 Deployment

### Production Considerations

1. **Use a production WSGI server** (Gunicorn/uWSGI)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app


2. **Set up proper database** (Replace JSON with PostgreSQL/MongoDB)
3. **Use Redis for chat history** (Instead of in-memory storage)
4. **Enable HTTPS** with SSL certificates
5. **Set up background job manager** (Celery instead of threading)
6. **Configure CORS** for API access
7. **Add rate limiting** to prevent abuse

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LlamaParse**: Advanced document parsing
- **Qdrant**: Vector database
- **OpenRouter**: LLM API aggregation
- **Twilio**: WhatsApp Business API
- **SentenceTransformers**: Embedding models

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Email: your-email@example.com

## 🗺️ Roadmap

- [ ] Voice message support via WhatsApp
- [ ] Multi-language support
- [ ] Analytics dashboard
- [ ] API endpoints for integrations
- [ ] Export/import chatbot data
- [ ] Team collaboration features
- [ ] Custom embedding models
- [ ] Fine-tuned LLM support

---

**Built with ❤️ using Flask, AI, and WhatsApp**













































