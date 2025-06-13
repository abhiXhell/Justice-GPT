# Indian Labour & Consumer Law AI Chatbot

A local AI reasoning chatbot specialized in Indian Labour and Consumer Court law, running entirely offline using Deepseek R1 model via Ollama.

## Prerequisites

1. Python 3.9 or higher
2. Ollama installed on your system
3. Git (optional, for cloning the repository)

## Setup Instructions

### 1. Install Python
- Download and install Python 3.9+ from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

### 2. Install Ollama
- Download Ollama from [ollama.ai](https://ollama.ai/download)
- Install and run Ollama
- Pull the Deepseek R1 model:
```bash
ollama pull deepseek-coder:latest
```

### 3. Project Setup
1. Clone or download this repository
2. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Running the Application
1. Ensure Ollama is running in the background
2. Start the Streamlit application:
```bash
streamlit run app.py
```

### 5. Using the Chatbot
1. Upload relevant legal documents (PDFs) through the web interface
2. Wait for the documents to be processed and indexed
3. Start asking questions about Indian Labour and Consumer Court law

## Project Structure
```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── src/
│   ├── document_processor.py  # Document processing and indexing
│   ├── rag_engine.py         # RAG implementation
│   └── llm_interface.py      # Ollama/Deepseek interface
└── data/
    └── documents/           # Directory for storing uploaded documents
```

## Notes
- The chatbot is designed to work offline and only with the provided legal documents
- All processing happens locally on your machine
- The system is optimized for Indian Labour and Consumer Court law queries 