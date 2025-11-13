🧠 RAG Chatbot with Citation

A Retrieval-Augmented Generation (RAG) based chatbot that can answer user queries using local documents, while also providing source citations for transparency.
This project integrates LlamaIndex, Ollama LLM, and Hugging Face Embeddings to build a self-contained, offline-capable document question-answering system with a simple and interactive interface.

🚀 Features

📄 Document-Aware Responses – Answers are grounded in uploaded files.

📚 Citation Support – Each response includes the filename sources.

⚡ Fast Local Inference – Runs fully offline using Ollama models (e.g., llama3.2:latest).

🧩 RAG Pipeline – Built on top of LlamaIndex for modularity and extendability.

💻 Simple Web Interface – Clean frontend for chatting with your documents.

🏗️ Project Structure
RAG-Chatbot-with-Citation/
│
├── data/                     # Folder containing user-uploaded or reference documents
│   ├── doc1.pdf
│   └── notes.txt
│
├── backend/
│   └── app.py                # Flask-based backend API for RAG responses
│
├── rag_pipeline.py            # Main pipeline defining embeddings, LLM, and query engine
│
├── run_app.py                 # Runs both backend + frontend interface
│
├── requirements.txt           # Required Python dependencies
│
└── README.md                  # Documentation (this file)

⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/<your-username>/RAG-Chatbot-with-Citation.git
cd RAG-Chatbot-with-Citation

2. Create and Activate a Virtual Environment
python -m venv venv
# Activate the environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Install Ollama

Download and install Ollama from https://ollama.com/download

Once installed, pull your model (example: llama3.2):

ollama pull llama3.2

5. Prepare Your Documents

Place any .txt, .pdf, or .md files inside the /data folder.
These documents will serve as your knowledge base.

6. Run the App
python run_app.py


Your chatbot will be available at:

http://127.0.0.1:7860

🧠 RAG Architecture Overview

Retrieval-Augmented Generation (RAG) combines information retrieval with language generation:

Document Ingestion – Local documents are loaded and preprocessed.

Embedding Generation – Each document chunk is converted into a dense vector using sentence-transformers/all-MiniLM-L6-v2.

Vector Indexing – The vectors are stored in a VectorStoreIndex (from LlamaIndex).

Retrieval – For each user query, the top relevant document chunks are retrieved using similarity search.

Augmented Generation – The retrieved text is combined with the user query and passed to the LLM (Ollama) for response generation.

Citation Mapping – The final output displays the sources used for the answer.

🧩 Core Components

LlamaIndex – Manages document indexing and retrieval.

Langchain Embedding (HuggingFace) – Creates embeddings for semantic similarity.

Ollama LLM – Runs a local large language model (like llama3.2).

Flask – Lightweight backend for the API endpoints.

Gradio Interface – Interactive web UI for chatting and viewing citations.

🧪 Example Usage

Query:

“What is the purpose of reinforcement learning?”

Response:

Reinforcement learning focuses on training agents to make sequences of decisions by rewarding good behavior and penalizing poor choices.

Sources:

📘 machine_learning_notes.txt
📘 AI_research_paper.pdf

🛠️ Customization

You can change the embedding model in rag_pipeline.py:

HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


Swap the LLM model to another Ollama model:

Ollama(model="mistral:latest")


Adjust retrieval depth:

query_engine = index.as_query_engine(similarity_top_k=5)

🤖 Future Enhancements

✅ Multi-user session tracking

✅ Chat history with memory persistence

✅ Improved citation ranking

✅ UI enhancements for better document management

👨‍💻 Author

Abdullah Bin Noor Tazim
AI Engineer | Machine Learning Enthusiast
