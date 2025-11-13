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
```
RAG-Chatbot-with-Citation/
│
├── data/                          # Folder containing user-uploaded or reference documents
│   ├── sample_policies/           # Default document directory
│   │   ├── *.pdf                 # PDF documents
│   │   └── *.txt                  # Text documents
│   └── vector_store/              # Vector store cache (auto-generated)
│
├── src/                           # Core source code modules
│   ├── rag_pipeline.py           # Main RAG pipeline orchestrator
│   ├── data_loader.py             # Document loading and chunking
│   ├── embedding.py               # Embedding model initialization
│   ├── llm_setup.py               # LLM (Ollama) configuration
│   ├── query_engine.py             # Query engine and reranking
│   └── utils.py                   # Logging and utility functions
│
├── logs/                          # Application logs
│   ├── app.log                    # Main application log
│   └── run_app.log                # Runtime log
│
├── run_app.py                     # Main entry point (Streamlit interface)
├── requirements.txt                # Python dependencies
└── README.md                       # This documentation
```

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

Start the Streamlit application:
```bash
streamlit run run_app.py
```

Or use Python directly:
```bash
python run_app.py
```

Your chatbot will be available at:

**http://127.0.0.1:8501** (Streamlit default port)

The application will automatically:
- Load documents from `data/sample_policies/`
- Initialize the embedding model and vector index
- Start the Ollama LLM connection
- Launch the interactive web interface

**Note:** Make sure Ollama is running before starting the app. You can verify by running `ollama list` in a separate terminal.

🧠 RAG Architecture Overview

Retrieval-Augmented Generation (RAG) combines information retrieval with language generation:

Document Ingestion – Local documents are loaded and preprocessed.

Embedding Generation – Each document chunk is converted into a dense vector using `intfloat/e5-large-v2` (configurable in `src/embedding.py`).

Vector Indexing – The vectors are stored in a VectorStoreIndex (from LlamaIndex).

Retrieval – For each user query, the top relevant document chunks are retrieved using similarity search.

Augmented Generation – The retrieved text is combined with the user query and passed to the LLM (Ollama) for response generation.

Citation Mapping – The final output displays the sources used for the answer.

🧩 Core Components

- **LlamaIndex** – Manages document indexing, vector storage, and retrieval
- **Langchain Embedding (HuggingFace)** – Creates embeddings for semantic similarity using `intfloat/e5-large-v2`
- **Ollama LLM** – Runs a local large language model (default: `llama3.2:1b`)
- **CrossEncoder Reranker** – Optional reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2` for improved relevance
- **Streamlit** – Interactive web UI for chatting and viewing citations
- **Session-based Chat History** – Each user session maintains its own conversation history

🧪 Example Usage

Query:

“What is the purpose of reinforcement learning?”

Response:

Reinforcement learning focuses on training agents to make sequences of decisions by rewarding good behavior and penalizing poor choices.

Sources:

📘 machine_learning_notes.txt
📘 AI_research_paper.pdf

🛠️ Customization

**Change the Embedding Model**

Edit `src/embedding.py` or pass parameters to `RAGPipeline`:
```python
rag = RAGPipeline(
    docs_folder="data/sample_policies",
    embedding_model="sentence-transformers/all-mpnet-base-v2"  # Change here
)
```

**Swap the LLM Model**

Edit `src/llm_setup.py` or pass parameters:
```python
rag = RAGPipeline(
    docs_folder="data/sample_policies",
    llm_model="mistral:latest"  # Change here
)
```

**Adjust Retrieval Parameters**

Edit `src/rag_pipeline.py`:
```python
result = rag_pipeline.ask(query, top_k=10)  # Retrieve more documents
```

**Disable Reranking**

For faster responses (slightly lower quality):
```python
rag = RAGPipeline(
    docs_folder="data/sample_policies",
    use_rerank=False
)
```

🤖 Features & Notes

✅ **Per-Session Chat History** – Each Streamlit session maintains its own conversation history, isolated from other users

✅ **Citation Support** – Every response includes source document citations with page numbers when available

✅ **Reranking** – Optional CrossEncoder reranking improves answer relevance

✅ **Offline Operation** – Fully local inference, no external API calls required

✅ **Modular Architecture** – Clean separation of concerns for easy customization

**Future Enhancements**

- [ ] Persistent chat history across sessions (database storage)
- [ ] Multi-user authentication and session management
- [ ] Document upload interface
- [ ] Export chat history to PDF/JSON
- [ ] Advanced citation ranking algorithms
- [ ] Support for more document formats (Word, Excel, etc.)

👨‍💻 Author

Abdullah Bin Noor Tazim
AI Engineer | Machine Learning Enthusiast
