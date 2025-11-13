from utils import logger, log_exceptions, log_time
from data_loader import load_documents, split_documents
from embedding import initialize_embeddings
from llm_setup import initialize_llm
from query_engine import build_vector_index, create_query_engine, ask_question, initialize_reranker
from llama_index.core import Settings
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater")

class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline.
    Combines document ingestion, embeddings, LLM, vector search, and optional reranking.
    """

    @log_exceptions
    @log_time
    def __init__(self,
                 docs_folder="data/sample_policies",
                 embedding_model="intfloat/e5-large-v2",
                 llm_model="llama3.2:1b",
                 use_rerank=True,
                 device="cpu"):
        logger.info("Initializing RAG Pipeline...")

        # 1. Load & split documents
        self.docs = load_documents(docs_folder)
        self.chunked_docs = split_documents(self.docs)

        # 2. Initialize embeddings
        self.embed_model = initialize_embeddings(model_name=embedding_model, device=device)
        # Ensure LlamaIndex uses our embedding model (avoid default OpenAI dependency)
        if self.embed_model is not None:
            Settings.embed_model = self.embed_model

        # 3. Initialize LLM
        self.llm = initialize_llm(model_name=llm_model)

        # 4. Build vector index
        self.index = build_vector_index(self.chunked_docs)

        # 5. Initialize reranker (optional)
        self.use_rerank = use_rerank
        if use_rerank:
            initialize_reranker()

        # 6. Create query engine
        self.query_engine = create_query_engine(self.index, self.llm)

        # 7. Initialize conversation memory (optional)
        self.chat_history = []  # simple memory as list of dicts
        logger.info("RAG Pipeline initialized successfully.")

    @log_exceptions
    def ask(self, query, top_k=5):
        """
        Ask a question using the RAG pipeline.
        Returns answer + citations and updates chat history.
        """
        logger.info(f"Received query: {query}")
        result = ask_question(
            self.query_engine,
            query,
            use_rerank=self.use_rerank,
            top_k=top_k
        )

        # Update chat history
        self.chat_history.append({"query": query, "answer": result["answer"], "citations": result["citations"]})

        return result

    @log_exceptions
    def get_history(self):
        """
        Returns the conversation history.
        """
        return self.chat_history
