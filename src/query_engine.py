from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from sentence_transformers import CrossEncoder
from utils import logger, log_exceptions, log_time
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater")
# Global reranker variable (optional)
cross_encoder = None

@log_exceptions
@log_time
def build_vector_index(chunked_docs):
    """
    Build a VectorStoreIndex from chunked documents.
    Returns the index object.
    """
    try:
        logger.info(f"Building vector index from {len(chunked_docs)} document chunks")
        index = VectorStoreIndex.from_documents(chunked_docs)
        logger.info("Vector index created successfully")
        return index
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}", exc_info=True)
        return None


@log_exceptions
def initialize_reranker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """
    Initialize a CrossEncoder for optional reranking.
    """
    global cross_encoder
    try:
        cross_encoder = CrossEncoder(model_name)
        logger.info(f"CrossEncoder reranker initialized: {model_name}")
        return cross_encoder
    except Exception as e:
        logger.error(f"Failed to initialize reranker: {e}", exc_info=True)
        return None


@log_exceptions
def rerank_nodes(nodes, query, top_k=5):
    """
    Rerank nodes based on query using CrossEncoder.
    Returns top_k nodes.
    """
    if cross_encoder is None:
        logger.warning("Reranker not initialized, skipping rerank")
        return nodes[:top_k]

    try:
        scores = cross_encoder.predict([(node.node.text, query) for node in nodes])
        ranked_nodes = [node for _, node in sorted(zip(scores, nodes), reverse=True)]
        return ranked_nodes[:top_k]
    except Exception as e:
        logger.error(f"Failed to rerank nodes: {e}", exc_info=True)
        return nodes[:top_k]


@log_exceptions
def create_query_engine(index, llm):
    """
    Create a query engine from a VectorStoreIndex and LLM.
    """
    try:
        query_engine = index.as_query_engine(llm=llm)
        logger.info("Query engine created successfully")
        return query_engine
    except Exception as e:
        logger.error(f"Failed to create query engine: {e}", exc_info=True)
        return None


@log_exceptions
def ask_question(query_engine, query, use_rerank=False, top_k=5):
    """
    Ask a query to the RAG system.
    Returns a dict: {'answer': str, 'citations': list[str]}
    """
    if query_engine is None:
        logger.error("Query engine is None, cannot process query")
        return {"answer": "Query engine not initialized.", "citations": []}

    try:
        response = query_engine.query(query)
        top_nodes = response.source_nodes

        # Optional reranking
        if use_rerank:
            top_nodes = rerank_nodes(top_nodes, query, top_k=top_k)

        # Collect answer and citations
        answer_text = response.response
        citations = []
        for node in top_nodes:
            md = node.node.metadata
            src = md.get("source", "unknown_file")
            page = md.get("page")
            if page:
                citations.append(f"{src} (page {page})")
            else:
                citations.append(src)
        citations = list(set(citations))

        logger.info(f"Query processed successfully: {query}")
        return {"answer": answer_text, "citations": citations}

    except Exception as e:
        logger.error(f"Failed to process query: {query} | {e}", exc_info=True)
        return {"answer": "Failed to process query.", "citations": []}
