"""
run_app.py
-----------
Runs the RAG Chatbot with a clean Streamlit web interface.
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------------
# Setup Paths
# ---------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import RAG pipeline
from src.rag_pipeline import RAGPipeline  # type: ignore  # pylint: disable=wrong-import-position
from src.utils import setup_logger  # type: ignore  # pylint: disable=wrong-import-position


class AppException(Exception):
    """Custom application exception for user-friendly error reporting."""

    def __init__(self, message: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original


# ---------------------------------------------------------------------------------
# Initialize pipeline and logger
# ---------------------------------------------------------------------------------
LOG_PATH = BASE_DIR / "logs" / "app.log"
logger = setup_logger(str(LOG_PATH))

st.set_page_config(
    page_title="RAG Chatbot with Citation",
    layout="centered",
    page_icon="🧠",
)

st.title("🧠 RAG-based Chatbot with Citations")
st.markdown("Ask any question from your knowledge base and get responses with document citations.")


# ---------------------------------------------------------------------------------
# Initialize RAG Pipeline
# ---------------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline() -> RAGPipeline:
    try:
        rag = RAGPipeline(docs_folder=str(BASE_DIR / "data" / "sample_policies"))
        logger.info("RAG pipeline initialized successfully.")
        return rag
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Pipeline initialization failed: %s", exc)
        st.error("Failed to initialize RAG pipeline. Check logs for details.")
        raise AppException("Pipeline initialization failed.", exc) from exc


rag_pipeline = load_pipeline()


# ---------------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict[str, object]] = []

user_query = st.text_input("💬 Enter your question:", placeholder="What is the refund policy?")

col1, col2 = st.columns([1, 1])

with col1:
    ask_clicked = st.button("Ask", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear History", use_container_width=True)

if clear_clicked:
    st.session_state.chat_history.clear()
    st.success("Chat history cleared.")

if ask_clicked:
    if user_query.strip():
        with st.spinner("Generating answer..."):
            try:
                result = rag_pipeline.ask(user_query)
                answer = result.get("answer", "No answer found.")
                sources = result.get("citations") or result.get("sources") or []

                st.session_state.chat_history.append(
                    {"query": user_query, "answer": answer, "sources": sources}
                )

                logger.info("Response generated for query: %s", user_query)
            except AppException as exc:
                logger.error("AppException while processing query: %s", exc)
                st.error(str(exc))
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Unexpected error while processing query '%s': %s", user_query, exc)
                st.error("An unexpected error occurred while processing your question.")
    else:
        st.warning("Please enter a valid question.")

# ---------------------------------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------------------------------
for chat in reversed(st.session_state.chat_history):
    with st.expander(f"❓ {chat['query']}"):
        st.write(chat["answer"])
        sources = chat.get("sources") or []
        if sources:
            st.markdown("**📚 Sources:**")
            for src in sources:
                st.markdown(f"- `{src}`")

# ---------------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------------
st.markdown("---")
st.caption("🚀 Powered by Local RAG Pipeline • Built with Streamlit")

