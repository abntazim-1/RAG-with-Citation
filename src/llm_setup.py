from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from utils import logger, log_exceptions, log_time
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater")

@log_exceptions
@log_time


def initialize_llm(
    model_name="llama3.2:1b",
    system_prompt=None,
    temperature=0.0,
    max_tokens=2000,
    context_window=2048,
    keep_alive="1m",
    chunk_size=1024,
    chunk_overlap=128,
    similarity_top_k=5,
    similarity_cutoff=0.75,
    response_mode="compact"
):
    """
    Initialize LLM (Ollama) and LlamaIndex Settings.
    Returns the LLM object.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a factual QA assistant.\n"
            "- Answer ONLY based on the retrieved context.\n"
            "- Cite the source document (filename) for every factual statement.\n"
            "- If the answer is unknown, say 'I don't know.'\n"
            "- Keep answers concise and precise."
        )

    try:
        logger.info(f"Initializing LLM: {model_name}")
        llm = Ollama(
            model=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            keep_alive=keep_alive
        )

        # Update LlamaIndex global settings
        Settings.llm = llm
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        Settings.similarity_top_k = similarity_top_k
        Settings.node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
        Settings.response_mode = "compact"
        Settings.system_prompt = system_prompt

        logger.info("LLM and Settings initialized successfully")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        return None
