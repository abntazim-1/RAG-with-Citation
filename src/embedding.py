from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from utils import logger, log_exceptions, log_time
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater")
@log_exceptions
@log_time
def initialize_embeddings(model_name="intfloat/e5-large-v2", device="cpu", batch_size=32):
    """
    Initialize HuggingFace embeddings wrapped in LangchainEmbedding.
    Returns LangchainEmbedding object.
    """
    try:
        logger.info(f"Initializing embeddings: {model_name} | device={device} | batch_size={batch_size}")
        hf_embed = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": batch_size}
        )
        embed_model = LangchainEmbedding(hf_embed)
        logger.info("Embeddings initialized successfully")
        return embed_model
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
        return None
