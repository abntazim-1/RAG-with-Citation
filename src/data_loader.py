import os
import pdfplumber
from llama_index.core import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import setup_logger, log_exceptions, log_time

# Setup logger
logger = setup_logger("logs/data_loader.log")


@log_exceptions
@log_time
def load_documents(folder_path="data/sample_policies"):
    """
    Load PDF and TXT documents from the folder, store text in Document objects with metadata.
    Returns a list of Document objects.
    """
    docs = []

    if not os.path.exists(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        return docs

    files = os.listdir(folder_path)
    if not files:
        logger.warning(f"No files found in folder: {folder_path}")

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            if filename.lower().endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            docs.append(Document(
                                text=text,
                                metadata={"source": filename, "page": i+1}
                            ))
                logger.info(f"Loaded PDF: {filename}")
            elif filename.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    if text.strip():
                        docs.append(Document(text=text, metadata={"source": filename}))
                logger.info(f"Loaded TXT: {filename}")
            else:
                logger.warning(f"Unsupported file type skipped: {filename}")
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}", exc_info=True)

    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


@log_exceptions
@log_time
def split_documents(docs, chunk_size=1024, chunk_overlap=128):
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Returns a list of chunked Document objects preserving metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?"]
    )

    chunked_docs = []
    for doc in docs:
        try:
            chunks = splitter.split_text(doc.text)
            for c in chunks:
                chunked_docs.append(Document(text=c, metadata=doc.metadata))
        except Exception as e:
            logger.error(f"Failed to split document {doc.metadata.get('source')}: {e}", exc_info=True)

    logger.info(f"Total document chunks created: {len(chunked_docs)}")
    return chunked_docs
