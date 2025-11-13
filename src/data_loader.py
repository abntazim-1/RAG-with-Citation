import os
from pathlib import Path
import pdfplumber
from llama_index.core import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import setup_logger, log_exceptions, log_time

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater")
# Setup logger
logger = setup_logger("logs/data_loader.log")


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".conf",
    ".html",
    ".htm",
}


def _read_text_file(file_path: str, filename: str) -> str | None:
    """Attempt to read a text file using a sequence of encodings."""
    encodings = ("utf-8", "utf-16", "latin-1")
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as handle:
                content = handle.read()
                if content.strip():
                    return content
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # pragma: no cover - logged for visibility
            logger.error(f"Failed to load {filename} with encoding {encoding}: {exc}", exc_info=True)
            return None

    # Fallback: read as binary and decode ignoring errors
    try:
        with open(file_path, "rb") as handle:
            raw = handle.read()
        content = raw.decode("utf-8", errors="ignore")
        if content.strip():
            logger.warning(f"Decoded {filename} using utf-8 with ignored errors.")
            return content
    except Exception as exc:
        logger.error(f"Failed to load {filename}: {exc}", exc_info=True)

    return None


def _is_supported_text_file(filename: str) -> bool:
    """Determine whether a filename is a supported text format."""
    extension = Path(filename).suffix.lower()
    return extension in TEXT_EXTENSIONS or extension == ""


@log_exceptions
@log_time
def load_documents(folder_path="data/sample_policies"):
    """
    Load PDF and text-based documents from the folder, storing text in Document objects with metadata.

    Supported text formats include: txt, md, markdown, rst, log, csv, tsv, json, yaml, yml, ini, cfg, conf, html, htm.
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
                                metadata={"source": filename, "page": i + 1}
                            ))
                logger.info(f"Loaded PDF: {filename}")
            elif _is_supported_text_file(filename):
                text = _read_text_file(file_path, filename)
                if text:
                    docs.append(Document(text=text, metadata={"source": filename}))
                    logger.info(f"Loaded text file: {filename}")
                else:
                    logger.warning(f"No readable text found in {filename}")
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
