import hashlib
import json
import shutil
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMOTIONS_WORKBOOK,
    KNOWLEDGE_BASE_FILES,
    KNOWLEDGE_BASE_PDFS,
    MANIFEST_PATH,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    RAG_MIN_RELEVANCE,
    RETRIEVAL_LIMIT,
)
from app.emotions import (
    EmotionRecord,
    KnowledgeBaseError,
    emotion_records_to_documents,
    load_emotion_emojis,
    load_emotion_records,
)


@dataclass
class RagServices:
    knowledge_base: "ChromaKnowledgeBase"
    llm: ChatOpenAI | None
    emotions: list[EmotionRecord] = dataclass_field(default_factory=list)
    emotion_emojis: dict[str, str] = dataclass_field(default_factory=dict)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index_manifest(knowledge_paths: tuple[Path, ...]) -> dict:
    documents = []
    for path in knowledge_paths:
        if not path.is_file():
            raise KnowledgeBaseError(
                f"Required knowledge-base file is missing: {path.name}"
            )
        documents.append(
            {
                "name": path.name,
                "sha256": file_sha256(path),
            }
        )
    return {
        "documents": documents,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "collection": COLLECTION_NAME,
    }


def read_index_manifest() -> dict | None:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_and_split_documents(
    pdf_paths: tuple[Path, ...],
    emotions_workbook_path: Path | None = None,
) -> list[Document]:
    loaded_documents: list[Document] = []
    for pdf_path in pdf_paths:
        try:
            documents = PyMuPDFLoader(str(pdf_path)).load()
        except Exception as exc:
            raise KnowledgeBaseError(
                f"Could not read knowledge-base PDF: {pdf_path.name}"
            ) from exc
        if not any(document.page_content.strip() for document in documents):
            raise KnowledgeBaseError(
                f"No searchable text was found in knowledge-base PDF: {pdf_path.name}"
            )
        for document in documents:
            document.metadata["source"] = pdf_path.name
        loaded_documents.extend(documents)

    if emotions_workbook_path is not None:
        loaded_documents.extend(
            emotion_records_to_documents(
                load_emotion_records(emotions_workbook_path),
                emotions_workbook_path.name,
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [
        chunk
        for chunk in splitter.split_documents(loaded_documents)
        if chunk.page_content.strip()
    ]
    if not chunks:
        raise KnowledgeBaseError("The local knowledge base contains no searchable text.")
    return chunks


def create_embeddings() -> OpenAIEmbeddings:
    if not OPENAI_API_KEY:
        raise KnowledgeBaseError(
            "OPENAI_API_KEY is required to create OpenAI embeddings."
        )
    try:
        return OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=2,
        )
    except Exception as exc:
        raise KnowledgeBaseError(
            "OpenAI embeddings could not be initialised. "
            "Check OPENAI_API_KEY and network connectivity."
        ) from exc


class ChromaKnowledgeBase:
    def __init__(
        self,
        vector_store: Chroma,
        minimum_relevance: float = RAG_MIN_RELEVANCE,
    ):
        self.vector_store = vector_store
        self.minimum_relevance = minimum_relevance

    def retrieve(self, query: str) -> list[Document]:
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=RETRIEVAL_LIMIT,
            )
        except Exception as exc:
            raise KnowledgeBaseError(
                "The local Chroma knowledge base could not be searched."
            ) from exc
        return [
            document
            for document, score in results
            if score >= self.minimum_relevance
        ]


def build_or_load_knowledge_base() -> ChromaKnowledgeBase:
    expected_manifest = build_index_manifest(KNOWLEDGE_BASE_FILES)
    embeddings = create_embeddings()
    existing_manifest = read_index_manifest()

    if existing_manifest == expected_manifest and CHROMA_DIR.is_dir():
        try:
            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
            )
            if vector_store._collection.count() > 0:
                return ChromaKnowledgeBase(vector_store)
        except Exception:
            pass

    try:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        chunks = load_and_split_documents(KNOWLEDGE_BASE_PDFS, EMOTIONS_WORKBOOK)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
        MANIFEST_PATH.write_text(
            json.dumps(expected_manifest, indent=2),
            encoding="utf-8",
        )
        return ChromaKnowledgeBase(vector_store)
    except KnowledgeBaseError:
        raise
    except Exception as exc:
        raise KnowledgeBaseError(
            "The local Chroma knowledge base could not be created."
        ) from exc


def create_llm() -> ChatOpenAI | None:
    if not OPENAI_API_KEY:
        return None
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0,
        max_tokens=220,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=2,
    )


def initialise_services() -> RagServices:
    return RagServices(
        knowledge_base=build_or_load_knowledge_base(),
        llm=create_llm(),
        emotions=load_emotion_records(),
        emotion_emojis=load_emotion_emojis(),
    )

