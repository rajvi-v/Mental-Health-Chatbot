from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document

from app.chat import (
    answer_chat,
    build_search_query,
    format_context,
    format_recent_history,
    generate_grounded_answer,
    has_safety_concern,
    is_greeting,
    limit_answer_sentences,
)
from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    EMOJI_SHEET_NAME,
    GREETING_RESPONSE,
    NO_MATCH_RESPONSE,
    OPENAI_TIMEOUT_SECONDS,
    RETRIEVAL_LIMIT,
    STATIC_DIR,
)
from app.emotions import (
    EmotionRecord,
    emoji_code_to_character,
    emotion_records_to_documents,
    group_emotions,
    load_emotion_emojis,
    load_emotion_records,
)
from app.rag import (
    ChromaKnowledgeBase,
    KnowledgeBaseError,
    RagServices,
    build_index_manifest,
    build_or_load_knowledge_base,
    create_embeddings,
    create_llm,
    initialise_services,
    load_and_split_documents,
)
from app.schemas import ChatRequest, ChatResponse, EmotionGroup


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_services = None
    app.state.rag_error = None
    try:
        app.state.rag_services = initialise_services()
    except KnowledgeBaseError as exc:
        app.state.rag_error = str(exc)
    yield


app = FastAPI(title="Student Support Chat", version="3.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/emotions", response_model=list[EmotionGroup])
async def emotions(request: Request) -> list[EmotionGroup]:
    services = getattr(request.app.state, "rag_services", None)
    if services is not None:
        return group_emotions(services.emotions, services.emotion_emojis)

    error = getattr(request.app.state, "rag_error", None)
    detail = "The emotions knowledge base is unavailable."
    if error:
        detail = f"{detail} {error}"
    raise HTTPException(status_code=503, detail=detail)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request) -> ChatResponse:
    content = await answer_chat(chat_request, request)
    return ChatResponse(
        id=str(uuid4()),
        content=content,
        category="Knowledge base",
        actions=[],
        safetyWarning=has_safety_concern(chat_request.message),
        createdAt=datetime.now(timezone.utc).isoformat(),
    )


