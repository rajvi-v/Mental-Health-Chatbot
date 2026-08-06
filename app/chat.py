import re

from fastapi import HTTPException, Request
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.config import (
    GREETING_PATTERN,
    GREETING_RESPONSE,
    NO_MATCH_RESPONSE,
    SAFETY_PATTERNS,
    SYSTEM_PROMPT,
)
from app.rag import KnowledgeBaseError
from app.schemas import ChatRequest, HistoryMessage


def has_safety_concern(message: str) -> bool:
    return any(pattern.search(message) for pattern in SAFETY_PATTERNS)


def is_greeting(message: str) -> bool:
    return bool(GREETING_PATTERN.fullmatch(message))


def build_search_query(chat: ChatRequest) -> str:
    previous_user_message = next(
        (
            history_message.content
            for history_message in reversed(chat.history)
            if history_message.role == "user"
        ),
        None,
    )
    if previous_user_message:
        return f"{previous_user_message} {chat.message}"
    return chat.message


def format_recent_history(history: list[HistoryMessage]) -> str:
    recent = history[-6:]
    if not recent:
        return "No previous conversation."
    return "\n".join(
        f"{message.role.title()}: {message.content[:1_000]}"
        for message in recent
    )


def format_context(documents: list[Document]) -> str:
    return "\n\n---\n\n".join(
        document.page_content.strip()
        for document in documents
    )


def limit_answer_sentences(content: str, maximum: int = 4) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    return " ".join(sentences[:maximum]).strip()


async def generate_grounded_answer(
    llm: ChatOpenAI | None,
    chat: ChatRequest,
    documents: list[Document],
) -> str:
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The OpenAI API key is not configured. "
                "Add OPENAI_API_KEY to your .env file."
            ),
        )

    user_prompt = f"""Knowledge-base context:
{format_context(documents)}

Recent conversation:
{format_recent_history(chat.history)}

Current question:
{chat.message}

Answer the current question using only the knowledge-base context."""

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
    except APITimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The answer service took too long to respond. Please try again.",
        ) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail="The answer service is busy or its usage limit has been reached.",
        ) from exc
    except APIConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="The answer service is currently unavailable. Please try again.",
        ) from exc
    except APIStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail="The answer service could not complete the request.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The answer service could not complete the request.",
        ) from exc

    content = response.content
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(
            status_code=502,
            detail="The answer service returned an empty response.",
        )
    return limit_answer_sentences(content)


async def answer_chat(chat_request: ChatRequest, request: Request) -> str:
    if is_greeting(chat_request.message):
        return GREETING_RESPONSE

    services = getattr(request.app.state, "rag_services", None)
    if services is None:
        error = getattr(request.app.state, "rag_error", None)
        detail = "The local knowledge base is unavailable."
        if error:
            detail = f"{detail} {error}"
        raise HTTPException(status_code=503, detail=detail)

    try:
        documents = services.knowledge_base.retrieve(
            build_search_query(chat_request)
        )
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not documents:
        return NO_MATCH_RESPONSE

    return await generate_grounded_answer(
        services.llm,
        chat_request,
        documents,
    )
