from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content must not be blank")
        return value


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    id: str
    role: Literal["assistant"] = "assistant"
    content: str
    category: str
    actions: list[str]
    safetyWarning: bool
    createdAt: str


class EmotionOption(BaseModel):
    emotion: str
    reason: str
    advice: str
    services: str


class EmotionGroup(BaseModel):
    group: str
    emoji: str = ""
    options: list[EmotionOption]
