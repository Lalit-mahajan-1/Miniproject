# app/api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict

class ChatTurn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatTurn] = Field(default_factory=list)
    k: int = 4

class ChatResponse(BaseModel):
    reply: str
    history: List[ChatTurn]