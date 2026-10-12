
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

load_dotenv()

project = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

client = project.get_openai_client(
    agent_name=os.environ["AGENT_NAME"]
)

app = FastAPI(title="RAG Learning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = None


@app.get("/")
def home():
    return {"message": "RAG Learning API is running"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        # Start a new conversation, or continue an existing one.
        conversation_id = request.conversation_id

        if not conversation_id:
            conversation = client.conversations.create()
            conversation_id = conversation.id

        response = client.responses.create(
            conversation=conversation_id,
            input=request.question,
        )

        return {
            "question": request.question,
            "answer": response.output_text,
            "conversation_id": conversation_id,
        }

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="The AI agent request failed.",
        ) from error
