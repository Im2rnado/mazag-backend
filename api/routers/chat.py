"""
Chat Router — AI conversation with SSE streaming support.

Endpoints:
  POST /chat/message  → Non-streaming, returns full response at once
  POST /chat/stream   → SSE streaming, tokens sent as they're generated
  GET  /chat/history/{session_id} → Retrieve conversation history
"""

import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.schemas import ChatMessageRequest, ChatMessageResponse, ConversationHistoryResponse, MessageRecord
from api.database import get_db
from api.services import chatbot, knowledge
from api.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(body: ChatMessageRequest):
    """
    Non-streaming chat endpoint.
    Waits for the full LLM response before returning.
    """
    db = get_db()
    settings = get_settings()

    # Load conversation history from MongoDB
    history = await chatbot.load_history(body.session_id, db)

    # Retrieve RAG context
    context = knowledge.retrieve_context(body.message, top_k=settings.rag_top_k)

    # Generate response
    response_text, analysis = await chatbot.generate_response(
        message=body.message,
        history=history,
        context=context,
    )

    # Persist both messages
    await chatbot.save_message(body.session_id, "user", body.message, db)
    await chatbot.save_message(body.session_id, "assistant", response_text, db)

    guardrail_triggered = analysis.get("guardrail") is not None

    return ChatMessageResponse(
        response=response_text,
        session_id=body.session_id,
        analysis=analysis,
        guardrail_triggered=guardrail_triggered,
    )


@router.post("/stream")
async def stream_message(body: ChatMessageRequest):
    """
    SSE streaming chat endpoint.
    Sends tokens as they arrive using Server-Sent Events format.

    Frontend consumes this with fetch() + ReadableStream or EventSource.
    Each SSE event:
      data: <token_text>\\n\\n
    Final event:
      data: [DONE]\\n\\n
    """
    db = get_db()
    settings = get_settings()

    history = await chatbot.load_history(body.session_id, db)
    context = knowledge.retrieve_context(body.message, top_k=settings.rag_top_k)

    # Save user message immediately
    await chatbot.save_message(body.session_id, "user", body.message, db)

    accumulated_response = []

    async def event_generator():
        nonlocal accumulated_response
        try:
            async for token in chatbot.stream_response(
                message=body.message,
                history=history,
                context=context,
            ):
                accumulated_response.append(token)
                # SSE format: "data: <payload>\n\n"
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"

            # Save complete assistant response to MongoDB
            full_response = "".join(accumulated_response)
            await chatbot.save_message(body.session_id, "assistant", full_response, db)

            # Signal stream end
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_payload = json.dumps({"error": str(e)})
            yield f"data: {error_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/history/{session_id}", response_model=ConversationHistoryResponse)
async def get_history(session_id: str):
    """Retrieve conversation history for a session."""
    db = get_db()
    cursor = db["messages"].find(
        {"session_id": session_id},
        {"_id": 0, "session_id": 1, "role": 1, "content": 1, "created_at": 1}
    ).sort("created_at", 1).limit(100)

    messages = await cursor.to_list(length=100)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found or empty")

    return ConversationHistoryResponse(
        session_id=session_id,
        messages=[
            MessageRecord(
                session_id=session_id,
                role=m["role"],
                content=m["content"],
                created_at=m["created_at"],
            )
            for m in messages
        ],
    )
