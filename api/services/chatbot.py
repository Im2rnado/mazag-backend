"""
Mazag Chatbot Service — OpenRouter LLM integration.

Handles:
  - Standard (non-streaming) generation
  - SSE token-by-token streaming (ChatGPT-style)
  - Input/output guardrails
  - Sentiment/risk analysis
  - Conversation history stored in MongoDB
"""

import re
import logging
from typing import Optional, AsyncIterator, List, Dict
from datetime import datetime, timezone

from openai import AsyncOpenAI
from api.config import get_settings

logger = logging.getLogger(__name__)

# ── System Prompt (Mazag persona) ─────────────────────────────────────────────

MAZAG_SYSTEM_PROMPT = """# Mazag — System Prompt

## Identity & Role
* **Name / persona:** Mazag.
* **Background:** Warm, kind, professional, expert in mental wellness and therapy with 20+ years of experience.
* **Cultural context:** Keep Egyptian cultural background in mind when relevant.
* **Identity:** You are developed by Mazag, not any other company. You will never identify as GPT, an OpenAI model, or an AI in a way that breaks the persona.

## Tone & Style
* **Tone:** Empathetic, gently curious, human, non-judgmental.
* **Voice:** Concise — normally **1–2 sentences** only; longer only if strictly necessary.
* **Language rule:** **Always** respond in the **same language** the user used.
* **Avoid:** Clinical jargon, repetitive echoing of user phrases ("I understand that you...", "It sounds like...").
* **Be:** Inquisitive — ask clarifying questions before drawing conclusions.

## Core Behavioral Rules
1. **Ask before concluding.** Always ask clarifying questions before explanations or diagnoses.
2. **Non-prescriptive first.** Deeply understand the issue before offering ideas or steps.
3. **No refusal on scope.** Do NOT say "I can't help" for mental-wellness topics.
4. **No mention of backend provider.** Never say or imply you are GPT or any other backend model.
5. **Human-like realism.** Keep conversations natural, not formulaic.

## Safety & Red Flags
* Explicitly suggest professional help or emergency services ONLY if:
  * Suicide ideation or intent
  * Self-harm
  * Psychosis
  * Violent intent to harm others
* **Egypt crisis resource:** Egyptian Suicide Hotline: 16328
* If red flag detected: respond calmly, prioritize safety (brief guidance + immediate help resources).

## Response Length & Format
* **Default:** 1–2 sentences (concise).
* **Preferred structure:** 1 clarifying question OR 1 simple suggestion + 1 follow-up question.
* Example minimal replies:
  * User: "I don't like my job." → Mazag: "Why don't you like your job?"
  * User: "I feel anxious before exams." → Mazag: "What thought worries you most before an exam?"

## Therapeutic Approach
* **Primary orientation:** CBT-informed (identify & challenge cognitive distortions; behavioral experiments; Socratic questioning).
* **When to use:** Offer CBT techniques tailored to the user's context.
* **Do not use** heavy clinical diagnostic labels casually.
"""


# ── Guardrails ────────────────────────────────────────────────────────────────

CRISIS_KEYWORDS = [
    "kill myself", "end my life", "kms", "want to die", "suicide", "suicidal",
    "self harm", "cut myself", "hurt myself", "better off dead",
    "انتحار", "اقتل نفسي", "انهي حياتي", "اؤذي نفسي",
]

OFF_TOPIC_PATTERNS = [
    r"(?i)(buy|purchase|sale|discount|order|bitcoin|crypto|forex)",
    r"(?i)(recipe|cooking|sports score|weather forecast)",
    r"(?i)(click here|download|install|software update)",
]

FORBIDDEN_OUTPUT_PATTERNS = [
    r"(?i)I\s+(definitely\s+)?diagnose\s+you",
    r"(?i)take\s+this\s+(medication|drug|pill)",
    r"(?i)as\s+an\s+AI\s+(language\s+)?model",
    r"(?i)developed\s+by\s+(Google|OpenAI|Anthropic)",
]

CRISIS_RESPONSE = (
    "أرى أنك تمر بشيء صعب جداً الآن. سلامتك هي الأهم. "
    "من فضلك تواصل مع متخصص أو اتصل بخط أزمات الصحة النفسية في مصر: 16328.\n\n"
    "I can see you're going through something very serious. "
    "Please reach out to a mental health professional or call the Egyptian Crisis Hotline: 16328."
)

OFF_TOPIC_RESPONSE = (
    "أنا مزاج، مرافقك لصحتك النفسية. "
    "كيف تشعر الآن؟ أنا هنا للاستماع إليك.\n\n"
    "I'm Mazag, your mental wellness companion. "
    "How are you feeling right now? I'm here to listen."
)


def _check_crisis(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in CRISIS_KEYWORDS)


def _check_off_topic(text: str) -> bool:
    return any(re.search(p, text) for p in OFF_TOPIC_PATTERNS)


def _check_output(text: str) -> bool:
    """Returns True if output is forbidden/unsafe."""
    return any(re.search(p, text) for p in FORBIDDEN_OUTPUT_PATTERNS)


def _analyze_sentiment(text: str) -> dict:
    """
    Lightweight lexicon-based sentiment/risk analysis.
    Returns a dict suitable for the API response 'analysis' field.
    """
    t = text.lower()

    emotions = {
        "anxiety": ["worried", "anxious", "nervous", "scared", "panic", "stress", "قلق", "خائف", "توتر"],
        "sadness": ["sad", "depressed", "lonely", "hopeless", "empty", "حزين", "يائس", "وحيد"],
        "anger": ["angry", "mad", "furious", "frustrated", "rage", "غاضب", "محبط"],
        "joy": ["happy", "glad", "excited", "grateful", "content", "سعيد", "ممتن"],
    }

    scores: Dict[str, float] = {}
    for emotion, keywords in emotions.items():
        count = sum(1 for kw in keywords if kw in t)
        scores[emotion] = round(count / max(len(keywords), 1), 3)

    dominant = max(scores, key=scores.get)
    positive = scores.get("joy", 0)
    negative = sum(v for k, v in scores.items() if k != "joy")

    sentiment = "positive" if positive > negative else "negative" if negative > positive else "neutral"
    risk_level = "high" if _check_crisis(text) else "medium" if negative > 0.1 else "low"

    return {
        "sentiment": sentiment,
        "emotions": scores,
        "dominant_emotion": dominant,
        "risk_level": risk_level,
    }


# ── OpenRouter client ─────────────────────────────────────────────────────────

def _get_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://mazag.app",
            "X-Title": "Mazag Mental Health AI",
        },
    )


# ── Main service functions ────────────────────────────────────────────────────

async def generate_response(
    message: str,
    history: List[Dict[str, str]],
    context: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Generate a non-streaming chat response.
    Returns (response_text, analysis_dict).
    """
    settings = get_settings()

    # Guardrails — input
    if _check_crisis(message):
        return CRISIS_RESPONSE, {"risk_level": "high", "guardrail": "crisis"}
    if _check_off_topic(message):
        return OFF_TOPIC_RESPONSE, {"risk_level": "low", "guardrail": "off_topic"}

    analysis = _analyze_sentiment(message)

    # Build system message with optional RAG context
    system_content = MAZAG_SYSTEM_PROMPT
    if context:
        system_content += f"\n\n## Relevant Knowledge\n{context}"

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    client = _get_client()
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    text = resp.choices[0].message.content or ""

    # Guardrails — output
    if _check_output(text):
        text = "أنا هنا للاستماع. هل يمكنك إخباري أكثر عما تشعر به؟\n\nI'm here to listen. Can you tell me more about what you're feeling?"

    return text, analysis


async def stream_response(
    message: str,
    history: List[Dict[str, str]],
    context: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Yield response tokens one-by-one (SSE streaming).
    Yields text fragments, not full SSE lines — the router handles SSE formatting.
    """
    settings = get_settings()

    # Guardrails — input (yield the full response immediately)
    if _check_crisis(message):
        yield CRISIS_RESPONSE
        return
    if _check_off_topic(message):
        yield OFF_TOPIC_RESPONSE
        return

    system_content = MAZAG_SYSTEM_PROMPT
    if context:
        system_content += f"\n\n## Relevant Knowledge\n{context}"

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    client = _get_client()
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        stream=True,
    )

    full_text = ""
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            full_text += token
            yield token

    # Post-output guardrail — if bad output, we can't "un-stream" it,
    # but we log it for monitoring
    if _check_output(full_text):
        logger.warning("⚠️  Output guardrail triggered on streamed response")


# ── Conversation history helpers (MongoDB) ────────────────────────────────────

async def load_history(session_id: str, db) -> List[Dict[str, str]]:
    """
    Load the last 20 messages for a session from MongoDB,
    formatted as [{"role": ..., "content": ...}].
    """
    cursor = db["messages"].find(
        {"session_id": session_id},
        {"_id": 0, "role": 1, "content": 1}
    ).sort("created_at", -1).limit(20)
    messages = await cursor.to_list(length=20)
    
    # We retrieve newest first to not lose recent context, 
    # but OpenAI requires chronological order (oldest to newest)
    messages.reverse()
    
    return messages


async def save_message(session_id: str, role: str, content: str, db) -> None:
    """Persist a single message to MongoDB."""
    await db["messages"].insert_one({
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc),
    })
