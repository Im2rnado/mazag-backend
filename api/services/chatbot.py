"""
Mazag Chatbot Service — OpenRouter LLM integration.

Handles:
  - Standard (non-streaming) generation
  - SSE token-by-token streaming (ChatGPT-style)
  - Input/output guardrails
  - Sentiment/risk analysis
  - Conversation history stored in MongoDB
  - Tool calling for therapist recommendations
"""

import re
import json
import logging
from typing import Optional, AsyncIterator, List, Dict, Any, Union
from datetime import datetime, timezone

from openai import AsyncOpenAI
from transformers import pipeline
from api.config import get_settings
from api.services import therapists

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

THERAPIST_TOOL = {
    "type": "function",
    "function": {
        "name": "find_therapists",
        "description": "Search for therapists based on user criteria like location, max price, or specialization. Call this tool WHENEVER the user asks for a therapist recommendation or asks to find a therapist.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or neighborhood (e.g., 'Dokki', 'Maadi')."
                },
                "max_price": {
                    "type": "integer",
                    "description": "Maximum session price in EGP."
                },
                "keyword": {
                    "type": "string",
                    "description": "Keyword related to specialization or issue (e.g., 'addiction', 'CBT', 'family')."
                }
            }
        }
    }
}


def _check_crisis(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in CRISIS_KEYWORDS)


def _check_off_topic(text: str) -> bool:
    return any(re.search(p, text) for p in OFF_TOPIC_PATTERNS)


def _check_output(text: str) -> bool:
    """Returns True if output is forbidden/unsafe."""
    return any(re.search(p, text) for p in FORBIDDEN_OUTPUT_PATTERNS)


_emotion_classifier = None

def _get_emotion_classifier():
    global _emotion_classifier
    if _emotion_classifier is None:
        try:
            logger.info("Loading transformer emotion model...")
            # We use j-hartmann's emotion model which predicts 7 emotions
            _emotion_classifier = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
            logger.info("Transformer emotion model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            _emotion_classifier = "failed"
    return _emotion_classifier

def _analyze_sentiment(text: str) -> dict:
    t = text.lower()
    
    # 1. Base keyword scoring (acts as a robust fallback and supports Arabic)
    emotions = {
        "anxiety": ["worried", "anxious", "nervous", "scared", "panic", "stress", "قلق", "خائف", "توتر"],
        "sadness": ["sad", "depressed", "lonely", "hopeless", "empty", "حزين", "يائس", "وحيد"],
        "anger": ["angry", "mad", "furious", "frustrated", "rage", "غاضب", "محبط"],
        "joy": ["happy", "glad", "excited", "grateful", "content", "سعيد", "ممتن"],
    }

    scores: Dict[str, float] = {k: 0.0 for k in emotions.keys()}
    for emotion, keywords in emotions.items():
        count = sum(1 for kw in keywords if kw in t)
        scores[emotion] = round(count / max(len(keywords), 1), 3)

    # 2. Advanced Transformer scoring (English)
    classifier = _get_emotion_classifier()
    if classifier and classifier != "failed":
        try:
            # Safely truncate text to avoid max length issues (512 tokens)
            safe_text = text[:1500] 
            hf_results = classifier(safe_text)[0]
            
            for res in hf_results:
                label = res['label']
                score = res['score']
                
                # Map model labels to our unified schema
                if label == 'joy':
                    scores['joy'] += score
                elif label == 'sadness':
                    scores['sadness'] += score
                elif label == 'anger' or label == 'disgust':
                    scores['anger'] += score
                elif label == 'fear':
                    scores['anxiety'] += score
            
            # Normalize to 0-1 range
            max_score = max(scores.values()) if max(scores.values()) > 0 else 1
            scores = {k: round(v / max_score, 3) for k, v in scores.items()}
            
        except Exception as e:
            logger.error(f"Transformer inference error: {e}")

    dominant = max(scores, key=scores.get) if max(scores.values()) > 0 else "neutral"
    
    # Calculate overall sentiment polarity
    positive = scores.get("joy", 0)
    negative = scores.get("sadness", 0) + scores.get("anger", 0) + scores.get("anxiety", 0)

    sentiment = "positive" if positive > negative else "negative" if negative > positive else "neutral"
    risk_level = "high" if _check_crisis(text) else "medium" if negative > 0.5 else "low"

    return {
        "sentiment": sentiment,
        "emotions": scores,
        "dominant_emotion": dominant,
        "risk_level": risk_level,
    }


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


async def generate_response(
    message: str,
    history: List[Dict[str, str]],
    context: Optional[str] = None,
) -> tuple[str, dict, List[Dict[str, Any]]]:
    """
    Generate a non-streaming chat response.
    Returns (response_text, analysis_dict, recommended_therapists_list).
    """
    settings = get_settings()

    if _check_crisis(message):
        return CRISIS_RESPONSE, {"risk_level": "high", "guardrail": "crisis"}, []
    if _check_off_topic(message):
        return OFF_TOPIC_RESPONSE, {"risk_level": "low", "guardrail": "off_topic"}, []

    analysis = _analyze_sentiment(message)

    system_content = MAZAG_SYSTEM_PROMPT
    if context:
        system_content += f"\n\n## Relevant Knowledge\n{context}"

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    client = _get_client()
    
    # 1. First API call (with tools)
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        tools=[THERAPIST_TOOL],
        tool_choice="auto"
    )
    
    response_msg = resp.choices[0].message
    recommended_therapists = []
    
    # 2. Check for tool calls
    if response_msg.tool_calls:
        # Append the assistant's tool call request to messages
        messages.append(response_msg.model_dump())
        
        for tool_call in response_msg.tool_calls:
            if tool_call.function.name == "find_therapists":
                args = json.loads(tool_call.function.arguments)
                found = therapists.search_therapists(
                    location=args.get("location"),
                    max_price=args.get("max_price"),
                    keyword=args.get("keyword")
                )
                recommended_therapists = found
                
                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps([{"id": t["id"], "name": t["name"], "price": t["price"], "location": t["location"]} for t in found])
                })
                
        # 3. Second API call to formulate the final answer based on tool output
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        text = resp.choices[0].message.content or ""
    else:
        text = response_msg.content or ""

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    if _check_output(text):
        text = "أنا هنا للاستماع. هل يمكنك إخباري أكثر عما تشعر به؟\n\nI'm here to listen. Can you tell me more about what you're feeling?"

    return text, analysis, recommended_therapists


async def stream_response(
    message: str,
    history: List[Dict[str, str]],
    context: Optional[str] = None,
) -> AsyncIterator[Union[str, dict]]:
    """
    Yields either text chunks (str) or a dictionary {"therapists": [...]}.
    """
    settings = get_settings()

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
    
    # 1. Initiate stream with tools
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        tools=[THERAPIST_TOOL],
        tool_choice="auto",
        stream=True,
    )

    tool_calls = []
    full_text = ""
    
    # 2. Process first stream (might be text, might be a tool call)
    async for chunk in stream:
        delta = chunk.choices[0].delta
        
        # Accumulate tool calls if present
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                # Initialize new tool call in our tracker if needed
                while len(tool_calls) <= tool_call.index:
                    tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                
                # Append data to the active tool call
                tc = tool_calls[tool_call.index]
                if tool_call.id: tc["id"] += tool_call.id
                if tool_call.function.name: tc["function"]["name"] += tool_call.function.name
                if tool_call.function.arguments: tc["function"]["arguments"] += tool_call.function.arguments
        
        # If it's normal text, yield it
        elif delta.content:
            token = delta.content
            full_text += token
            yield token

    # 3. If a tool call was detected and fully streamed, execute it
    if tool_calls:
        # Reconstruct the assistant's tool_call message for context
        assistant_msg = {
            "role": "assistant", 
            "content": None, 
            "tool_calls": [
                {
                    "id": tc["id"], 
                    "type": "function", 
                    "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                } for tc in tool_calls
            ]
        }
        messages.append(assistant_msg)
        
        recommended_therapists = []
        for tc in tool_calls:
            if tc["function"]["name"] == "find_therapists":
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                    
                found = therapists.search_therapists(
                    location=args.get("location"),
                    max_price=args.get("max_price"),
                    keyword=args.get("keyword")
                )
                recommended_therapists.extend(found)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps([{"id": t["id"], "name": t["name"], "price": t["price"], "location": t["location"]} for t in found])
                })
                
        # Yield the structured data event so the frontend can show the cards
        if recommended_therapists:
            yield {"therapists": recommended_therapists}
            
        # 4. Stream the second LLM response based on tool execution
        second_stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            stream=True,
        )
        
        async for chunk in second_stream:
            token = chunk.choices[0].delta.content
            if token:
                full_text += token
                yield token

    if _check_output(full_text):
        logger.warning("⚠️  Output guardrail triggered on streamed response")


# ── Conversation history helpers (MongoDB) ────────────────────────────────────

async def load_history(session_id: str, db) -> List[Dict[str, str]]:
    cursor = db["messages"].find(
        {"session_id": session_id},
        {"_id": 0, "role": 1, "content": 1}
    ).sort("created_at", -1).limit(20)
    messages = await cursor.to_list(length=20)
    messages.reverse()
    return messages


async def save_message(session_id: str, role: str, content: str, db) -> None:
    await db["messages"].insert_one({
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc),
    })
