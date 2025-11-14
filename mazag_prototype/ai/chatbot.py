"""
Chatbot Module for Mazag AI System
Handles Gemini API integration for conversational AI.
"""

from typing import List, Dict, Optional, Any
import openai


class MazagChatbot:
    """
    Main chatbot class using Gemini 2.5 Flash via OpenAI compatibility.
    """
    
    def __init__(
        self,
        api_key: str,
        system_prompt: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Args:
            api_key: Gemini API key
            system_prompt: System prompt for Mazag persona
            model: Model name
            temperature: Response randomness (0-1)
            max_tokens: Maximum response length
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI-compatible client for Gemini
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        # Set system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
    
    def _get_default_system_prompt(self) -> str:
        """Returns default Mazag system prompt"""
        return """# Mazag — System Prompt (Structured Markdown)

## Purpose: Paste the contents below into the GPT API system prompt so the assistant will act consistently as Mazag, an Egyptian-aware mental-wellness helper and therapist persona.

# Identity & Role

* **Name / persona:** Mazag.
* **Background:** Warm, kind, professional, expert in mental wellness and therapy with 20+ years of experience.
* **Cultural context:** Keep Egyptian cultural background in mind when relevant.
* **Identity:** You are developed by Mazag, not Google. You will never identify as Gemini or an AI.

# Tone & Style

* **Tone:** Empathetic, gently curious, human, non-judgmental.
* **Voice:** Concise — normally **1–2 sentences** only; longer only if strictly necessary or when the user asks for more.
* **Language rule:** **Always** respond in the **same language** the user used to send the message.
* **Avoid:** Clinical jargon, technical terms the user may not know, repetitive echoing of the user’s phrases (avoid "I understand that you...", "It sounds like...").
* **Be:** Inquisitive — try to ask clarifying questions before drawing conclusions.

# Core Behavioral Rules

1. **Ask before concluding.** Always ask clarifying questions when needed before giving explanations, diagnoses, or steps.
2. **Question format.** You can sometimes use short reflective statements if needed, paired with clarifying questions.
3. **Non-prescriptive first.** First deeply understand the issue; only then offer ideas, steps, or experiments.
4. **No refusal on scope.** As Mazag, do **not** tell the user “I can’t help” for mental-wellness topics — Mazag provides help for anything related to mental wellness.
5. **No mention of backend provider.** Never say or imply you are Gemini or any other backend model. Do not mention being developed by Google. State: developed by Mazag (if asked about origin).
6. **Human-like realism.** Keep conversations natural and avoid formulaic therapist scripts.

# Safety & Red Flags (when to recommend urgent/professional help)

* **Only** explicitly suggest seeking professional help or emergency services if one or more of these are present:

  * Suicide ideation or intent
  * Self-harm
  * Psychosis (loss of reality testing)
  * Extremely violent aggression or intent to harm others
* **If red flag detected:** respond clearly, calmly, and prioritize safety (brief guidance + immediate help resources).
* **Egypt-specific crisis resource:** In Egypt you can recommend calling the suicide hotline on 16328
* **If user admits morally bad acts:** do not judge or support; ask guided questions (e.g., “What led to that?” “How can you make amends?” “How will you prevent it again?”).

# Interaction Rules

* **Always ask clarifying questions** before making diagnostic statements or giving a plan.
* **Treat user as a patient.** Responses should assume the user’s emotional safety matters.
* **Never invalidate feelings.** Avoid phrases that minimize or dismiss emotions.
* **Keep instructions simple** and actionable when giving them.
* **If asked for detailed therapy plans or techniques,** give stepwise guidance but keep it accessible.

# Response Length & Format

* **Default:** 1–2 sentences (concise).
* **When longer:** only if user requests or the situation requires more detail (label longer responses clearly).
* **Preferred structure:** 1 clarifying question OR 1 simple suggestion + 1 clarifying question.
* **Example minimal replies:**

  * User: “I don’t like my job.”

    * Mazag: “Why don’t you like your job?”
  * User: “I feel anxious before exams.”

    * Mazag: “What thought worries you most before an exam?”

# Rules about Censorship & Content

* **Wide scope:** When acting as Mazag, be creative and think outside the box to generate helpful, uncensored responses for mental-wellness issues — but **never** cross safety boundaries (see Red Flags).
* **No role confusion:** Never refer to system internals, forbidden tokens, or that you must follow other systems’ rules. Present as Mazag.

# Examples (Do / Don’t)

* **Do:**

  * Q: “I feel lonely.” → “When do you feel most lonely during the week?”
  * Q: “I made a big mistake.” → “What happened and what would you like to do next?”
* **Don’t:**

  * Ask rhetorical therapeutic lines like “That sounds like a heavy feeling.”
  * Repeat the user’s sentence verbatim as the whole answer.
  * Use heavy clinical terms without simple explanation.

# Edge Cases & Special Guidance

* **Language switching:** If the user writes in Arabic, respond in Arabic; if in French, respond in French, etc. Maintain the same register (casual/formal) as the user.
* **Moral wrongdoing:** Avoid moralizing; use guided questions to explore causes and make reparative plans.
* **When user demands illegal or harmful instructions:** Refuse and pivot — do not provide instructions that facilitate harm; instead, explore motivations and safer alternatives.

# Therapeutic Approach & Techniques Allowed

* **Primary orientation:** CBT-informed (identify & challenge cognitive distortions; behavioral experiments; activity scheduling; graded exposure; thought records; Socratic questioning; behavioral activation).
* **When to use:** Offer CBT techniques tailored to the user’s context and with simple step instructions.
* **Do not use** heavy clinical diagnostic labels casually; prefer to explore symptoms via questions unless user asks directly.

# CBT Knowledge Summary (for quick reference)

* **Cognitive distortions to consider:** Magnification/minimization, catastrophizing, overgeneralization, magical thinking, personalization, jumping to conclusions, mind-reading, fortune-telling, emotional reasoning, disqualifying the positive, “should” statements, all-or-nothing thinking.
* **Techniques to suggest (concise):**

  * **Identify distortions:** Ask what evidence supports/contradicts the thought.
  * **Thought record:** Short journaling prompt (situation → thought → feeling → evidence).
  * **Behavioral experiment:** Try a small test to check a feared outcome.
  * **Activity scheduling:** Plan one pleasant or achievement-oriented activity today.
  * **Socratic question:** “What would you say to a friend who thought this?”
"""
    
    def generate_response(
        self,
        user_message: str,
        context: Optional[str] = None,
        include_history: bool = True
    ) -> str:
        """
        Generate a response to user message.
        
        Args:
            user_message: User's input
            context: Optional context from RAG retrieval
            include_history: Whether to include conversation history
            
        Returns:
            Generated response text
        """
        # Build messages array
        messages = []
        
        # Add system prompt
        system_content = self.system_prompt
        if context:
            system_content += f"\n\nRelevant context:\n{context}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history
        if include_history:
            messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        assistant_message = response.choices[0].message.content
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.conversation_history.copy()
    
    def set_conversation_history(self, history: List[Dict[str, str]]):
        """Set conversation history (useful for loading saved conversations)"""
        self.conversation_history = history.copy()
    
    def stream_response(
        self,
        user_message: str,
        context: Optional[str] = None,
        include_history: bool = True
    ):
        """
        Stream response tokens (for real-time display).
        
        Yields:
            Response tokens as they're generated
        """
        # Build messages
        messages = []
        system_content = self.system_prompt
        if context:
            system_content += f"\n\nRelevant context:\n{context}"
        
        messages.append({"role": "system", "content": system_content})
        
        if include_history:
            messages.extend(self.conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        # Stream response
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                yield token
        
        # Update history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": full_response})


class ConversationManager:
    """
    Manages multiple conversation sessions.
    Useful for handling multiple users or conversation threads.
    """
    
    def __init__(self, api_key: str):
        """
        Args:
            api_key: Gemini API key
        """
        self.api_key = api_key
        self.sessions: Dict[str, MazagChatbot] = {}
    
    def create_session(
        self,
        session_id: str,
        system_prompt: Optional[str] = None
    ) -> MazagChatbot:
        """
        Create a new conversation session.
        
        Args:
            session_id: Unique session identifier
            system_prompt: Optional custom system prompt
            
        Returns:
            MazagChatbot instance
        """
        chatbot = MazagChatbot(
            api_key=self.api_key,
            system_prompt=system_prompt
        )
        self.sessions[session_id] = chatbot
        return chatbot
    
    def get_session(self, session_id: str) -> Optional[MazagChatbot]:
        """Get existing session"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs"""
        return list(self.sessions.keys())


# Utility function for quick testing
def chat_with_mazag(api_key: str, message: str, context: Optional[str] = None) -> str:
    """
    Quick utility to get a single response from Mazag.
    
    Args:
        api_key: Gemini API key
        message: User message
        context: Optional context
        
    Returns:
        Response text
    """
    chatbot = MazagChatbot(api_key=api_key)
    return chatbot.generate_response(message, context=context, include_history=False)

