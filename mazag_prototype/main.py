"""
Main Mazag Integration Module
Combines RAG, AI, and recommendation systems into a unified flow.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json

# Import all modules (using relative imports)
from .rag.chunker import DocumentChunker
from .rag.embedder import create_embedder, BaseEmbedder
from .rag.vector_store import FAISSVectorStore, SimpleVectorStore
from .ai.guardrails import GuardrailsSystem, FilterResult
from .ai.chatbot import MazagChatbot
from .ai.analyzer import create_analyzer, TextAnalyzer, AnalysisResult
from .recommend.recommender import TherapistRecommender, PatientProfile, TherapistProfile


@dataclass
class MazagConfig:
    """Configuration for Mazag system"""
    api_key: str
    
    # RAG settings
    chunking_strategy: str = "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 5
    embedding_model: str = "sentence-transformer"
    vector_store_type: str = "faiss"
    similarity_metric: str = "cosine"
    rag_top_k: int = 1
    
    # AI settings
    gemini_model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 500
    
    # Analysis settings
    analysis_method: str = "lexicon"
    
    # Guardrails
    strict_mode: bool = False
    
    # Recommendation
    recommendation_top_k: int = 5


class MazagEngine:
    """
    Main Mazag engine that orchestrates all components.
    """
    
    def __init__(self, config: MazagConfig):
        """
        Args:
            config: MazagConfig object with settings
        """
        self.config = config
        
        # Initialize components
        self._init_rag()
        self._init_ai()
        self._init_guardrails()
        self._init_analyzer()
        
        # Conversation state
        self.current_analysis: Optional[AnalysisResult] = None
    
    def _init_rag(self):
        """Initialize RAG components"""
        # Chunker
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            strategy=self.config.chunking_strategy
        )
        
        # Embedder
        embedder_kwargs = {}
        if self.config.embedding_model == "sentence-transformer":
            embedder_kwargs["model_name"] = "all-MiniLM-L6-v2"
        elif self.config.embedding_model == "gemini":
            embedder_kwargs["api_key"] = self.config.api_key
        
        self.embedder = create_embedder(
            self.config.embedding_model,
            **embedder_kwargs
        )
        
        # Vector store
        if self.config.vector_store_type == "faiss":
            self.vector_store = FAISSVectorStore(
                embedding_dim=self.embedder.get_embedding_dim(),
                metric=self.config.similarity_metric
            )
        else:
            self.vector_store = SimpleVectorStore(
                metric=self.config.similarity_metric
            )
    
    def _init_ai(self):
        """Initialize AI chatbot"""
        self.chatbot = MazagChatbot(
            api_key=self.config.api_key,
            model=self.config.gemini_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
    
    def _init_guardrails(self):
        """Initialize guardrails system"""
        self.guardrails = GuardrailsSystem(
            strict_mode=self.config.strict_mode
        )
    
    def _init_analyzer(self):
        """Initialize text analyzer"""
        analyzer_kwargs = {}
        if self.config.analysis_method == "gemini":
            analyzer_kwargs["api_key"] = self.config.api_key
        
        self.analyzer = create_analyzer(
            self.config.analysis_method,
            **analyzer_kwargs
        )
    
    def add_knowledge(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add knowledge to the RAG system.
        
        Args:
            text: Document text
            metadata: Optional metadata for the document
        """
        # Chunk the text
        chunks = self.chunker.chunk_text(text, metadata or {})
        
        # Embed chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed(chunk_texts)
        
        # Store in vector store
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        metadata_list = [chunk.metadata for chunk in chunks]
        
        self.vector_store.add_batch(
            chunk_ids=chunk_ids,
            texts=chunk_texts,
            embeddings=embeddings,
            metadata_list=metadata_list
        )
    
    def process_message(
        self,
        user_message: str,
        use_rag: bool = True,
        analyze_sentiment: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user message through the full Mazag pipeline.
        
        Args:
            user_message: User's input message
            use_rag: Whether to use RAG for context retrieval
            analyze_sentiment: Whether to analyze sentiment
            
        Returns:
            Dict with response and metadata
        """
        result = {
            "user_message": user_message,
            "response": None,
            "guardrail_check": None,
            "analysis": None,
            "retrieved_context": None,
            "error": None
        }
        
        try:
            # Step 1: Input guardrails
            guardrail_result = self.guardrails.check_input(user_message)
            result["guardrail_check"] = guardrail_result
            
            if not guardrail_result["allowed"]:
                # Handle blocked input
                if guardrail_result["critical_issue"]:
                    result["response"] = self.guardrails.get_crisis_response()
                else:
                    # Find block reason
                    for check in guardrail_result["results"]:
                        if check["result"] == "block":
                            if "off-topic" in check["reason"].lower():
                                result["response"] = self.guardrails.get_off_topic_response()
                            else:
                                result["response"] = check["reason"]
                            break
                return result
            
            # Step 2: Sentiment/emotion analysis
            if analyze_sentiment:
                analysis = self.analyzer.analyze(user_message)
                result["analysis"] = {
                    "sentiment": analysis.sentiment,
                    "emotions": analysis.emotions,
                    "tone": analysis.tone,
                    "confidence": analysis.confidence,
                    "indicators": analysis.indicators,
                    "risk_level": analysis.risk_level
                }
                self.current_analysis = analysis
                
                # If high risk, add to context
                if analysis.risk_level == "high":
                    guardrail_result["should_warn"] = True
            
            # Step 3: RAG retrieval
            context = None
            if use_rag and self.vector_store.size() > 0:
                # Embed query
                query_embedding = self.embedder.embed(user_message)
                
                # Retrieve relevant chunks
                retrieved = self.vector_store.search(
                    query_embedding,
                    k=self.config.rag_top_k
                )
                
                if retrieved:
                    # Combine retrieved texts
                    context = "\n\n".join([r["text"] for r in retrieved])
                    result["retrieved_context"] = retrieved
            
            # Step 4: Generate response
            response = self.chatbot.generate_response(
                user_message,
                context=context
            )
            
            # Step 5: Output guardrails
            output_check = self.guardrails.check_output(response)
            
            if not output_check["allowed"]:
                # Regenerate or use fallback
                response = (
                    "I hear you. Can you tell me more about what you're experiencing?"
                )
            
            result["response"] = response
            
        except Exception as e:
            result["error"] = str(e)
            result["response"] = (
                "I apologize, I'm having trouble processing that right now. "
                "Could you rephrase what you're feeling?"
            )
        
        return result
    
    def get_therapist_recommendations(
        self,
        patient_concerns: List[str],
        therapist_profiles: List[Dict[str, Any]],
        preferred_language: Optional[str] = None,
        preferred_approach: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get therapist recommendations based on patient needs.
        
        Args:
            patient_concerns: List of concerns (anxiety, depression, etc.)
            therapist_profiles: List of therapist profile dicts
            preferred_language: Preferred language
            preferred_approach: Preferred therapy approaches
            
        Returns:
            List of recommendations
        """
        # Convert therapist dicts to TherapistProfile objects
        therapists = []
        for prof in therapist_profiles:
            # Embed therapist description
            desc = prof.get("description", "")
            if desc:
                embedding = self.embedder.embed(desc)
            else:
                embedding = None
            
            therapists.append(TherapistProfile(
                therapist_id=prof.get("id", prof.get("name", "unknown")),
                name=prof.get("name", "Unknown"),
                specialties=prof.get("specialties", []),
                languages=prof.get("languages", ["English"]),
                approach=prof.get("approach", []),
                description=desc,
                embedding=embedding,
                metadata=prof
            ))
        
        # Create patient profile
        patient_desc = ", ".join(patient_concerns)
        patient_embedding = self.embedder.embed(patient_desc)
        
        patient = PatientProfile(
            patient_id="current_user",
            concerns=patient_concerns,
            description=patient_desc,
            preferred_language=preferred_language,
            preferred_approach=preferred_approach,
            embedding=patient_embedding,
            sentiment_vector=None
        )
        
        # Get recommendations
        recommender = TherapistRecommender(therapists)
        recommendations = recommender.recommend(
            patient,
            top_k=self.config.recommendation_top_k
        )
        
        # Convert to dicts
        return [
            {
                "therapist": rec.therapist.metadata,
                "score": rec.score,
                "similarity_breakdown": rec.similarity_breakdown,
                "match_reasons": rec.match_reasons
            }
            for rec in recommendations
        ]
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.chatbot.reset_conversation()
        self.current_analysis = None
    
    def save_vector_store(self, path: str):
        """Save RAG vector store"""
        self.vector_store.save(path)
    
    def load_vector_store(self, path: str):
        """Load RAG vector store"""
        if self.config.vector_store_type == "faiss":
            self.vector_store = FAISSVectorStore.load(path)
        else:
            raise NotImplementedError("SimpleVectorStore doesn't support loading")


# Quick utility functions
def create_mazag_engine(api_key: str, **kwargs) -> MazagEngine:
    """
    Quick factory to create Mazag engine.
    
    Args:
        api_key: Gemini API key
        **kwargs: Additional config options
        
    Returns:
        MazagEngine instance
    """
    config = MazagConfig(api_key=api_key, **kwargs)
    return MazagEngine(config)


def chat_with_mazag_full(
    api_key: str,
    message: str,
    knowledge_base: Optional[List[str]] = None
) -> str:
    """
    Complete Mazag interaction with single message.
    
    Args:
        api_key: Gemini API key
        message: User message
        knowledge_base: Optional documents for RAG
        
    Returns:
        Response text
    """
    engine = create_mazag_engine(api_key)
    
    # Add knowledge if provided
    if knowledge_base:
        for doc in knowledge_base:
            engine.add_knowledge(doc)
    
    # Process message
    result = engine.process_message(message)
    
    return result["response"]

