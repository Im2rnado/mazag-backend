"""
Sentiment and Emotion Analysis Module for Mazag
Analyzes user text for sentiment, emotions, and psychological indicators.
"""

from typing import Dict, List, Optional, Any
import re
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Results of text analysis"""
    sentiment: str  # positive, negative, neutral
    emotions: Dict[str, float]  # emotion: confidence
    tone: str  # anxious, calm, angry, sad, etc.
    confidence: float
    indicators: List[str]  # detected psychological indicators
    risk_level: str  # low, medium, high


class TextAnalyzer:
    """
    Analyzes text for sentiment, emotion, and psychological indicators.
    Supports multiple analysis backends.
    """
    
    def __init__(self, method: str = "transformers"):
        """
        Args:
            method: Analysis method ('transformers', 'text2emotion', 'lexicon', 'gemini')
        """
        self.method = method
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """Initialize the selected analyzer"""
        if self.method == "transformers":
            self._init_transformers()
        elif self.method == "text2emotion":
            self._init_text2emotion()
        elif self.method == "lexicon":
            self._init_lexicon()
        elif self.method == "gemini":
            # Gemini-based will be initialized per-call
            pass
        else:
            raise ValueError(f"Unknown analysis method: {self.method}")
    
    def _init_transformers(self):
        """Initialize transformers-based analyzer"""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers"
            )
        
        # Sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Emotion analysis pipeline
        try:
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
        except:
            # Fallback if emotion model not available
            self.emotion_analyzer = None
    
    def _init_text2emotion(self):
        """Initialize text2emotion analyzer"""
        try:
            import text2emotion as te
        except ImportError:
            raise ImportError(
                "text2emotion not installed. "
                "Install with: pip install text2emotion"
            )
        self.te = te
    
    def _init_lexicon(self):
        """Initialize lexicon-based analyzer"""
        # Define emotion lexicons (keywords for each emotion)
        self.emotion_lexicons = {
            "anxiety": [
                "worried", "anxious", "nervous", "fear", "scared", "panic",
                "stress", "tense", "uneasy", "concerned", "dread",
                "قلق", "خائف", "متوتر", "خوف", "توتر"
            ],
            "sadness": [
                "sad", "depressed", "down", "unhappy", "miserable", "hopeless",
                "lonely", "empty", "worthless", "grief", "despair",
                "حزين", "مكتئب", "يائس", "وحيد"
            ],
            "anger": [
                "angry", "mad", "furious", "irritated", "frustrated", "rage",
                "annoyed", "hostile", "resentful", "bitter",
                "غاضب", "منزعج", "محبط", "غضب"
            ],
            "joy": [
                "happy", "joyful", "excited", "glad", "cheerful", "delighted",
                "pleased", "content", "satisfied", "grateful",
                "سعيد", "فرح", "مبسوط", "ممتن"
            ],
            "fear": [
                "afraid", "terrified", "frightened", "horror", "panic",
                "phobia", "terror", "dread",
                "خائف", "رعب", "فزع"
            ]
        }
        
        # Psychological indicators
        self.indicator_patterns = {
            "cognitive_distortion": [
                r"\b(always|never|everyone|no one|everything|nothing)\b",
                r"\bshould\b.{0,20}\bhave\b",
                r"\bwhat if\b",
            ],
            "negative_self_talk": [
                r"\bI('m| am)\s+(worthless|useless|failure|stupid|idiot)\b",
                r"\bI\s+can't\b",
                r"\bI\s+hate\s+myself\b",
            ],
            "low_motivation": [
                r"\bno\s+(energy|motivation|point)\b",
                r"\bdon't\s+(care|want to|feel like)\b",
                r"\btired\s+of\s+everything\b",
            ],
            "sleep_issues": [
                r"\bcan't\s+sleep\b",
                r"\b(insomnia|sleepless)\b",
                r"\bsleep\s+too\s+much\b",
            ],
            "social_withdrawal": [
                r"\b(avoid|avoiding)\s+(people|everyone|friends)\b",
                r"\bdon't\s+want\s+to\s+(see|talk to)\s+anyone\b",
                r"\bisolat(e|ing)\b",
            ]
        }
    
    def analyze(self, text: str) -> AnalysisResult:
        """
        Analyze text for sentiment, emotions, and indicators.
        
        Args:
            text: Text to analyze
            
        Returns:
            AnalysisResult object
        """
        if self.method == "transformers":
            return self._analyze_transformers(text)
        elif self.method == "text2emotion":
            return self._analyze_text2emotion(text)
        elif self.method == "lexicon":
            return self._analyze_lexicon(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _analyze_transformers(self, text: str) -> AnalysisResult:
        """Analyze using transformers models"""
        # Get sentiment
        sent_result = self.sentiment_analyzer(text[:512])[0]  # Limit length
        sentiment = "positive" if sent_result["label"] == "POSITIVE" else "negative"
        
        # Get emotions
        emotions = {}
        if self.emotion_analyzer:
            emotion_results = self.emotion_analyzer(text[:512])[0]
            for result in emotion_results:
                emotions[result["label"].lower()] = result["score"]
            
            # Determine dominant emotion as tone
            tone = max(emotions.items(), key=lambda x: x[1])[0]
        else:
            emotions = {"unknown": 1.0}
            tone = "neutral"
        
        # Extract indicators using lexicon patterns
        indicators = self._extract_indicators_lexicon(text)
        
        # Determine risk level
        risk_level = self._assess_risk_level(text, emotions, indicators)
        
        return AnalysisResult(
            sentiment=sentiment,
            emotions=emotions,
            tone=tone,
            confidence=sent_result["score"],
            indicators=indicators,
            risk_level=risk_level
        )
    
    def _analyze_text2emotion(self, text: str) -> AnalysisResult:
        """Analyze using text2emotion library"""
        emotions_dict = self.te.get_emotion(text)
        
        # Determine sentiment from emotions
        positive_emotions = emotions_dict.get("Happy", 0) + emotions_dict.get("Surprise", 0)
        negative_emotions = emotions_dict.get("Sad", 0) + emotions_dict.get("Angry", 0) + emotions_dict.get("Fear", 0)
        
        if positive_emotions > negative_emotions:
            sentiment = "positive"
        elif negative_emotions > positive_emotions:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Convert to lowercase keys
        emotions = {k.lower(): v for k, v in emotions_dict.items()}
        
        # Determine tone (dominant emotion)
        if emotions:
            tone = max(emotions.items(), key=lambda x: x[1])[0]
        else:
            tone = "neutral"
        
        # Extract indicators
        indicators = self._extract_indicators_lexicon(text)
        
        # Assess risk
        risk_level = self._assess_risk_level(text, emotions, indicators)
        
        # Calculate confidence (average of emotion scores)
        confidence = sum(emotions.values()) / len(emotions) if emotions else 0.5
        
        return AnalysisResult(
            sentiment=sentiment,
            emotions=emotions,
            tone=tone,
            confidence=confidence,
            indicators=indicators,
            risk_level=risk_level
        )
    
    def _analyze_lexicon(self, text: str) -> AnalysisResult:
        """Analyze using keyword lexicons"""
        text_lower = text.lower()
        
        # Count emotion keywords
        emotion_scores = {}
        for emotion, keywords in self.emotion_lexicons.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = count
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotion_scores.items()}
        else:
            emotions = {"neutral": 1.0}
        
        # Determine sentiment
        positive_score = emotions.get("joy", 0)
        negative_score = emotions.get("sadness", 0) + emotions.get("anger", 0) + emotions.get("fear", 0) + emotions.get("anxiety", 0)
        
        if positive_score > negative_score:
            sentiment = "positive"
        elif negative_score > positive_score:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Tone is dominant emotion
        tone = max(emotions.items(), key=lambda x: x[1])[0]
        
        # Extract indicators
        indicators = self._extract_indicators_lexicon(text)
        
        # Assess risk
        risk_level = self._assess_risk_level(text, emotions, indicators)
        
        # Confidence based on keyword density
        confidence = min(total / 10, 1.0)
        
        return AnalysisResult(
            sentiment=sentiment,
            emotions=emotions,
            tone=tone,
            confidence=confidence,
            indicators=indicators,
            risk_level=risk_level
        )
    
    def _extract_indicators_lexicon(self, text: str) -> List[str]:
        """Extract psychological indicators using pattern matching"""
        if not hasattr(self, 'indicator_patterns'):
            return []
        
        indicators = []
        for indicator, patterns in self.indicator_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    indicators.append(indicator)
                    break
        
        return indicators
    
    def _assess_risk_level(
        self,
        text: str,
        emotions: Dict[str, float],
        indicators: List[str]
    ) -> str:
        """
        Assess mental health risk level based on analysis.
        
        Returns:
            'low', 'medium', or 'high'
        """
        risk_score = 0
        
        # Check for crisis keywords
        crisis_keywords = [
            "suicide", "kill myself", "end my life", "want to die",
            "self harm", "cut myself", "hurt myself",
            "انتحار", "اقتل نفسي"
        ]
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in crisis_keywords):
            return "high"
        
        # Check emotion intensity
        negative_emotions = ["sadness", "anger", "fear", "anxiety", "sad", "angry", "fear"]
        for emotion in negative_emotions:
            if emotion in emotions and emotions[emotion] > 0.6:
                risk_score += 1
        
        # Check indicators
        high_risk_indicators = ["negative_self_talk", "low_motivation", "social_withdrawal"]
        for indicator in indicators:
            if indicator in high_risk_indicators:
                risk_score += 1
        
        # Determine level
        if risk_score >= 3:
            return "high"
        elif risk_score >= 1:
            return "medium"
        else:
            return "low"


class GeminiAnalyzer(TextAnalyzer):
    """
    Uses Gemini API for advanced sentiment and emotion analysis.
    """
    
    def __init__(self, api_key: str):
        """
        Args:
            api_key: Gemini API key
        """
        self.api_key = api_key
        self.method = "gemini"
        
        import openai
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    
    def analyze(self, text: str) -> AnalysisResult:
        """Analyze using Gemini API"""
        
        prompt = f"""Analyze the following text for sentiment, emotions, and psychological indicators.

Text: "{text}"

Provide your analysis in this exact format:

SENTIMENT: [positive/negative/neutral]
EMOTIONS: [list emotions with confidence scores 0-1, e.g., sadness:0.8, anxiety:0.6]
TONE: [primary emotional tone]
CONFIDENCE: [0-1]
INDICATORS: [list psychological indicators like cognitive_distortion, negative_self_talk, etc.]
RISK_LEVEL: [low/medium/high]

Be concise and clinical in your assessment."""
        
        response = self.client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        result_text = response.choices[0].message.content
        
        # Parse the structured response
        return self._parse_gemini_response(result_text)
    
    def _parse_gemini_response(self, response: str) -> AnalysisResult:
        """Parse Gemini's structured response"""
        
        # Default values
        sentiment = "neutral"
        emotions = {}
        tone = "neutral"
        confidence = 0.5
        indicators = []
        risk_level = "low"
        
        # Parse each field
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith("SENTIMENT:"):
                sentiment = line.split(":", 1)[1].strip().lower()
            
            elif line.startswith("EMOTIONS:"):
                emotions_str = line.split(":", 1)[1].strip()
                # Parse emotion:score pairs
                for pair in emotions_str.split(","):
                    if ":" in pair:
                        emotion, score = pair.split(":")
                        try:
                            emotions[emotion.strip()] = float(score.strip())
                        except:
                            pass
            
            elif line.startswith("TONE:"):
                tone = line.split(":", 1)[1].strip().lower()
            
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    pass
            
            elif line.startswith("INDICATORS:"):
                indicators_str = line.split(":", 1)[1].strip()
                indicators = [i.strip() for i in indicators_str.split(",") if i.strip()]
            
            elif line.startswith("RISK_LEVEL:"):
                risk_level = line.split(":", 1)[1].strip().lower()
        
        return AnalysisResult(
            sentiment=sentiment,
            emotions=emotions,
            tone=tone,
            confidence=confidence,
            indicators=indicators,
            risk_level=risk_level
        )


# Factory function
def create_analyzer(method: str = "lexicon", **kwargs) -> TextAnalyzer:
    """
    Create an analyzer instance.
    
    Args:
        method: 'transformers', 'text2emotion', 'lexicon', or 'gemini'
        **kwargs: Additional arguments (e.g., api_key for gemini)
        
    Returns:
        TextAnalyzer instance
    """
    if method == "gemini":
        if "api_key" not in kwargs:
            raise ValueError("api_key required for gemini analyzer")
        return GeminiAnalyzer(api_key=kwargs["api_key"])
    else:
        return TextAnalyzer(method=method)

