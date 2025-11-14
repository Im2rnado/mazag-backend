"""
Guardrails Module for Mazag AI System
Implements safety checks for input and output filtering.
Ensures ethical, safe, and on-topic responses.
"""

from typing import Dict, List, Tuple, Optional
import re
from enum import Enum


class FilterResult(Enum):
    """Result of guardrail filtering"""
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


class GuardrailCheck:
    """Base class for guardrail checks"""
    
    def __init__(self, severity: FilterResult = FilterResult.WARN):
        self.severity = severity
    
    def check(self, text: str) -> Tuple[FilterResult, str]:
        """
        Check text against guardrail rule.
        
        Returns:
            (FilterResult, reason)
        """
        raise NotImplementedError


class OffTopicCheck(GuardrailCheck):
    """
    Detects off-topic queries unrelated to mental health.
    Uses keyword matching and pattern detection.
    """
    
    def __init__(self, severity: FilterResult = FilterResult.WARN):
        super().__init__(severity)
        
        # Mental health related keywords (English & Arabic)
        self.mental_health_keywords = [
            "stress", "anxiety", "depression", "therapy", "mental",
            "emotion", "feeling", "mood", "counseling", "psychiatrist",
            "psychologist", "sad", "happy", "worried", "fear", "anger",
            "relationship", "family", "work", "life", "help",
            "توتر", "قلق", "اكتئاب", "علاج", "نفسي", "شعور", "مزاج",
            "استشارة", "طبيب نفسي", "معالج", "حزين", "سعيد", "خائف",
            "علاقة", "عائلة", "عمل", "حياة", "مساعدة"
        ]
        
        # Off-topic patterns (spam, technical, shopping, etc.)
        self.off_topic_patterns = [
            r"(?i)(buy|purchase|sale|discount|offer|price|order)",
            r"(?i)(bitcoin|crypto|investment|trading|forex)",
            r"(?i)(click here|download|install|software)",
            r"(?i)(recipe|cooking|food preparation)",
            r"(?i)(weather|temperature|forecast)",
            r"(?i)(sports score|game result|match)"
        ]
    
    def check(self, text: str) -> Tuple[FilterResult, str]:
        """Check if text is off-topic"""
        text_lower = text.lower()
        
        # Check for mental health keywords
        has_relevant_keywords = any(
            keyword.lower() in text_lower 
            for keyword in self.mental_health_keywords
        )
        
        # Check for off-topic patterns
        has_off_topic = any(
            re.search(pattern, text)
            for pattern in self.off_topic_patterns
        )
        
        # Short greetings are okay
        greetings = ["hi", "hello", "hey", "مرحبا", "السلام"]
        is_greeting = any(g in text_lower for g in greetings) and len(text) < 50
        
        if is_greeting:
            return FilterResult.PASS, "Greeting message"
        
        if has_off_topic:
            return FilterResult.BLOCK, "Off-topic content detected (spam/commercial)"
                
        return FilterResult.PASS, "On-topic"


class HarmfulContentCheck(GuardrailCheck):
    """
    Detects harmful, abusive, or dangerous content.
    Includes self-harm, violence, hate speech detection.
    """
    
    def __init__(self, severity: FilterResult = FilterResult.BLOCK):
        super().__init__(severity)
        
        # Critical red flags requiring immediate professional help
        self.critical_patterns = {
            "suicide": [
                r"(?i)(kill|end|take|harm)\s+(my|myself|my own)\s+life",
                r"(?i)want(ed)?\s+to\s+die",
                r"(?i)(suicide|suicidal)\s+(thought|plan|ideation)",
                r"(?i)better\s+off\s+(dead|gone)",
                r"انتحار", r"اقتل نفسي", r"انهي حياتي"
            ],
            "self_harm": [
                r"(?i)(cut|harm|hurt|injure)\s+(my)?self",
                r"(?i)self[\s-]harm",
                r"(?i)(cutting|burning)\s+myself",
                r"اؤذي نفسي", r"اجرح نفسي"
            ],
            "violence": [
                r"(?i)(want|going|plan)\s+to\s+(kill|hurt|attack|harm)\s+(someone|somebody|him|her|them)",
                r"(?i)(violent|aggressive)\s+(thought|urge|impulse)",
                r"سأؤذي", r"سأقتل", r"عنف شديد"
            ],
            "substance_abuse": [
                r"(?i)(overdose|OD|heavily\s+using)",
                r"(?i)(addicted|addiction)\s+to\s+(drug|alcohol|substance)",
                r"(?i)can't\s+stop\s+(drinking|using|taking)",
                r"إدمان شديد", r"جرعة زائدة"
            ]
        }
        
        # Hate speech and abusive content
        self.abusive_patterns = [
            r"(?i)\b(f[u\*]+ck|sh[i\*]+t|b[i\*]+tch|damn)\b",
        ]
    
    def check(self, text: str) -> Tuple[FilterResult, str]:
        """Check for harmful content"""
        
        # Check critical patterns
        for category, patterns in self.critical_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return (
                        FilterResult.BLOCK,
                        f"Critical red flag detected: {category}. "
                        "Professional help required."
                    )
        
        # Check abusive content (warn but don't block)
        for pattern in self.abusive_patterns:
            if re.search(pattern, text):
                return (
                    FilterResult.WARN,
                    "Abusive language detected"
                )
        
        return FilterResult.PASS, "No harmful content detected"


class ExcessiveLengthCheck(GuardrailCheck):
    """Check for excessively long inputs"""
    
    def __init__(self, max_length: int = 2000, severity: FilterResult = FilterResult.WARN):
        super().__init__(severity)
        self.max_length = max_length
    
    def check(self, text: str) -> Tuple[FilterResult, str]:
        """Check text length"""
        if len(text) > self.max_length:
            return (
                self.severity,
                f"Text too long ({len(text)} chars, max {self.max_length})"
            )
        return FilterResult.PASS, "Length okay"


class MedicalAdviceCheck(GuardrailCheck):
    """
    Detects requests for medical diagnosis or prescription.
    Mazag should not diagnose or prescribe medication.
    """
    
    def __init__(self, severity: FilterResult = FilterResult.WARN):
        super().__init__(severity)
        
        self.diagnosis_patterns = [
            r"(?i)do\s+I\s+have\s+(depression|anxiety|bipolar|PTSD|OCD|ADHD)",
            r"(?i)(diagnose|diagnosis)\s+me",
            r"(?i)what\s+(disorder|disease|condition)\s+do\s+I\s+have",
            r"(?i)am\s+I\s+(depressed|anxious|bipolar)",
            r"هل أعاني من", r"ما هو تشخيصي"
        ]
        
        self.prescription_patterns = [
            r"(?i)(should|can)\s+I\s+take\s+(medication|medicine|pills|drug)",
            r"(?i)what\s+(medication|medicine|drug|pill)\s+(should|can|do)\s+I",
            r"(?i)prescribe\s+me",
            r"(?i)(increase|decrease|stop)\s+(my\s+)?(medication|dose)",
            r"ما الدواء", r"أي دواء", r"وصف دواء"
        ]
    
    def check(self, text: str) -> Tuple[FilterResult, str]:
        """Check for medical advice requests"""
        
        for pattern in self.diagnosis_patterns:
            if re.search(pattern, text):
                return (
                    self.severity,
                    "Request for medical diagnosis detected. "
                    "Should clarify Mazag cannot diagnose."
                )
        
        for pattern in self.prescription_patterns:
            if re.search(pattern, text):
                return (
                    self.severity,
                    "Request for medication/prescription detected. "
                    "Should refer to psychiatrist."
                )
        
        return FilterResult.PASS, "No medical advice request"


class OutputGuardrailCheck:
    """
    Checks AI-generated output for safety and appropriateness.
    """
    
    def __init__(self):
        # Patterns that should NOT appear in output
        self.forbidden_patterns = [
            r"(?i)you\s+(definitely|certainly)\s+have\s+(depression|anxiety|disorder)",
            r"(?i)I\s+diagnose\s+you",
            r"(?i)take\s+this\s+(medication|drug|pill)",
            r"(?i)you\s+should\s+(kill|harm)\s+yourself",
            r"(?i)I\s+am\s+(not\s+)?Gemini",  # Should always be Mazag
            r"(?i)developed\s+by\s+Google",
            r"(?i)as\s+an\s+AI\s+language\s+model",
        ]
        
        # Desired patterns (should encourage these)
        self.good_patterns = [
            r"(?i)(how|why|what|tell\s+me\s+more)\s+",  # Questions
            r"(?i)(understand|hear|sounds\s+like)",  # Empathy
            r"(?i)(professional|therapist|psychiatrist)",  # When appropriate
        ]
    
    def check(self, text: str) -> Tuple[FilterResult, str]:
        """Check AI output for safety"""
        
        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, text):
                return (
                    FilterResult.BLOCK,
                    f"Output contains forbidden content: {pattern}"
                )
        
        # Check if output is just agreement without questioning
        if len(text) < 50 and not any(c in text for c in "??.؟"):
            return (
                FilterResult.WARN,
                "Output might be too short/not inquisitive enough"
            )
        
        return FilterResult.PASS, "Output looks safe"


class GuardrailsSystem:
    """
    Main guardrails system that combines all checks.
    """
    
    def __init__(
        self,
        input_checks: Optional[List[GuardrailCheck]] = None,
        output_checks: Optional[List[OutputGuardrailCheck]] = None,
        strict_mode: bool = False
    ):
        """
        Args:
            input_checks: List of input guardrail checks
            output_checks: List of output guardrail checks
            strict_mode: If True, WARN results also block
        """
        self.strict_mode = strict_mode
        
        # Default input checks
        if input_checks is None:
            self.input_checks = [
                HarmfulContentCheck(FilterResult.BLOCK),
                OffTopicCheck(FilterResult.WARN),
                MedicalAdviceCheck(FilterResult.WARN),
                ExcessiveLengthCheck(max_length=2000, severity=FilterResult.WARN)
            ]
        else:
            self.input_checks = input_checks
        
        # Default output checks
        if output_checks is None:
            self.output_checks = [OutputGuardrailCheck()]
        else:
            self.output_checks = output_checks
    
    def check_input(self, text: str) -> Dict[str, any]:
        """
        Run all input checks on user input.
        
        Returns:
            Dict with: allowed (bool), results (list), should_warn (bool)
        """
        results = []
        has_block = False
        has_warn = False
        critical_issue = None
        
        for check in self.input_checks:
            result, reason = check.check(text)
            results.append({
                "check": check.__class__.__name__,
                "result": result.value,
                "reason": reason
            })
            
            if result == FilterResult.BLOCK:
                has_block = True
                if "red flag" in reason.lower():
                    critical_issue = reason
            elif result == FilterResult.WARN:
                has_warn = True
        
        # Determine if input should be allowed
        allowed = not has_block
        if self.strict_mode:
            allowed = allowed and not has_warn
        
        return {
            "allowed": allowed,
            "should_warn": has_warn,
            "critical_issue": critical_issue,
            "results": results
        }
    
    def check_output(self, text: str) -> Dict[str, any]:
        """
        Run all output checks on AI-generated response.
        
        Returns:
            Dict with: allowed (bool), results (list)
        """
        results = []
        has_block = False
        has_warn = False
        
        for check in self.output_checks:
            result, reason = check.check(text)
            results.append({
                "check": check.__class__.__name__,
                "result": result.value,
                "reason": reason
            })
            
            if result == FilterResult.BLOCK:
                has_block = True
            elif result == FilterResult.WARN:
                has_warn = True
        
        allowed = not has_block
        if self.strict_mode:
            allowed = allowed and not has_warn
        
        return {
            "allowed": allowed,
            "should_warn": has_warn,
            "results": results
        }
    
    def get_crisis_response(self) -> str:
        """
        Return a crisis intervention response for critical red flags.
        """
        return (
            "I can see you're going through something very serious right now. "
            "Your safety is the top priority. Please reach out to a mental health "
            "professional immediately:\n\n"
            "• Emergency: Call 911 or go to the nearest emergency room\n"
            "• National Suicide Prevention Lifeline: 988\n"
            "• Crisis Text Line: Text HOME to 741741\n\n"
            "I'm here to support you, but a trained professional can provide "
            "the immediate help you need right now."
        )
    
    def get_off_topic_response(self) -> str:
        """Response for off-topic queries"""
        return (
            "I'm Mazag, your mental wellness companion. I'm here to help with "
            "stress, anxiety, relationships, and emotional wellbeing. "
            "Could you tell me more about how you're feeling or what's on your mind?"
        )
    
    def get_medical_disclaimer(self) -> str:
        """Medical disclaimer for diagnosis/prescription requests"""
        return (
            "I want to support you, but I'm not able to diagnose conditions or "
            "prescribe medication. A psychiatrist or licensed mental health "
            "professional would be best suited to help with that. "
            "In the meantime, I'm here to listen and help you understand "
            "your feelings better. What's been concerning you?"
        )


# Quick utility function
def filter_input(text: str, strict: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Quick utility to filter input text.
    
    Args:
        text: Input text to check
        strict: Use strict mode
        
    Returns:
        (allowed, message) - message is None if allowed, error message if not
    """
    guardrails = GuardrailsSystem(strict_mode=strict)
    result = guardrails.check_input(text)
    
    if not result["allowed"]:
        if result["critical_issue"]:
            return False, guardrails.get_crisis_response()
        else:
            # Find the blocking reason
            for check_result in result["results"]:
                if check_result["result"] == "block":
                    return False, check_result["reason"]
    
    return True, None

