"""
Pydantic Schemas for Mazag API.
Defines request bodies and response models for all endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    device_id: str = Field(..., description="Unique device identifier (UUID from client)")
    name: str = Field(..., min_length=1, max_length=100)


class UserResponse(BaseModel):
    user_id: str
    device_id: str
    name: str
    created_at: datetime


# ── Onboarding ────────────────────────────────────────────────────────────────

class OnboardingRequest(BaseModel):
    user_id: str
    primary_concern: List[str] = []
    severity_level: int = Field(default=3, ge=1, le=10)
    therapy_experience: str = ""
    therapy_approach: List[str] = []
    mood_patterns: str = ""
    sleep_quality: int = Field(default=5, ge=1, le=10)
    support_system: str = ""
    wellness_goals: List[str] = []
    preferred_exercises: List[str] = []
    communication_style: str = ""


class OnboardingResponse(BaseModel):
    success: bool
    message: str = "Onboarding saved"


class OnboardingDataResponse(BaseModel):
    user_id: str
    primary_concern: List[str]
    severity_level: int
    therapy_experience: str
    therapy_approach: List[str]
    mood_patterns: str
    sleep_quality: int
    support_system: str
    wellness_goals: List[str]
    preferred_exercises: List[str]
    communication_style: str
    completed_at: datetime


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(..., description="UUID for this conversation session")
    user_id: Optional[str] = None


class ChatMessageResponse(BaseModel):
    response: str
    session_id: str
    analysis: Optional[dict] = None
    guardrail_triggered: bool = False


class MessageRecord(BaseModel):
    session_id: str
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime


class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageRecord]


# ── Therapists ────────────────────────────────────────────────────────────────

class TherapistSchema(BaseModel):
    id: str
    name: str
    title: str
    specialization: str
    gender: str
    price: float
    rating: Optional[float] = None
    review_count: Optional[int] = None
    bio: Optional[str] = None
    languages: List[str] = []
    age_groups: List[str] = []
    years_of_experience: Optional[int] = None
    qualifications: List[str] = []
    approach: Optional[str] = None
    available_slots: List[str] = []


class RecommendedTherapistSchema(TherapistSchema):
    match_score: float = 0.0
    match_reasons: List[str] = []


class BookingRequest(BaseModel):
    therapist_id: str
    user_id: str
    datetime: str  # ISO string


class BookingResponse(BaseModel):
    booking_id: str
    therapist_id: str
    user_id: str
    datetime: str
    status: str = "confirmed"
    created_at: datetime


# ── Exercises ─────────────────────────────────────────────────────────────────

class ExerciseSchema(BaseModel):
    id: str
    title: str
    type: str
    duration_minutes: Optional[int] = None
    description: Optional[str] = None
    difficulty: Optional[str] = None
    benefits: List[str] = []
    steps: Optional[List[str]] = None
