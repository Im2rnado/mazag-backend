"""
Onboarding Router — save and retrieve user onboarding profiles.

The frontend calls POST /onboarding after the user completes the
onboarding flow. The profile is used by the therapist recommender.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from api.schemas import OnboardingRequest, OnboardingResponse, OnboardingDataResponse
from api.database import get_db

router = APIRouter(prefix="/onboarding", tags=["Onboarding"])


@router.post("", response_model=OnboardingResponse)
async def save_onboarding(body: OnboardingRequest):
    """Save (or update) a user's onboarding profile."""
    db = get_db()
    now = datetime.now(timezone.utc)

    doc = {
        "user_id": body.user_id,
        "primary_concern": body.primary_concern,
        "severity_level": body.severity_level,
        "therapy_experience": body.therapy_experience,
        "therapy_approach": body.therapy_approach,
        "mood_patterns": body.mood_patterns,
        "sleep_quality": body.sleep_quality,
        "support_system": body.support_system,
        "wellness_goals": body.wellness_goals,
        "preferred_exercises": body.preferred_exercises,
        "communication_style": body.communication_style,
        "completed_at": now,
    }

    # Upsert by user_id
    await db["onboarding"].update_one(
        {"user_id": body.user_id},
        {"$set": doc},
        upsert=True,
    )
    return OnboardingResponse(success=True)


@router.get("/{user_id}", response_model=OnboardingDataResponse)
async def get_onboarding(user_id: str):
    """Retrieve a user's onboarding profile."""
    db = get_db()
    doc = await db["onboarding"].find_one({"user_id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Onboarding profile not found")

    return OnboardingDataResponse(
        user_id=doc["user_id"],
        primary_concern=doc.get("primary_concern", []),
        severity_level=doc.get("severity_level", 5),
        therapy_experience=doc.get("therapy_experience", ""),
        therapy_approach=doc.get("therapy_approach", []),
        mood_patterns=doc.get("mood_patterns", ""),
        sleep_quality=doc.get("sleep_quality", 5),
        support_system=doc.get("support_system", ""),
        wellness_goals=doc.get("wellness_goals", []),
        preferred_exercises=doc.get("preferred_exercises", []),
        communication_style=doc.get("communication_style", ""),
        completed_at=doc["completed_at"],
    )
