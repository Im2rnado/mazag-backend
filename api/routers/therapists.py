"""
Therapists Router.

Endpoints:
  GET /therapists             → All therapists with optional filters
  GET /therapists/recommended → AI-ranked therapists for a user's profile
  POST /bookings              → Book a session slot
  GET /bookings/{user_id}     → Get user's bookings
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException
from api.schemas import TherapistSchema, RecommendedTherapistSchema, BookingRequest, BookingResponse
from api.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Therapists"])

# ── Load therapists.json once at module import ────────────────────────────────

_THERAPISTS_PATH = Path(__file__).parent.parent.parent.parent / "mazag" / "assets" / "data" / "therapists.json"

def _load_therapists() -> List[dict]:
    try:
        with open(_THERAPISTS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Normalise camelCase keys → snake_case for internal use
        normalized = []
        for t in raw:
            normalized.append({
                "id": t.get("id", ""),
                "name": t.get("name", ""),
                "title": t.get("title", ""),
                "specialization": t.get("specialization", ""),
                "gender": t.get("gender", ""),
                "price": t.get("price", 0),
                "rating": t.get("rating"),
                "review_count": t.get("reviewCount"),
                "bio": t.get("bio", ""),
                "languages": t.get("languages", []),
                "age_groups": t.get("ageGroups", []),
                "years_of_experience": t.get("yearsOfExperience"),
                "qualifications": t.get("qualifications", []),
                "approach": t.get("approach", ""),
                "available_slots": t.get("availableSlots", []),
                # Also keep original camelCase fields for recommender text search
                "reviewCount": t.get("reviewCount"),
                "ageGroups": t.get("ageGroups", []),
                "yearsOfExperience": t.get("yearsOfExperience"),
                "availableSlots": t.get("availableSlots", []),
            })
        return normalized
    except FileNotFoundError:
        logger.warning(f"therapists.json not found at {_THERAPISTS_PATH}. Returning empty list.")
        return []

_THERAPISTS: List[dict] = _load_therapists()


def _to_schema(t: dict) -> TherapistSchema:
    return TherapistSchema(
        id=t["id"],
        name=t["name"],
        title=t.get("title", ""),
        specialization=t.get("specialization", ""),
        gender=t.get("gender", ""),
        price=t.get("price", 0),
        rating=t.get("rating"),
        review_count=t.get("review_count"),
        bio=t.get("bio"),
        languages=t.get("languages", []),
        age_groups=t.get("age_groups", []),
        years_of_experience=t.get("years_of_experience"),
        qualifications=t.get("qualifications", []),
        approach=t.get("approach"),
        available_slots=t.get("available_slots", []),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/therapists", response_model=List[TherapistSchema])
async def get_therapists(
    gender: Optional[str] = Query(None, description="Filter by gender (Male/Female)"),
    language: Optional[str] = Query(None, description="Filter by language (Arabic/English/French)"),
    specialization: Optional[str] = Query(None, description="Keyword in specialization"),
    search: Optional[str] = Query(None, description="Search in name, bio, or specialization"),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
):
    """Return all therapists with optional filtering."""
    results = _THERAPISTS[:]

    if gender:
        results = [t for t in results if t.get("gender", "").lower() == gender.lower()]
    if language:
        results = [t for t in results if language.lower() in [l.lower() for l in t.get("languages", [])]]
    if specialization:
        results = [t for t in results if specialization.lower() in t.get("specialization", "").lower()]
    if search:
        q = search.lower()
        results = [
            t for t in results
            if q in t.get("name", "").lower()
            or q in t.get("bio", "").lower()
            or q in t.get("specialization", "").lower()
        ]
    if min_price is not None:
        results = [t for t in results if t.get("price", 0) >= min_price]
    if max_price is not None:
        results = [t for t in results if t.get("price", 0) <= max_price]

    return [_to_schema(t) for t in results]


@router.get("/therapists/recommended", response_model=List[RecommendedTherapistSchema])
async def get_recommended_therapists(
    user_id: str = Query(..., description="User ID to fetch onboarding profile"),
):
    """
    Return therapists ranked by relevance to the user's onboarding profile.
    Falls back to unranked list if no onboarding found.
    """
    db = get_db()
    onboarding = await db["onboarding"].find_one({"user_id": user_id})

    if not onboarding:
        # No onboarding — return all therapists with default score
        return [
            RecommendedTherapistSchema(**_to_schema(t).model_dump(), match_score=0.5, match_reasons=["Available therapist"])
            for t in _THERAPISTS
        ]

    from api.services.recommender import recommend_therapists

    ranked = recommend_therapists(
        therapists=_THERAPISTS,
        concerns=onboarding.get("primary_concern", []),
        preferred_language=None,  # Don't filter by language, just score
        preferred_approaches=onboarding.get("therapy_approach", []),
        top_k=len(_THERAPISTS),
    )

    return [
        RecommendedTherapistSchema(
            **_to_schema(t).model_dump(),
            match_score=t.get("match_score", 0.0),
            match_reasons=t.get("match_reasons", []),
        )
        for t in ranked
    ]


@router.get("/therapists/{therapist_id}", response_model=TherapistSchema)
async def get_therapist(therapist_id: str):
    """Return a single therapist by ID."""
    t = next((t for t in _THERAPISTS if t["id"] == therapist_id), None)
    if not t:
        raise HTTPException(status_code=404, detail="Therapist not found")
    return _to_schema(t)


@router.post("/bookings", response_model=BookingResponse)
async def book_session(body: BookingRequest):
    """Book a session with a therapist."""
    db = get_db()
    booking_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Verify therapist exists
    therapist = next((t for t in _THERAPISTS if t["id"] == body.therapist_id), None)
    if not therapist:
        raise HTTPException(status_code=404, detail="Therapist not found")

    doc = {
        "booking_id": booking_id,
        "therapist_id": body.therapist_id,
        "user_id": body.user_id,
        "datetime": body.datetime,
        "status": "confirmed",
        "created_at": now,
    }
    await db["bookings"].insert_one(doc)

    return BookingResponse(
        booking_id=booking_id,
        therapist_id=body.therapist_id,
        user_id=body.user_id,
        datetime=body.datetime,
        status="confirmed",
        created_at=now,
    )


@router.get("/bookings/{user_id}", response_model=List[BookingResponse])
async def get_bookings(user_id: str):
    """Retrieve all bookings for a user."""
    db = get_db()
    cursor = db["bookings"].find(
        {"user_id": user_id},
        {"_id": 0}
    ).sort("created_at", -1)
    docs = await cursor.to_list(length=50)
    return [
        BookingResponse(
            booking_id=d["booking_id"],
            therapist_id=d["therapist_id"],
            user_id=d["user_id"],
            datetime=d["datetime"],
            status=d.get("status", "confirmed"),
            created_at=d["created_at"],
        )
        for d in docs
    ]
