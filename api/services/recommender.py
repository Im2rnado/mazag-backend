"""
Therapist Recommendation Service.

Uses multi-metric scoring to rank therapists by relevance to a user's
onboarding profile (concerns, language, severity, therapy approach).
Adapted from the Mazag prototype recommender.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _jaccard(set1: set, set2: set) -> float:
    """Jaccard similarity for two sets of strings."""
    if not set1 and not set2:
        return 0.5
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity (assumes L2-normalised inputs → just dot product)."""
    v1 = v1.flatten()
    v2 = v2.flatten()
    dot = float(np.dot(v1, v2))
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def _normalize_therapist_text(therapist: Dict[str, Any]) -> str:
    """Build a combined text blob for embedding a therapist."""
    parts = [
        therapist.get("specialization", ""),
        therapist.get("bio", ""),
        therapist.get("approach", ""),
        " ".join(therapist.get("qualifications", [])),
    ]
    return " ".join(p for p in parts if p)


def recommend_therapists(
    therapists: List[Dict[str, Any]],
    concerns: List[str],
    preferred_language: Optional[str] = None,
    preferred_approaches: Optional[List[str]] = None,
    top_k: int = 12,
    use_embeddings: bool = True,
) -> List[Dict[str, Any]]:
    """
    Score and rank therapists for a patient profile.

    Scoring breakdown (weights):
      40% — embedding cosine similarity (patient concerns text vs therapist description)
      30% — specialization keyword match (Jaccard)
      20% — language match
      10% — therapy approach match (Jaccard)

    Returns list of therapist dicts augmented with `match_score` and `match_reasons`.
    """
    if not therapists:
        return []

    patient_concern_text = ", ".join(concerns) if concerns else "general mental health support"
    patient_concerns_set = {c.lower() for c in concerns}
    patient_lang = preferred_language.lower() if preferred_language else None
    patient_approaches_set = {a.lower() for a in (preferred_approaches or [])}

    # Build embeddings if possible
    concern_embedding: Optional[np.ndarray] = None
    therapist_embeddings: Dict[str, np.ndarray] = {}

    if use_embeddings:
        try:
            from api.services.embedder import get_embedder
            embedder = get_embedder()
            concern_embedding = embedder.embed_one(patient_concern_text)

            for th in therapists:
                th_text = _normalize_therapist_text(th)
                if th_text:
                    therapist_embeddings[th["id"]] = embedder.embed_one(th_text)
        except Exception as e:
            logger.warning(f"Embedding failed for recommendation, using keyword-only: {e}")
            use_embeddings = False

    results = []
    for th in therapists:
        scores: Dict[str, float] = {}
        reasons: List[str] = []

        # 1. Embedding similarity
        if use_embeddings and concern_embedding is not None and th["id"] in therapist_embeddings:
            emb_score = _cosine(concern_embedding, therapist_embeddings[th["id"]])
            # cosine of L2-normed = dot product; remap from [-1,1] to [0,1]
            emb_score = (emb_score + 1) / 2
            scores["embedding"] = emb_score
        else:
            scores["embedding"] = 0.5  # neutral fallback

        # 2. Specialization keyword match
        specialization = th.get("specialization", "").lower()
        qualifications = [q.lower() for q in th.get("qualifications", [])]
        bio = th.get("bio", "").lower()

        spec_keywords: set = set()
        spec_keywords.add(specialization)
        for qual in qualifications:
            spec_keywords.update(qual.split())

        specialty_score = _jaccard(patient_concerns_set, spec_keywords)
        # Also boost if any concern word appears in bio
        bio_matches = [c for c in patient_concerns_set if c in bio]
        if bio_matches:
            specialty_score = min(specialty_score + 0.2 * len(bio_matches), 1.0)
        scores["specialty"] = specialty_score

        if specialty_score > 0.1:
            reasons.append(f"Specializes in areas related to: {', '.join(patient_concerns_set)}")

        # 3. Language match
        th_langs = {lang.lower() for lang in th.get("languages", [])}
        if patient_lang:
            lang_score = 1.0 if patient_lang in th_langs else 0.0
            if lang_score == 1.0:
                reasons.append(f"Speaks {preferred_language}")
        else:
            lang_score = 0.5  # neutral
        scores["language"] = lang_score

        # 4. Approach / therapy style match
        th_approach_text = th.get("approach", "").lower()
        approach_keywords: set = set()
        for word in th_approach_text.split():
            if len(word) > 4:
                approach_keywords.add(word)

        if patient_approaches_set:
            approach_score = _jaccard(patient_approaches_set, approach_keywords)
        else:
            # Check common approaches in bio/approach text
            common = ["cbt", "mindfulness", "psychodynamic", "emdr", "dbt", "motivational"]
            found = [a for a in common if a in th_approach_text]
            approach_score = min(len(found) * 0.15, 1.0) if found else 0.3
            if found:
                reasons.append(f"Uses {', '.join(found).upper()} techniques")
        scores["approach"] = approach_score

        # Weighted final score
        final_score = (
            0.40 * scores["embedding"]
            + 0.30 * scores["specialty"]
            + 0.20 * scores["language"]
            + 0.10 * scores["approach"]
        )

        if not reasons:
            reasons.append("Good overall match for your wellness needs")

        result = {
            **th,
            "match_score": round(final_score, 4),
            "match_reasons": reasons,
        }
        results.append(result)

    # Sort descending by score
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results[:top_k]
