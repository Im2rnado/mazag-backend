"""
Recommendation System for Mazag
Matches users with therapists using multiple similarity techniques.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SimilarityMetric(Enum):
    """Available similarity metrics"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"


@dataclass
class TherapistProfile:
    """Therapist profile with metadata"""
    therapist_id: str
    name: str
    specialties: List[str]
    languages: List[str]
    approach: List[str]  # e.g., CBT, mindfulness, psychodynamic
    description: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PatientProfile:
    """Patient profile/needs"""
    patient_id: str
    concerns: List[str]  # anxiety, depression, relationships, etc.
    description: str
    preferred_language: Optional[str] = None
    preferred_approach: Optional[List[str]] = None
    embedding: Optional[np.ndarray] = None
    sentiment_vector: Optional[np.ndarray] = None


@dataclass
class RecommendationResult:
    """Result of therapist recommendation"""
    therapist: TherapistProfile
    score: float
    similarity_breakdown: Dict[str, float]  # scores from different metrics
    match_reasons: List[str]  # why this therapist was recommended


class SimilarityCalculator:
    """
    Calculates similarity between vectors using various metrics.
    """
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Cosine similarity (range: -1 to 1, normalized to 0 to 1).
        """
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Normalize to 0-1 range
        return (similarity + 1) / 2
    
    @staticmethod
    def euclidean_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Euclidean distance converted to similarity (range: 0 to 1).
        Closer = higher similarity.
        """
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        distance = np.linalg.norm(vec1 - vec2)
        # Convert to similarity (inverse distance)
        return 1 / (1 + distance)
    
    @staticmethod
    def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Dot product similarity (normalized).
        """
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        dot = np.dot(vec1, vec2)
        # Normalize by vector magnitudes
        max_possible = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if max_possible == 0:
            return 0.0
        
        return dot / max_possible
    
    @staticmethod
    def manhattan_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Manhattan distance converted to similarity.
        """
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        distance = np.sum(np.abs(vec1 - vec2))
        # Convert to similarity
        return 1 / (1 + distance)
    
    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """
        Jaccard similarity for sets (e.g., specialties, keywords).
        """
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def calculate(
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: SimilarityMetric
    ) -> float:
        """
        Calculate similarity using specified metric.
        
        Args:
            vec1: First vector
            vec2: Second vector
            metric: Similarity metric to use
            
        Returns:
            Similarity score (0 to 1)
        """
        if metric == SimilarityMetric.COSINE:
            return SimilarityCalculator.cosine_similarity(vec1, vec2)
        elif metric == SimilarityMetric.EUCLIDEAN:
            return SimilarityCalculator.euclidean_similarity(vec1, vec2)
        elif metric == SimilarityMetric.DOT_PRODUCT:
            return SimilarityCalculator.dot_product_similarity(vec1, vec2)
        elif metric == SimilarityMetric.MANHATTAN:
            return SimilarityCalculator.manhattan_similarity(vec1, vec2)
        else:
            raise ValueError(f"Unknown metric: {metric}")


class TherapistRecommender:
    """
    Recommends therapists based on patient needs using ensemble methods.
    """
    
    def __init__(
        self,
        therapists: List[TherapistProfile],
        embedding_weight: float = 0.4,
        specialty_weight: float = 0.3,
        language_weight: float = 0.2,
        approach_weight: float = 0.1
    ):
        """
        Args:
            therapists: List of therapist profiles
            embedding_weight: Weight for embedding similarity
            specialty_weight: Weight for specialty matching
            language_weight: Weight for language matching
            approach_weight: Weight for approach matching
        """
        self.therapists = therapists
        self.embedding_weight = embedding_weight
        self.specialty_weight = specialty_weight
        self.language_weight = language_weight
        self.approach_weight = approach_weight
        
        # Normalize weights
        total = sum([
            embedding_weight, specialty_weight,
            language_weight, approach_weight
        ])
        self.embedding_weight /= total
        self.specialty_weight /= total
        self.language_weight /= total
        self.approach_weight /= total
    
    def recommend(
        self,
        patient: PatientProfile,
        top_k: int = 5,
        metrics: Optional[List[SimilarityMetric]] = None,
        ensemble_method: str = "weighted_average"
    ) -> List[RecommendationResult]:
        """
        Recommend therapists for a patient.
        
        Args:
            patient: Patient profile
            top_k: Number of recommendations to return
            metrics: List of similarity metrics to use (for ensemble)
            ensemble_method: How to combine metrics ('weighted_average', 'max', 'min')
            
        Returns:
            List of recommendations sorted by score
        """
        if metrics is None:
            metrics = [
                SimilarityMetric.COSINE,
                SimilarityMetric.EUCLIDEAN,
                SimilarityMetric.DOT_PRODUCT
            ]
        
        recommendations = []
        
        for therapist in self.therapists:
            # Calculate similarity scores
            similarity_scores = self._calculate_similarity(
                patient, therapist, metrics
            )
            
            # Calculate feature-based scores
            feature_scores = self._calculate_feature_scores(patient, therapist)
            
            # Combine scores
            final_score = self._combine_scores(
                similarity_scores,
                feature_scores,
                ensemble_method
            )
            
            # Generate match reasons
            match_reasons = self._generate_match_reasons(
                patient, therapist, feature_scores
            )
            
            recommendations.append(RecommendationResult(
                therapist=therapist,
                score=final_score,
                similarity_breakdown={
                    **similarity_scores,
                    **feature_scores
                },
                match_reasons=match_reasons
            ))
        
        # Sort by score and return top k
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:top_k]
    
    def _calculate_similarity(
        self,
        patient: PatientProfile,
        therapist: TherapistProfile,
        metrics: List[SimilarityMetric]
    ) -> Dict[str, float]:
        """Calculate embedding similarity using multiple metrics"""
        scores = {}
        
        # Only calculate if both have embeddings
        if patient.embedding is not None and therapist.embedding is not None:
            for metric in metrics:
                if metric == SimilarityMetric.JACCARD:
                    continue  # Skip Jaccard for embeddings
                
                score = SimilarityCalculator.calculate(
                    patient.embedding,
                    therapist.embedding,
                    metric
                )
                scores[f"embedding_{metric.value}"] = score
        
        return scores
    
    def _calculate_feature_scores(
        self,
        patient: PatientProfile,
        therapist: TherapistProfile
    ) -> Dict[str, float]:
        """Calculate scores based on explicit features"""
        scores = {}
        
        # Specialty matching (Jaccard similarity)
        if patient.concerns and therapist.specialties:
            patient_concerns = set(c.lower() for c in patient.concerns)
            therapist_specs = set(s.lower() for s in therapist.specialties)
            scores["specialty_match"] = SimilarityCalculator.jaccard_similarity(
                patient_concerns, therapist_specs
            )
        else:
            scores["specialty_match"] = 0.0
        
        # Language matching
        if patient.preferred_language and therapist.languages:
            lang_lower = patient.preferred_language.lower()
            therapist_langs = [l.lower() for l in therapist.languages]
            scores["language_match"] = 1.0 if lang_lower in therapist_langs else 0.0
        else:
            scores["language_match"] = 0.5  # Neutral if not specified
        
        # Approach matching
        if patient.preferred_approach and therapist.approach:
            patient_approach = set(a.lower() for a in patient.preferred_approach)
            therapist_approach_set = set(a.lower() for a in therapist.approach)
            scores["approach_match"] = SimilarityCalculator.jaccard_similarity(
                patient_approach, therapist_approach_set
            )
        else:
            scores["approach_match"] = 0.5  # Neutral if not specified
        
        return scores
    
    def _combine_scores(
        self,
        similarity_scores: Dict[str, float],
        feature_scores: Dict[str, float],
        method: str
    ) -> float:
        """Combine similarity and feature scores using ensemble method"""
        
        # Aggregate embedding similarities
        embedding_scores = [
            v for k, v in similarity_scores.items()
            if k.startswith("embedding_")
        ]
        
        if embedding_scores:
            if method == "weighted_average":
                embedding_score = np.mean(embedding_scores)
            elif method == "max":
                embedding_score = np.max(embedding_scores)
            elif method == "min":
                embedding_score = np.min(embedding_scores)
            else:
                embedding_score = np.mean(embedding_scores)
        else:
            embedding_score = 0.0
        
        # Combine with weighted average
        final_score = (
            self.embedding_weight * embedding_score +
            self.specialty_weight * feature_scores.get("specialty_match", 0.0) +
            self.language_weight * feature_scores.get("language_match", 0.5) +
            self.approach_weight * feature_scores.get("approach_match", 0.5)
        )
        
        return final_score
    
    def _generate_match_reasons(
        self,
        patient: PatientProfile,
        therapist: TherapistProfile,
        feature_scores: Dict[str, float]
    ) -> List[str]:
        """Generate human-readable reasons for the match"""
        reasons = []
        
        # Specialty matches
        if feature_scores.get("specialty_match", 0) > 0.3:
            matching_specs = set(
                c.lower() for c in patient.concerns
            ) & set(
                s.lower() for s in therapist.specialties
            )
            if matching_specs:
                reasons.append(
                    f"Specializes in: {', '.join(matching_specs)}"
                )
        
        # Language match
        if feature_scores.get("language_match", 0) == 1.0:
            reasons.append(
                f"Speaks {patient.preferred_language}"
            )
        
        # Approach match
        if feature_scores.get("approach_match", 0) > 0.3:
            matching_approaches = set(
                a.lower() for a in (patient.preferred_approach or [])
            ) & set(
                a.lower() for a in therapist.approach
            )
            if matching_approaches:
                reasons.append(
                    f"Uses: {', '.join(matching_approaches)}"
                )
        
        # High overall similarity
        if not reasons:
            reasons.append("Good overall match based on your needs")
        
        return reasons
    
    def recommend_batch(
        self,
        patients: List[PatientProfile],
        top_k: int = 5
    ) -> Dict[str, List[RecommendationResult]]:
        """
        Recommend therapists for multiple patients.
        
        Args:
            patients: List of patient profiles
            top_k: Number of recommendations per patient
            
        Returns:
            Dict mapping patient_id to recommendations
        """
        results = {}
        for patient in patients:
            results[patient.patient_id] = self.recommend(patient, top_k=top_k)
        return results


class ContentBasedRecommender:
    """
    Simple content-based recommender using only text descriptions.
    Useful when embeddings aren't available yet.
    """
    
    def __init__(self, therapists: List[TherapistProfile]):
        """
        Args:
            therapists: List of therapist profiles
        """
        self.therapists = therapists
    
    def recommend(
        self,
        patient_concerns: List[str],
        preferred_language: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[TherapistProfile, float, List[str]]]:
        """
        Recommend based on keyword matching.
        
        Args:
            patient_concerns: List of patient concerns/keywords
            preferred_language: Preferred language
            top_k: Number of recommendations
            
        Returns:
            List of (therapist, score, reasons) tuples
        """
        recommendations = []
        patient_concerns_lower = set(c.lower() for c in patient_concerns)
        
        for therapist in self.therapists:
            score = 0.0
            reasons = []
            
            # Match specialties
            therapist_specs = set(s.lower() for s in therapist.specialties)
            matches = patient_concerns_lower & therapist_specs
            
            if matches:
                score += len(matches) * 0.5
                reasons.append(f"Specializes in: {', '.join(matches)}")
            
            # Match in description
            desc_lower = therapist.description.lower()
            desc_matches = [c for c in patient_concerns_lower if c in desc_lower]
            if desc_matches:
                score += len(desc_matches) * 0.3
            
            # Language bonus
            if preferred_language:
                lang_lower = preferred_language.lower()
                therapist_langs = [l.lower() for l in therapist.languages]
                if lang_lower in therapist_langs:
                    score += 1.0
                    reasons.append(f"Speaks {preferred_language}")
            
            if score > 0 or not reasons:
                if not reasons:
                    reasons = ["Available therapist"]
                recommendations.append((therapist, score, reasons))
        
        # Sort and return top k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]


# Utility function
def recommend_therapists(
    patient_description: str,
    therapist_profiles: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Quick utility to get therapist recommendations.
    
    Args:
        patient_description: Text description of patient needs
        therapist_profiles: List of therapist profile dicts
        top_k: Number of recommendations
        
    Returns:
        List of recommendation dicts
    """
    # Convert dicts to TherapistProfile objects
    therapists = []
    for prof in therapist_profiles:
        therapists.append(TherapistProfile(
            therapist_id=prof.get("id", prof.get("name", "unknown")),
            name=prof.get("name", "Unknown"),
            specialties=prof.get("specialties", []),
            languages=prof.get("languages", ["English"]),
            approach=prof.get("approach", []),
            description=prof.get("description", ""),
            metadata=prof
        ))
    
    # Use content-based recommender
    recommender = ContentBasedRecommender(therapists)
    
    # Extract concerns from description (simple keyword extraction)
    common_concerns = [
        "anxiety", "depression", "stress", "relationship",
        "family", "work", "trauma", "addiction", "sleep", "panic"
    ]
    patient_concerns = [
        concern for concern in common_concerns
        if concern in patient_description.lower()
    ]
    
    results = recommender.recommend(patient_concerns, top_k=top_k)
    
    # Convert to dicts
    return [
        {
            "therapist": therapist.metadata,
            "score": score,
            "reasons": reasons
        }
        for therapist, score, reasons in results
    ]

