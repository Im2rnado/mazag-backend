import json
import os
from typing import List, Dict, Any

THERAPISTS_FILE = os.path.join(
    os.path.dirname(__file__), 
    "..", "..", "..", "mazag", "assets", "data", "therapists.json"
)

def load_therapists() -> List[Dict[str, Any]]:
    try:
        with open(THERAPISTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading therapists from {THERAPISTS_FILE}: {e}")
        return []

def search_therapists(location: str = None, max_price: int = None, keyword: str = None, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Search for therapists based on given criteria.
    """
    therapists = load_therapists()
    results = []

    for t in therapists:
        # Filter by max price
        if max_price is not None and t.get("price", float('inf')) > max_price:
            continue
            
        # Filter by location (case insensitive substring)
        if location:
            loc = t.get("location", "").lower()
            if location.lower() not in loc:
                continue
                
        # Filter by keyword in bio, specialization, approach
        if keyword:
            kw = keyword.lower()
            text_to_search = f"{t.get('bio', '')} {t.get('specialization', '')} {t.get('approach', '')}".lower()
            if kw not in text_to_search:
                continue
                
        results.append(t)

    # Sort by rating descending
    results.sort(key=lambda x: x.get("rating", 0), reverse=True)
    return results[:limit]
