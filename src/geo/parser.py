"""Natural language parser for extracting spatial proximity constraints from queries."""

import re
from dataclasses import dataclass, field
from typing import Any

from src.geo.gazetteer import HAWAII_ISLANDS, HAWAII_LANDMARKS, SpatialGazetteer

DEFAULT_RADIUS_TOWN_MILES = 12.0
DEFAULT_RADIUS_ISLAND_MILES = 35.0


@dataclass
class GeoConstraint:
    """Represents extracted geographic constraints for a search query."""
    has_geo: bool
    raw_query: str
    clean_query: str
    anchor_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    radius_miles: float | None = None
    island: str | None = None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Human-readable description of this geographic constraint."""
        if not self.has_geo:
            return "No geographic constraint."
        parts = []
        if self.radius_miles and self.anchor_name:
            parts.append(f"Within {self.radius_miles:.1f} miles of {self.anchor_name}")
        elif self.anchor_name:
            parts.append(f"Near {self.anchor_name}")
        if self.latitude and self.longitude:
            parts.append(f"[{self.latitude:.4f}° N, {abs(self.longitude):.4f}° W]")
        return " ".join(parts)


class GeoQueryParser:
    """Extracts geographic proximity, radius, and anchor entities from natural language queries."""

    def __init__(self, gazetteer: SpatialGazetteer | None = None):
        self.gazetteer = gazetteer or SpatialGazetteer()

    def parse(self, query: str) -> GeoConstraint:
        """Parse natural language query into clean semantic text and structured GeoConstraint."""
        raw = query.strip()
        q_lower = raw.lower()

        # 1. Parse Radius Distance (e.g. "within 10 miles", "within 5 mi", "under 15 km")
        radius_miles = None
        radius_match = re.search(
            r"\b(?:within|under|less than|up to)\s+(\d+(?:\.\d+)?)\s*(miles?|mi|km)\b",
            q_lower,
        )
        if not radius_match:
            # Pattern like "10 miles from..." or "5 mi radius"
            radius_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(miles?|mi|km)\s*(?:from|away from|radius|of)\b", q_lower)

        matched_radius_str = ""
        if radius_match:
            val = float(radius_match.group(1))
            unit = radius_match.group(2)
            radius_miles = val if "km" not in unit else val * 0.621371
            matched_radius_str = radius_match.group(0)

        # 2. Parse Island Mentions (e.g. "in maui", "on oahu", "in big island")
        island_match = re.search(r"\b(?:in|on)\s+(maui|oahu|kauai|big island|hawaii island|lanai|molokai)\b", q_lower)
        mentioned_island = island_match.group(1) if island_match else None

        # 3. Parse Anchor Location Name
        # Check explicit preposition anchors: "from X", "near X", "around X", "in X"
        anchor_name = None
        anchor_lat = None
        anchor_lon = None
        anchor_island = None
        matched_anchor_str = ""
        notes = []

        # Find candidate anchor by checking landmark names present in query
        all_landmarks = sorted(list(HAWAII_LANDMARKS.keys()) + list(HAWAII_ISLANDS.keys()), key=len, reverse=True)
        detected_landmark = None

        for lm in all_landmarks:
            pattern = rf"\b{re.escape(lm)}\b"
            if re.search(pattern, q_lower):
                detected_landmark = lm
                break

        if detected_landmark:
            loc_info = self.gazetteer.lookup(detected_landmark)
            if loc_info:
                anchor_name = loc_info["name"]
                anchor_lat = loc_info["lat"]
                anchor_lon = loc_info["lon"]
                anchor_island = loc_info["island"]
                matched_anchor_str = detected_landmark

                # Default radius if none specified
                if radius_miles is None:
                    radius_miles = DEFAULT_RADIUS_ISLAND_MILES if loc_info.get("is_island") else DEFAULT_RADIUS_TOWN_MILES

        # Check for island discrepancy (e.g. "Hilo in Maui")
        if anchor_island and mentioned_island:
            discrepancy = self.gazetteer.check_island_discrepancy(anchor_island, mentioned_island)
            if discrepancy:
                notes.append(discrepancy)

        # 4. Construct Clean Semantic Query
        # Remove distance phrases, anchor prepositions, and location names from query
        clean = raw
        if matched_radius_str:
            clean = re.sub(re.escape(matched_radius_str), " ", clean, flags=re.IGNORECASE)

        if matched_anchor_str:
            # Remove preposition + anchor (e.g. "from hilo", "near waikiki", "in maui")
            prep_pattern = rf"\b(?:from|near|around|in|on|at)\s+{re.escape(matched_anchor_str)}\b"
            if re.search(prep_pattern, clean, flags=re.IGNORECASE):
                clean = re.sub(prep_pattern, " ", clean, flags=re.IGNORECASE)
            else:
                clean = re.sub(rf"\b{re.escape(matched_anchor_str)}\b", " ", clean, flags=re.IGNORECASE)

        if mentioned_island and mentioned_island.lower() != (matched_anchor_str or "").lower():
            clean = re.sub(rf"\b(?:in|on)\s+{re.escape(mentioned_island)}\b", " ", clean, flags=re.IGNORECASE)

        # Clean trailing prepositions, excess punctuation, and extra whitespace
        clean = re.sub(r"\b(within|from|near|in|on|around|of)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s+", " ", clean).strip()

        # If clean query became empty (e.g. query was just "Hilo, Hawaii"), restore original query
        if not clean:
            clean = raw

        has_geo = bool(anchor_lat is not None and anchor_lon is not None and radius_miles is not None)

        return GeoConstraint(
            has_geo=has_geo,
            raw_query=raw,
            clean_query=clean,
            anchor_name=anchor_name,
            latitude=anchor_lat,
            longitude=anchor_lon,
            radius_miles=radius_miles,
            island=anchor_island,
            notes=notes,
        )
