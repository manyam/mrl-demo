"""Hawaii spatial gazetteer, island registry, and corpus centroid lookup."""

import re
from typing import Any

# Major Hawaii Island Centers
HAWAII_ISLANDS = {
    "oahu": {"name": "Oahu", "lat": 21.4389, "lon": -158.0001},
    "maui": {"name": "Maui", "lat": 20.7984, "lon": -156.3319},
    "big island": {"name": "Big Island (Hawaii)", "lat": 19.5429, "lon": -155.6659},
    "hawaii island": {"name": "Big Island (Hawaii)", "lat": 19.5429, "lon": -155.6659},
    "hawaii": {"name": "Big Island (Hawaii)", "lat": 19.5429, "lon": -155.6659},
    "kauai": {"name": "Kauai", "lat": 22.0964, "lon": -159.5261},
    "molokai": {"name": "Molokai", "lat": 21.1444, "lon": -157.0226},
    "lanai": {"name": "Lanai", "lat": 20.8278, "lon": -156.9208},
}

# Major Hawaii Towns & Landmarks
HAWAII_LANDMARKS = {
    # Big Island
    "hilo": {"island": "Big Island (Hawaii)", "lat": 19.7121, "lon": -155.0817},
    "kailua-kona": {"island": "Big Island (Hawaii)", "lat": 19.6533, "lon": -155.9945},
    "kona": {"island": "Big Island (Hawaii)", "lat": 19.6533, "lon": -155.9945},
    "waimea (big island)": {"island": "Big Island (Hawaii)", "lat": 20.0203, "lon": -155.6661},
    "kamuela": {"island": "Big Island (Hawaii)", "lat": 20.0203, "lon": -155.6661},
    "waikoloa": {"island": "Big Island (Hawaii)", "lat": 19.9367, "lon": -155.7906},
    "volcano": {"island": "Big Island (Hawaii)", "lat": 19.4319, "lon": -155.2344},
    "pahoa": {"island": "Big Island (Hawaii)", "lat": 19.4975, "lon": -154.9458},
    "captain cook": {"island": "Big Island (Hawaii)", "lat": 19.4950, "lon": -155.9139},
    "honokaa": {"island": "Big Island (Hawaii)", "lat": 20.0789, "lon": -155.4642},
    "hawi": {"island": "Big Island (Hawaii)", "lat": 20.2392, "lon": -155.8328},
    "naalehu": {"island": "Big Island (Hawaii)", "lat": 19.0633, "lon": -155.5833},

    # Maui
    "kahului": {"island": "Maui", "lat": 20.8852, "lon": -156.4612},
    "lahaina": {"island": "Maui", "lat": 20.9083, "lon": -156.6784},
    "kihei": {"island": "Maui", "lat": 20.7644, "lon": -156.4450},
    "wailuku": {"island": "Maui", "lat": 20.8911, "lon": -156.5050},
    "wailea": {"island": "Maui", "lat": 20.6900, "lon": -156.4420},
    "paia": {"island": "Maui", "lat": 20.9161, "lon": -156.3725},
    "hana": {"island": "Maui", "lat": 20.7575, "lon": -155.9883},
    "makawao": {"island": "Maui", "lat": 20.8572, "lon": -156.3139},
    "kula": {"island": "Maui", "lat": 20.7922, "lon": -156.3283},

    # Oahu
    "honolulu": {"island": "Oahu", "lat": 21.2980, "lon": -157.8442},
    "waikiki": {"island": "Oahu", "lat": 21.2799, "lon": -157.8312},
    "kailua": {"island": "Oahu", "lat": 21.3955, "lon": -157.7408},
    "kaneohe": {"island": "Oahu", "lat": 21.4181, "lon": -157.8036},
    "haleiwa": {"island": "Oahu", "lat": 21.6070, "lon": -158.0895},
    "north shore": {"island": "Oahu", "lat": 21.6400, "lon": -158.0600},
    "wahiawa": {"island": "Oahu", "lat": 21.5028, "lon": -158.0236},
    "pearl city": {"island": "Oahu", "lat": 21.3972, "lon": -157.9750},
    "kapolei": {"island": "Oahu", "lat": 21.3358, "lon": -158.0800},
    "waipahu": {"island": "Oahu", "lat": 21.3867, "lon": -158.0092},
    "aiea": {"island": "Oahu", "lat": 21.3853, "lon": -157.9317},
    "laie": {"island": "Oahu", "lat": 21.6483, "lon": -157.9225},
    "waimanalo": {"island": "Oahu", "lat": 21.3469, "lon": -157.7183},

    # Kauai
    "lihue": {"island": "Kauai", "lat": 21.9811, "lon": -159.3711},
    "kapaa": {"island": "Kauai", "lat": 22.0886, "lon": -159.3375},
    "poipu": {"island": "Kauai", "lat": 21.8744, "lon": -159.4550},
    "koloa": {"island": "Kauai", "lat": 21.9039, "lon": -159.4650},
    "hanalei": {"island": "Kauai", "lat": 22.2033, "lon": -159.5008},
    "princeville": {"island": "Kauai", "lat": 22.2217, "lon": -159.4892},
}


class SpatialGazetteer:
    """Hawaii geographical gazetteer supporting landmark lookup and corpus-driven geocoding."""

    def __init__(self, corpus: list[dict[str, Any]] | None = None):
        self._corpus_centroids: dict[str, tuple[float, float]] = {}
        if corpus:
            self._index_corpus(corpus)

    def _index_corpus(self, corpus: list[dict[str, Any]]) -> None:
        """Calculate dynamic centroid coordinates for every town/city occurring in the corpus."""
        city_points: dict[str, list[tuple[float, float]]] = {}
        for doc in corpus:
            addr = doc.get("address", "")
            lat = doc.get("latitude")
            lon = doc.get("longitude")
            if lat is None or lon is None:
                continue
            m = re.search(r",\s*([^,]+),\s*HI\b", addr, flags=re.IGNORECASE)
            if m:
                city = m.group(1).strip().lower()
                if city not in city_points:
                    city_points[city] = []
                city_points[city].append((float(lat), float(lon)))

        for city, pts in city_points.items():
            avg_lat = sum(p[0] for p in pts) / len(pts)
            avg_lon = sum(p[1] for p in pts) / len(pts)
            self._corpus_centroids[city] = (avg_lat, avg_lon)

    def lookup(self, name: str) -> dict[str, Any] | None:
        """Resolve a city, town, island, or landmark name into coordinates and island metadata."""
        clean = name.strip().lower()

        # 1. Check known landmark/city registry
        if clean in HAWAII_LANDMARKS:
            entry = HAWAII_LANDMARKS[clean]
            return {
                "name": clean.title(),
                "island": entry["island"],
                "lat": entry["lat"],
                "lon": entry["lon"],
                "is_island": False,
            }

        # 2. Check major islands
        if clean in HAWAII_ISLANDS:
            entry = HAWAII_ISLANDS[clean]
            return {
                "name": entry["name"],
                "island": entry["name"],
                "lat": entry["lat"],
                "lon": entry["lon"],
                "is_island": True,
            }

        # 3. Check dynamic corpus centroids
        if clean in self._corpus_centroids:
            lat, lon = self._corpus_centroids[clean]
            # Infer island from latitude
            island_inferred = self.infer_island(lat, lon)
            return {
                "name": clean.title(),
                "island": island_inferred,
                "lat": lat,
                "lon": lon,
                "is_island": False,
            }

        return None

    @staticmethod
    def infer_island(lat: float, lon: float) -> str:
        """Infer Hawaiian island from coordinate bounds."""
        if lat < 20.3:
            return "Big Island (Hawaii)"
        elif 20.5 <= lat <= 21.1 and -156.8 <= lon <= -155.9:
            return "Maui"
        elif 21.2 <= lat <= 21.8 and -158.4 <= lon <= -157.6:
            return "Oahu"
        elif lat >= 21.8 and lon <= -159.2:
            return "Kauai"
        elif 21.0 <= lat <= 21.3 and -157.4 <= lon <= -156.7:
            return "Molokai"
        elif 20.7 <= lat <= 20.95 and -157.1 <= lon <= -156.8:
            return "Lanai"
        return "Hawaii"

    @staticmethod
    def check_island_discrepancy(anchor_island: str, mentioned_island: str) -> str | None:
        """Check if an anchor town is said to be on the wrong island, generating a clarification note."""
        a_norm = anchor_island.lower()
        m_norm = mentioned_island.lower()

        # Check if they reference different islands
        if ("maui" in m_norm and "maui" not in a_norm) or \
           ("oahu" in m_norm and "oahu" not in a_norm) or \
           ("kauai" in m_norm and "kauai" not in a_norm) or \
           ("big island" in m_norm and "big island" not in a_norm and "hawaii" not in a_norm):
            return (
                f"Geographic Clarification: The anchor location is situated on {anchor_island}, "
                f"not {mentioned_island.title()}. Showing matching places within the target radius on {anchor_island}."
            )
        return None
