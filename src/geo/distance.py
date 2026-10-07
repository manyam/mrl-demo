"""Haversine distance calculation and spatial proximity filtering."""

import math
from typing import Any

import numpy as np

EARTH_RADIUS_MILES = 3958.7613
EARTH_RADIUS_KM = 6371.0088


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    unit: str = "miles",
) -> float:
    """Calculate great-circle distance between two points on the Earth surface using Haversine formula."""
    r = EARTH_RADIUS_MILES if unit == "miles" else EARTH_RADIUS_KM

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    a_clamped = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a_clamped), math.sqrt(1.0 - a_clamped))

    return r * c


def build_corpus_coords_array(corpus: list[dict[str, Any]]) -> np.ndarray:
    """Pre-extract (lat, lon) into a contiguous 2D float32 numpy array for sub-millisecond filtering."""
    coords = np.zeros((len(corpus), 2), dtype=np.float32)
    for i, doc in enumerate(corpus):
        lat = doc.get("latitude")
        lon = doc.get("longitude")
        coords[i, 0] = float(lat) if lat is not None else 0.0
        coords[i, 1] = float(lon) if lon is not None else 0.0
    return coords


def compute_distances_vectorized(
    corpus_coords: np.ndarray,
    anchor_lat: float,
    anchor_lon: float,
) -> np.ndarray:
    """Compute distance in miles from anchor to all points in corpus in ~0.3ms."""
    phi1 = np.radians(anchor_lat)
    phi2 = np.radians(corpus_coords[:, 0])
    dphi = np.radians(corpus_coords[:, 0] - anchor_lat)
    dlam = np.radians(corpus_coords[:, 1] - anchor_lon)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(dlam / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return (EARTH_RADIUS_MILES * c).astype(np.float32)


def filter_corpus_by_proximity(
    corpus: list[dict[str, Any]],
    anchor_lat: float,
    anchor_lon: float,
    max_radius_miles: float,
    corpus_coords: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Filter corpus to businesses within max_radius_miles of anchor coordinates.

    Returns a list of tuples: (corpus_index, distance_in_miles).
    """
    if corpus_coords is None:
        corpus_coords = build_corpus_coords_array(corpus)

    dists = compute_distances_vectorized(corpus_coords, anchor_lat, anchor_lon)
    valid_indices = np.where(dists <= max_radius_miles)[0]

    return [(int(idx), round(float(dists[idx]), 2)) for idx in valid_indices]
