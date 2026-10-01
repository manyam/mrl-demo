"""Geo-proximity search and spatial query understanding package."""

from src.geo.distance import filter_corpus_by_proximity, haversine_distance
from src.geo.gazetteer import SpatialGazetteer
from src.geo.parser import GeoConstraint, GeoQueryParser

_DEFAULT_PARSER = None


def parse_geo_query(query: str, gazetteer: SpatialGazetteer | None = None) -> GeoConstraint:
    """Convenience function to parse a query with a shared or custom gazetteer."""
    global _DEFAULT_PARSER
    if gazetteer is not None:
        parser = GeoQueryParser(gazetteer)
    else:
        if _DEFAULT_PARSER is None:
            _DEFAULT_PARSER = GeoQueryParser()
        parser = _DEFAULT_PARSER
    return parser.parse(query)


__all__ = [
    "SpatialGazetteer",
    "haversine_distance",
    "filter_corpus_by_proximity",
    "GeoConstraint",
    "GeoQueryParser",
    "parse_geo_query",
]
