"""Unit tests for spatial gazetteer, Haversine distance, and geo-proximity query parsing."""

import unittest
from src.geo import (
    GeoConstraint,
    GeoQueryParser,
    SpatialGazetteer,
    filter_corpus_by_proximity,
    haversine_distance,
    parse_geo_query,
)


class TestGeoModule(unittest.TestCase):

    def setUp(self):
        self.gazetteer = SpatialGazetteer()
        self.parser = GeoQueryParser(self.gazetteer)

    def test_haversine_distance(self):
        """Verify Haversine distance between Hilo and Kona (~59 miles)."""
        hilo_lat, hilo_lon = 19.7121, -155.0817
        kona_lat, kona_lon = 19.6533, -155.9945
        dist = haversine_distance(hilo_lat, hilo_lon, kona_lat, kona_lon, unit="miles")
        self.assertAlmostEqual(dist, 59.5, delta=2.0)

        # Zero distance check
        self.assertAlmostEqual(haversine_distance(hilo_lat, hilo_lon, hilo_lat, hilo_lon), 0.0)

    def test_gazetteer_lookup(self):
        """Verify landmark and island coordinate lookups."""
        hilo = self.gazetteer.lookup("hilo")
        self.assertIsNotNone(hilo)
        self.assertEqual(hilo["island"], "Big Island (Hawaii)")
        self.assertAlmostEqual(hilo["lat"], 19.7121, delta=0.01)

        lahaina = self.gazetteer.lookup("lahaina")
        self.assertIsNotNone(lahaina)
        self.assertEqual(lahaina["island"], "Maui")

        maui = self.gazetteer.lookup("maui")
        self.assertIsNotNone(maui)
        self.assertTrue(maui["is_island"])

    def test_island_discrepancy_check(self):
        """Verify discrepancy detection when an anchor town is placed on the wrong island."""
        note = self.gazetteer.check_island_discrepancy("Big Island (Hawaii)", "maui")
        self.assertIsNotNone(note)
        self.assertIn("Big Island", note)
        self.assertIn("Maui", note)

        # Consistent island should return None
        no_note = self.gazetteer.check_island_discrepancy("Maui", "maui")
        self.assertIsNone(no_note)

    def test_parse_geo_query_user_example(self):
        """Verify user's exact query: 'shaved ice cream within 10 miles from hilo in maui'."""
        query = "shaved ice cream within 10 miles from hilo in maui"
        geo = self.parser.parse(query)

        self.assertTrue(geo.has_geo)
        self.assertEqual(geo.clean_query, "shaved ice cream")
        self.assertEqual(geo.anchor_name, "Hilo")
        self.assertEqual(geo.radius_miles, 10.0)
        self.assertAlmostEqual(geo.latitude, 19.7121, delta=0.01)
        self.assertAlmostEqual(geo.longitude, -155.0817, delta=0.01)
        self.assertGreater(len(geo.notes), 0)
        self.assertIn("Big Island", geo.notes[0])
        self.assertIn("Maui", geo.notes[0])

    def test_parse_geo_query_various_formats(self):
        """Verify various geo proximity query formulations."""
        # Radius with 'km'
        g1 = self.parser.parse("poke bowls within 15 km of honolulu")
        self.assertTrue(g1.has_geo)
        self.assertEqual(g1.anchor_name, "Honolulu")
        self.assertAlmostEqual(g1.radius_miles, 15 * 0.621371, delta=0.1)
        self.assertEqual(g1.clean_query, "poke bowls")

        # Anchor with 'near' and default radius
        g2 = self.parser.parse("coffee shops near waikiki")
        self.assertTrue(g2.has_geo)
        self.assertEqual(g2.anchor_name, "Waikiki")
        self.assertGreater(g2.radius_miles, 0.0)

        # Pure semantic query with no geographic constraints
        g3 = self.parser.parse("gluten free breakfast pastries")
        self.assertFalse(g3.has_geo)
        self.assertEqual(g3.clean_query, "gluten free breakfast pastries")

    def test_filter_corpus_by_proximity(self):
        """Verify spatial filtering on mock corpus."""
        mock_corpus = [
            {"name": "Hilo Spot", "latitude": 19.7121, "longitude": -155.0817},  # 0.0 miles
            {"name": "Nearby Hilo", "latitude": 19.7200, "longitude": -155.0900},  # ~0.7 miles
            {"name": "Kona Spot", "latitude": 19.6533, "longitude": -155.9945},  # ~59.5 miles
        ]
        matches = filter_corpus_by_proximity(
            corpus=mock_corpus,
            anchor_lat=19.7121,
            anchor_lon=-155.0817,
            max_radius_miles=10.0,
        )
        matched_names = [mock_corpus[idx]["name"] for idx, dist in matches]
        self.assertIn("Hilo Spot", matched_names)
        self.assertIn("Nearby Hilo", matched_names)
        self.assertNotIn("Kona Spot", matched_names)


if __name__ == "__main__":
    unittest.main()
