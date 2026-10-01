"""Unit tests for review scoring, keyword extraction, and top review selection."""

import unittest
from src.data.reviews import extract_keywords, score_review, select_top_review


class TestReviewModule(unittest.TestCase):

    def test_extract_keywords(self):
        query = "quiet beachfront coffee shop with vegan options in Hawaii"
        keywords = extract_keywords(query)
        self.assertIn("quiet", keywords)
        self.assertIn("beachfront", keywords)
        self.assertIn("coffee", keywords)
        self.assertIn("vegan", keywords)
        self.assertIn("options", keywords)
        self.assertNotIn("with", keywords)
        self.assertNotIn("in", keywords)

    def test_score_review(self):
        # Empty/short review should have negative score
        self.assertEqual(score_review({"text": "Short"}), -1.0)

        # Informative 5-star review matching keywords
        review = {
            "text": "Great quiet beachfront coffee shop! Amazing vegan pastries and excellent espresso right by the ocean.",
            "rating": 5.0,
        }
        score = score_review(review, query_terms={"vegan", "coffee", "beachfront"})
        self.assertGreater(score, 0.7)

    def test_select_top_review(self):
        candidates = [
            {"text": "Too short.", "rating": 5.0},
            {
                "text": "Regular coffee and sandwiches, nothing too special here.",
                "rating": 3.0,
            },
            {
                "text": "Delicious vegan muffins and smooth cold brew in a peaceful beachfront cafe setting. Loved the view!",
                "rating": 5.0,
            },
        ]

        top_rev = select_top_review(candidates, query="vegan beachfront coffee")
        self.assertIsNotNone(top_rev)
        self.assertIn("vegan muffins", top_rev["text"])
        self.assertEqual(top_rev["rating"], 5.0)

    def test_select_top_review_fallback(self):
        # When no reviews match query specifically, selects highest base quality
        candidates = [
            {"text": "Great spot for breakfast with friends and family. Friendly staff.", "rating": 5.0}
        ]
        top = select_top_review(candidates, query="surfboard rental")
        self.assertIsNotNone(top)
        self.assertIn("Great spot", top["text"])


if __name__ == "__main__":
    unittest.main()
