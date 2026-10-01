"""Unit tests for generative search summarization and YAML configuration using standard unittest."""

import unittest
from src.summary import (
    BaseSearchSummarizer,
    ExtractiveSummarizer,
    FallbackChainSummarizer,
    create_summarizer,
)
from src.summary.prompt import (
    build_chat_messages,
    build_summary_prompt,
    extract_citations,
    format_documents_context,
)
from src.utils.config import ConfigDict, load_config


class TestSummaryModule(unittest.TestCase):

    def setUp(self):
        self.sample_documents = [
            {
                "name": "Island Brew Coffeehouse",
                "address": "377 Keahole St, Honolulu, HI 96825",
                "category": "Coffee shop, Cafe",
                "avg_rating": "4.6",
                "description": "Waterfront cafe with organic coffee and vegan breakfast bites.",
                "score": 0.9235,
            },
            {
                "name": "Java Kai",
                "address": "4-1384 Kuhio Hwy, Kapaʻa, HI 96746",
                "category": "Coffee shop, Breakfast restaurant",
                "avg_rating": "4.7",
                "description": "Artisan roasted beans, vegan pastries, and acai bowls.",
                "score": 0.9231,
            },
            {
                "name": "Surfers Coffee",
                "address": "63 S Kamehameha Hwy, Wahiawa, HI 96786",
                "category": "Coffee shop, Non-profit",
                "avg_rating": "4.8",
                "description": "Community hub with espresso and vegan treats.",
                "score": 0.9102,
            },
        ]

    def test_config_loading(self):
        """Verify loading from configs/config.yaml and default fallbacks."""
        config = load_config("configs/config.yaml")
        self.assertIsInstance(config, ConfigDict)
        self.assertEqual(config.retrieval.model_path, "checkpoints/best_model")
        self.assertEqual(config.retrieval.default_dimension, 128)
        self.assertTrue(config.summary.enabled)
        self.assertEqual(config.summary.local.model_name, "google/gemma-2-2b-it")
        self.assertEqual(config.summary.ollama.model_name, "gemma2:2b")
        self.assertEqual(config.summary.google.model_name, "gemini-2.0-flash")

    def test_prompt_formatting(self):
        """Verify documents context and prompt construction."""
        query = "quiet beachfront coffee shop with vegan options"
        context = format_documents_context(self.sample_documents)
        self.assertIn("[1] Name: Island Brew Coffeehouse", context)
        self.assertIn("377 Keahole St, Honolulu, HI 96825", context)
        self.assertIn("[2] Name: Java Kai", context)
        self.assertIn("[3] Name: Surfers Coffee", context)

        prompt = build_summary_prompt(query, self.sample_documents)
        self.assertIn("--- RETRIEVED BUSINESSES & TOP REVIEWS ---", prompt)
        self.assertIn(query, prompt)

        messages = build_chat_messages(query, self.sample_documents)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_citation_extraction(self):
        """Verify regex extraction of bracketed citations."""
        text = "Visit [1] for ocean views, [2] for pastries, and [1] again later."
        citations = extract_citations(text)
        self.assertEqual(citations, [1, 2])

    def test_extractive_summarizer(self):
        """Verify extractive fallback produces grounded summary with addresses and citations."""
        summarizer = ExtractiveSummarizer()
        self.assertTrue(summarizer.is_available())
        self.assertEqual(summarizer.name, "extractive")

        query = "oceanfront vegan coffee"
        summary = summarizer.generate(query, self.sample_documents)
        self.assertIn("Island Brew Coffeehouse", summary)
        self.assertIn("[1]", summary)
        self.assertIn("377 Keahole St, Honolulu, HI 96825", summary)

        # Verify streaming
        chunks = list(summarizer.stream_generate(query, self.sample_documents))
        streamed_text = "".join(chunks)
        self.assertIn("Island Brew Coffeehouse", streamed_text)

    def test_fallback_chain_routing(self):
        """Verify FallbackChainSummarizer gracefully resolves to ExtractiveSummarizer."""
        chain = FallbackChainSummarizer([ExtractiveSummarizer()])
        self.assertTrue(chain.is_available())
        summary = chain.generate("test query", self.sample_documents)
        self.assertGreater(len(summary), 0)

    def test_create_summarizer_factory(self):
        """Verify create_summarizer factory produces configured backends."""
        config = load_config("configs/config.yaml")
        ext_sum = create_summarizer(config, backend_override="extractive")
        self.assertIsInstance(ext_sum, ExtractiveSummarizer)

        auto_sum = create_summarizer(config, backend_override="auto")
        self.assertIsInstance(auto_sum, FallbackChainSummarizer)


if __name__ == "__main__":
    unittest.main()
