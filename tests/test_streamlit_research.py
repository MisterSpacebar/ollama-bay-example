"""Streamlit AppTest smoke test for the dataset-free research screen.

This runs the Streamlit script (not a graphical browser session) and mocks
network responses; actual Ollama/browser checks remain separate manual steps.
"""
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class _FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


def fake_get(url, **kwargs):
    if url.endswith("/api/health"):
        return _FakeResponse({"model": "mock-ollama"})
    if url.endswith("/api/local-datasets"):
        return _FakeResponse({"groups": []})
    if url.endswith("/api/research/catalog"):
        return _FakeResponse({
            "enabled": True,
            "available": [{
                "artifact": "gam_scores", "relative_path": "environmental_gam/paired_scores.csv",
                "description": "Paired research model scores",
            }],
            "missing": [],
        })
    raise AssertionError("Unexpected HTTP request: " + url)


class StreamlitResearchScreenTests(unittest.TestCase):
    def test_research_browser_without_uploaded_dataset(self):
        with patch("requests.get", side_effect=fake_get):
            app = AppTest.from_file("streamlit_app.py").run(timeout=30)
        self.assertEqual(len(app.exception), 0, [str(x.message) for x in app.exception])
        labels = [item.label for item in app.selectbox]
        self.assertIn("Research result", labels)
        self.assertNotIn("Chart type", labels)


if __name__ == "__main__":
    unittest.main()
