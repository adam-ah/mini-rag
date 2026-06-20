import json
import os
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "tests", "query_rescue_corpus")
os.environ["SEARCH_OUTPUT"] = FIXTURE
os.environ["SEARCH_INPUT"] = FIXTURE

from corpus import Corpus
from settings import AISettings, Settings


class QueryRescueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus().load(text_dir=FIXTURE, source=FIXTURE)

    def assert_top(self, question, expected):
        result = self.corpus.search_result(question, limit=5)
        self.assertTrue(result.hits, question)
        self.assertIn(expected, result.hits[0]["relpath"], question)
        return result

    def test_natural_british_query_is_rescued(self):
        result = self.assert_top("what pressure do my tyres need?", "tyre_specs")
        self.assertTrue(result.analysis.rescued)
        self.assertIn(("tyre", "tire"),
                      [(e.source, e.target) for e in result.analysis.expansions])
        self.assertTrue(result.suggestions)

    def test_expected_phrasings_rank_specs_first(self):
        for question in ("recommended tyre pressure", "tire pressure", "tyre psi"):
            with self.subTest(question=question):
                self.assert_top(question, "tyre_specs")

    def test_answer_context_contains_both_pressures(self):
        used, _ = self.corpus.gather("what pressure do my tyres need?")
        context = "\n".join(item["body"].lower() for item in used)
        self.assertIn("36 psi", context)
        self.assertIn("42 psi", context)

    def test_pressure_distractors_remain_correct(self):
        self.assert_top("fuel injector pressure procedure", "fuel_pressure")
        self.assert_top("engine oil pressure limit", "engine_pressure")

    def test_exact_query_is_not_reported_as_rescued(self):
        result = self.assert_top("tire pressure", "tyre_specs")
        self.assertFalse(result.analysis.rescued)
        self.assertEqual(result.analysis.expansions, ())


class QueryRescueApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SEARCH_OUTPUT"] = FIXTURE
        os.environ["SEARCH_INPUT"] = FIXTURE
        import app
        cls.app_module = app

    def setUp(self):
        self.old_corpus = self.app_module.CORPUS[0]
        self.app_module.CORPUS[0] = Corpus().load(text_dir=FIXTURE, source=FIXTURE)
        self.settings_patch = mock.patch.object(
            self.app_module.settings_service, "get",
            return_value=Settings(ai=AISettings(backend="extractive")),
        )
        self.settings_patch.start()
        self.client = self.app_module.app.test_client()

    def tearDown(self):
        self.settings_patch.stop()
        self.app_module.CORPUS[0] = self.old_corpus

    def test_search_api_exposes_rescue_metadata(self):
        response = self.client.get("/api/search", query_string={"q": "what pressure do my tyres need?"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["rescued"])
        self.assertIn({"source": "tyre", "target": "tire", "reason": "spelling"}, data["expansions"])
        self.assertTrue(data["suggestions"])
        self.assertIn("tyre_specs", data["hits"][0]["relpath"])

    def test_stream_exposes_expansion_and_suggestions(self):
        response = self.client.post("/api/ask_stream", json={"q": "what pressure do my tyres need?"})
        self.assertEqual(response.status_code, 200)
        events = []
        for frame in response.get_data(as_text=True).split("\n\n"):
            if frame.startswith("data: "):
                events.append(json.loads(frame[6:]))
        self.assertIn("query_expansion", [event["type"] for event in events])
        self.assertIn("suggestions", [event["type"] for event in events])
        done = next(event for event in events if event["type"] == "done")
        self.assertTrue(done["rescued"])
        self.assertTrue(done["suggestions"])

    def test_sync_answer_uses_same_rescued_source_as_search(self):
        question = "what pressure do my tyres need?"
        search = self.client.get("/api/search", query_string={"q": question}).get_json()
        answer_response = self.client.post("/api/ask", json={"q": question})
        self.assertEqual(answer_response.status_code, 200)
        answer = answer_response.get_json()
        self.assertEqual(answer["sources"][0]["relpath"], search["hits"][0]["relpath"])
        self.assertTrue(answer["rescued"])
        self.assertTrue(answer["suggestions"])

    def test_answer_keeps_only_cited_sources_and_renumbers_from_one(self):
        used = [
            {"relpath": f"document-{i}.txt", "sprint": "", "body": f"passage {i}"}
            for i in range(1, 7)
        ]
        answer, sources = self.app_module.compact_citations(
            "Front is 36 PSI [2]; rear is 42 PSI [2][3][6].", used
        )
        self.assertEqual(answer, "Front is 36 PSI [1]; rear is 42 PSI [1][2][3].")
        self.assertEqual([source["n"] for source in sources], [1, 2, 3])
        self.assertEqual(
            [source["relpath"] for source in sources],
            ["document-2.txt", "document-3.txt", "document-6.txt"],
        )


if __name__ == "__main__":
    unittest.main()
