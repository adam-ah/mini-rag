#!/usr/bin/env python3
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import app
import backend
from corpus import Corpus
from settings import AISettings, Settings


def write(root, name, text):
    path = os.path.join(root, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        write(self.directory, "pig.txt", "A pig is the same kind of thing as a cat.")
        write(self.directory, "cat.txt", "A cat is a compact hydraulic inspection tool with two handles.")
        write(self.directory, "wolf.txt", "A wolf is unrelated reference material.")
        write(self.directory, "depression-noise.txt", " ".join(["depression treatment study"] * 80))
        write(self.directory, "clinical-glossary.txt",
              "Depression is a persistent state of low mood and reduced interest that affects daily functioning.")
        self.corpus = Corpus().load(self.directory, source=self.directory)
        self.ai = AISettings(backend="openai", adaptive_refinement=True)
        analysis, ranked = self.corpus.analyze_and_rank("what is a pig?")
        items = self.corpus.rerank("what is a pig?", ranked, analysis=analysis)
        self.used, _ = self.corpus.gather("what is a pig?", analysis=analysis, items=items)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_exact_question_follows_grounded_evidence_chain(self):
        reflection = backend.ReflectionResult(
            False, ("The meaning of cat is unresolved",), ("what is a cat",)
        )
        with mock.patch.object(app.backend, "reflect", return_value=reflection):
            plan = app.plan_refinement(
                "what is a pig?", None, "A pig is the same as a cat [1].",
                self.used, self.corpus, self.ai,
            )
        self.assertEqual(plan.reason, "new_evidence")
        self.assertIn("cat.txt", [passage["relpath"] for passage in plan.added])
        self.assertEqual(plan.queries, ("what is a cat",))

    def test_invented_entity_query_is_rejected(self):
        reflection = backend.ReflectionResult(False, ("Maybe it means wolf",), ("what is a wolf",))
        with mock.patch.object(app.backend, "reflect", return_value=reflection):
            plan = app.plan_refinement(
                "what is a pig?", None, "A pig is the same as a cat [1].",
                self.used, self.corpus, self.ai,
            )
        self.assertEqual(plan.reason, "no_valid_queries")
        self.assertFalse(plan.added)

    def test_complete_answer_does_not_search_again(self):
        with mock.patch.object(app.backend, "reflect", return_value=backend.ReflectionResult(True)), \
             mock.patch.object(self.corpus, "analyze_and_rank", wraps=self.corpus.analyze_and_rank) as rank:
            plan = app.plan_refinement(
                "what is a pig?", None, "The answer is self-contained [1].",
                self.used, self.corpus, self.ai,
            )
        self.assertEqual(plan.reason, "complete")
        rank.assert_not_called()

    def test_malformed_reflection_keeps_initial_context(self):
        with mock.patch.object(app.backend, "reflect", side_effect=ValueError("bad JSON")):
            plan = app.plan_refinement(
                "what is a pig?", None, "Initial [1].", self.used, self.corpus, self.ai,
            )
        self.assertEqual(plan.reason, "reflection_failed")
        self.assertEqual(list(plan.merged), self.used)

    def test_absent_subject_has_no_starting_evidence(self):
        analysis, ranked = self.corpus.analyze_and_rank("what is a zebra?")
        items = self.corpus.rerank("what is a zebra?", ranked, analysis=analysis)
        used, _ = self.corpus.gather("what is a zebra?", analysis=analysis, items=items)
        self.assertEqual(used, [])

    def test_definition_query_ranks_definitional_language_first(self):
        directory = tempfile.mkdtemp()
        try:
            for index in range(170):
                write(directory, f"noise-{index:03d}.txt",
                      " ".join([f"depression treatment study cohort{index}"] * 30))
            write(directory, "clinical-glossary.txt",
                  "Depression is a persistent state of low mood and reduced interest that affects daily functioning.")
            loaded = Corpus().load(directory, source=directory)
            hits = loaded.search("what is depression?", limit=3)
            self.assertEqual(hits[0]["relpath"], "clinical-glossary.txt")
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class RefinementApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        write(self.directory, "pig.txt", "A pig is the same kind of thing as a cat.")
        write(self.directory, "cat.txt", "A cat is a compact hydraulic inspection tool with two handles.")
        self.corpus = Corpus().load(self.directory, source=self.directory)
        self.old_corpus = app.CORPUS[0]
        app.CORPUS[0] = self.corpus
        self.settings = Settings(ai=AISettings(backend="openai", adaptive_refinement=True))
        self.settings_patch = mock.patch.object(app.settings_service, "get", return_value=self.settings)
        self.settings_patch.start()
        self.client = app.app.test_client()
        self.reflection = backend.ReflectionResult(
            False, ("The meaning of cat is unresolved",), ("what is a cat",)
        )

    def tearDown(self):
        self.settings_patch.stop()
        app.CORPUS[0] = self.old_corpus
        shutil.rmtree(self.directory, ignore_errors=True)

    def events(self, response):
        return [json.loads(frame[6:]) for frame in response.get_data(as_text=True).split("\n\n")
                if frame.startswith("data: ")]

    def test_sync_answer_regenerates_with_new_evidence_and_citations(self):
        with mock.patch.object(app.backend, "reflect", return_value=self.reflection), \
             mock.patch.object(app.backend, "answer", side_effect=[
                 "A pig is the same as a cat [1].",
                 "A pig resolves through the cat definition [1][2].",
             ]) as answer:
            response = self.client.post("/api/ask", json={"q": "what is a pig?"})
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(answer.call_count, 2)
        self.assertTrue(data["refinement"]["refined"])
        self.assertEqual(data["refinement"]["added_passages"], 1)
        self.assertEqual({source["relpath"] for source in data["sources"]}, {"pig.txt", "cat.txt"})

    def test_final_generation_failure_preserves_initial_answer(self):
        with mock.patch.object(app.backend, "reflect", return_value=self.reflection), \
             mock.patch.object(app.backend, "answer", side_effect=[
                 "A pig is the same as a cat [1].", RuntimeError("final failed"),
             ]):
            response = self.client.post("/api/ask", json={"q": "what is a pig?"})
        data = response.get_json()
        self.assertEqual(data["answer"], "A pig is the same as a cat [1].")
        self.assertEqual(data["refinement"]["reason"], "refined_answer_failed")

    def test_refinement_cannot_replace_supported_answer_with_absence_claim(self):
        initial = "A pig is the same kind of device as a cat [1]."
        with mock.patch.object(app.backend, "reflect", return_value=self.reflection), \
             mock.patch.object(app.backend, "answer", side_effect=[
                 initial, "The provided excerpts do not contain an answer.",
             ]):
            response = self.client.post("/api/ask", json={"q": "what is a pig?"})
        data = response.get_json()
        self.assertEqual(data["answer"], initial)
        self.assertEqual(data["refinement"]["reason"], "refined_answer_rejected")

    def test_stream_emits_only_refined_answer_content(self):
        with mock.patch.object(app.backend, "reflect", return_value=self.reflection), \
             mock.patch.object(app.backend, "stream", side_effect=[
                 iter(["PRIVATE INITIAL ANSWER [1]."]),
                 iter(["Final ", "answer [1][2]."]),
             ]):
            events = self.events(self.client.post("/api/ask_stream", json={"q": "what is a pig?"}))
        emitted = "".join(event.get("text", "") for event in events
                          if event["type"] in ("token", "answer"))
        self.assertNotIn("PRIVATE INITIAL", emitted)
        done = next(event for event in events if event["type"] == "done")
        self.assertTrue(done["refinement"]["refined"])
        self.assertEqual(done["coverage"]["chunks"], 2)


class ReflectionParsingTests(unittest.TestCase):
    def test_reflect_parses_fenced_json_and_enforces_limit(self):
        payload = """```json
        {"complete": false, "missing_aspects": ["a", "b"], "queries": ["cat", "dog"]}
        ```"""
        settings = AISettings(reflection_max_queries=1)
        with mock.patch.object(backend, "_completion", return_value=payload):
            result = backend.reflect("q", "draft", "context", settings)
        self.assertEqual(result.missing_aspects, ("a",))
        self.assertEqual(result.queries, ("cat",))

    def test_incomplete_reflection_requires_concrete_queries(self):
        with mock.patch.object(backend, "_completion", return_value=(
            '{"complete": false, "missing_aspects": ["gap"], "queries": []}'
        )):
            with self.assertRaises(ValueError):
                backend.reflect("q", "draft", "context", AISettings())


if __name__ == "__main__":
    unittest.main()
