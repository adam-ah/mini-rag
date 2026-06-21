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
        write(self.directory, "therapy-glossary.txt",
              "Dialectical Behavior Therapy (DBT) is a structured treatment framework.")
        write(self.directory, "therapy-usage.txt",
              "DBT emotion regulation skills are taught in groups.")
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

    def test_mixed_query_is_reduced_to_grounded_concepts(self):
        reflection = backend.ReflectionResult(
            False,
            ("More detail is needed",),
            ("what does cat entail for wolf",),
        )
        queries = app.validate_reflection_queries(
            "what is a pig?", self.used, reflection, self.corpus
        )
        self.assertEqual(queries, ("cat",))

    def test_expanded_passage_text_is_grounding_evidence(self):
        used = [{**self.used[0], "body": self.used[0]["body"] + " A gasket controls pressure."}]
        reflection = backend.ReflectionResult(
            False,
            ("The gasket is unresolved",),
            ("what does the gasket entail",),
        )
        queries = app.validate_reflection_queries(
            "what is a pig?", used, reflection, self.corpus
        )
        self.assertEqual(queries, ("gasket",))

    def test_new_concept_is_accepted_when_corpus_connects_it_to_evidence(self):
        directory = tempfile.mkdtemp()
        try:
            write(directory, "base.txt", "A flux valve controls vibration in the assembly.")
            write(directory, "related.txt",
                  "Adaptive resonance tuning calibrates a flux valve to control vibration.")
            loaded = Corpus().load(directory, source=directory)
            used = [next(p for p in loaded.chunks if p["relpath"] == "base.txt")]
            query = "adaptive resonance tuning for flux valve vibration"
            reflection = backend.ReflectionResult(False, ("A related method may add detail",), (query,))
            queries = app.validate_reflection_queries(
                "how does a flux valve control vibration?", used, reflection, loaded
            )
            self.assertEqual(queries, (query,))
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_passage_identity_ignores_expansion_body_shape(self):
        first = {"relpath": "guide.txt", "ord": 4, "body": "short"}
        expanded = {"relpath": "guide.txt", "ord": 4, "body": "short plus neighbours"}
        self.assertEqual(app.passage_key(first), app.passage_key(expanded))

    def test_queries_allow_framing_and_corpus_derived_acronym_expansion(self):
        used = [next(
            passage for passage in self.corpus.chunks
            if passage["relpath"] == "therapy-usage.txt"
        )]
        reflection = backend.ReflectionResult(False, ("therapy details",), (
            "what is dialectical behavior therapy",
            "what are the specific emotion regulation skills in DBT",
        ))
        queries = app.validate_reflection_queries(
            "how are DBT emotion regulation skills used", used, reflection, self.corpus
        )
        self.assertEqual(queries, reflection.queries)

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
        refined_question = answer.call_args_list[1].args[0]
        self.assertIn("Initial answer", refined_question)
        self.assertIn("materially expand", refined_question)
        self.assertIn("Name newly supported concepts", refined_question)
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

    def test_stream_emits_initial_then_refined_with_divider(self):
        with mock.patch.object(app.backend, "reflect", return_value=self.reflection), \
             mock.patch.object(app.backend, "stream", side_effect=[
                 iter(["PRIVATE INITIAL ANSWER [1]."]),
                 iter(["Final ", "answer [1][2]."]),
             ]):
            events = self.events(self.client.post("/api/ask_stream", json={"q": "what is a pig?"}))
        check_idx = next(i for i, e in enumerate(events)
                         if e["type"] == "step" and "unresolved evidence" in e["label"])
        initial_tokens = [e.get("text", "") for i, e in enumerate(events)
                          if e["type"] == "token" and i < check_idx]
        self.assertTrue(initial_tokens, "Initial answer tokens should appear before refinement")
        self.assertIn("PRIVATE INITIAL ANSWER [1].", initial_tokens)
        dividers = [e for e in events if e["type"] == "answer_divider"]
        self.assertEqual(len(dividers), 1, "Exactly one divider separating sections")
        divider_idx = next(i for i, e in enumerate(events) if e["type"] == "answer_divider")
        refined_tokens = [e.get("text", "") for i, e in enumerate(events)
                          if e["type"] == "token" and i > divider_idx]
        self.assertEqual(refined_tokens, ["Final ", "answer [1][2]."],
                         "Refined model tokens should stream without buffering or rechunking")
        source_indexes = [i for i, event in enumerate(events) if event["type"] == "sources"]
        refined_answer_idx = max(i for i, event in enumerate(events) if event["type"] == "answer")
        self.assertEqual(len(source_indexes), 2)
        self.assertLess(source_indexes[1], refined_answer_idx,
                        "Refined source count must arrive before final Markdown rendering")
        done = next(event for event in events if event["type"] == "done")
        self.assertTrue(done["refinement"]["refined"])
        self.assertEqual(done["coverage"]["chunks"], 2)

    def test_stream_discards_rejected_refinement(self):
        initial = "A pig is the same kind of device as a cat [1]."
        with mock.patch.object(app.backend, "reflect", return_value=self.reflection), \
             mock.patch.object(app.backend, "stream", side_effect=[
                 iter([initial]),
                 iter(["The provided excerpts do not contain an answer."]),
             ]):
            events = self.events(self.client.post("/api/ask_stream", json={"q": "what is a pig?"}))
        self.assertTrue(any(event["type"] == "answer_divider" for event in events))
        self.assertTrue(any(event["type"] == "refinement_discard" for event in events))
        done = next(event for event in events if event["type"] == "done")
        self.assertEqual(done["refinement"]["reason"], "refined_answer_rejected")


class ReflectionParsingTests(unittest.TestCase):
    def test_reflection_prompt_seeks_novel_complementary_evidence(self):
        self.assertIn("new evidence", backend.REFLECTION_SYSTEM_PROMPT)
        self.assertIn("Do not merely rephrase", backend.REFLECTION_SYSTEM_PROMPT)
        self.assertIn("strongly related concept", backend.REFLECTION_SYSTEM_PROMPT)

    def test_answer_prompt_forbids_cross_excerpt_relationship_inference(self):
        with mock.patch.object(backend, "_completion", return_value="answer") as completion:
            backend.answer("q", "context", 1, AISettings())
        system_prompt = completion.call_args.args[0]
        self.assertIn("Do not infer a relationship", system_prompt)
        self.assertIn("explicitly connects", system_prompt)
        self.assertNotIn("treatment", system_prompt.lower())
        self.assertNotIn("referral", system_prompt.lower())

    def test_reflection_instruction_checks_unsupported_relationships(self):
        def completion(system_prompt, _user_content, _settings, _max_tokens,
                       _temperature, _json_mode):
            if "unsupported relationship" in system_prompt and "explicitly connect" in system_prompt:
                return json.dumps({
                    "complete": False,
                    "missing_aspects": ["The claimed relationship lacks evidence"],
                    "queries": ["grounded relationship"],
                })
            return json.dumps({"complete": True, "missing_aspects": [], "queries": []})

        with mock.patch.object(backend, "_completion", side_effect=completion):
            result = backend.reflect(
                "How does one concept affect another?",
                "The draft claims the concepts are linked.",
                "The excerpts discuss each concept separately.",
                AISettings(),
            )
        self.assertFalse(result.complete)
        self.assertEqual(result.queries, ("grounded relationship",))

    def test_reflection_instruction_treats_named_alternatives_as_a_gap(self):
        def completion(system_prompt, _user_content, _settings, _max_tokens,
                       _temperature, _json_mode):
            if "named alternatives" in system_prompt and "definition" in system_prompt:
                return json.dumps({
                    "complete": False,
                    "missing_aspects": ["The likely named concept is not defined"],
                    "queries": ["canonical concept"],
                })
            return json.dumps({"complete": True, "missing_aspects": [], "queries": []})

        with mock.patch.object(backend, "_completion", side_effect=completion):
            result = backend.reflect(
                "What is the user's approximate term?",
                "That exact term is not present, but the excerpts name a likely canonical concept.",
                "The evidence links the approximate term to the canonical concept.",
                AISettings(),
            )
        self.assertFalse(result.complete)
        self.assertEqual(result.queries, ("canonical concept",))

    def test_reflect_retries_empty_response_with_larger_budget(self):
        settings = AISettings(reflection_max_tokens=400)
        with mock.patch.object(backend, "_completion", side_effect=[
            "",
            '{"complete": true, "missing_aspects": [], "queries": []}',
        ]) as completion:
            result = backend.reflect("q", "draft", "context", settings)
        self.assertTrue(result.complete)
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(completion.call_args_list[0].args[3], 400)
        self.assertEqual(completion.call_args_list[1].args[3], 2000)

    def test_reflect_retries_truncated_json_with_larger_budget(self):
        settings = AISettings(reflection_max_tokens=400)
        with mock.patch.object(backend, "_completion", side_effect=[
            '{"complete": false, "missing',
            '{"complete": false, "missing_aspects": ["treatment"], '
            '"queries": ["what is the treatment"]}',
        ]) as completion:
            result = backend.reflect("q", "draft", "context", settings)
        self.assertFalse(result.complete)
        self.assertEqual(result.queries, ("what is the treatment",))
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(completion.call_args_list[1].args[3], 2000)

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
