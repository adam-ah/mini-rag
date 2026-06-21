#!/usr/bin/env python3
import json, os, re, unittest
from dataclasses import replace
from unittest import mock
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["SEARCH_OUTPUT"] = os.path.join(HERE, "tests", "small_corpus")
os.environ["SEARCH_INPUT"] = os.path.join(HERE, "tests", "small_corpus")
import app
import backend
import corpus
from settings import AISettings, Settings

BIG = os.path.join(corpus.HERE, "tests", "big_corpus")
SMALL = os.path.join(corpus.HERE, "tests", "small_corpus")
AI_PIPELINE = os.path.join(corpus.HERE, "tests", "ai_pipeline_corpus")
BIG_C = corpus.Corpus().load(text_dir=BIG, source=BIG)
SMALL_C = corpus.Corpus().load(text_dir=SMALL, source=SMALL)
AI_PIPELINE_C = corpus.Corpus().load(text_dir=AI_PIPELINE, source=AI_PIPELINE)


def live_ai_settings():
    """Build test-only settings; never fall back to the application's provider."""
    base_url = os.environ.get("AI_TEST_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return None
    return AISettings(
        backend="openai",
        base_url=base_url,
        model=os.environ.get("AI_TEST_MODEL", "local").strip() or "local",
        api_key=os.environ.get("AI_TEST_API_KEY", ""),
        temperature=float(os.environ.get("AI_TEST_TEMPERATURE", "0.2")),
        max_tokens=int(os.environ.get("AI_TEST_MAX_TOKENS", "2500")),
        timeout_seconds=int(os.environ.get("AI_TEST_TIMEOUT", "180")),
        adaptive_refinement=False,
    )


LIVE_AI_SETTINGS = live_ai_settings()


def ai_available():
    return LIVE_AI_SETTINGS is not None and backend.test_connection(LIVE_AI_SETTINGS)[0]


def chars(used):
    return sum(len(u["body"]) for u in used)


@unittest.skipUnless(BIG_C.N > 0, "tests/big_corpus missing")
class BigCorpusE2E(unittest.TestCase):
    def test_corpus_loaded(self):
        self.assertEqual(BIG_C.files, 13)
        self.assertGreater(BIG_C.N, 100)

    def test_distinctive_term_ranks_right_file(self):
        self.assertIn("ch04-01-what-is-ownership",
                      BIG_C.search("ownership rules moving a value", limit=5)[0]["relpath"])
        self.assertIn("ch13-02-iterators",
                      BIG_C.search("iterator adaptor map and collect", limit=5)[0]["relpath"])

    def test_deep_question_returns_large_contiguous_context(self):
        used, _ = BIG_C.gather("explain how ownership and borrowing work in detail")
        self.assertGreater(chars(used), 9000)
        self.assertGreater(max(len(u["body"]) for u in used), corpus.CHUNK,
                           "a how-answer excerpt should merge neighbouring chunks into a section")

    def test_factual_question_stays_lean(self):
        used, _ = BIG_C.gather("what is the keyword for a mutable variable")
        self.assertLess(chars(used), 9000)

    def test_budget_adapts_deep_vs_factual(self):
        deep = chars(BIG_C.gather("explain how ownership and borrowing work in detail")[0])
        fact = chars(BIG_C.gather("what is the keyword for a mutable variable")[0])
        self.assertGreater(deep, fact)

    def test_global_question_spans_multiple_files(self):
        _, n_files = BIG_C.gather("summarize ownership references slices and structs across the chapters")
        self.assertGreaterEqual(n_files, 3)

    def test_topics_and_documents_from_corpus(self):
        self.assertEqual(len(BIG_C.documents()), 13)
        self.assertTrue(BIG_C.topics())

    def test_no_screens_when_corpus_has_no_wireframes(self):
        self.assertEqual(BIG_C.relevant_screens("ownership and borrowing"), [])


@unittest.skipUnless(SMALL_C.N > 0, "tests/small_corpus missing")
class SmallCorpusE2E(unittest.TestCase):
    def test_corpus_loaded(self):
        self.assertEqual(SMALL_C.files, 3)
        self.assertEqual(len(SMALL_C.documents()), 3)

    def test_how_question_returns_content(self):
        used, _ = SMALL_C.gather("how do I write and run a hello world program")
        self.assertTrue(used)
        self.assertGreater(chars(used), 0)


class DocServingE2E(unittest.TestCase):
    def test_doc_resolves_real_md(self):
        rel = next(c["relpath"] for c in BIG_C.chunks if c["relpath"].endswith(".md"))
        self.assertIsNotNone(app.safe_path(BIG, rel, app.DOC_EXTS))

    def test_doc_rejects_traversal(self):
        self.assertIsNone(app.safe_path(BIG, "../app.py", app.DOC_EXTS))
        self.assertIsNone(app.safe_path(BIG, "../../../etc/passwd", app.DOC_EXTS))

    def test_doc_rejects_bad_extension(self):
        self.assertIsNone(app.safe_path(BIG, "nope.png", app.DOC_EXTS))

    def test_doc_rejects_excluded_paths(self):
        self.assertIsNone(app.safe_path(BIG, "Sample Projects Data/x.txt", app.DOC_EXTS))
        self.assertIsNone(app.safe_path(BIG, "Sample Business Rules/rates.csv", app.DOC_EXTS))

    def test_doc_md_renders_client_side(self):
        html = app.render_doc("x.md", "a **bold** word\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n")
        self.assertIn('src="/render.js"', html)
        self.assertIn("MD.render(window.__DOC)", html)
        self.assertIn('window.__KIND="md"', html)
        self.assertIn("**bold**", html)

    def test_doc_txt_is_raw_client_side(self):
        html = app.render_doc("x.txt", "**left as-is**")
        self.assertIn('window.__KIND="raw"', html)
        self.assertIn("MD.esc(window.__DOC)", html)
        self.assertIn("**left as-is**", html)


@unittest.skipUnless(BIG_C.N > 0, "tests/big_corpus missing")
@unittest.skipUnless(
    os.environ.get("RUN_LIVE_AI_TESTS") == "1" and ai_available(),
    "live AI tests require RUN_LIVE_AI_TESTS=1 and a reachable AI_TEST_BASE_URL",
)
class AnswerE2E(unittest.TestCase):
    def test_openai_synthesis_returns_text(self):
        used, _ = BIG_C.gather("explain how ownership works")
        ctx = app.build_context(used)
        answer = backend.answer("explain how ownership works", ctx, len(used), LIVE_AI_SETTINGS)
        self.assertGreater(len(answer), 40)

    def test_openai_stream_yields_tokens(self):
        used, _ = BIG_C.gather("what are the ownership rules")
        ctx = app.build_context(used)
        toks = list(backend.stream("what are the ownership rules", ctx, len(used), LIVE_AI_SETTINGS))
        self.assertTrue(toks)
        self.assertGreater(len("".join(toks)), 20)


@unittest.skipUnless(AI_PIPELINE_C.N > 0, "tests/ai_pipeline_corpus missing")
@unittest.skipUnless(
    os.environ.get("RUN_LIVE_AI_TESTS") == "1" and ai_available(),
    "live AI tests require RUN_LIVE_AI_TESTS=1 and a reachable AI_TEST_BASE_URL",
)
class AdaptivePipelineAI(unittest.TestCase):
    def setUp(self):
        self.old_corpus = app.CORPUS[0]
        app.CORPUS[0] = AI_PIPELINE_C
        self.ai = replace(LIVE_AI_SETTINGS, adaptive_refinement=True)
        self.settings_patch = mock.patch.object(
            app.settings_service, "get", return_value=Settings(ai=self.ai)
        )
        self.settings_patch.start()
        self.client = app.app.test_client()

    def tearDown(self):
        self.settings_patch.stop()
        app.CORPUS[0] = self.old_corpus

    def ask(self, question):
        response = self.client.post("/api/ask", json={"q": question})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def stream_events(self, question):
        response = self.client.post("/api/ask_stream", json={"q": question})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return [json.loads(frame[6:]) for frame in response.get_data(as_text=True).split("\n\n")
                if frame.startswith("data: ")]

    def assert_citations_resolve(self, data):
        citations = {int(value) for value in re.findall(r"\[(\d+)\]", data["answer"])}
        self.assertTrue(citations)
        self.assertTrue(all(1 <= value <= len(data["sources"]) for value in citations))

    def test_exact_question_follows_reference_and_stays_grounded(self):
        data = self.ask("What is a pig?")
        self.assertTrue(data["refinement"]["checked"])
        self.assertTrue(data["refinement"]["refined"])
        self.assertIn("cat", " ".join(data["refinement"]["queries"]).lower())
        self.assertIn("hydraulic", data["answer"].lower())
        self.assertIn("two handles", data["answer"].lower())
        paths = {source["relpath"] for source in data["sources"]}
        self.assertIn("pig.txt", paths)
        self.assertIn("cat.txt", paths)
        self.assertNotIn("wolf.txt", paths)
        self.assertNotRegex(data["answer"].lower(), r"purple|satellite|wolf")
        self.assert_citations_resolve(data)

    def test_complete_answer_does_not_refine_and_absent_subject_stays_absent(self):
        complete = self.ask("What is a duck?")
        self.assertTrue(complete["refinement"]["checked"])
        self.assertFalse(complete["refinement"]["refined"])
        self.assertEqual(complete["refinement"]["added_passages"], 0)
        self.assertIn("yellow", complete["answer"].lower())
        self.assertIn("locking lid", complete["answer"].lower())
        absent = self.ask("What is a zebra?")
        self.assertEqual(absent["mode"], "none")
        self.assertEqual(absent["sources"], [])
        self.assertIn("no matching", absent["answer"].lower())

    def test_multi_part_answer_covers_both_documents(self):
        data = self.ask("What colour and shape is the emergency alignment signal?")
        self.assertIn("amber", data["answer"].lower())
        self.assertIn("triangular", data["answer"].lower())
        paths = {source["relpath"] for source in data["sources"]}
        self.assertIn("signal-colour.txt", paths)
        self.assertIn("signal-shape.txt", paths)
        self.assert_citations_resolve(data)

    def test_stream_shows_initial_then_refined(self):
        events = self.stream_events("What is a pig?")
        check_index = next(i for i, event in enumerate(events)
                           if event["type"] == "step" and "unresolved evidence" in event["label"])
        initial_token_indexes = [i for i, event in enumerate(events)
                                 if event["type"] == "token" and i < check_index]
        self.assertTrue(initial_token_indexes,
                        "Initial answer tokens should appear before refinement check")
        divider_index = next((i for i, event in enumerate(events)
                              if event["type"] == "answer_divider"), None)
        self.assertIsNotNone(divider_index, "Divider should mark refined section")
        refined_token_indexes = [i for i, event in enumerate(events)
                                 if event["type"] == "token" and i > divider_index]
        self.assertTrue(refined_token_indexes,
                        "Refined answer tokens should appear after divider")
        done = next(event for event in events if event["type"] == "done")
        self.assertTrue(done["refinement"]["refined"])
        final_answer_events = [event["text"] for event in reversed(events)
                               if event["type"] == "answer"]
        final_text = "\n\n---\n\n".join(final_answer_events) if final_answer_events else ""
        self.assertIn("hydraulic", final_text.lower())


class ReflectionReasonInStepEvent(unittest.TestCase):
    def setUp(self):
        self.old_corpus = app.CORPUS[0]
        app.CORPUS[0] = SMALL_C
        ai = AISettings(
            backend="openai", base_url="http://localhost:9999/v1",
            model="test", api_key="x", temperature=0.2, max_tokens=500,
            timeout_seconds=10, adaptive_refinement=True,
        )
        self.settings_patch = mock.patch.object(
            app.settings_service, "get", return_value=Settings(ai=ai)
        )
        self.settings_patch.start()
        self.client = app.app.test_client()

    def tearDown(self):
        self.settings_patch.stop()
        app.CORPUS[0] = self.old_corpus

    def _events(self, question):
        response = self.client.post("/api/ask_stream", json={"q": question})
        self.assertEqual(response.status_code, 200)
        return [json.loads(f[6:]) for f in response.get_data(as_text=True).split("\n\n")
                if f.startswith("data: ")]

    def test_complete_reason_surfaces_in_step(self):
        fake_reflection = app.backend.ReflectionResult(True)
        with mock.patch.object(backend, "stream", return_value=iter(("Some initial answer.",))), \
             mock.patch.object(app, "plan_refinement", return_value=app.RefinementPlan(
                 reason="complete", reflection=fake_reflection, merged=tuple(SMALL_C.chunks[:1]),
             )):
            events = self._events("test")
        steps = [e["label"] for e in events if e["type"] == "step"]
        no_evidence = [s for s in steps if "No additional evidence" in s]
        self.assertEqual(len(no_evidence), 1)
        self.assertIn("found no gaps", no_evidence[0].lower())

    def test_no_valid_queries_reason_surfaces_in_step(self):
        fake_reflection = app.backend.ReflectionResult(False, ("gap",), ("alt",))
        with mock.patch.object(backend, "stream", return_value=iter(("Some initial answer.",))), \
             mock.patch.object(app, "plan_refinement", return_value=app.RefinementPlan(
                 reason="no_valid_queries", reflection=fake_reflection,
                 merged=tuple(SMALL_C.chunks[:1]),
             )):
            events = self._events("test")
        steps = [e["label"] for e in events if e["type"] == "step"]
        no_evidence = [s for s in steps if "No additional evidence" in s]
        self.assertEqual(len(no_evidence), 1)
        self.assertIn("not grounded", no_evidence[0].lower())

    def test_no_new_evidence_reason_surfaces_in_step(self):
        fake_reflection = app.backend.ReflectionResult(False, ("gap",), ("valid query",))
        with mock.patch.object(backend, "stream", return_value=iter(("Some initial answer.",))), \
             mock.patch.object(app, "plan_refinement", return_value=app.RefinementPlan(
                 reason="no_new_evidence", reflection=fake_reflection,
                 queries=("valid query",), merged=tuple(SMALL_C.chunks[:1]),
             )):
            events = self._events("test")
        steps = [e["label"] for e in events if e["type"] == "step"]
        no_evidence = [s for s in steps if "No additional evidence" in s]
        self.assertEqual(len(no_evidence), 1)
        self.assertIn("no new", no_evidence[0].lower())


class ReflectionGapSurfaced(unittest.TestCase):
    def setUp(self):
        self.old_corpus = app.CORPUS[0]
        app.CORPUS[0] = SMALL_C
        ai = AISettings(
            backend="openai", base_url="http://localhost:9999/v1",
            model="test", api_key="x", temperature=0.2, max_tokens=500,
            timeout_seconds=10, adaptive_refinement=True,
        )
        self.settings_patch = mock.patch.object(
            app.settings_service, "get", return_value=Settings(ai=ai)
        )
        self.settings_patch.start()
        self.client = app.app.test_client()

    def tearDown(self):
        self.settings_patch.stop()
        app.CORPUS[0] = self.old_corpus

    def _events(self, question):
        response = self.client.post("/api/ask_stream", json={"q": question})
        self.assertEqual(response.status_code, 200)
        return [json.loads(f[6:]) for f in response.get_data(as_text=True).split("\n\n")
                if f.startswith("data: ")]

    def test_gaps_emitted_as_step_event_with_keywords_and_queries(self):
        fake_reflection = app.backend.ReflectionResult(
            False,
            ("stage 3 of the process", "exception handling"),
            ("process stage 3 procedure", "exception handling rule"),
        )
        with mock.patch.object(backend, "stream", return_value=iter(("Some initial answer.",))), \
             mock.patch.object(app, "plan_refinement", return_value=app.RefinementPlan(
                 reason="new_evidence", reflection=fake_reflection,
                 queries=("process stage 3 procedure", "exception handling rule"),
                 added=({'relpath': 'one.md', 'body': 'Extra.'},
                        {'relpath': 'two.md', 'body': 'More.'}),
                 merged=tuple(SMALL_C.chunks[:2]),
             )):
            events = self._events("test")
        gap_step = [e for e in events if e["type"] == "step" and "found gap" in e["label"].lower()]
        self.assertEqual(len(gap_step), 1, "Expected one 'found gap' step event")
        self.assertIn("stage 3 of the process", gap_step[0]["label"])
        self.assertIn("exception handling", gap_step[0]["label"])
        gap_step_index = events.index(gap_step[0])
        unwrapped = [e for e in events[:gap_step_index + 1] if e["type"] == "step" and "unresolved evidence" in e["label"]]
        self.assertTrue(unwrapped, "Gap step should appear after the 'Checking the answer' step")

    def test_no_gap_emitted_when_reflection_complete(self):
        fake_reflection = app.backend.ReflectionResult(True)
        with mock.patch.object(backend, "stream", return_value=iter(("Some initial answer.",))), \
             mock.patch.object(app, "plan_refinement", return_value=app.RefinementPlan(
                 reason="complete", reflection=fake_reflection, merged=tuple(SMALL_C.chunks[:1]),
             )):
            events = self._events("test")
        gap_steps = [e for e in events if e["type"] == "step" and "found gap" in e["label"].lower()]
        self.assertEqual(gap_steps, [])


class ReflectionContextUsesFullGatheredText(unittest.TestCase):
    def test_no_truncation_for_normal_gather_size(self):
        used = [
            {"relpath": "a.md", "sprint": "", "ord": 0, "html": False, "body": "x" * 15000}
            for _ in range(1)
        ]
        captured = {}
        def fake_reflect(question, draft, context, settings):
            captured["context"] = context
            return app.backend.ReflectionResult(True)
        settings = AISettings(
            backend="openai", base_url="http://localhost:9999/v1",
            model="test", api_key="x", temperature=0.2, max_tokens=500,
            timeout_seconds=10, adaptive_refinement=True,
        )
        with mock.patch.object(app.backend, "reflect", side_effect=fake_reflect):
            app.plan_refinement("q?", None, "draft", used, app.get_corpus(), settings)
        self.assertEqual(captured["context"], app.build_context(used),
                         "Reflection should see the full gathered context within normal size")

    def test_truncation_still_applied_for_pathologically_large_gather(self):
        used = [
            {"relpath": "a.md", "sprint": "", "ord": 0, "html": False, "body": "x" * 200000}
            for _ in range(1)
        ]
        captured = {}
        def fake_reflect(question, draft, context, settings):
            captured["context"] = context
            return app.backend.ReflectionResult(True)
        settings = AISettings(
            backend="openai", base_url="http://localhost:9999/v1",
            model="test", api_key="x", temperature=0.2, max_tokens=500,
            timeout_seconds=10, adaptive_refinement=True,
        )
        with mock.patch.object(app.backend, "reflect", side_effect=fake_reflect):
            app.plan_refinement("q?", None, "draft", used, app.get_corpus(), settings)
        self.assertLess(len(captured["context"]), len(app.build_context(used)),
                        "Reflection context should be capped when gather is pathologically large")


class StreamingSectionsMocked(unittest.TestCase):
    def setUp(self):
        self.old_corpus = app.CORPUS[0]
        app.CORPUS[0] = SMALL_C
        ai = AISettings(
            backend="openai", base_url="http://localhost:9999/v1",
            model="test", api_key="x", temperature=0.2, max_tokens=500,
            timeout_seconds=10, adaptive_refinement=True,
        )
        self.settings_patch = mock.patch.object(
            app.settings_service, "get", return_value=Settings(ai=ai)
        )
        self.settings_patch.start()
        self.client = app.app.test_client()

    def tearDown(self):
        self.settings_patch.stop()
        app.CORPUS[0] = self.old_corpus

    def _events(self, question):
        response = self.client.post("/api/ask_stream", json={"q": question})
        self.assertEqual(response.status_code, 200)
        return [json.loads(f[6:]) for f in response.get_data(as_text=True).split("\n\n")
                if f.startswith("data: ")]

    def test_stream_initial_tokens_before_refinement(self):
        with mock.patch.object(backend, 'stream') as mock_stream:
            call_count = [0]
            def side_effect(*args, **kwargs):
                if call_count[0] == 0:
                    call_count[0] += 1
                    yield from ("This is the ", "initial answer ", "with some content.")
                else:
                    call_count[0] += 1
                    yield from ("Here is the ", "refined answer ", "incorporating more evidence.")
            mock_stream.side_effect = side_effect
            with mock.patch.object(app, 'plan_refinement') as mock_plan:
                mock_plan.return_value = app.RefinementPlan(
                    reason="new_evidence",
                    reflection=app.backend.ReflectionResult(False, ("more details",), ("extra info",)),
                    queries=("extra info",),
                    added=({'relpath': 'extra.txt', 'body': 'Extra evidence here.'},),
                    merged=(
                        {'relpath': 'one.md', 'body': 'First passage.',
                         'tf': {}, 'ord': 0, 'sprint': '', 'html': False},
                        {'relpath': 'extra.txt', 'body': 'Extra evidence here.',
                         'tf': {}, 'ord': 1, 'sprint': '', 'html': False},
                    ),
                )
                events = self._events("test question")
        check_idx = next(i for i, e in enumerate(events)
                         if e["type"] == "step" and "unresolved evidence" in e["label"])
        initial_tokens = [e for i, e in enumerate(events) if e["type"] == "token" and i < check_idx]
        self.assertTrue(initial_tokens, "Initial tokens should appear before refinement")
        self.assertEqual(len(initial_tokens), 3)
        sources_idx = next(i for i, event in enumerate(events) if event["type"] == "sources")
        self.assertLess(sources_idx, check_idx)

    def test_stream_divider_separates_initial_from_refined(self):
        with mock.patch.object(backend, 'stream') as mock_stream:
            call_count = [0]
            def side_effect(*args, **kwargs):
                if call_count[0] == 0:
                    call_count[0] += 1
                    yield from ("Initial ", "answer ", "text.")
                else:
                    call_count[0] += 1
                    yield from ("Refined ", "answer ", "text with new details.")
            mock_stream.side_effect = side_effect
            with mock.patch.object(app, 'plan_refinement') as mock_plan:
                mock_plan.return_value = app.RefinementPlan(
                    reason="new_evidence",
                    reflection=app.backend.ReflectionResult(False, ("gap",), ("fill gap",)),
                    queries=("fill gap",),
                    added=({'relpath': 'extra.md', 'body': 'Extra.'},),
                    merged=(
                        {'relpath': 'one.md', 'body': 'One.', 'tf': {}, 'ord': 0, 'sprint': '', 'html': False},
                        {'relpath': 'extra.md', 'body': 'Extra.', 'tf': {}, 'ord': 1, 'sprint': '', 'html': False},
                    ),
                )
                events = self._events("test")
        dividers = [e for e in events if e["type"] == "answer_divider"]
        self.assertEqual(len(dividers), 1, "Exactly one divider separating sections")
        tokens_before = [e for i, e in enumerate(events)
                         if e["type"] == "token" and events.index(e) < events.index(dividers[0])]
        tokens_after = [e for i, e in enumerate(events)
                        if e["type"] == "token" and events.index(e) > events.index(dividers[0])]
        self.assertTrue(tokens_before, "Tokens should appear before divider")
        self.assertTrue(tokens_after, "Tokens should appear after divider")

    def test_stream_discards_refined_box_when_answer_rejected(self):
        with mock.patch.object(backend, 'stream') as mock_stream:
            call_count = [0]
            def side_effect(*args, **kwargs):
                if call_count[0] == 0:
                    call_count[0] += 1
                    yield from ("Good initial answer.")
                else:
                    call_count[0] += 1
                    yield from ("The excerpts do not contain the information needed.")
            mock_stream.side_effect = side_effect
            with mock.patch.object(app, 'plan_refinement') as mock_plan:
                mock_plan.return_value = app.RefinementPlan(
                    reason="new_evidence",
                    reflection=app.backend.ReflectionResult(False, ("gap",), ("fill gap",)),
                    queries=("fill gap",),
                    added=({'relpath': 'extra.md', 'body': 'Extra.'},),
                    merged=(
                        {'relpath': 'one.md', 'body': 'One.', 'tf': {}, 'ord': 0, 'sprint': '', 'html': False},
                        {'relpath': 'extra.md', 'body': 'Extra.', 'tf': {}, 'ord': 1, 'sprint': '', 'html': False},
                    ),
                )
                events = self._events("test")
        dividers = [e for e in events if e["type"] == "answer_divider"]
        self.assertEqual(len(dividers), 1, "Refined streaming starts before validation")
        discards = [e for e in events if e["type"] == "refinement_discard"]
        self.assertEqual(len(discards), 1, "Rejected refined answer must be removed")
        answers = [e for e in events if e["type"] == "answer"]
        sources = [e for e in events if e["type"] == "sources"]
        self.assertEqual(len(answers), 1, "Only one answer event when refined is rejected")
        self.assertEqual(len(sources), 1, "Only one sources event when refined is rejected")
        done = next(e for e in events if e["type"] == "done")
        self.assertEqual(done["refinement"]["reason"], "refined_answer_rejected")

    def test_stream_no_refinement_shows_tokens_only(self):
        with mock.patch.object(backend, 'stream') as mock_stream:
            mock_stream.return_value = iter(("A complete ", "answer without ", "gaps."))
            with mock.patch.object(app, 'plan_refinement') as mock_plan:
                mock_plan.return_value = app.RefinementPlan(
                    reason="complete",
                    reflection=app.backend.ReflectionResult(True),
                    merged=(
                        {'relpath': 'one.md', 'body': 'One.', 'tf': {}, 'ord': 0, 'sprint': '', 'html': False},
                    ),
                )
                events = self._events("test")
        dividers = [e for e in events if e["type"] == "answer_divider"]
        self.assertEqual(len(dividers), 0, "No divider when no refinement needed")
        tokens = [e for e in events if e["type"] == "token"]
        self.assertTrue(tokens, "Should have token events")
        done = next(e for e in events if e["type"] == "done")
        self.assertEqual(done.get("refinement", {}).get("reason"), "complete")


class ChunkTokensHelper(unittest.TestCase):
    def test_splits_on_whitespace_and_reassembles(self):
        text = "Hello world this is a longer sentence that should be chunked."
        chunks = list(app.chunk_tokens(text, size=20))
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertLessEqual(max(len(c) for c in chunks), 22)

    def test_short_text_yields_single_chunk(self):
        chunks = list(app.chunk_tokens("hello", size=120))
        self.assertEqual(chunks, ["hello"])

    def test_preserves_exact_text_when_no_spaces(self):
        chunks = list(app.chunk_tokens("abcdefghij", size=4))
        reconstructed = "".join(chunks)
        self.assertEqual(reconstructed, "abcdefghij")
        for c in chunks:
            self.assertLessEqual(len(c), 4)


if __name__ == "__main__":
    unittest.main()
