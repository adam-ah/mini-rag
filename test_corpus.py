#!/usr/bin/env python3
import os, tempfile, shutil, unittest
from unittest import mock
import corpus
from settings import RetrievalSettings, Settings


def write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def long_doc(term, n_paras):
    return "\n\n".join(f"{term} paragraph {i} " + ("filler text here. " * 18) for i in range(n_paras))


class TokenizeTests(unittest.TestCase):
    def test_tokenize_keeps_short_alnum_terms(self):
        self.assertEqual(corpus.tokenize("AI and 3D, v2!"), ["ai", "and", "3d", "v2"])

    def test_tokenize_drops_single_chars(self):
        self.assertEqual(corpus.tokenize("a I x to"), ["to"])

    def test_query_terms_retains_ai(self):
        self.assertIn("ai", corpus.query_terms("What is the role of AI in the system?"))

    def test_query_terms_drops_stopwords(self):
        terms = corpus.query_terms("what is the role of the author")
        self.assertNotIn("the", terms)
        self.assertIn("author", terms)
        self.assertIn("role", terms)

    def test_query_terms_drops_retrieval_framing(self):
        terms = corpus.query_terms("explain the specific definition and details of widgets")
        self.assertEqual(terms, ["widget"])

    def test_query_terms_drops_generic_approach_framing(self):
        terms = corpus.query_terms("what is the focus on emotions approach in CBT")
        self.assertEqual(terms, ["focu", "emotion", "cbt"])

    def test_hyphenated_term_also_exposes_its_components(self):
        self.assertEqual(
            corpus.tokenize("binge-purge cycle"),
            ["binge-purge", "binge", "purge", "cycle"],
        )

    def test_slug_like_term_does_not_expose_generic_components(self):
        self.assertEqual(
            corpus.tokenize("term-that-does-not-exist-in-the-corpus"),
            ["term-that-does-not-exist-in-the-corpus"],
        )


class QueryModeTests(unittest.TestCase):
    def test_explain_and_steps_are_how(self):
        self.assertEqual(corpus.query_mode("explain the steps for borrowing"), "how")
        self.assertEqual(corpus.query_mode("how does ownership work"), "how")

    def test_fact_lookup_is_exact(self):
        self.assertEqual(corpus.query_mode("what keyword declares a variable"), "exact")

    def test_multi_facet_questions_receive_deeper_context(self):
        self.assertEqual(
            corpus.query_mode("what is the calibration for flux-valve used for vibration control"),
            "how",
        )

    def test_summary_is_global(self):
        self.assertEqual(corpus.query_mode("summarize ownership across the chapters"), "global")


class ExclusionTests(unittest.TestCase):
    def test_is_excluded(self):
        configured = Settings(retrieval=RetrievalSettings(
            exclude_patterns=["Private Fixtures", "Ignored Rules"]))
        with mock.patch.object(corpus.settings_service, "get", return_value=configured):
            self.assertTrue(corpus.is_excluded("Folder/Private Fixtures/x.txt"))
            self.assertTrue(corpus.is_excluded("Folder/Ignored Rules/rates.md"))
            self.assertFalse(corpus.is_excluded("Folder/Notes/x.md"))


class ChunkTests(unittest.TestCase):
    def test_short_text_one_chunk(self):
        self.assertEqual(len(corpus.chunk_text("one para.\n\ntwo para.")), 1)

    def test_long_text_splits(self):
        self.assertGreater(len(corpus.chunk_text(long_doc("x", 30))), 1)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        configured = Settings(retrieval=RetrievalSettings(
            exclude_patterns=["Sample Projects Data"]))
        self.settings_patch = mock.patch.object(corpus.settings_service, "get", return_value=configured)
        self.settings_patch.start()
        self.dir = tempfile.mkdtemp()
        write(self.dir, "Topic A/Engines.md",
              "The cycle engine applies a sequence pattern across many similar elements using the signature matcher.")
        write(self.dir, "Topic A/Duration.md",
              "Activity duration equals quantity divided by productivity rate divided by resource count.")
        write(self.dir, "Topic A/Sample Projects Data/proj.txt",
              "Confidential sample data that must never be searchable in this index.")
        write(self.dir, "Topic B/Widgets.md", long_doc("widget", 12))
        write(self.dir, "Topic B/WidgetRef.md", "A single widget reference paragraph.")
        self.c = corpus.Corpus().load(self.dir, source=self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        self.settings_patch.stop()

    def test_excluded_not_loaded(self):
        self.assertEqual(self.c.files, 4)
        for ch in self.c.chunks:
            self.assertFalse(corpus.is_excluded(ch["relpath"]))

    def test_search_ranks_relevant_doc_first(self):
        hits = self.c.search("signature matcher", limit=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["relpath"], "Topic A/Engines.md")

    def test_search_finds_duration(self):
        hits = self.c.search("duration productivity rate", limit=5)
        self.assertEqual(hits[0]["relpath"], "Topic A/Duration.md")

    def test_excluded_data_not_searchable(self):
        for term in ["Confidential sample", "never be searchable"]:
            for h in self.c.search(term, limit=50):
                self.assertNotIn("Sample Projects Data", h["relpath"])

    def test_sprint_filter(self):
        hits = self.c.search("widget", limit=50, sprint="Topic B")
        self.assertTrue(hits)
        self.assertTrue(all(h["sprint"] == "Topic B" for h in hits))


class GatherTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        write(self.dir, "Guide.md", "\n\n".join(self.para(i) for i in range(10)))
        write(self.dir, "Notes.md", "\n\n".join(self.para(100 + i) for i in range(3)))
        self.c = corpus.Corpus().load(self.dir, source=self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def para(i):
        uniq = " ".join(f"w{i}n{j}" for j in range(170))
        return f"zphrase step marker{i} {uniq}."

    def test_each_paragraph_is_its_own_chunk(self):
        guide = [c for c in self.c.chunks if c["relpath"] == "Guide.md"]
        self.assertEqual(len(guide), 10)
        self.assertEqual([c["ord"] for c in guide], list(range(10)))

    def test_how_query_expands_to_contiguous_neighbours(self):
        used, _ = self.c.gather("explain step marker5")
        self.assertTrue(used)
        body = used[0]["body"]
        neighbours = sum(f"marker{j}" in body for j in (3, 4, 5, 6, 7))
        self.assertGreaterEqual(neighbours, 3)

    def test_how_sends_more_context_than_exact(self):
        deep, _ = self.c.gather("explain the step process in detail")
        flat, _ = self.c.gather("zphrase")
        self.assertGreater(sum(len(u["body"]) for u in deep),
                           sum(len(u["body"]) for u in flat))

    def test_query_prioritizes_passages_connecting_requested_and_rare_facets(self):
        directory = tempfile.mkdtemp()
        try:
            relation = (
                "Flux-valve behavior can be used for vibration control. "
                + "case formulation filler " * 180
            )
            procedure = (
                "Flux and valve calibration uses TARGET_STAGED_PROCEDURE. "
                + "clinical protocol filler " * 180
            )
            generic = (
                "Calibration is used for vibration control. "
                + "generic skills filler " * 180
            )
            write(directory, "Relation.md", relation)
            write(directory, "Specific procedure.md", procedure)
            write(directory, "Generic skills.md", generic)
            loaded = corpus.Corpus().load(directory, source=directory)
            used, _ = loaded.gather(
                "what is the calibration for flux-valve used for vibration control?",
                char_budget=7000,
            )
            bodies = "\n".join(passage["body"] for passage in used)
            self.assertIn("TARGET_STAGED_PROCEDURE", bodies)
            self.assertIn("TARGET_STAGED_PROCEDURE", used[0]["body"])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_override_max_chunks(self):
        used, _ = self.c.gather("zphrase", max_chunks=2)
        self.assertLessEqual(len(used), 2)

    def test_override_expansion_radius(self):
        default, _ = self.c.gather("explain step marker5")
        narrow, _ = self.c.gather("explain step marker5", expand_radius=0)
        self.assertGreater(sum(len(p["body"]) for p in default),
                           sum(len(p["body"]) for p in narrow))

    def test_override_char_budget(self):
        used, _ = self.c.gather("zphrase", char_budget=2500)
        total = sum(len(u["body"]) for u in used)
        self.assertGreaterEqual(total, 1)
        self.assertLessEqual(total, 2500)

    def test_override_per_file_cap(self):
        used, _ = self.c.gather("zphrase", per_file_cap=1, max_chunks=99)
        counts = {}
        for u in used:
            counts[u["relpath"]] = counts.get(u["relpath"], 0) + 1
        self.assertTrue(all(v <= 1 for v in counts.values()))


class HtmlMappingTests(unittest.TestCase):
    def setUp(self):
        self.text = tempfile.mkdtemp()
        self.src = tempfile.mkdtemp()
        write(self.text, "Screens/screen1.md", "A start screen with zones and grouping controls.")
        write(self.text, "Screens/brief.md", "A textual brief, not a wireframe screen.")
        write(self.src, "Screens/screen1.html", "<html><body>screen1 wireframe</body></html>")
        write(self.src, "Screens/brief.docx", "not html")
        self.c = corpus.Corpus().load(self.text, source=self.src)

    def tearDown(self):
        shutil.rmtree(self.text, ignore_errors=True)
        shutil.rmtree(self.src, ignore_errors=True)

    def test_wireframe_detected(self):
        screen = [c for c in self.c.chunks if c["relpath"].endswith("screen1.md")]
        self.assertTrue(screen)
        self.assertEqual(screen[0]["html"], "Screens/screen1.html")
        self.assertEqual(self.c.wireframes, 1)

    def test_nonwireframe_has_no_html(self):
        brief = [c for c in self.c.chunks if c["relpath"].endswith("brief.md")]
        self.assertTrue(brief)
        self.assertIsNone(brief[0]["html"])

    def test_search_hit_carries_html(self):
        hits = self.c.search("start screen zones grouping", limit=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["html"], "Screens/screen1.html")


class TemplateCssTests(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
        with open(path, encoding="utf-8") as f:
            self.html = f.read()

    def test_answer_strong_is_dark_not_white(self):
        self.assertIn(".answer{color:var(--ink)}", self.html)
        self.assertIn(".answer strong{color:var(--ink)", self.html)
        self.assertNotIn(".answer strong{color:#fff", self.html)
        self.assertNotIn(".answer{color:#fff", self.html)

    def test_doc_links_wired(self):
        self.assertIn("function docHref", self.html)
        self.assertIn("docHref(s.relpath)", self.html)
        self.assertIn("docHref(h.relpath)", self.html)

    def test_single_enter_submits(self):
        self.assertIn("e.key==='Enter'&&!e.shiftKey", self.html)

    def test_answer_shows_spinner_while_waiting(self):
        self.assertIn(".awaiting{", self.html)
        self.assertIn('class="awaiting"', self.html)
        self.assertIn("waiting for the model", self.html)

    def test_refined_answer_has_separate_box_and_spinner(self):
        self.assertIn('id="refinedAnswerCard"', self.html)
        self.assertIn('id="refinedAnswer"', self.html)
        self.assertIn('id="refinedSpinner"', self.html)
        self.assertIn("Refining answer with additional evidence", self.html)
        self.assertIn("src-initial-", self.html)
        self.assertIn("src-refined-", self.html)
        self.assertIn("wireCites('#answer','initial')", self.html)
        self.assertIn("wireCites('#refinedAnswer','refined')", self.html)

    def test_working_panel_collapses_on_done(self):
        self.assertIn("function setStepsCollapsed", self.html)
        self.assertIn('id="stepsToggle"', self.html)
        self.assertIn("setStepsCollapsed(true)", self.html)
        self.assertIn("scrollIntoView", self.html)

    def test_examples_come_from_the_corpus(self):
        self.assertIn("m.topics", self.html)
        self.assertIn("m.documents", self.html)
        self.assertNotIn("const EXAMPLES", self.html)

    def test_shared_renderer_loaded(self):
        self.assertIn('src="/render.js"', self.html)
        self.assertIn("MD.render(", self.html)
        self.assertIn(".answer table{", self.html)


if __name__ == "__main__":
    unittest.main()
