#!/usr/bin/env python3
import unittest
import corpus
import eval as ev

CORPUS = corpus.Corpus().load(text_dir=ev.EVAL_CORPUS, source=ev.EVAL_CORPUS)


@unittest.skipUnless(CORPUS.N > 0, "tests/big_corpus missing")
class RetrievalEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = ev.measure(CORPUS, ev.load_eval())

    def test_answer_coverage(self):
        self.assertGreaterEqual(self.m["coverage"], 0.9, self.m["rows"])

    def test_recall_at_10(self):
        self.assertGreaterEqual(self.m["recall@10"], 0.9, self.m["rows"])

    def test_recall_at_5(self):
        self.assertGreaterEqual(self.m["recall@5"], 0.85, self.m["rows"])

    def test_mrr(self):
        self.assertGreaterEqual(self.m["mrr"], 0.8, self.m["rows"])

    def test_context_stays_bounded(self):
        self.assertLess(self.m["avg_chars"], 14000)

    def test_low_duplicate_excerpts(self):
        self.assertLessEqual(self.m["dup_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
