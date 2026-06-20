#!/usr/bin/env python3
import os, re, glob, math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from settings import settings_service
from env_loader import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(HERE, ".env"))

OUTPUT = os.environ.get("SEARCH_OUTPUT", os.path.join(HERE, "output"))
INPUT = os.environ.get("SEARCH_INPUT", os.path.join(HERE, "input"))
CHUNK, MIN_CHUNK = 1300, 60
K1, B = 1.5, 0.75
META_WEIGHT = float(os.environ.get("SEARCH_META_WEIGHT", "3"))
ALIAS_WEIGHT = float(os.environ.get("SEARCH_ALIAS_WEIGHT", "0.4"))
SCREEN_MAX = int(os.environ.get("SEARCH_SCREEN_MAX", "4"))
SCREEN_FLOOR = float(os.environ.get("SEARCH_SCREEN_FLOOR", "0.2"))
RERANK_POOL = int(os.environ.get("SEARCH_RERANK_POOL", "150"))
DUP_SIM = float(os.environ.get("SEARCH_DUP_SIM", "0.7"))
MODE = {
    "exact":  {"rel": 0.55, "floor": 1, "nmax": 6,  "ceil": 8000,  "cap": 3, "expand": 0, "dedup": True},
    "how":    {"rel": 0.33, "floor": 2, "nmax": 14, "ceil": 24000, "cap": 0, "expand": 2, "dedup": False},
    "ui":     {"rel": 0.45, "floor": 1, "nmax": 12, "ceil": 14000, "cap": 4, "expand": 0, "dedup": True},
    "global": {"rel": 0.30, "floor": 3, "nmax": 16, "ceil": 20000, "cap": 4, "expand": 1, "dedup": True},
}
GLOBAL_CUES = ("across", "role of", "summar", "compare", "overview", " all ", " every ",
               "throughout", "end-to-end", "end to end", "everything", "list ", " each ")
UI_CUES = ("screen", "wireframe", "canvas", "mockup", "button", " ui ", " view ", "layout")
HOW_CUES = ("how do", "how does", "how is", "how are", "explain", "walk me",
            "workflow", "step", "process", "sequence")

@dataclass(frozen=True)
class QueryExpansion:
    source: str
    target: str
    reason: str       # "spelling" or "document_acronym"
    weight: float

@dataclass(frozen=True)
class QueryAnalysis:
    original: str
    terms: tuple[str, ...]
    weighted_terms: tuple[tuple[str, float], ...]
    expansions: tuple[QueryExpansion, ...]
    rescued: bool

@dataclass(frozen=True)
class SearchResult:
    hits: tuple[dict, ...]
    analysis: QueryAnalysis
    suggestions: tuple[str, ...]

def query_mode(question):
    q = " " + question.lower().strip() + " "
    if any(c in q for c in UI_CUES):
        return "ui"
    if any(c in q for c in GLOBAL_CUES):
        return "global"
    if any(c in q for c in HOW_CUES):
        return "how"
    return "exact"


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")
TAG_RE = re.compile(r"</?[a-zA-Z][^>\n]{0,300}>")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
ACR_FWD = re.compile(r"\b([A-Z][A-Z0-9]{1,7})\s*\(([A-Za-z][^)\n]{3,60})\)")
ACR_REV = re.compile(r"\b([A-Z][a-z][A-Za-z /\-]{3,50}?)\s*\(([A-Z][A-Z0-9]{1,7})\)")
STOPWORDS = set(
    "the a an and or but if then else of for to in on at by with from as is are was were "
    "be been being do does did has have had this that these those it its i you he she we they "
    "them his her our your their what which who whom how why when where can could should would "
    "will shall may might must not no yes about into over under than too very just also more "
    "most some any all each both few there here me my mine us "
    "only nothing pls please concise verbose briefly bullet bullets markdown tldr give show tell".split()
)


def tokenize(text):
    res = TOKEN_RE.findall(text.lower())
    # print(f"DEBUG: tokenize('{text[:20]}...') -> {res}")
    return res


def stem(w):
    if len(w) <= 4 or any(c in w for c in "_-0123456789"):
        return w
    if w.endswith("ies") and len(w) > 5:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ing") and len(w) > 6:
        return w[:-3]
    if w.endswith("ed") and len(w) > 5:
        return w[:-2]
    if w.endswith("es") and len(w) > 5:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def stems(text):
    return [stem(t) for t in tokenize(text)]


def strip_html(text):
    return TAG_RE.sub(" ", text)

class VocabularyIndex:
    def __init__(self, corpus_df, body_df):
        self.by_len = defaultdict(set)
        self.by_first_last = defaultdict(set)
        self.deletes = defaultdict(set)
        self.vocab = set()
        self.body_df = body_df
        
        for term in corpus_df:
            if self._is_suitable(term) and body_df.get(term, 0):
                self.vocab.add(term)
                self.by_len[len(term)].add(term)
                self.by_first_last[(term[0], term[-1])].add(term)
                for sig in self._get_deletes(term):
                    self.deletes[sig].add(term)

    def _is_suitable(self, term):
        if len(term) < 4:
            return False
        if term.isdigit():
            return False
        # Basic OCR garbage detection: too many non-alphanumeric
        if sum(not c.isalnum() for c in term) > len(term) // 3:
            return False
        return True

    def _get_deletes(self, word):
        return [word[:i] + word[i+1:] for i in range(len(word))]

    def candidates(self, term, max_dist=1, limit=20):
        if term in self.vocab:
            return set() # Not OOV
        
        cands = set()
        # Edit distance 1: deletions, replacements, insertions
        # Using deletion signatures for dist 1
        sigs = self._get_deletes(term) + [term]
        for sig in sigs:
            cands |= self.deletes.get(sig, set())
            
        # Same-edge candidates cover common substitutions (tyre/tire) without
        # scanning every similarly-sized vocabulary term on each request.
        cands |= self.by_first_last.get((term[0], term[-1]), set())
        filtered = [t for t in cands
                    if abs(len(term) - len(t)) <= max_dist and self._levenshtein(term, t) <= max_dist]
        filtered.sort(key=lambda t: (-self.body_df.get(t, 0), t))
        return set(filtered[:limit])

    def _levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if not s2:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            cur = [i + 1]
            for j, c2 in enumerate(s2):
                cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
            prev = cur
        return prev[-1]

class QueryAnalyzer:
    def __init__(self, corpus):
        self.corpus = corpus

    def analyze(self, question):
        original = question
        q_terms = query_terms(question)
        
        expansions = []
        weighted = {t: 1.0 for t in q_terms}
        corrections_count = 0

        for t in q_terms:
            if corrections_count >= 2:
                break

            # Acronym expansions are learned from the indexed documents, not a
            # maintained synonym list.
            document_targets = sorted(
                target for target in self.corpus.document_expansions.get(t, ())
                if target in self.corpus.df
            )
            if document_targets:
                for target in document_targets[:2]:
                    reason = "document_acronym"
                    weight = ALIAS_WEIGHT
                    weighted[target] = max(weighted.get(target, 0.0), weight)
                    expansions.append(QueryExpansion(t, target, reason, weight))
                corrections_count += 1
                continue

            if t in self.corpus.vocab.vocab or len(t) < 4:
                continue

            max_dist = 1 if len(t) <= 7 else 2
            candidates = self.corpus.vocab.candidates(t, max_dist=max_dist)
            ranked_cands = []
            for cand in candidates:
                dist = self.corpus.vocab._levenshtein(t, cand)
                freq = self.corpus.body_df.get(cand, 0)
                score = (1.0 / (1.0 + dist)) * 10 + min(freq, 100) / 100.0
                if t[:1] == cand[:1]: score += 1.0
                if t[-1:] == cand[-1:]: score += 1.0
                ranked_cands.append((cand, score))
            if not ranked_cands:
                continue
            ranked_cands.sort(key=lambda x: x[1], reverse=True)
            best_cand, best_score = ranked_cands[0]
            if len(ranked_cands) > 1:
                second_score = ranked_cands[1][1]
                if best_score <= second_score * 1.08:
                    continue
            elif best_score < 2.0:
                continue
            expansions.append(QueryExpansion(source=t, target=best_cand, reason="spelling", weight=0.75))
            weighted[best_cand] = max(weighted.get(best_cand, 0.0), 0.75)
            corrections_count += 1
        
        return QueryAnalysis(
            original=original,
            terms=tuple(q_terms),
            weighted_terms=tuple(weighted.items()),
            expansions=tuple(expansions),
            rescued=False
        )



def is_excluded(rel):
    patterns = settings_service.get().retrieval.exclude_patterns
    return any(p.lower() in rel.lower() for p in patterns)


def html_stems(source):
    out = {}
    if not source or not os.path.isdir(source):
        return out
    for p in glob.glob(os.path.join(source, "**", "*.html"), recursive=True):
        rel = os.path.relpath(p, source)
        if is_excluded(rel):
            continue
        out[os.path.splitext(rel)[0]] = rel
    return out


def query_terms(question):
    toks = tokenize(question)
    base = [t for t in toks if len(t) >= 2 and t not in STOPWORDS] or toks
    out, seen = [], set()
    for t in base:
        s = stem(t)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _is_separator(row):
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    return cells and all(set(c) <= set("-: ") and "-" in c for c in cells)


def _split_table(table):
    rows = table.split("\n")
    head = rows[:2] if len(rows) >= 2 and _is_separator(rows[1]) else rows[:1]
    body = rows[len(head):]
    parts, cur, base_len = [], list(head), sum(len(r) for r in head)
    cur_len = base_len
    for r in body:
        if cur_len + len(r) > CHUNK and len(cur) > len(head):
            parts.append("\n".join(cur))
            cur, cur_len = list(head), base_len
        cur.append(r)
        cur_len += len(r) + 1
    if len(cur) > len(head):
        parts.append("\n".join(cur))
    return parts or [table]


def _segments(text):
    lines = text.split("\n")
    segs, para, stack, heading = [], [], [], ""

    def flush_para():
        nonlocal para
        if para:
            block = "\n".join(para).strip()
            if block:
                segs.append(("text", block, heading))
            para = []

    i = 0
    while i < len(lines):
        line = lines[i]
        m = HEADING_RE.match(line)
        if m:
            flush_para()
            lvl, title = len(m.group(1)), m.group(2).strip()
            stack[:] = [(l, t) for l, t in stack if l < lvl] + [(lvl, title)]
            heading = " › ".join(t for _, t in stack)
            segs.append(("text", line.strip(), heading))
            i += 1
        elif line.strip().startswith("|"):
            flush_para()
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            segs.append(("table", "\n".join(lines[i:j]), heading))
            i = j
        elif line.strip() == "":
            flush_para()
            i += 1
        else:
            para.append(line)
            i += 1
    flush_para()
    return segs


def chunk_text(text):
    chunks, cur, cur_len, cur_head = [], [], 0, None

    def flush():
        nonlocal cur, cur_len, cur_head
        if cur:
            block = "\n\n".join(cur).strip()
            if block:
                chunks.append({"text": block, "heading": cur_head})
        cur, cur_len, cur_head = [], 0, None

    for kind, seg, heading in _segments(text):
        if kind == "table" and len(seg) > CHUNK:
            flush()
            for part in _split_table(seg):
                chunks.append({"text": part, "heading": heading})
            continue
        if len(seg) > CHUNK:
            flush()
            for k in range(0, len(seg), CHUNK):
                chunks.append({"text": seg[k:k + CHUNK], "heading": heading})
            continue
        if cur and cur_len + len(seg) + 2 > CHUNK:
            flush()
        if not cur:
            cur_head = heading
        cur.append(seg)
        cur_len += len(seg) + 2
    flush()

    out = []
    for c in chunks:
        if len(c["text"]) < MIN_CHUNK and out:
            out[-1]["text"] += "\n\n" + c["text"]
        else:
            out.append(c)
    return out


class Corpus:
    def __init__(self):
        self.chunks = []
        self.postings = defaultdict(list)
        self.df = Counter()
        self.body_df = Counter()
        self.N = 0
        self.avgdl = 1.0
        self.files = 0
        self.wireframes = 0
        self.document_expansions = defaultdict(set)
        self.by_file = defaultdict(list)
        self.vocab = None
        self.analyzer = None

    def _add_document_expansion(self, acronym, expansion):
        a = stem(acronym.lower())
        ws = {stem(w) for w in tokenize(expansion) if len(w) > 2 and w not in STOPWORDS}
        ws.discard(a)
        if ws:
            self.document_expansions[a] |= ws

    def load(self, text_dir=OUTPUT, source=OUTPUT):
        wf = html_stems(source)
        paths = []
        for ext in ("*.md", "*.txt", "*.csv", "*.json"):
            paths += glob.glob(os.path.join(text_dir, "**", ext), recursive=True)
        total_len = 0
        for p in sorted(paths):
            rel = os.path.relpath(p, text_dir)
            if is_excluded(rel):
                continue
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    text = strip_html(fh.read())
            except Exception as e:
                continue
            if not text.strip():
                continue
            for m in ACR_FWD.finditer(text):
                self._add_document_expansion(m.group(1), m.group(2))
            for m in ACR_REV.finditer(text):
                self._add_document_expansion(m.group(2), m.group(1))
            self.files += 1
            sprint = rel.split(os.sep)[0]
            kind = os.path.splitext(p)[1].lstrip(".")
            html = wf.get(os.path.splitext(rel)[0])
            path_meta = os.path.splitext(os.path.basename(rel))[0] + " " + rel.replace(os.sep, " ")
            for i, c in enumerate(chunk_text(text)):
                content, heading = c["text"], c["heading"] or ""
                tf = {}
                content_terms = stems(content)
                for t in content_terms:
                    tf[t] = tf.get(t, 0.0) + 1.0
                for t in set(content_terms):
                    self.body_df[t] += 1
                dl = int(sum(tf.values()))
                for mt in stems(path_meta + " " + heading):
                    tf[mt] = tf.get(mt, 0.0) + META_WEIGHT
                idx = len(self.chunks)
                self.chunks.append({"relpath": rel, "sprint": sprint, "kind": kind, "ord": i,
                                    "body": content, "heading": heading, "tf": tf, "dl": dl, "html": html})
                for term, n in tf.items():
                    self.postings[term].append((idx, n))
                    self.df[term] += 1
                total_len += dl
        for gi, ch in enumerate(self.chunks):
            self.by_file[ch["relpath"]].append(gi)
        self.N = len(self.chunks)
        self.avgdl = (total_len / self.N) if self.N else 1.0
        self.wireframes = len({c["html"] for c in self.chunks if c["html"]})
        self.vocab = VocabularyIndex(self.df, self.body_df)
        self.analyzer = QueryAnalyzer(self)
        return self

    def query_terms(self, question):
        return query_terms(question)

    def documents(self, limit=40):
        out, seen = [], set()
        for ch in self.chunks:
            rel = ch["relpath"]
            if rel in seen:
                continue
            seen.add(rel)
            name = os.path.splitext(os.path.basename(rel))[0]
            out.append(re.sub(r"[\s_-]+", " ", name).strip())
            if len(out) >= limit:
                break
        return out

    def topics(self, limit=6):
        out, seen, per_file = [], set(), Counter()
        for ch in self.chunks:
            if not ch["heading"]:
                continue
            h = ch["heading"].split(" › ")[-1].strip().lstrip("# ").strip()
            k = h.lower()
            if len(h) < 5 or h.isdigit() or k in seen or per_file[ch["relpath"]] >= 1:
                continue
            seen.add(k)
            per_file[ch["relpath"]] += 1
            out.append(h)
            if len(out) >= limit:
                break
        return out

    def weighted_terms(self, question, rescue=True):
        if not rescue:
            return [(t, 1.0) for t in query_terms(question)]
        return list(self.analyzer.analyze(question).weighted_terms)

    def _score(self, question, rescue=True):
        return self._score_terms(self.weighted_terms(question, rescue=rescue))

    def _score_terms(self, weighted_terms):
        scores = defaultdict(float)
        for t, w in weighted_terms:
            df = self.df.get(t, 0)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for idx, tf in self.postings[t]:
                dl = self.chunks[idx]["dl"]
                denom = tf + K1 * (1 - B + B * dl / self.avgdl)
                scores[idx] += w * idf * (tf * (K1 + 1)) / denom
        return scores

    def rank(self, question, sprint=None, rescue=True):
        if rescue:
            return self.analyze_and_rank(question, sprint)[1]
        return self._rank_terms(self.weighted_terms(question, rescue=False), sprint)

    def _rank_terms(self, weighted_terms, sprint=None):
        scores = self._score_terms(weighted_terms)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for idx, sc in ranked:
            ch = self.chunks[idx]
            if sprint and ch["sprint"] != sprint:
                continue
            out.append((ch, sc))
        return out

    def search(self, question, limit=20, sprint=None):
        return list(self.search_result(question, limit=limit, sprint=sprint).hits)

    def search_result(self, question, limit=20, sprint=None):
        analysis, ranked = self.analyze_and_rank(question, sprint)
        items = self.rerank(question, self.candidate_pool(question, ranked, analysis), analysis=analysis)
        hits = tuple({**ch, "score": sc} for ch, sc in items[:limit])
        suggestions = self.suggest_questions(question, analysis, hits)
        return SearchResult(hits=hits, analysis=analysis, suggestions=suggestions)

    def relevant_screens(self, question, sprint=None, max_screens=SCREEN_MAX, floor=SCREEN_FLOOR):
        ranked = self.rank(question, sprint)
        if not ranked:
            return []
        cutoff = floor * ranked[0][1]
        seen, out = set(), []
        for ch, sc in ranked:
            if sc < cutoff:
                break
            if not ch["html"] or ch["html"] in seen:
                continue
            seen.add(ch["html"])
            out.append({"relpath": ch["relpath"], "html": ch["html"],
                        "name": os.path.splitext(os.path.basename(ch["relpath"]))[0]})
            if len(out) >= max_screens:
                break
        return out

    def analyze_and_rank(self, question, sprint=None):
        proposed = self.analyzer.analyze(question)
        original = self._rank_terms([(t, 1.0) for t in proposed.terms], sprint)
        if not proposed.expansions:
            return proposed, original

        expanded = self._rank_terms(proposed.weighted_terms, sprint)
        if not expanded:
            return proposed, original

        original_coverage = self._top_concept_coverage(original, proposed, include_expansions=False)
        expanded_coverage = self._top_concept_coverage(expanded, proposed, include_expansions=True)
        corrected_sources = {e.source for e in proposed.expansions}
        original_corrected = self._top_source_coverage(original, corrected_sources, proposed, False)
        expanded_corrected = self._top_source_coverage(expanded, corrected_sources, proposed, True)
        if (not original or expanded_coverage > original_coverage or
                (expanded_coverage >= original_coverage and expanded_corrected > original_corrected)):
            return replace(proposed, rescued=True), expanded
        return proposed, original

    @staticmethod
    def _top_concept_coverage(ranked, analysis, include_expansions):
        if not ranked:
            return 0.0
        terms = set(ranked[0][0]["tf"])
        targets = defaultdict(set)
        if include_expansions:
            for expansion in analysis.expansions:
                targets[expansion.source].add(expansion.target)
        covered = sum(1 for term in analysis.terms
                      if term in terms or bool(targets.get(term, set()) & terms))
        return covered / len(analysis.terms) if analysis.terms else 1.0

    @staticmethod
    def _top_source_coverage(ranked, sources, analysis, include_expansions):
        if not ranked or not sources:
            return 0.0
        terms = set(ranked[0][0]["tf"])
        targets = defaultdict(set)
        if include_expansions:
            for expansion in analysis.expansions:
                targets[expansion.source].add(expansion.target)
        covered = sum(1 for source in sources
                      if source in terms or bool(targets.get(source, set()) & terms))
        return covered / len(sources)

    def definition_terms(self, question, analysis=None):
        if not re.search(r"\b(?:what\s+(?:is|are)|define|definition\s+of|meaning\s+of)\b",
                         question.lower()):
            return ()
        return tuple(analysis.terms) if analysis else tuple(query_terms(question))

    def definition_hit(self, chunk, terms):
        body = chunk["body"].lower()
        return any(re.search(
            rf"\b{re.escape(term)}\s+(?:is\s+(?:a|an|the)|means\b|refers\s+to\b|is\s+defined\s+as\b)",
            body,
        ) for term in terms)

    def candidate_pool(self, question, ranked, analysis=None, limit=RERANK_POOL):
        pool = list(ranked[:limit])
        terms = self.definition_terms(question, analysis)
        if not terms:
            return pool
        seen = {(chunk["relpath"], chunk["ord"]) for chunk, _ in pool}
        extras = []
        for chunk, score in ranked[limit:]:
            key = (chunk["relpath"], chunk["ord"])
            if key not in seen and self.definition_hit(chunk, terms):
                seen.add(key)
                extras.append((chunk, score))
                if len(extras) >= 30:
                    break
        return pool + extras

    def rerank(self, question, candidates, analysis=None):
        qterms = list(analysis.terms) if analysis else query_terms(question)
        if analysis and analysis.rescued:
            qterms += [e.target for e in analysis.expansions]
        qset = set(qterms)
        qtok = [t for t in tokenize(question) if t not in STOPWORDS]
        bigrams = [f"{qtok[i]} {qtok[i + 1]}" for i in range(len(qtok) - 1)]
        definition_terms = self.definition_terms(question, analysis)
        out = []
        for ch, bm in candidates:
            cterms = set(ch["tf"])
            cov = len(qset & cterms) / len(qset) if qset else 0.0
            head_hit = bool(qset & set(stems(ch["heading"])))
            name_hit = bool(qset & set(stems(os.path.basename(ch["relpath"]))))
            body_l = ch["body"].lower()
            phrase_hit = any(bg in body_l for bg in bigrams)
            definition_hit = self.definition_hit(ch, definition_terms) if definition_terms else False
            boost = (1 + 0.6 * cov + 0.4 * head_hit + 0.4 * name_hit +
                     0.6 * phrase_hit + 2.0 * definition_hit)
            out.append((ch, bm * boost))
        out.sort(key=lambda kv: kv[1], reverse=True)
        return out

    def suggest_questions(self, question, analysis, hits, limit=3):
        if not analysis.rescued or not analysis.expansions or not hits:
            return ()
        replacements = {e.source: e.target for e in analysis.expansions}
        words = re.findall(r"[A-Za-z0-9_-]+|[^A-Za-z0-9_-]+", question)
        rewritten = []
        for word in words:
            source = stem(word.lower()) if re.match(r"[A-Za-z0-9_]", word) else None
            target = replacements.get(source)
            if target:
                if word.lower().endswith("s") and not target.endswith("s"):
                    target += "s"
                rewritten.append(target.capitalize() if word[:1].isupper() else target)
            else:
                rewritten.append(word)
        suggestions = []
        rewritten_question = "".join(rewritten).strip()
        if rewritten_question.lower() != question.strip().lower():
            suggestions.append(rewritten_question)

        top = hits[0]
        targets = [e.target for e in analysis.expansions]
        exact = [t for t in analysis.terms if t in top["tf"] and t not in replacements]
        key_terms = []
        for term in targets + exact:
            if term not in key_terms and len(term) > 2:
                key_terms.append(term)
        phrase = " ".join(key_terms[:3])
        heading = (top.get("heading") or "").split(" › ")[-1].strip()
        if heading and phrase:
            suggestions.append(f"What does {heading} say about {phrase}?")

        # Units are read from the matched passage instead of a global list.
        units = []
        for unit in re.findall(r"\b\d+(?:\.\d+)?\s*([A-Za-z°%]{1,6})\b", top["body"]):
            normalized = unit.upper()
            if normalized not in units:
                units.append(normalized)
        if phrase and units:
            suggestions.append(f"What is the {phrase} in {' and '.join(units[:2])}?")

        out = []
        seen = set()
        for suggestion in suggestions:
            suggestion = suggestion[:120].strip()
            key = " ".join(query_terms(suggestion))
            if suggestion and key not in seen:
                seen.add(key)
                out.append(suggestion)
            if len(out) >= limit:
                break
        return tuple(out)

    def gather(self, question, sprint=None, char_budget=None, per_file_cap=None,
               max_chunks=None, ranked=None, analysis=None, items=None):
        cfg = MODE[query_mode(question)]
        nmax = max_chunks if max_chunks is not None else cfg["nmax"]
        ceil = char_budget if char_budget is not None else cfg["ceil"]
        cap = per_file_cap if per_file_cap is not None else cfg["cap"]
        expand, dedup, rel, floor = cfg["expand"], cfg["dedup"], cfg["rel"], cfg["floor"]
        
        if items is None:
            if ranked is None:
                analysis, ranked = self.analyze_and_rank(question, sprint)
            elif analysis is None:
                analysis, _ = self.analyze_and_rank(question, sprint)
            items = self.rerank(question, self.candidate_pool(question, ranked, analysis), analysis=analysis)
        if not items:
            return [], 0
        cutoff = rel * items[0][1]
        seeds, per_file, seen, sel = [], Counter(), set(), []
        for ch, sc in items:
            if len(seeds) >= nmax:
                break
            if len(seeds) >= floor and sc < cutoff:
                break
            if cap and per_file[ch["relpath"]] >= cap:
                continue
            if dedup:
                key = re.sub(r"\s+", " ", ch["body"])[:160].strip().lower()
                if key in seen:
                    continue
                cterms = set(ch["tf"])
                if any(jaccard(cterms, s) > DUP_SIM for s in sel):
                    continue
                seen.add(key)
                sel.append(cterms)
            seeds.append((ch, sc))
            per_file[ch["relpath"]] += 1
        used = self._expand(seeds, expand, ceil) if expand else self._pack(seeds, ceil)
        return used, len({u["relpath"] for u in used})

    def _pack(self, seeds, budget):
        used, total = [], 0
        for ch, sc in seeds:
            if used and total + len(ch["body"]) > budget:
                continue
            used.append({**ch, "score": sc})
            total += len(ch["body"])
        return used

    def _expand(self, seeds, radius, budget):
        raw = defaultdict(list)
        for ch, sc in seeds:
            idxs = self.by_file.get(ch["relpath"])
            if not idxs:
                continue
            lo = max(0, ch["ord"] - radius)
            hi = min(len(idxs) - 1, ch["ord"] + radius)
            raw[ch["relpath"]].append((lo, hi, sc, ch))
        ranges = []
        for rel, spans in raw.items():
            spans.sort(key=lambda s: s[0])
            clo, chi, cscore, cseed = spans[0]
            for lo, hi, sc, seed in spans[1:]:
                if lo <= chi + 1:
                    chi = max(chi, hi)
                    if sc > cscore:
                        cscore, cseed = sc, seed
                else:
                    ranges.append((rel, clo, chi, cscore, cseed))
                    clo, chi, cscore, cseed = lo, hi, sc, seed
            ranges.append((rel, clo, chi, cscore, cseed))
        ranges.sort(key=lambda r: r[3], reverse=True)
        used, total = [], 0
        for rel, lo, hi, score, seed in ranges:
            idxs = self.by_file[rel]
            body = "\n\n".join(self.chunks[idxs[k]]["body"] for k in range(lo, hi + 1))
            if used and total + len(body) > budget:
                continue
            used.append({**seed, "body": body, "score": score})
            total += len(body)
        return used
