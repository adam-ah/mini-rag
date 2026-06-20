# Two-Pass Query Rescue and Suggested Questions

## Goal

Make natural questions retrieve the right document passages even when the user uses different spelling, terminology, inflection, or units from the source documents.

The motivating regression is:

- `what pressure do my tyres need?` currently ranks unrelated engine-pressure passages.
- `tyre psi?` finds the correct motorcycle tyre specification.
- The documents use `tire`, while the user uses `tyre`.

After this work, the first query must retrieve the passage containing the cold front and rear pressures: 36 PSI front and 42 PSI rear.

## Constraints

- Preserve BM25 as the primary retrieval mechanism.
- Do not call an AI model on every search.
- Do not silently replace valid user terms.
- Expand only unmatched or demonstrably weak terms.
- Keep search deterministic and fast on the current in-memory corpus.
- Expose expansions to the user instead of hiding them.
- Do not add organization-specific synonym constants to `corpus.py`.
- Existing retrieval evaluation scores must not regress.

## Desired user experience

For a rescued query, show a small message above results or in the Working panel:

> Expanded search: `tyre` → `tire`

If retrieval remains weak, show clickable alternatives:

- What are the front and rear tire pressures?
- What tire pressure should be used when cold?
- Show tire pressure in PSI and kPa.

Clicking an alternative should place it in the question box and run the search or answer flow.

## Architecture

Add a query-analysis layer between `query_terms()` and BM25 scoring.

```text
User question
    │
    ▼
Normalize and identify exact corpus terms
    │
    ├── Strong first pass ───────────────► use original ranking
    │
    └── Weak/OOV terms
            │
            ▼
       Generate bounded expansions
            │
            ▼
       Second BM25 pass + rerank
            │
            ├── Improved ────────────────► use rescued ranking
            └── Still weak ──────────────► results + suggested questions
```

Introduce explicit result objects rather than passing anonymous term lists:

```python
@dataclass(frozen=True)
class QueryExpansion:
    source: str
    target: str
    reason: str       # "spelling", "alias", "unit", "inflection"
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
```

Keep compatibility wrappers temporarily if changing every caller at once is too risky.

## Phase 1: Add regression fixtures before changing retrieval

Files:

- Add a small deterministic vehicle fixture under `tests/query_rescue_corpus/`.
- Add `test_query_rescue.py`.
- Extend `test_browser.py`.

Fixture contents should include:

1. A tyre/tire section containing:
   - `Tire air pressure (measured on cold tires)`
   - `Front: 250 kPa (36 psi)`
   - `Rear: 290 kPa (42 psi)`
2. Several distractors containing:
   - Engine oil pressure.
   - Fuel injector pressure.
   - Compression pressure.

Tests to write first:

1. Demonstrate the current failure: the natural British-English question does not rank the tyre section first.
2. Require these queries to rank the tyre section first after implementation:
   - `what pressure do my tyres need?`
   - `recommended tyre pressure`
   - `tire pressure`
   - `tyre psi`
3. Require the selected answer context to contain both `36 psi` and `42 psi`.
4. Require unrelated pressure queries to remain correct:
   - `fuel injector pressure procedure`
   - `engine oil pressure limit`
5. Add a benchmark row to `eval_set.json` only if it can use a stable bundled corpus. Otherwise keep a separate rescue benchmark.

Do not weaken assertions to merely require the expected passage somewhere in the top 20. Require rank 1 for this focused fixture.

## Phase 2: Build a corpus vocabulary index

File:

- `corpus.py`

Tasks:

1. After loading chunks, retain corpus-level term statistics already available in `self.df`.
2. Build lookup structures for correction candidates:
   - Terms grouped by length.
   - Terms grouped by first and last character.
   - Optional deletion signatures for edit-distance-one matching.
3. Exclude unsuitable correction targets:
   - Pure numbers.
   - Very short terms below four characters, except configured aliases/units.
   - Extremely rare OCR garbage.
   - Terms containing mostly punctuation or malformed OCR artifacts.
4. Give useful document terms preference over metadata-only terms. Track body document frequency separately if required.
5. Keep the structure immutable after corpus construction so concurrent searches remain safe.

Acceptance criteria:

- Candidate lookup does not scan the entire vocabulary for every query term.
- Corpus loading remains acceptably fast on the current manuals.
- Search latency does not materially change for queries where all terms match exactly.

## Phase 3: Implement bounded OOV correction

File:

- `corpus.py`, preferably in a small `QueryAnalyzer` class.

OOV means a normalized query term does not exist in the corpus vocabulary.

Algorithm:

1. Normalize the question with existing tokenization and stemming.
2. Separate exact corpus terms from OOV terms.
3. For each OOV term of at least four characters, find candidates within a bounded edit distance:
   - Length 4–7: maximum edit distance 1.
   - Length 8 or more: maximum edit distance 2, with a strong similarity threshold.
4. Rank candidates using:
   - Lower edit distance.
   - Similar prefix/suffix.
   - Higher body document frequency.
   - Co-occurrence with the exact query terms.
5. Accept a correction only when:
   - The best candidate exceeds a minimum score.
   - It is clearly better than the second candidate.
   - It occurs in the corpus body, not only a filename.
6. Keep the original term and add the correction at a lower weight, for example:
   - Original exact term: `1.0`.
   - Accepted spelling correction: `0.75`.
7. Limit corrections to two per query.

For the motivating query, `tyre` should be OOV and resolve to `tire` with edit distance one.

Do not use unrestricted fuzzy matching for terms already in the corpus. If both `form` and `from` exist, the system must respect the user’s exact term.

Tests:

- `tyre → tire` is accepted.
- An exact `tyre` corpus term prevents replacement.
- Ambiguous corrections are rejected.
- OCR garbage does not become a suggestion.
- Short terms such as `AI`, `ID`, and `PSI` are not fuzzy-corrected.
- Correction count and weights are bounded.

## Phase 4: Derive expansions from the corpus

Use only runtime evidence from the indexed corpus:

1. Corpus-derived acronym definitions already collected by `Corpus`.
2. Bounded OOV spelling candidates from the corpus vocabulary.
3. Measurement tokens extracted from matched passages for question suggestions.

Rules:

- Do not maintain a synonym dictionary in code, JSON, or settings.
- Unit suggestions must come from matched text, not a global unit list.
- Record the expansion reason in `QueryExpansion`.

The `tyre → tire` regression passes through corpus-driven OOV correction.

## Phase 5: Define weak retrieval and run a second pass

Do not run rescue merely because the user used an OOV word. Measure whether it improves retrieval.

First-pass signals:

- Number of query terms present in the vocabulary.
- Query-term coverage in the top hit.
- Top-hit BM25/rerank score.
- Score gap between the first and following hits.
- Whether top results agree on a document/heading.

Suggested initial policy:

Run a rescue pass when at least one meaningful query term is OOV and either:

- Fewer than half the meaningful concepts are represented in the top hit, or
- The top results are diffuse across unrelated documents/headings.

Implementation:

1. Produce the original ranking.
2. Generate accepted expansions.
3. Produce an expanded ranking.
4. Compare both rankings using term coverage and rerank score.
5. Use the expanded ranking only if it improves the objective score by a documented margin.
6. Store `rescued=True` only when the expanded ranking is selected.

Do not compare raw BM25 scores across differently sized queries without normalization. Prefer coverage and rank-quality features.

Tests:

- The tyre query chooses the rescued ranking.
- A strong exact query keeps its original ranking.
- A proposed correction that worsens coverage is rejected.
- Search and answer gathering use the same selected analysis/ranking.

## Phase 6: Prevent `search()` and `gather()` from disagreeing

Current search and answer retrieval can independently rank the same question. Refactor so one analyzed ranking can be reused.

Tasks:

1. Add a method such as:

```python
analysis, ranked = corpus.analyze_and_rank(question, sprint=None)
```

2. Let `search()` accept or produce this analysis.
3. Let `gather()` accept the same ranked results and analysis.
4. In `/api/ask_stream`, analyze once and reuse the ranking for Working steps and context assembly.
5. Ensure reranking occurs exactly once per candidate set.

Acceptance criteria:

- The UI cannot show a rescued search while the answer uses the unexpanded query.
- Search-only and Ask rank the same first passage for the same query and filter.

## Phase 7: Add transparent API metadata

Update API responses without breaking existing consumers.

`GET /api/search` response addition:

```json
{
  "query": "what pressure do my tyres need?",
  "expansions": [
    {"source": "tyre", "target": "tire", "reason": "spelling"}
  ],
  "rescued": true,
  "suggestions": [
    "What are the front and rear tire pressures?"
  ],
  "hits": []
}
```

Streaming additions:

- Emit a `query_expansion` event after analysis and before collating passages.
- Include expansions and suggestions in the final `done` event.
- Example Working step: `Expanded search: tyre → tire`.

Never claim that a spelling correction is certain. The wording should say “Expanded search,” not “Corrected your mistake.”

## Phase 8: Generate bounded suggested questions

Suggestions should be grounded in retrieved passages, not invented freely.

Generate suggestions only when:

- Retrieval used a rescue expansion, or
- Retrieval remains weak after rescue.

Deterministic sources:

1. Accepted query expansion plus the original intent.
2. Headings from the top passages.
3. High-value units or entities present in top passages, such as PSI and kPa.
4. Existing corpus-derived document/topic metadata.

Rules:

- Return at most three suggestions.
- Keep each under 120 characters.
- Do not include suggestions that normalize to the same term set.
- Do not include facts not present in retrieved passages.
- Avoid raw OCR fragments and incomplete headings.
- Prefer question templates appropriate to the original intent:
  - `what` → “What is …?”
  - `how` → “How do I …?”
  - `compare` → “Compare …”

Optional later enhancement:

- If deterministic rescue still has no usable results and an AI backend is configured, ask the model for up to three search rewrites using only corpus vocabulary/headings supplied in the prompt.
- Keep this disabled by default.
- Validate model output and rerun retrieval before showing a suggestion.
- Never treat an unverified model rewrite as an answer.

## Phase 9: Update the GUI

Files:

- `templates/index.html`
- `test_browser.py`
- Possibly extract search UI code into a dedicated JavaScript file if the template becomes difficult to test.

Tasks:

1. Add a compact expansion notice to Search results and the Working panel.
2. Add a “Try asking” section containing suggestion buttons.
3. Clicking a suggestion should:
   - Set the textarea value.
   - Start the same workflow the user originally selected where practical.
4. Clear old expansions and suggestions at the start of every new request.
5. Escape all expansion and suggestion strings before inserting them into HTML.
6. Keep the original user question visible; do not replace it with the rewritten form.

Browser E2E cases:

1. Search for `what pressure do my tyres need?`.
2. Assert an expansion notice contains `tyre → tire`.
3. Assert the first source is the tyre-pressure fixture.
4. Assert results contain 36 PSI and 42 PSI.
5. Assert at least one grounded suggested question is shown.
6. Click a suggestion and assert a new request is issued.
7. Run Ask and assert Working contains the expansion step.
8. Assert the streamed answer cites the tyre-pressure source.
9. Assert there are no page errors or console errors.

Use a deterministic mocked streaming backend, as the existing browser test does. Do not depend on a live external AI service.

## Phase 10: Performance and safety limits

Add instrumentation or tests for:

- Query-analysis duration.
- Number of fuzzy candidates examined.
- Whether rescue ran.
- Original and rescued top-document identifiers.

Limits:

- Maximum 20 meaningful input terms.
- Maximum two fuzzy corrections.
- Maximum five candidates evaluated per OOV term after indexed filtering.
- Maximum three suggestions.
- No recursive rewriting.
- One rescue pass only.

Avoid logging entire private questions by default. Debug logging may report counts and expansion pairs.

## Phase 11: Test matrix

### Unit tests

- Vocabulary construction.
- OOV identification.
- Edit-distance candidate selection.
- Ambiguity rejection.
- Alias weighting.
- Unit expansion rules.
- Weak-retrieval detection.
- Original-versus-rescued ranking selection.
- Suggestion deduplication and length limits.

### Retrieval integration tests

- British/American spelling.
- Singular/plural mismatch.
- Acronym/expanded-form mismatch.
- Unit/name mismatch.
- Distractor-heavy documents.
- Exact queries that must not be rewritten.

### API tests

- Expansion metadata in search responses.
- Expansion event in streamed answers.
- Same ranking used by search and gather.
- Empty and no-rescue queries retain the existing response shape.

### Browser E2E tests

- Rescued Search-only workflow.
- Rescued streamed Ask workflow.
- Suggested-question click workflow.
- No browser page or console errors.

### Regression tests

Run:

```bash
python3 -m unittest -v \
  test_corpus test_e2e test_eval test_query_rescue \
  test_settings test_ui_contract test_browser
node test_stream.js
node test_render.js
python3 eval.py
```

## Recommended implementation sequence

1. Add the focused corpus and failing regression tests.
2. Add vocabulary lookup structures.
3. Implement OOV correction and weighted expansions.
4. Add first-pass/rescue-pass selection.
5. Refactor search and gather to share one analyzed ranking.
6. Add API expansion metadata and streaming events.
7. Add deterministic grounded suggestions.
8. Add GUI notices and suggestion buttons.
9. Add headless-Chrome E2E coverage.
10. Measure performance and run the complete regression suite.

## Definition of done

- `what pressure do my tyres need?` ranks the cold tyre-pressure passage first.
- Its answer context contains 36 PSI front and 42 PSI rear.
- The UI reports `tyre → tire` transparently.
- Search-only and Ask use the same rescued ranking.
- Up to three grounded alternative questions are offered when useful.
- Exact strong queries are not unnecessarily expanded.
- Distractor pressure queries continue finding engine/injector material correctly.
- Query rescue adds no model dependency and no material latency to normal exact searches.
- Unit, retrieval, API, and real headless-browser E2E tests pass without console errors.
