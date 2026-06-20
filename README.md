# Document Search

A small local search + Q&A app over a folder of your own documents (references, books, specs,
notes, schemas, spreadsheets). Ask a question; it finds the relevant passages and answers from
them — including **global** questions ("what's the role of X?") that need synthesis across many
documents, and **deep "how/explain" questions** that need a full, ordered answer pulled from one
place.

## How it works
- **`input/` → `output/`.** Drop any source files (`.docx`, `.pdf`, `.xlsx`, `.pptx`, `.html`,
  `.md`, `.txt`, …) into **`input/`**. `./convert.sh` (and every `./start.sh`) syncs them into
  **`output/`**: new/changed files are converted, and files you **remove** from `input/` have
  their `output/` counterpart **deleted**. `output/` is the only thing the app reads at runtime.
- **In memory, no database, no embeddings.** At startup `output/` is loaded, structure-aware
  chunked (by markdown heading; tables kept whole with their header repeated), and indexed with
  **BM25**. Each chunk also indexes lightweight metadata (filename, folder, heading path) and the
  corpus's own acronym definitions (harvested from the text, e.g. *WBS → work breakdown
  structure*) so queries match more than raw body text. Light stemming; metadata never pollutes
  the displayed text or the length normalization.
- **Retrieve → rerank → assemble, sized to the question.** BM25 top-150 → a cheap feature
  reranker (term coverage, heading/filename hits, phrase match) → an
  assembly step whose size **adapts to the question**. A rule-based **query mode**
  (`exact` / `how` / `ui` / `global`) picks the *shape*; the *amount* is driven by the relevance
  curve so a one-line fact and a ten-page summary each get what they need:
  - **How much** — passages are kept while their rerank score stays near the top hit's
    (`score ≥ rel × top`), bounded by a `floor` (always send a little) and a `ceil`
    (model-context cap). A concentrated query whose scores cliff after the top hit collapses to a
    couple of passages; a diffuse query whose scores stay high grows toward the ceiling. Same code,
    both extremes.
  - **What shape** — `exact` / `ui` / `global` pack a *diverse* set of excerpts (per-file cap +
    near-duplicate skip), best when the answer is spread across documents. **`how`** (triggered by
    "explain", "steps", "how does", "walk me through", "process", …) switches to **depth**: it
    lifts the per-file cap and the near-duplicate skip, and for each top hit pulls in the
    **neighbouring chunks** so the model receives whole contiguous sections instead of scattered
    fragments — the right shape for a step-by-step or "explain the process" answer.
- **Scope.** You control what's searchable by what you put in `input/`. As a safety net, a small
  skip list (`EXCLUDE` in `corpus.py`) is ignored by `convert.py` and refused by the `/doc` and
  `/wireframe` routes even if dropped in.
- **Pluggable answer backend** (`SEARCH_BACKEND`):
  - `openai` (**default**) — a local/remote **OpenAI-compatible** chat server (llama.cpp,
    LM Studio, vLLM, Ollama). Called over plain HTTP — no SDK.
  - `claude` — the Anthropic API.
  - `extractive` — no LLM; returns the gathered passages.
  - If the LLM backend is unreachable, it falls back to the extractive answer automatically.

## Run (default: local OpenAI-compatible model)
```bash
cd <project dir>
python3 -m pip install -r requirements.txt
./start.sh                     # → http://127.0.0.1:5000
```

### Prerequisites
- **Python 3.8+**
- **Pandoc** (optional but recommended for `.docx` and `.html` conversion).
  - **macOS**: `brew install pandoc`
  - **Ubuntu/Debian**: `sudo apt-get install pandoc`
  - **Windows**: `winget install jgm.pandoc` or download from [pandoc.org](https://pandoc.org/installing.html)

Point it at your server / pick the model:
```bash
OPENAI_BASE_URL=http://localhost:8080/v1 \
SEARCH_MODEL=<served-model-name> \
./start.sh
```
- `OPENAI_BASE_URL` — default `http://localhost:8080/v1`. **Include `/v1`** if your server
  expects it; the app posts to `<base>/chat/completions`.
- `SEARCH_MODEL` — model name sent in the request (default `local`).
- `OPENAI_API_KEY` — only if your server enforces one.

Use Claude instead: `SEARCH_BACKEND=claude ANTHROPIC_API_KEY=sk-ant-... ./start.sh` (needs `pip install anthropic`).

### Configuration (`.env`)
Instead of exporting variables by hand, copy `.env.example` to `.env` and set your endpoint and
key there. `corpus.py` loads `.env` at startup (real environment variables still win), and
`start.sh` / `test.sh` source it too. `.env` is gitignored so your key isn't committed. Common
setups:
- **Local OpenAI-compatible server:** `OPENAI_BASE_URL=http://<host>:8080/v1`, `SEARCH_MODEL=local`, key blank.
- **OpenAI:** `OPENAI_BASE_URL=https://api.openai.com/v1`, `SEARCH_MODEL=gpt-4o-mini`, `OPENAI_API_KEY=sk-...`.
- **OpenRouter:** `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, `SEARCH_MODEL=anthropic/claude-3.5-sonnet`, `OPENAI_API_KEY=sk-or-...`.

### Settings and AI Backend
You can configure the AI backend, endpoint, and model directly from the GUI using the **Settings** button.
- **Persistence**: Settings are saved to `settings.json`.
- **Precedence**: Environment variables (e.g., `SEARCH_BACKEND`) always take precedence over GUI settings.
- **Security**: API keys are never returned to the browser or logged.

### Syncing Documents
Instead of restarting the app, you can now sync your `input/` folder while the app is running:
- Use the **Sync documents** button (to be added to GUI) or call `POST /api/corpus/sync`.
- The app will convert new/changed files and prune removed ones atomically.
- You can monitor progress via `GET /api/corpus/status`.
- If a sync fails, the previous searchable corpus remains active.

### OCR Warning
If a PDF contains no extractable text, the sync status will flag it as requiring OCR.

### Tuning retrieval (env vars)
- Behaviour is **mode-driven** (`MODE` in `corpus.py`). Each mode sets the relevance cutoff
  (`rel`, fraction of the top score below which passages are dropped), the minimum/maximum passage
  count (`floor` / `nmax`), the char ceiling (`ceil`), the per-file cap (`cap`, `0` = unlimited),
  the neighbour-expansion radius (`expand`, `0` = off), and near-duplicate skipping (`dedup`).
  Lower `rel` and higher `ceil` widen an answer; higher `rel` and lower `nmax` tighten it. The
  `gather()` overrides `char_budget` / `per_file_cap` / `max_chunks` map to `ceil` / `cap` / `nmax`.
- `SEARCH_RERANK_POOL` — BM25 candidates fed to the reranker (default `150`).
- `SEARCH_META_WEIGHT` (default `3`) — weight of filename/heading metadata in the index.
- `SEARCH_ALIAS_WEIGHT` (default `0.4`) — weight of acronym-expansion terms.
- `SEARCH_DUP_SIM` (default `0.7`) — Jaccard above which a chunk is skipped as a near-duplicate
  (only in modes where `dedup` is on).
- `PORT` (default `5000`), `SEARCH_BACKEND` (`openai` | `claude` | `extractive`).

## UI
- Press **Enter** to ask; **Shift+Enter** for a newline.
- While answering, a live **Working** panel shows each step — interpreting the question,
  how many passages matched, which files were collated (with the list), then synthesizing —
  followed by the answer streaming in token by token, with clickable `[n]` citations (grouped
  citations like `[1, 3]` are each clickable; a stray number out of range stays plain text).
- **Relevant UI screens** — if your corpus includes HTML wireframes/screens, a question that
  relates to them shows a card listing the matching screens with an **open wireframe ↗** link that
  serves the original `.html` (route `/wireframe/<path>`, scoped to `output/`). Search-only results
  link wireframes the same way. Questions unrelated to screens surface none.
- **Clickable references** — every source `[n]` path (and every search match) is a link that opens
  the converted document in a new tab (`/doc/<path>`, scoped to `output/`; excluded paths refused),
  rendered by the **same client-side renderer as the answer** (`render.js`) — one markdown
  renderer, so the two views never drift. `.txt`/`.csv` references show verbatim.
- **Answer respects format/length.** Ask "table only", "one line", or "bullets" and the model
  produces exactly that; ask to "explain the steps" or "walk me through" a process and it gives a
  thorough, ordered answer; otherwise it gives a concise synthesis. Obvious format/filler words
  (`only`, `nothing`, `concise`, `bullets`, `pls`, …) are dropped from the search query so they
  don't pull table-heavy or off-topic chunks. The streamed answer renders markdown — **tables**,
  headings, lists, bold, inline code — not raw text.
- **Search only** shows ranked matches without calling the model.
- **Query rescue** retries weak searches with bounded corpus-aware spelling correction and
  document-derived acronym expansion. The UI
  shows changes such as `tyre → tire` and offers grounded, clickable alternative questions.
- **Adaptive refinement** reviews every synthesized answer for unresolved, evidence-grounded
  concepts and cross-references. When an excerpt says, for example, that one term is the same as
  another, the app searches for the referenced term, adds genuinely new passages, and regenerates
  the answer once. Proposed searches are restricted to terms already present in the question or
  retrieved excerpts; absent subjects are reported as absent rather than guessed semantically.

## Tests
```bash
python3 -m pip install -r requirements-dev.txt
./test.sh
```
`test.sh` runs locally without contacting the application's configured provider. Live model tests
are opt-in and use a separate endpoint exclusively:
```bash
RUN_LIVE_AI_TESTS=1 \
AI_TEST_BASE_URL=http://10.0.0.10:8080/v1 \
AI_TEST_MODEL=local \
./test.sh
```
`AI_TEST_API_KEY` is available when the dedicated test server requires authentication. The live
tests never fall back to `OPENAI_BASE_URL`, the GUI settings, or their API key. No server is
started by the test suite.
- `test_corpus.py` — corpus-independent unit tests on synthetic fixtures (tokenize, query mode,
  chunking, BM25 ranking, exclusion, the dynamic budget: section expansion, deep-vs-factual sizing,
  and the `gather()` overrides) + template guards.
- `test_render.js` — corpus-independent unit tests for `render.js` under Node (tables, headings,
  lists, bold, and citation gating) — the shared renderer the Python suite can't reach. `test.sh`
  runs it when `node` is present.
- `test_e2e.py` / `test_eval.py` / `eval.py` — run against **bundled, downloaded corpora** in
  `tests/big_corpus` (13 markdown chapters) and `tests/small_corpus` (3 short files), so they're
  deterministic and independent of whatever you've loaded into `output/`. `test_e2e.py` checks
  ranking, the deep-vs-factual budget contrast, section expansion, corpus-derived topics, and the
  doc-serving guards; `eval_set.json` + `test_eval.py` are a 15-query retrieval benchmark
  (`python3 eval.py` prints recall/MRR/coverage/avg-chars; the test fails the build if they
  regress). The two answer tests call only `AI_TEST_BASE_URL` and skip unless explicitly enabled
  and that endpoint is reachable.

## Adding / updating / removing documents
Everything is driven by the `input/` folder:
```bash
# add a file
cp my-new-spec.docx "input/Some Folder/"
./convert.sh            # → converts it into output/

# update a file → edit it in input/, then ./convert.sh  (reconverts just that one)
# remove a file → delete it from input/, then ./convert.sh  (deletes its output/ counterpart)
./convert.sh --force    # reconvert everything regardless of dates
```
`convert.sh` is a **full sync**: each `output/` file mirrors its `input/` source's name and
modification time, so only missing/out-of-date files are converted; any `output/` file whose
source no longer exists is pruned (and empty folders removed). `./start.sh` runs the same sync
on every launch, so a normal restart already reflects whatever you changed in `input/`.

Converters: `pandoc` (.docx/.html), openpyxl/pandas (.xlsx/.xls), markitdown (.pptx),
extract-msg (.msg), PyMuPDF (.pdf); `.csv/.json/.md/.txt` are copied. `convert.sh` installs the
Python ones if missing and warns if `pandoc` is absent.

## Files
- `input/` — **you** put source documents here (any format). The one folder you manage.
- `output/` — generated: converted `.md`/`.txt` + raw wireframe `.html`. The only thing the app reads.
- `app.py` — Flask backend (`/api/search`, `/api/ask`, `/api/ask_stream`, `/api/meta`, `/doc`, `/wireframe`).
- `corpus.py` — in-memory corpus: structure-aware chunking, metadata/acronym indexing, BM25,
  reranker, query modes, diversity packing and section expansion, screen selection.
- `convert.py` — `input/`→`output/` sync (convert/copy, mtime mirroring, prune).
- `tests/big_corpus/`, `tests/small_corpus/` — bundled markdown corpora the test suite runs against.
- `eval.py` / `eval_set.json` — retrieval benchmark over `tests/big_corpus`.
- `render.js` — the single client-side markdown renderer (tables, headings, lists, bold, code,
  citations), served at `/render.js` and used by both the answer and the `/doc` view.
- `templates/index.html` — single-page UI, no external dependencies (works offline).
- `start.sh` — sync + launch. `convert.sh` — sync only. `test.sh` — prereqs + tests.
</content>
</invoke>
