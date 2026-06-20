# Plan: Stream Initial Answer, Append Refinement Below

The goal is to show the initial answer immediately via streaming, then—if refinement finds new evidence—append a clearly labeled refined section below it. The user can read the initial answer while refinement happens.

## Current State (and a bug)

In `api_ask_stream`, both the "new_evidence" and "no refinement" branches set `stream = None`, making the token-yielding for-loop dead code that never fires. The entire response is buffered and sent as a single `type: 'answer'` at the end.

## Changes

### 1. `app.py` — `api_ask_stream` (gen function)

#### Stream the initial answer tokens

Replace:
```python
initial_answer = "".join(backend.stream(q, context, len(used), s.ai))
```

With a loop that yields each token immediately:
```python
initial_answer = ""
for tok in backend.stream(q, context, len(used), s.ai):
    initial_answer += tok
    yield sse({"type": "token", "text": tok})
```

Keep the existing guard: `if not initial_answer.strip(): raise ValueError(...)`.

#### After refinement decision — always send compacted initial answer first

After `plan_refinement` returns (and any refinement step messages like "Follow-up search..." and "Found N additional passages..."):

- Compact the initial answer and yield it as `type: 'answer'` so it renders cleanly with markdown.
- Yield its sources (compact_citations filtered set).

This happens in **all** LLM paths, not just the refinement path.

#### Branch A: New evidence found (`reason == "new_evidence"`)

1. Send divider: `yield sse({"type": "answer_divider", "text": "<section-divider>"})` — signals frontend to lock current rendered HTML into an "Initial answer" block and prepare a new section below.

2. Stream refined tokens: call `backend.stream(q, build_context(final_used), len(final_used), s.ai)` in a loop, accumulating into `refined_answer`, yielding each as `type: 'token'`.

3. **Absence check**: If the refined answer is an absence-answer but the initial was not, discard it (fallback to initial only). In this case, do NOT send the divider and do NOT stream refined tokens — fall back before step 1. Move the `absence_answer` check to happen *after* buffering the refined answer with `"".join(backend.stream(...))`, then decide:
   - Rejected → skip divider + streaming entirely. Initial answer already sent. Set `answer_text = initial_answer`, `final_used = used`.
   - Accepted → send divider, stream refined tokens as shown above.

4. After refined tokens finish, compact the refined text and yield it as `type: 'answer'` (second answer block). Yield its sources from `compact_citations(refined_answer, final_used)` — these are a *second* source list, numbered independently starting at 1.

5. Final coverage uses `len(final_used)`. Set `answer_text = initial_answer + "\n\n---\n\n" + refined_answer`.

#### Branch B: No new evidence (all other reasons)

No divider, no refined streaming. The compacted initial answer already covers everything.

`answer_text = initial_answer`, coverage from `len(used)`.

#### Remove dead code

Delete the existing `stream = None` / for-tok loop block entirely — it was never executed and is superseded by the new streaming logic.

#### Done event

`type: 'done'` includes:
- `answer_text`: the final answer (initial + refined if applicable)
- `sources`: sources from compact_citations of the *last* answer segment (refined if present, initial otherwise)
- `coverage`, `mode`, `refinement`, etc. as before

#### Error path

If an exception occurs after initial tokens were streamed:
- If `initial_answer` has content → keep it visible, show error step message, yield compacted initial answer + sources + done with `mode: "extractive-fallback"`. The divider and refined tokens simply never arrive.
- If no initial answer yet → same as current fallback to extractive.

### 2. `templates/index.html` — Frontend

#### State changes

Add state fields:
```js
let initialHtml = "";       // Rendered HTML of the initial answer
let inRefinement = false;   // True after divider received
```

#### Token handling (two modes)

**Mode 1 — Initial answer streaming** (`inRefinement == false`):
```js
else if(m.type === 'token') {
    state.answerText += m.text;
    $('#answer').innerHTML = MD.render(state.answerText);
}
```

**Mode 2 — Refined answer streaming** (`inRefinement == true`):
```js
// Accumulate refined tokens and render *below* the preserved initial block
state.refinedText += m.text;
$('#answer').innerHTML = initialHtml + '<div class="refined-answer">' + MD.render(state.refinedText) + '</div>';
```

#### Divider handling

```js
else if(m.type === 'answer_divider') {
    inRefinement = true;
    initialHtml = $('#answer').innerHTML;  // Lock current rendered HTML
    state.refinedText = "";
    $('#answer').innerHTML =
        '<div class="initial-answer">' + initialHtml + '</div>' +
        '<div class="refining-placeholder"><span class="spin"></span> Refining answer with additional evidence…</div>';
}
```

#### Answer handling (final compacted versions)

Two `type: 'answer'` events can fire. Distinguish by `inRefinement`:

**Before refinement** — first answer (compacted initial):
```js
else if(m.type === 'answer' && !inRefinement) {
    state.answerText = m.text;
    $('#answer').innerHTML = MD.render(state.answerText);
}
```

**After refinement** — second answer (compacted refined text replaces placeholder):
```js
else if(m.type === 'answer' && inRefinement) {
    state.refinedText = m.text;
    $('#answer').innerHTML =
        '<div class="initial-answer">' + initialHtml + '</div>' +
        '<div class="refined-answer"><h3>Refined Answer</h3>' + MD.render(m.text) + '</div>';
}
```

#### Sources handling (two lists)

First `type: 'sources'` → render above the answer or below initial section. Second `type: 'sources'` → append below refined section.

Use a counter to distinguish:

```js
let sourcesCount = 0;
else if(m.type === 'sources') {
    sourcesCount++;
    if(sourcesCount === 1) {
        // Sources for initial answer — render in existing sourcesCard (or a new card)
        renderSources(m.sources);  // Existing function, unchanged behavior
    } else {
        // Sources for refined answer — append to sources or show separately
        renderRefinedSources(m.sources);
    }
}
```

Add `renderRefinedSources` that appends source items to the existing sources list (they naturally have their own [1], [2] numbering scoped to the refined section).

#### CSS additions

```css
.initial-answer { margin-bottom: 8px; }
.refined-answer { border-top: 2px solid var(--accent); padding-top: 14px; }
.refined-answer h3 { font-size: 14px; color: var(--accent); text-transform: uppercase; letter-spacing: .6px; margin: 0 0 10px; }
.refining-placeholder { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 13.5px; padding: 8px 0; }
```

#### wireCites update

After `done`, call `wireCites()` as before — it wires `.cite` links to source list items by `[n]`. Since both sections have independent numbering, citations from the initial section map to the first batch of sources, and refined citations map to the second. This works naturally if sources are in order (initial first, then refined).

## Event Sequence

### With refinement (new evidence accepted)

```
type: step   — "Interpreting question..."
type: step   — "Searched N passages..."
type: step   — "Query mode..."
type: step   — "Collated N passages..."
type: step   — "Built prompt..."
type: step   — "Creating an initial answer..."
type: token  — "The concept of o"       ← streams immediately
type: token  — "wnership is a ..."
...
type: step   — "Checking the answer for unresolved evidence"
type: step   — "Follow-up search: ..."
type: step   — "Found N additional relevant passage(s); refining answer"
type: answer — <compacted initial answer>          ← first complete render
type: sources— <initial sources list>
type: answer_divider — "<section-divider>"         ← locks initial, shows placeholder
type: token  — "Building on the..."                ← streams refined answer below
type: token  — " previous analysis..."
...
type: answer — <compacted refined answer>          ← replaces placeholder
type: sources— <refined sources list>              ← appended to source card
type: done   — final metadata
```

### Without refinement (no new evidence)

```
... tokens stream ...
type: step   — "Checking the answer for unresolved evidence"
type: step   — "No additional evidence needed"
type: answer — <compacted initial answer>
type: sources— <sources list>
type: done   — final metadata
```

### Refined answer rejected (absence detected)

Same as "without refinement" — initial answer stays, no divider appears.

## Tests

### New unit test (mocked, in `test_e2e.py`)

`test_stream_initial_answer_tokens_then_refined_section`: Mock `backend.stream` to return tokens on first call (initial) and different tokens on second call (refined). Verify:
- Token events appear **before** the "Checking the answer" step
- Exactly one `answer_divider` event appears between initial and refined tokens
- Two `type: 'answer'` events appear (compacted versions)
- Two `type: 'sources'` events appear
- Final `done` contains `refinement.refined == true`

### New unit test — refinement rejected

Same mock, but return an absence-answer on the second call. Verify:
- No `answer_divider` event
- Only one `answer` / `sources` pair
- `done.refinement.reason == "refined_answer_rejected"`

### Replace existing live test

Replace `test_stream_hides_draft_until_refinement_finishes` with `test_stream_shows_initial_then_refined`:
- Assert some token events appear before the "additional relevant" step
- Assert an `answer_divider` exists after refinement steps
- Assert refined tokens appear after the divider
- Assert final answer contains content from both phases

### Remove dead test assertion

The existing assertion `self.assertTrue(all(index > found_index for index in output_indexes))` contradicts the new behavior and must be removed with its parent test.
