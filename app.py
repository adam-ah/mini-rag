#!/usr/bin/env python3
import os, re, json, time, urllib.request, sys
from flask import Flask, request, jsonify, Response, stream_with_context, abort
from corpus import Corpus, OUTPUT, RERANK_POOL, is_excluded, query_mode
from settings import settings_service, AISettings
import backend
import ingest
from dataclasses import asdict, dataclass

DOC_EXTS = (".md", ".txt", ".csv", ".json")

HERE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CSRF_TOKEN = os.urandom(16).hex()
CORPUS = [Corpus().load()]
coordinator = ingest.IngestionCoordinator(CORPUS)

def get_corpus():
    return CORPUS[0]

def build_context(used):
    out = []
    for i, h in enumerate(used, 1):
        loc = h["relpath"] + (f" › {h['heading']}" if h.get("heading") else "")
        tag = " [UI wireframe]" if h.get("html") else ""
        out.append(f"[{i}] ({loc}){tag}\n{h['body'].strip()}\n")
    return "\n".join(out)


def user_prompt(question, context, n):
    return (
        f"Question: {question}\n\n"
        f"Excerpts ({n} of them, from across the reference material):\n{context}\n\n"
        "Answer the question using the excerpts above, honoring any format or length the user "
        "asked for. Cite [n] where you draw on an excerpt."
    )


def extractive_answer(used):
    lines = ["_Extractive answer — the most relevant passages, not a synthesized response. "
             "Configure an LLM backend (or fix its endpoint) for a written answer._\n"]
    for i, h in enumerate(used[:6], 1):
        snippet = re.sub(r"\n{3,}", "\n\n", h["body"].strip())
        if len(snippet) > 700:
            snippet = snippet[:700].rsplit(" ", 1)[0] + " …"
        lines.append(f"**[{i}] {h['relpath']}**\n\n{snippet}")
    return "\n\n".join(lines)


def preview(body):
    return re.sub(r"\s+", " ", body).strip()[:280]


def sources_of(used):
    return [{"n": i + 1, "relpath": h["relpath"], "sprint": h["sprint"],
             "preview": preview(h["body"]), "html": h.get("html")}
            for i, h in enumerate(used)]


def compact_citations(answer, used):
    """Keep cited excerpts only and renumber citations for this response from one."""
    cited = []
    for match in re.finditer(r"\[(\d+)\]", answer or ""):
        old = int(match.group(1))
        if 1 <= old <= len(used) and old not in cited:
            cited.append(old)
    if not cited:
        return answer, sources_of(used)
    mapping = {old: new for new, old in enumerate(cited, 1)}
    normalized = re.sub(
        r"\[(\d+)\]",
        lambda match: f"[{mapping[int(match.group(1))]}]"
        if int(match.group(1)) in mapping else match.group(0),
        answer,
    )
    selected = [used[old - 1] for old in cited]
    return normalized, sources_of(selected)


def expansion_data(analysis):
    if not analysis.rescued:
        return []
    return [{"source": e.source, "target": e.target, "reason": e.reason}
            for e in analysis.expansions]


@dataclass(frozen=True)
class RefinementPlan:
    reflection: object = None
    queries: tuple = ()
    added: tuple = ()
    merged: tuple = ()
    reason: str = "disabled"


def passage_key(passage):
    return (passage.get("relpath"), passage.get("ord"), passage.get("body"))


def validate_reflection_queries(question, used, reflection, corpus):
    """Keep only non-speculative queries grounded in the question or retrieved evidence."""
    original = set(corpus.query_terms(question))
    grounded = set(original)
    for passage in used:
        grounded.update(passage.get("tf", {}))
    accepted, seen = [], set()
    for proposed in reflection.queries:
        terms = tuple(corpus.query_terms(proposed))
        key = " ".join(terms)
        if not terms or key in seen or set(terms) == original:
            continue
        # Every meaningful query term must be traceable to existing evidence. This
        # deliberately rejects guessed synonyms and entities.
        if not set(terms).issubset(grounded):
            continue
        seen.add(key)
        accepted.append(proposed)
    return tuple(accepted)


def plan_refinement(question, sprint, initial_answer, used, corpus, ai_settings):
    if not ai_settings.adaptive_refinement or ai_settings.backend == "extractive":
        return RefinementPlan(merged=tuple(used), reason="disabled")
    reflection_context = build_context(used)
    if len(reflection_context) > ai_settings.reflection_context_budget:
        reflection_context = reflection_context[:ai_settings.reflection_context_budget]
    try:
        reflection = backend.reflect(question, initial_answer, reflection_context, ai_settings)
    except Exception as error:
        print(f"Reflection error: {type(error).__name__}: {error}", file=sys.stderr)
        return RefinementPlan(merged=tuple(used), reason="reflection_failed")
    if reflection.complete:
        return RefinementPlan(reflection=reflection, merged=tuple(used), reason="complete")
    queries = validate_reflection_queries(question, used, reflection, corpus)
    if not queries:
        return RefinementPlan(reflection=reflection, merged=tuple(used), reason="no_valid_queries")

    existing = {passage_key(p) for p in used}
    added, added_chars = [], 0
    per_query_budget = max(1000, ai_settings.refinement_context_budget // len(queries))
    for follow_up in queries:
        analysis, ranked = corpus.analyze_and_rank(follow_up, sprint)
        candidates = corpus.candidate_pool(follow_up, ranked, analysis)
        items = corpus.rerank(follow_up, candidates, analysis=analysis)
        candidates, _ = corpus.gather(
            follow_up, sprint=sprint, analysis=analysis, items=items,
            char_budget=per_query_budget, max_chunks=6,
        )
        follow_terms = set(corpus.query_terms(follow_up))
        for passage in candidates:
            key = passage_key(passage)
            if key in existing or not (follow_terms & set(passage.get("tf", {}))):
                continue
            size = len(passage.get("body", ""))
            if added and added_chars + size > ai_settings.refinement_context_budget:
                continue
            existing.add(key)
            added.append(passage)
            added_chars += size
    if not added:
        return RefinementPlan(reflection=reflection, queries=queries, merged=tuple(used),
                              reason="no_new_evidence")
    return RefinementPlan(reflection=reflection, queries=queries, added=tuple(added),
                          merged=tuple(used) + tuple(added), reason="new_evidence")


def refinement_data(plan):
    reflection = plan.reflection
    return {
        "checked": reflection is not None,
        "refined": plan.reason == "new_evidence",
        "reason": plan.reason,
        "missing_aspects": list(reflection.missing_aspects) if reflection else [],
        "queries": list(plan.queries),
        "added_passages": len(plan.added),
    }


def absence_answer(answer):
    text = re.sub(r"\s+", " ", (answer or "").lower())
    return any(phrase in text for phrase in (
        "do not contain", "does not contain", "don't contain", "not contain an answer",
        "not provided in the excerpts", "insufficient information", "cannot answer from",
        "can't answer from", "no information in the excerpts",
    ))




def validate_settings_payload(data):
    if not isinstance(data, dict):
        raise ValueError("Payload must be a JSON object")
    # Check for unknown fields in AI settings
    ai = data.get("ai", {})
    if isinstance(ai, dict):
        allowed_ai = {"backend", "base_url", "model", "api_key", "temperature", "max_tokens",
                      "timeout_seconds", "adaptive_refinement", "reflection_max_queries",
                      "reflection_max_tokens", "reflection_context_budget",
                      "refinement_context_budget"}
        for k in ai:
            if k not in allowed_ai:
                raise ValueError(f"Unknown AI setting: {k}")
    # Check for unknown fields in Retrieval settings
    ret = data.get("retrieval", {})
    if isinstance(ret, dict):
        allowed_ret = {"exclude_patterns"}
        for k in ret:
            if k not in allowed_ret:
                raise ValueError(f"Unknown retrieval setting: {k}")


@app.get("/api/settings")
def api_settings():
    return jsonify(settings_service.get_public())


@app.put("/api/settings")
def put_settings():
    if request.headers.get("X-CSRF-Token") != CSRF_TOKEN:
        abort(403, "Invalid CSRF token")
    
    data = request.get_json(force=True, silent=True) or {}
    try:
        validate_settings_payload(data)
        
        # Handle partial updates for AI settings
        current_ai = asdict(settings_service.get().ai)
        new_ai = data.get("ai", {})
        if isinstance(new_ai, dict):
            # API key: "omit means keep, explicit null/empty means clear"
            if "api_key" in new_ai:
                current_ai["api_key"] = new_ai["api_key"] or ""
            else:
                # Keep existing key
                pass
            for k, v in new_ai.items():
                if k != "api_key":
                    current_ai[k] = v
        
        # The current GUI does not edit retrieval settings. An empty object must
        # not erase exclusions while saving unrelated AI settings.
        new_ret = data.get("retrieval") or asdict(settings_service.get().retrieval)
        
        settings_service.save(current_ai, new_ret)
        return jsonify({"status": "saved"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Settings save failed: {type(e).__name__}: {e}", file=sys.stderr)
        return jsonify({"error": "Internal server error"}), 500


@app.delete("/api/settings")
def reset_settings():
    if request.headers.get("X-CSRF-Token") != CSRF_TOKEN:
        abort(403, "Invalid CSRF token")
    data = request.get_json(force=True, silent=True) or {}
    if data.get("confirm") != "RESET":
        return jsonify({"error": "Reset confirmation required"}), 400
    try:
        settings_service.reset()
        return jsonify({"status": "reset"})
    except Exception as e:
        print(f"Settings reset failed: {type(e).__name__}: {e}", file=sys.stderr)
        return jsonify({"error": "Internal server error"}), 500


@app.post("/api/settings/test")
def test_settings():
    data = request.get_json(force=True, silent=True) or {}
    ai_data = data.get("ai", {})
    if not isinstance(ai_data, dict):
        return jsonify({"error": "Invalid AI settings"}), 400
    
    try:
        test_ai = AISettings(**ai_data)
        ok, msg = backend.test_connection(test_ai)
        return jsonify({"ok": ok, "message": msg})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.get("/api/settings/models")
def api_models():
    s = settings_service.get()
    if s.ai.backend != "openai":
        return jsonify({"error": "Model discovery only supported for OpenAI-compatible backends"}), 400
    try:
        req = urllib.request.Request(s.ai.base_url + "/models", method="GET")
        if s.ai.api_key:
            req.add_header("Authorization", f"Bearer {s.ai.api_key}")
        with urllib.request.urlopen(req, timeout=2.0) as r:
            data = json.loads(r.read().decode("utf-8"))
            models = [m["id"] for m in data.get("data", [])]
            return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/corpus/sync")
def api_sync():
    if request.headers.get("X-CSRF-Token") != CSRF_TOKEN:
        abort(403, "Invalid CSRF token")
    ok, msg = coordinator.sync()
    return jsonify({"ok": ok, "message": msg})


@app.get("/api/corpus/status")
def api_status():
    status = coordinator.get_status()
    res_data = None
    if status.result:
        res_data = {
            "converted": status.result.converted,
            "copied": status.result.copied,
            "skipped": status.result.skipped,
            "failed": status.result.failed,
            "pruned": status.result.pruned,
            "errors": [{"relpath": e.relpath, "message": e.message, "status": e.status} for e in status.result.errors]
        }
    return jsonify({
        "state": status.state,
        "message": status.message,
        "start_time": status.start_time,
        "end_time": status.end_time,
        "result": res_data
    })
def reachable():
    s = settings_service.get()
    ok, _ = backend.test_connection(s.ai)
    return ok


@app.get("/api/meta")
def api_meta():
    s = settings_service.get()
    corpus = get_corpus()
    meta = {"chunks": corpus.N, "files": corpus.files,
            "sprints": sorted({c["sprint"] for c in corpus.chunks}),
            "documents": corpus.documents(), "topics": corpus.topics(),
            "backend": s.ai.backend, "model": s.ai.model}
    if s.ai.backend == "openai":
        meta["endpoint"] = s.ai.base_url
        meta["reachable"] = reachable()
    return jsonify(meta)


@app.get("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip()
    sprint = request.args.get("sprint") or None
    if not q:
        return jsonify({"hits": []})
    result = get_corpus().search_result(q, limit=20, sprint=sprint)
    return jsonify({"query": q, "rescued": result.analysis.rescued,
                    "expansions": expansion_data(result.analysis),
                    "suggestions": list(result.suggestions), "hits": [
        {"relpath": h["relpath"], "sprint": h["sprint"], "preview": preview(h["body"]),
         "html": h.get("html")} for h in result.hits]})


@app.post("/api/ask")
def api_ask():
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("q") or "").strip()
    sprint = data.get("sprint") or None
    if not q:
        return jsonify({"error": "empty question"}), 400
    corpus = get_corpus()
    analysis, ranked = corpus.analyze_and_rank(q, sprint)
    items = corpus.rerank(q, corpus.candidate_pool(q, ranked, analysis), analysis=analysis)
    used, n_files = corpus.gather(q, sprint=sprint, analysis=analysis, items=items)
    if not used:
        return jsonify({"answer": "No matching passages found.", "sources": [], "mode": "none"})
    context = build_context(used)
    s = settings_service.get()
    if s.ai.backend == "extractive":
        answer, mode = extractive_answer(used), "extractive"
        final_used = used
        refinement = RefinementPlan(merged=tuple(used), reason="disabled")
    else:
        try:
            initial_answer = backend.answer(q, context, len(used), s.ai)
            if not initial_answer.strip():
                raise ValueError("model returned no content")
            refinement = plan_refinement(q, sprint, initial_answer, used, corpus, s.ai)
            final_used = list(refinement.merged)
            answer = initial_answer
            if refinement.reason == "new_evidence":
                try:
                    candidate = backend.answer(q, build_context(final_used), len(final_used), s.ai)
                    if absence_answer(candidate) and not absence_answer(initial_answer):
                        final_used = used
                        refinement = RefinementPlan(
                            reflection=refinement.reflection, queries=refinement.queries,
                            added=refinement.added, merged=tuple(used),
                            reason="refined_answer_rejected",
                        )
                    else:
                        answer = candidate
                except Exception as error:
                    print(f"Refined answer error: {type(error).__name__}: {error}", file=sys.stderr)
                    refinement = RefinementPlan(
                        reflection=refinement.reflection, queries=refinement.queries,
                        added=refinement.added, merged=refinement.merged,
                        reason="refined_answer_failed",
                    )
            mode = f"{s.ai.backend}:{s.ai.model}" + (f" @ {s.ai.base_url}" if s.ai.backend == "openai" else "")
        except Exception as e:
            print(f"Backend error: {type(e).__name__}: {e}", file=sys.stderr)
            answer = extractive_answer(used) + f"\n\n_({s.ai.backend} backend failed: {type(e).__name__}: {e})_"
            mode = "extractive-fallback"
            final_used = used
            refinement = RefinementPlan(merged=tuple(used), reason="answer_failed")
    suggestions = corpus.suggest_questions(q, analysis,
        tuple({**ch, "score": score} for ch, score in items[:20]))
    answer, sources = compact_citations(answer, final_used)
    return jsonify({"answer": answer, "sources": sources,
                    "screens": corpus.relevant_screens(q, sprint),
                    "mode": mode, "coverage": {"chunks": len(final_used),
                                                 "files": len({u['relpath'] for u in final_used})},
                    "rescued": analysis.rescued, "expansions": expansion_data(analysis),
                    "suggestions": list(suggestions), "refinement": refinement_data(refinement)})


def sse(obj):
    return "data: " + json.dumps(obj) + "\n\n"


@app.post("/api/ask_stream")
def api_ask_stream():
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("q") or "").strip()
    sprint = data.get("sprint") or None
    if len(q) > 10000:
        return jsonify({"error": "question too long"}), 400

    def gen():
        if not q:
            yield sse({"type": "error", "message": "empty question"})
            return
        corpus = get_corpus()
        terms = corpus.query_terms(q)
        yield sse({"type": "step", "label": f"Interpreting question — key terms: {', '.join(terms) or '(none)'}"})
        analysis, ranked = corpus.analyze_and_rank(q, sprint)
        expansions = expansion_data(analysis)
        if expansions:
            yield sse({"type": "query_expansion", "expansions": expansions})
            pairs = ", ".join(f"{e['source']} → {e['target']}" for e in expansions)
            yield sse({"type": "step", "label": f"Expanded search: {pairs}"})
        yield sse({"type": "step",
                   "label": f"Searched {corpus.N:,} passages in {corpus.files} files — {len(ranked)} matched"})
        yield sse({"type": "step", "label": f"Query mode: {query_mode(q)} (reranking, then diverse selection)"})
        items = corpus.rerank(q, corpus.candidate_pool(q, ranked, analysis), analysis=analysis)
        used, n_files = corpus.gather(q, sprint=sprint, analysis=analysis, items=items)
        hit_views = tuple({**ch, "score": score} for ch, score in items[:20])
        suggestions = corpus.suggest_questions(q, analysis, hit_views)
        if suggestions:
            yield sse({"type": "suggestions", "suggestions": list(suggestions)})
        if not used:
            yield sse({"type": "sources", "sources": []})
            yield sse({"type": "answer", "text": "No matching passages found in the reference material."})
            yield sse({"type": "done", "mode": "none", "coverage": {"chunks": 0, "files": 0},
                       "rescued": analysis.rescued, "expansions": expansions,
                       "suggestions": list(suggestions)})
            return
        files = sorted({u["relpath"] for u in used})
        yield sse({"type": "step", "label": f"Collated {len(used)} passages across {n_files} files", "files": files})
        screens = corpus.relevant_screens(q, sprint)
        if screens:
            yield sse({"type": "screens", "screens": screens})
            yield sse({"type": "step",
                       "label": f"Relevant UI wireframe(s): {', '.join(s['name'] for s in screens)}"})
        context = build_context(used)
        coverage = {"chunks": len(used), "files": n_files}
        s = settings_service.get()
        if s.ai.backend == "extractive":
            answer, sources = compact_citations(extractive_answer(used), used)
            yield sse({"type": "step", "label": "No LLM backend — returning the gathered passages"})
            yield sse({"type": "answer", "text": answer})
            yield sse({"type": "sources", "sources": sources})
            yield sse({"type": "done", "mode": "extractive", "coverage": coverage,
                       "rescued": analysis.rescued, "expansions": expansions,
                       "suggestions": list(suggestions),
                       "refinement": refinement_data(RefinementPlan(merged=tuple(used), reason="disabled"))})
            return
        chars = len(context)
        target = f"{s.ai.model} @ {s.ai.base_url}" if s.ai.backend == "openai" else f"Claude {s.ai.model}"
        yield sse({"type": "step",
                   "label": f"Built prompt: {len(used)} excerpts, {chars:,} chars (~{chars // 4:,} tokens)"})
        yield sse({"type": "step", "label": f"Creating an initial answer with {target}"})
        t0 = time.time()
        initial_answer = ""
        try:
            # Buffer the draft instead of exposing it. The selected answer is the
            # only one sent to the browser, so refinement never visibly rewrites it.
            initial_answer = "".join(backend.stream(q, context, len(used), s.ai))
            if not initial_answer.strip():
                raise ValueError("model returned no content")
            mode = f"{s.ai.backend}:{s.ai.model}" + (f" @ {s.ai.base_url}" if s.ai.backend == "openai" else "")
            yield sse({"type": "step", "label": "Checking the answer for unresolved evidence"})
            refinement = plan_refinement(q, sprint, initial_answer, used, corpus, s.ai)
            final_used = list(refinement.merged)
            if refinement.queries:
                yield sse({"type": "step",
                           "label": "Follow-up search: " + "; ".join(refinement.queries)})
            if refinement.reason == "new_evidence":
                yield sse({"type": "step",
                           "label": f"Found {len(refinement.added)} additional relevant passage(s); refining answer"})
                final_context = build_context(final_used)
                candidate = "".join(backend.stream(q, final_context, len(final_used), s.ai))
                if absence_answer(candidate) and not absence_answer(initial_answer):
                    answer_text = initial_answer
                    final_used = used
                    refinement = RefinementPlan(
                        reflection=refinement.reflection, queries=refinement.queries,
                        added=refinement.added, merged=tuple(used),
                        reason="refined_answer_rejected",
                    )
                else:
                    answer_text = candidate
                stream = None
            else:
                yield sse({"type": "step", "label": "No additional evidence needed"})
                stream = None
                answer_text = initial_answer
            got, nchunks, nchars = False, 0, 0
            if stream is None:
                got = bool(answer_text)
            else:
                for tok in stream:
                    if not got:
                        got = True
                        yield sse({"type": "step",
                                   "label": f"Receiving final response from {s.ai.model} (waited {time.time() - t0:.1f}s)…"})
                    nchunks += 1
                    nchars += len(tok)
                    answer_text += tok
                    yield sse({"type": "token", "text": tok})
            elapsed = time.time() - t0
            if not got:
                yield sse({"type": "answer", "text": "(model returned no content)"})
                yield sse({"type": "step", "label": f"No content returned ({elapsed:.1f}s)"})
            else:
                if stream is not None:
                    yield sse({"type": "step",
                               "label": f"Received {nchars:,} chars in {nchunks:,} chunks ({elapsed:.1f}s)"})
                answer_text, sources = compact_citations(answer_text, final_used)
                yield sse({"type": "answer", "text": answer_text})
                yield sse({"type": "sources", "sources": sources})
            final_coverage = {"chunks": len(final_used),
                              "files": len({u["relpath"] for u in final_used})}
            yield sse({"type": "done", "mode": mode,
                       "rescued": analysis.rescued, "expansions": expansions,
                       "suggestions": list(suggestions), "coverage": final_coverage,
                       "refinement": refinement_data(refinement)})
        except Exception as e:
            fallback = initial_answer or extractive_answer(used)
            answer, sources = compact_citations(fallback, used)
            yield sse({"type": "step",
                       "label": f"{s.ai.backend} backend failed ({type(e).__name__}: {str(e)[:80]}) — "
                                + ("keeping the initial answer" if initial_answer else "falling back to passages")})
            yield sse({"type": "answer", "text": answer})
            yield sse({"type": "sources", "sources": sources})
            yield sse({"type": "done", "mode": "extractive-fallback", "coverage": coverage,
                       "rescued": analysis.rescued, "expansions": expansions,
                       "suggestions": list(suggestions),
                       "refinement": {"checked": bool(initial_answer), "refined": False,
                                      "reason": "refined_answer_failed" if initial_answer else "answer_failed",
                                      "missing_aspects": [], "queries": [], "added_passages": 0}})

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if "text/html" in response.mimetype:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'unsafe-inline' 'self'; "
            "style-src 'unsafe-inline' 'self'; "
            "img-src 'self' data:; "
            "frame-ancestors 'none';"
        )
    return response

def safe_path(base, rel, exts):
    if is_excluded(rel):
        return None
    root = os.path.realpath(base)
    target = os.path.realpath(os.path.join(root, rel))
    if target != root and not target.startswith(root + os.sep):
        return None
    if not target.lower().endswith(exts) or not os.path.isfile(target):
        return None
    return target


def html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


DOC_CSS = """
body{font:15px/1.65 -apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1c2330;background:#f7f8fa;margin:0}
.doc{max-width:880px;margin:0 auto;padding:26px 26px 70px}
.crumb{color:#5b6675;font-size:12.5px;margin-bottom:20px;word-break:break-word}
h1,h2,h3,h4,h5,h6{line-height:1.25;margin:22px 0 8px;color:#1c2330}
h1{font-size:24px}h2{font-size:20px}h3{font-size:17px}h4{font-size:15px}
p{margin:0 0 11px}
a{color:#2563eb}
strong{font-weight:700}
code{background:#eef1f5;border:1px solid #d9dee7;border-radius:5px;padding:1px 5px;font-size:13px}
pre{background:#fff;border:1px solid #d9dee7;border-radius:10px;padding:14px 16px;overflow:auto;
white-space:pre-wrap;word-wrap:break-word;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre code{background:none;border:0;padding:0}
blockquote{margin:0 0 11px;padding:6px 14px;border-left:3px solid #d9dee7;color:#3a4452;background:#fff}
table{border-collapse:collapse;margin:12px 0;font-size:13px;display:block;overflow:auto;max-width:100%}
th,td{border:1px solid #d9dee7;padding:6px 10px;text-align:left;vertical-align:top}
th{background:#f1f3f6;font-weight:700}
tr:nth-child(even) td{background:#fafbfc}
ul,ol{margin:6px 0 11px;padding-left:24px}
li{margin:2px 0}
img{max-width:100%}
"""


def render_doc(rel, body):
    title = html_escape(rel)
    kind = "md" if rel.lower().endswith(".md") else "raw"
    payload = json.dumps(body).replace("</", "<\\/")
    return ("<!doctype html><html><head><meta charset=utf-8><title>" + title + "</title>"
            "<style>" + DOC_CSS + "</style><script src=\"/render.js\"></script></head>"
            "<body><div class=\"doc\"><div class=\"crumb\">" + title + "</div>"
            "<div class=\"doc-body\" id=\"docbody\"></div></div>"
            "<script>window.__DOC=" + payload + ";window.__KIND=\"" + kind + "\";"
            "document.getElementById('docbody').innerHTML="
            "window.__KIND==='md'?MD.render(window.__DOC):'<pre>'+MD.esc(window.__DOC)+'</pre>';"
            "</script></body></html>")


@app.get("/render.js")
def render_js():
    with open(os.path.join(HERE, "render.js"), encoding="utf-8") as f:
        return Response(f.read(), mimetype="application/javascript")


@app.get("/stream.js")
def stream_js():
    with open(os.path.join(HERE, "stream.js"), encoding="utf-8") as f:
        return Response(f.read(), mimetype="application/javascript")


@app.get("/wireframe/<path:rel>")
def wireframe(rel):
    target = safe_path(OUTPUT, rel, (".html",))
    if not target:
        abort(404)
    with open(target, encoding="utf-8", errors="replace") as f:
        raw_html = f.read()
    
    # Wrap in a sandboxed iframe
    wrapper = (
        f"<!doctype html><html><head><title>{html_escape(rel)}</title>"
        f"<style>body,html{{margin:0;padding:0;height:100%;overflow:hidden}} "
        f"iframe{{width:100%;height:100%;border:0}}</style></head>"
        f"<body><iframe sandbox srcdoc={json.dumps(raw_html)}></iframe></body></html>"
    )
    return Response(wrapper, mimetype="text/html")


@app.get("/doc/<path:rel>")
def doc(rel):
    target = safe_path(OUTPUT, rel, DOC_EXTS)
    if not target:
        abort(404)
    with open(target, encoding="utf-8", errors="replace") as f:
        body = f.read()
    return Response(render_doc(rel, body), mimetype="text/html")


@app.get("/")
def index():
    with open(os.path.join(HERE, "templates", "index.html"), encoding="utf-8") as f:
        content = f.read()
    content = content.replace("</head>", f"<script>window.CSRF_TOKEN='{CSRF_TOKEN}';</script></head>")
    return Response(content, mimetype="text/html")


if __name__ == "__main__":
    if get_corpus().N == 0:
        raise SystemExit(f"No documents loaded from {OUTPUT} — run ./convert.sh to build it.")
    
    s = settings_service.get()
    port = int(os.environ.get("PORT", "5000"))
    desc = (f"OpenAI-compatible {s.ai.model} @ {s.ai.base_url}" if s.ai.backend == "openai"
            else f"Claude {s.ai.model}" if s.ai.backend == "claude" else "extractive (no LLM)")
    print(f"Document search → http://127.0.0.1:{port}   "
          f"({get_corpus().N:,} passages / {get_corpus().files} files in memory; answers via {desc})")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
