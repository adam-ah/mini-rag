import os, json, urllib.request, re
from dataclasses import dataclass
from typing import Generator, Tuple, Optional
from settings import Settings, AISettings

def openai_payload(question: str, context: str, n: int, stream: bool, settings: AISettings) -> bytes:
    return json.dumps({
        "model": settings.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(question, context, n)},
        ],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "stream": stream,
    }).encode("utf-8")

def openai_headers(settings: AISettings) -> dict:
    h = {"Content-Type": "application/json"}
    if settings.api_key:
        h["Authorization"] = f"Bearer {settings.api_key}"
    return h

def user_prompt(question: str, context: str, n: int) -> str:
    return (
        f"Question: {question}\n\n"
        f"Excerpts ({n} of them, from across the reference material):\n{context}\n\n"
        "Answer the question using the excerpts above, honoring any format or length the user "
        "asked for. Cite [n] where you draw on an excerpt."
    )

SYSTEM_PROMPT = (
    "You answer questions using the numbered excerpts provided. "
    "Answer in the format and length the user asks for: if they request a table, one line, or "
    "bullets, produce exactly that and nothing else. If the user asks you to explain a process, "
    "procedure, or steps, give a thorough, well-ordered answer that draws on all the relevant "
    "excerpts rather than stopping at the first one. Otherwise give a concise answer, integrating "
    "across the excerpts when the question is broad rather than just summarizing the first one. "
    "Cite excerpts inline as [1], [2] where you draw on them, unless the requested format makes "
    "citations impractical. Use only the provided excerpts; if they do not contain the answer, "
    "say so. Some excerpts are UI wireframes (tagged [UI wireframe]); name a relevant screen so "
    "the reader can open it."
)

REFLECTION_SYSTEM_PROMPT = (
    "/no_think\n"
    "Return one JSON object immediately. Do not explain or reason step by step. Check whether the "
    "draft fully answers the question from the excerpts. Treat undefined terms, aliases, acronyms, "
    "cross-references, prerequisites, exceptions, stages, and unanswered question parts as gaps. "
    "Queries may use only concepts in the question, draft, or excerpts. Never add facts or guess "
    "synonyms. Return complete, missing_aspects, and queries. Use complete=true and empty arrays "
    "when there is no concrete gap."
)


@dataclass(frozen=True)
class ReflectionResult:
    complete: bool
    missing_aspects: Tuple[str, ...] = ()
    queries: Tuple[str, ...] = ()


def _completion(system_prompt: str, user_content: str, settings: AISettings,
                max_tokens: int, temperature: float = 0.0, json_mode: bool = False) -> str:
    if settings.backend == "openai":
        request_data = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            request_data["response_format"] = {"type": "json_object"}
            request_data["reasoning_effort"] = "low"
        payload = json.dumps(request_data).encode("utf-8")
        req = urllib.request.Request(settings.base_url + "/chat/completions", data=payload,
                                     headers=openai_headers(settings), method="POST")
        with urllib.request.urlopen(req, timeout=settings.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return (data["choices"][0]["message"]["content"] or "").strip()
    if settings.backend == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.api_key)
        response = client.messages.create(
            model=settings.model, max_tokens=max_tokens, system=system_prompt,
            temperature=temperature, messages=[{"role": "user", "content": user_content}],
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text").strip()
    return ""


def reflect(question: str, draft: str, context: str, settings: AISettings) -> ReflectionResult:
    prompt = (
        "/no_think\n"
        f"Question:\n{question}\n\nDraft answer:\n{draft}\n\n"
        f"Retrieved excerpts:\n{context}\n\n"
        f"Return at most {settings.reflection_max_queries} missing aspects and queries."
    )
    raw = _completion(REFLECTION_SYSTEM_PROMPT, prompt, settings,
                      settings.reflection_max_tokens, 0.0, True)
    fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1)
    data = None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            candidate, _ = decoder.raw_decode(raw[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            data = candidate
            break
    if not isinstance(data, dict) or not isinstance(data.get("complete"), bool):
        raise ValueError("Invalid reflection response")
    aspects = data.get("missing_aspects", [])
    queries = data.get("queries", [])
    if not isinstance(aspects, list) or not isinstance(queries, list):
        raise ValueError("Invalid reflection response")
    limit = settings.reflection_max_queries
    clean_aspects = tuple(x.strip()[:240] for x in aspects[:limit] if isinstance(x, str) and x.strip())
    clean_queries = tuple(x.strip()[:240] for x in queries[:limit] if isinstance(x, str) and x.strip())
    if data["complete"]:
        return ReflectionResult(True)
    if not clean_aspects or not clean_queries:
        raise ValueError("Incomplete reflection has no concrete gaps")
    return ReflectionResult(False, clean_aspects, clean_queries)

def test_connection(settings: AISettings) -> Tuple[bool, str]:
    if settings.backend == "extractive":
        return True, "Extractive mode is always available"
    try:
        req = urllib.request.Request(settings.base_url + "/models", method="GET")
        if settings.api_key:
            req.add_header("Authorization", f"Bearer {settings.api_key}")
        with urllib.request.urlopen(req, timeout=2.0) as r:
            if r.status < 500:
                return True, "Connection successful"
    except Exception as e:
        # Sanitize error
        err_msg = str(e)
        if "HTTPError" in err_msg:
            return False, "Server returned an error"
        if "Timeout" in err_msg or "timed out" in err_msg.lower():
            return False, "Connection timed out"
        return False, "Could not connect to backend"
    return False, "Unknown connection error"

def answer(question: str, context: str, count: int, settings: AISettings) -> str:
    if settings.backend == "extractive":
        return "" # Handled by app.py using extractive_answer
    
    return _completion(SYSTEM_PROMPT, user_prompt(question, context, count), settings,
                       settings.max_tokens, settings.temperature)

def stream(question: str, context: str, count: int, settings: AISettings) -> Generator[str, None, None]:
    if settings.backend == "openai":
        req = urllib.request.Request(settings.base_url + "/chat/completions",
                                     data=openai_payload(question, context, count, True, settings),
                                     headers=openai_headers(settings), method="POST")
        with urllib.request.urlopen(req, timeout=settings.timeout_seconds) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta
    
    elif settings.backend == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.api_key)
        with client.messages.stream(model=settings.model, max_tokens=settings.max_tokens, system=SYSTEM_PROMPT,
                                    messages=[{"role": "user", "content": user_prompt(question, context, count)}]) as s:
            for t in s.text_stream:
                yield t
