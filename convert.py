import os, sys, io, shutil, subprocess, time, tempfile
from dataclasses import dataclass, field
from typing import List, Optional
from corpus import INPUT, OUTPUT, is_excluded

CONVERT_EXT = {".docx", ".html", ".htm", ".xlsx", ".xls", ".pptx", ".msg", ".pdf"}
COPY_EXT = {".csv", ".json", ".md", ".txt", ".xer"}
FORCE = ("--force" in sys.argv) or os.environ.get("SEARCH_FORCE") == "1"
OCR_MESSAGE = "No extractable text; this PDF may require OCR and is not included in the corpus."

@dataclass
class FileResult:
    relpath: str
    status: str # converted, copied, unchanged, skipped, warning, failed
    output_paths: List[str] = field(default_factory=list)
    message: str = ""
    error_category: Optional[str] = None
    duration: float = 0.0

@dataclass
class SessionResult:
    converted: int = 0
    copied: int = 0
    skipped: int = 0
    failed: int = 0
    pruned: int = 0
    errors: List[FileResult] = field(default_factory=list)
    exit_code: int = 0

def safe_write(content, dest):
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(dest))
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, dest)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def stale(src, out):
    if FORCE or not os.path.exists(out):
        return True
    return int(os.path.getmtime(out)) != int(os.path.getmtime(src))


def stamp(out, src):
    t = os.path.getmtime(src)
    os.utime(out, (t, t))


def ensure(out):
    os.makedirs(os.path.dirname(out), exist_ok=True)


def out_for(rel, ext=None):
    base = os.path.join(OUTPUT, rel)
    return os.path.splitext(base)[0] + ext if ext else base


def pandoc(src, out, fmt=None):
    cmd = ["pandoc", src, "-t", "gfm", "-o", out, "--wrap=none"]
    if fmt:
        cmd[2:2] = ["-f", fmt]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)


def conv_xlsx(src):
    import openpyxl
    wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
    buf = io.StringIO()
    buf.write(f"# {os.path.basename(src)}\n")
    for ws in wb.worksheets:
        buf.write(f"\n## Sheet: {ws.title}\n\n")
        rows = list(ws.iter_rows(values_only=True))
        while rows and (rows[-1] is None or all(c is None for c in rows[-1])):
            rows.pop()
        emitted = 0
        for r in rows[:3000]:
            cells = ["" if c is None else str(c).replace("\n", " ").replace("|", "/") for c in r]
            if any(cells):
                buf.write("| " + " | ".join(cells) + " |\n")
                emitted += 1
                if emitted == 1:
                    buf.write("| " + " | ".join("---" for _ in cells) + " |\n")
    wb.close()
    return buf.getvalue()


def conv_xls(src):
    import pandas as pd
    xls = pd.ExcelFile(src)
    buf = io.StringIO()
    buf.write(f"# {os.path.basename(src)}\n")
    for sh in xls.sheet_names:
        buf.write(f"\n## Sheet: {sh}\n\n")
        buf.write(xls.parse(sh, header=None).head(2000).to_csv(index=False, header=False))
    return buf.getvalue()


def conv_pptx(src):
    from markitdown import MarkItDown
    return MarkItDown().convert(src).text_content


def conv_msg(src):
    import extract_msg
    m = extract_msg.Message(src)
    buf = io.StringIO()
    buf.write(f"# {m.subject or os.path.basename(src)}\n\n")
    buf.write(f"- From: {m.sender}\n- To: {m.to}\n- Date: {m.date}\n\n---\n\n")
    buf.write(m.body or "")
    res = buf.getvalue()
    m.close()
    return res


def conv_pdf(src):
    import fitz
    doc = fitz.open(src)
    text = "\n".join(pg.get_text("text") for pg in doc)
    doc.close()
    return text


def outputs_for(rel, ext):
    if ext in COPY_EXT:
        return [out_for(rel)]
    if ext == ".pdf":
        return [out_for(rel, ".txt")]
    if ext in (".html", ".htm"):
        return [out_for(rel, ".md"), out_for(rel)]
    if ext in CONVERT_EXT:
        return [out_for(rel, ".md")]
    return []


def produce(src, rel, ext, session):
    t0 = time.time()
    try:
        if ext in COPY_EXT:
            out = out_for(rel)
            ensure(out)
            shutil.copy2(src, out)
            stamp(out, src)
            return FileResult(rel, "copied", [out], duration=time.time()-t0)

        out = out_for(rel, ".txt" if ext == ".pdf" else ".md")
        ensure(out)

        content = ""
        if ext == ".docx":
            # pandoc produces a file, we need to read it to validate
            pandoc(src, out)
            with open(out, 'r', encoding='utf-8') as f: content = f.read()
        elif ext in (".html", ".htm"):
            pandoc(src, out, fmt="html")
            with open(out, 'r', encoding='utf-8') as f: content = f.read()
            raw = out_for(rel)
            ensure(raw)
            shutil.copy2(src, raw)
            stamp(raw, src)
        elif ext == ".xlsx":
            content = conv_xlsx(src)
        elif ext == ".xls":
            content = conv_xls(src)
        elif ext == ".pptx":
            content = conv_pptx(src)
        elif ext == ".msg":
            content = conv_msg(src)
        elif ext == ".pdf":
            content = conv_pdf(src)

        if not content.strip():
            safe_write("", out)
            stamp(out, src)
            return FileResult(rel, "warning", [out], OCR_MESSAGE, "OCR_NEEDED", time.time()-t0)

        safe_write(content, out)
        stamp(out, src)
        return FileResult(rel, "converted", [out], duration=time.time()-t0)
    except Exception as e:
        return FileResult(rel, "failed", [], str(e), type(e).__name__, time.time()-t0)


def check_deps():
    import importlib.util as u
    need = [m for m in ("flask", "fitz", "openpyxl", "markitdown", "extract_msg", "pandas", "anthropic") if not u.find_spec(m)]
    if need:
        print(f"Missing dependencies: {', '.join(need)}")
        print("Please run: python3 -m pip install -r requirements.txt")
        return False
    return True


def run_conversion() -> SessionResult:
    os.makedirs(INPUT, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)
    session = SessionResult()
    expected = set()
    missing_deps = set()

    tasks = []
    for root, _, files in os.walk(INPUT):
        for f in files:
            if f.startswith("~$") or f == ".DS_Store":
                continue
            src = os.path.join(root, f)
            rel = os.path.relpath(src, INPUT)
            if is_excluded(rel):
                continue
            ext = os.path.splitext(f)[1].lower()
            outs = outputs_for(rel, ext)
            if not outs:
                continue
            tasks.append((src, rel, ext, outs))
            expected.update(os.path.abspath(o) for o in outs)

    total = len(tasks)
    for i, (src, rel, ext, outs) in enumerate(tasks, 1):
        print(f"[{i}/{total}] Syncing {rel}...", end="\r", flush=True)
        try:
            if any(stale(src, o) for o in outs):
                res = produce(src, rel, ext, session)
                if res.status == "converted": session.converted += 1
                elif res.status == "copied": session.copied += 1
                elif res.status == "failed":
                    session.failed += 1
                    session.errors.append(res)
                elif res.status == "warning":
                    session.converted += 1
                    session.errors.append(res)
            else:
                session.skipped += 1
                if ext == ".pdf" and any(os.path.getsize(out) == 0 for out in outs):
                    session.errors.append(FileResult(rel, "warning", outs, OCR_MESSAGE, "OCR_NEEDED"))
        except (ImportError, ModuleNotFoundError) as e:
            missing_deps.add(str(e).split("'")[1] if "'" in str(e) else str(e))
        except FileNotFoundError:
            missing_deps.add("pandoc")
        except Exception as e:
            session.failed += 1
            session.errors.append(FileResult(rel, "failed", [], str(e), type(e).__name__))

    print(f"\nSyncing {total} files complete.")

    pruned = 0
    for root, _, files in os.walk(OUTPUT):
        for f in files:
            p = os.path.abspath(os.path.join(root, f))
            if p not in expected:
                os.remove(p); pruned += 1
    for root, _, _ in os.walk(OUTPUT, topdown=False):
        if root != OUTPUT and not os.listdir(root):
            os.rmdir(root)

    session.pruned = pruned
    session.exit_code = 1 if (session.failed > 0 or missing_deps) else 0
    return session

def main():
    res = run_conversion()
    print(f"output/: {res.converted} converted, {res.copied} copied, "
          f"{res.skipped} up-to-date, {res.pruned} pruned, {res.failed} errors")
    for warning in (result for result in res.errors if result.status == "warning"):
        print(f"WARN: {warning.relpath}: {warning.message}")
    if res.exit_code != 0:
        sys.exit(res.exit_code)


if __name__ == "__main__":
    main()
