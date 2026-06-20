#!/usr/bin/env python3
import os, json
import corpus

EVAL_CORPUS = os.path.join(corpus.HERE, "tests", "big_corpus")


def load_eval():
    with open(os.path.join(corpus.HERE, "eval_set.json"), encoding="utf-8") as f:
        return json.load(f)


def first_rank(hits, expect):
    for i, h in enumerate(hits, 1):
        if any(x.lower() in h["relpath"].lower() for x in expect):
            return i
    return None


def covered(items, expect):
    return any(any(x.lower() in u["relpath"].lower() for x in expect) for u in items)


def measure(c, data):
    h5 = h10 = cov = 0
    rr = 0.0
    chars = files = dups = 0
    rows = []
    for it in data:
        hits = c.search(it["q"], limit=10)
        fr = first_rank(hits, it["expect"])
        if fr:
            rr += 1.0 / fr
            h5 += fr <= 5
            h10 += fr <= 10
        used, nf = c.gather(it["q"])
        in_answer = covered(used, it["expect"])
        cov += in_answer
        chars += sum(len(u["body"]) for u in used)
        files += nf
        sets = [set(u["tf"]) for u in used]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if corpus.jaccard(sets[i], sets[j]) > 0.5:
                    dups += 1
        rows.append((fr, in_answer, len(used), nf, it["q"]))
    n = len(data)
    return {
        "recall@5": h5 / n, "recall@10": h10 / n, "mrr": rr / n, "coverage": cov / n,
        "avg_chars": chars // n, "avg_files": files / n, "dup_pairs": dups, "rows": rows, "n": n,
    }


def main():
    c = corpus.Corpus().load(text_dir=EVAL_CORPUS, source=EVAL_CORPUS)
    if c.N == 0:
        print(f"No corpus loaded from {EVAL_CORPUS}")
        return
    m = measure(c, load_eval())
    for fr, in_ans, used, nf, q in m["rows"]:
        rank_mark = ("@" + str(fr)) if fr else "MISS"
        print(f"  search:{rank_mark:>5}  answer:{'✓' if in_ans else '✗'}  used={used:>2} files={nf:>2}  {q[:54]}")
    print(f"\n{m['n']} queries · search recall@5={m['recall@5']:.2f} recall@10={m['recall@10']:.2f} "
          f"MRR={m['mrr']:.2f}")
    print(f"answer coverage (expected file in excerpts sent)={m['coverage']:.2f}")
    print(f"avg chars sent={m['avg_chars']:,} · avg files/answer={m['avg_files']:.1f} · "
          f"near-dup excerpt pairs={m['dup_pairs']}")


if __name__ == "__main__":
    main()
