"""Dataset preparation: seeded subsets of GLUE SST-2 (validation) and SQuAD 2.0 (validation), plus a
synthetic, labelled tool-use set. Sampling functions are pure so they can be unit-tested without downloads."""
from __future__ import annotations
import hashlib, json, random, urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .tools import calculator


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_sst2(rows: list[dict], n: int, seed: int) -> list[dict]:
    """rows: {'sentence': str, 'label': 0|1}. Class-balanced, seeded."""
    rng = random.Random(seed)
    pos = [r for r in rows if r["label"] == 1]; neg = [r for r in rows if r["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    pick = pos[: n // 2] + neg[: n - n // 2]
    rng.shuffle(pick)
    return [{"id": f"sst2-{i:04d}", "sentence": r["sentence"].strip(),
             "label": "positive" if r["label"] == 1 else "negative"} for i, r in enumerate(pick)]


def sample_squad(rows: list[dict], n: int, seed: int, impossible_fraction: float = 0.5) -> list[dict]:
    """rows: {'id','context','question','answers': {'text': [...]}}. Impossible = no gold answers."""
    rng = random.Random(seed)
    imp = [r for r in rows if not r["answers"]["text"]]; ans = [r for r in rows if r["answers"]["text"]]
    rng.shuffle(imp); rng.shuffle(ans)
    n_imp = round(n * impossible_fraction)
    pick = imp[:n_imp] + ans[: n - n_imp]
    rng.shuffle(pick)
    return [{"id": r["id"], "context": r["context"], "question": r["question"],
             "answers": sorted(set(r["answers"]["text"])), "is_impossible": not r["answers"]["text"]} for r in pick]


def build_tool_set(seed: int, n: int) -> list[dict]:
    """Synthetic, labelled tool-use scenarios. Half REQUIRE exact arithmetic; half do NOT need a tool."""
    rng = random.Random(seed)
    rows = []
    for i in range(n // 2):
        kind = i % 3
        if kind == 0:
            a, b = rng.randint(100, 999), rng.randint(100, 999)
            q, expr = f"What is {a} multiplied by {b}?", f"{a} * {b}"
        elif kind == 1:
            a, p, b, q2 = rng.randint(12, 60), rng.randint(15, 95), rng.randint(5, 40), rng.randint(15, 95)
            q = f"A shop sells {a} pens at {p} cents each and {b} notebooks at {q2} cents each. How many cents is the total?"
            expr = f"{a} * {p} + {b} * {q2}"
        else:
            price, pct = rng.randint(200, 900), rng.choice([10, 15, 20, 25, 35])
            q = f"An item costs {price} dollars with a {pct} percent discount. What is the final price in dollars?"
            expr = f"{price} * (1 - {pct} / 100)"
        rows.append({"question": q, "gold": calculator(expr), "kind": "numeric", "tool_policy": "required",
                     "reference_expression": expr})
    words = ["planet", "river", "silver", "garden", "window", "basket", "candle", "forest"]
    for i in range(n - n // 2):
        if i % 2 == 0:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            while a == b:
                b = rng.randint(10, 99)
            rows.append({"question": f"Which number is larger, {a} or {b}?", "gold": str(max(a, b)), "kind": "numeric",
                         "tool_policy": "not_needed", "reference_expression": None})
        else:
            w = rng.choice(words)
            rows.append({"question": f"Write the word '{w}' in capital letters.", "gold": w.upper(), "kind": "text",
                         "tool_policy": "not_needed", "reference_expression": None})
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"tool-{i:04d}"
    return rows



# ---------------- real dataset sources (GitHub mirrors; no Hugging Face needed) ----------------
class DataIntegrityError(Exception):
    pass


SOURCES = {
    "squad_dev": {
        "url": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json",
        "sha256": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
        "what": "SQuAD 2.0 dev (official file from the SQuAD authors' repository)",
        "expect": {"questions": 11873, "unanswerable": 5945},
    },
    "sst2_dev": {
        "url": "https://raw.githubusercontent.com/YJiangcm/SST-2-sentiment-analysis/master/data/dev.tsv",
        "sha256": "c9ddb67ce3068540c5bb6562e1f34788302dd06d570e6ca0d5e61f7d5ef2bfad",
        "what": "SST-2 validation (standard 872-sentence binary split), third-party GitHub mirror",
        "expect": {"sentences": 872, "positive": 444, "negative": 428},
    },
}


def fetch(name: str, raw_dir) -> Path:
    """Download (or reuse) a raw source file and verify its SHA-256. A changed file fails loudly."""
    src = SOURCES[name]
    dest = Path(raw_dir) / Path(src["url"]).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(src["url"], timeout=120) as r:
            dest.write_bytes(r.read())
    got = sha256(dest)
    if got != src["sha256"]:
        raise DataIntegrityError(f"{name}: sha256 {got} != expected {src['sha256']} (file changed upstream?). "
                                 f"Delete {dest} to re-download, or update SOURCES after inspecting the file.")
    return dest


def flatten_squad_v2(obj: dict) -> list[dict]:
    """Official SQuAD v2.0 JSON -> flat rows {'id','context','question','answers':{'text':[...]}}."""
    rows = []
    for art in obj["data"]:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                texts = [a["text"] for a in qa.get("answers", [])]
                if bool(qa.get("is_impossible")) != (len(texts) == 0):
                    raise DataIntegrityError(f"inconsistent is_impossible for {qa['id']}")
                rows.append({"id": qa["id"], "context": para["context"], "question": qa["question"],
                             "answers": {"text": texts}})
    return rows


def parse_sst2_label_first(text: str) -> list[dict]:
    """Lines of 'label<TAB>sentence' (label 0/1) -> rows {'sentence','label'}. Header/blank lines are skipped."""
    rows = []
    for line in text.splitlines():
        lab, _, sent = line.partition("\t")
        if lab.strip() in ("0", "1") and sent.strip():
            rows.append({"sentence": sent.strip(), "label": int(lab)})
    return rows


def verify_counts(sst_rows, squad_rows):
    e = SOURCES["sst2_dev"]["expect"]
    got = {"sentences": len(sst_rows), "positive": sum(r["label"] == 1 for r in sst_rows),
           "negative": sum(r["label"] == 0 for r in sst_rows)}
    if got != e:
        raise DataIntegrityError(f"SST-2 stats {got} != expected {e}")
    e = SOURCES["squad_dev"]["expect"]
    got = {"questions": len(squad_rows), "unanswerable": sum(not r["answers"]["text"] for r in squad_rows)}
    if got != e:
        raise DataIntegrityError(f"SQuAD stats {got} != expected {e}")


def prepare_all(out_dir, n_sst2=100, n_squad=200, impossible_fraction=0.5, n_tools=40, seed=42, source="github"):
    """Build the three benchmark files.

    source='github': downloads the official SQuAD 2.0 dev file and an SST-2 validation mirror from GitHub,
                     verifies SHA-256 and published statistics (default; no extra dependencies).
    source='hf'    : uses Hugging Face `datasets` (needs requirements-data.txt and access to huggingface.co).
    """
    out = Path(out_dir)
    if source == "github":
        raw = out / "raw"
        sst_rows = parse_sst2_label_first(fetch("sst2_dev", raw).read_text(encoding="utf8"))
        squad_rows = flatten_squad_v2(json.loads(fetch("squad_dev", raw).read_text(encoding="utf8")))
        verify_counts(sst_rows, squad_rows)
        sources = {"sst2": SOURCES["sst2_dev"], "squad": SOURCES["squad_dev"], "tools": "synthetic (build_tool_set)"}
    elif source == "hf":
        from datasets import load_dataset  # lazy import
        sst_rows = [dict(r) for r in load_dataset("glue", "sst2", split="validation")]
        squad_rows = [dict(r) for r in load_dataset("rajpurkar/squad_v2", split="validation")]
        sources = {"sst2": "glue/sst2 validation (Hugging Face)", "squad": "rajpurkar/squad_v2 validation (Hugging Face)",
                   "tools": "synthetic (build_tool_set)"}
    else:
        raise ValueError(f"unknown source {source}")
    files = {"sst2.jsonl": sample_sst2(sst_rows, n_sst2, seed),
             "squad.jsonl": sample_squad(squad_rows, n_squad, seed, impossible_fraction),
             "tools.jsonl": build_tool_set(seed, n_tools)}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "source_mode": source,
                "sources": sources, "population": {"sst2_validation": len(sst_rows), "squad_v2_validation": len(squad_rows)},
                "files": {}}
    for name, rows in files.items():
        write_jsonl(out / name, rows)
        manifest["files"][name] = {"n": len(rows), "sha256": sha256(out / name)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
