import json
import re
import numpy as np
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Models ─────────────────────────────────────────────────
print("Loading models...")
BGE = SentenceTransformer("BAAI/bge-small-en-v1.5")
NLI = CrossEncoder(
    "cross-encoder/nli-deberta-v3-small",
    num_labels=3
)

chroma = chromadb.PersistentClient(
    path="data/step6_v14/chroma_db"
)
collection = chroma.get_collection("pramana_v14")
Path("data/step13_final").mkdir(parents=True, exist_ok=True)
EXPECTED_SENTENCES = 3362

# ── Thresholds (calibrated) ────────────────────────────────
ENTAIL_THRESHOLD = 0.35
CONTRA_THRESHOLD = 0.50
GATE1_FLOOR = 0.25

# ── Softmax (REQUIRED — do not use raw logits) ─────────────
def nli_softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

# ── Field name discovery ───────────────────────────────────
def discover_fields(cases):
    if not cases:
        return {}
    sample = cases[0]
    field_map = {}

    for candidate in ("case_id","CASE_ID","id","ID"):
        if candidate in sample:
            field_map["case_id"] = candidate
            break

    for candidate in ("title", "TITLE", "Title"):
        if candidate in sample:
            field_map["title"] = candidate
            break

    for candidate in ("story", "STORY", "Story"):
        if candidate in sample:
            field_map["story"] = candidate
            break

    for candidate in ("evidence_presented", "EVIDENCE_PRESENTED", "Evidence_Presented", "evidence", "EVIDENCE"):
        if candidate in sample:
            field_map["evidence"] = candidate
            break

    for candidate in (
        "decision_and_meaning","DECISION_AND_MEANING",
        "Decision_And_Meaning",
        "WHAT_THE_COURT_DECIDED_AND_WHAT_THAT_DECISION_MEANS_IN_SIMPLE_WORDS",
        "decision","DECISION"
    ):
        if candidate in sample:
            field_map["decision"] = candidate
            break

    for candidate in (
        "why_court_decided","WHY_COURT_DECIDED",
        "Why_Court_Decided",
        "WHY_THE_COURT_DECIDED_THAT_WAY",
        "why","WHY"
    ):
        if candidate in sample:
            field_map["why"] = candidate
            break

    print("Discovered fields:")
    for k, v in field_map.items():
        print(f"  {k:20s} → {v}")
    return field_map

# ── Load JSON safely ───────────────────────────────────────
def load_json_safe(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    for key in ("CASES", "results", "cases", "data"):
        if key in raw and isinstance(raw[key], list):
            return raw[key]
    raise ValueError(f"Cannot find list in {path}")

# ── ChromaDB source text ───────────────────────────────────
def get_source_text(case_id):
    results = collection.get(
        where={"case_id": case_id},
        include=["documents"]
    )
    if results["documents"]:
        return " ".join(results["documents"])
    return None

# ── Sentence splitting ─────────────────────────────────────
def split_sentences(text):
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [
        s.strip() for s in parts
        if len(s.strip()) > 20
        and re.search(r"[a-zA-Z]{3,}", s)
        and "not legal advice" not in s.lower()
        and "[source:" not in s.lower()
    ]

# ── Gate 1: cosine candidate finder ───────────────────────
def gate1_find_best(claim, source_text):
    source_sents = split_sentences(source_text)
    if not source_sents:
        return 0.0, None
    claim_emb = BGE.encode(
        claim, normalize_embeddings=True
    )
    src_embs = BGE.encode(
        source_sents, normalize_embeddings=True
    )
    sims = (src_embs @ claim_emb).tolist()
    best_idx = int(np.argmax(sims))
    return round(sims[best_idx], 4), source_sents[best_idx]

# ── Gate 2: NLI with softmax ───────────────────────────────
def gate2_nli(claim, source_sentence):
    logits = NLI.predict([(source_sentence, claim)])
    probs = nli_softmax(logits[0])
    contra_prob = float(probs[0])
    entail_prob = float(probs[1])

    if contra_prob >= CONTRA_THRESHOLD:
        verdict = "BLOCKED"
    elif entail_prob >= ENTAIL_THRESHOLD:
        verdict = "VERIFIED"
    else:
        verdict = "NEEDS_REVIEW"

    return verdict, round(entail_prob, 4), round(contra_prob, 4)

# ── Per-case verification ──────────────────────────────────
def verify_case(case, field_map, gate0_results):
    case_id = case.get(
        field_map.get("case_id", "case_id"), ""
    )

    # Build Gate 0 blocked set with normalisation
    gate0 = next(
        (r for r in gate0_results
         if r["case_id"] == case_id),
        None
    )
    gate0_blocked = set()
    if gate0:
        for d in gate0.get("blocked_details", []):
            raw = d["sentence"]
            gate0_blocked.add(raw[:80])
            gate0_blocked.add(raw[:60])
            gate0_blocked.add(
                re.sub(r'\s+', ' ', raw).strip()[:80]
            )

    source_text = get_source_text(case_id)
    if not source_text:
        return {
            "case_id": case_id,
            "status": "NO_SOURCE",
            "overall": "NEEDS_REVIEW",
            "verified": 0,
            "needs_review": 0,
            "blocked": 0,
            "total_sentences": 0,
            "details": []
        }

    fields_to_check = []
    for key, field_name in [
        ("title", field_map.get("title")),
        ("story", field_map.get("story")),
        ("evidence", field_map.get("evidence")),
        ("decision", field_map.get("decision")),
        ("why", field_map.get("why"))
    ]:
        if field_name and field_name in case:
            fields_to_check.append((key, case[field_name]))

    details = []
    verified = needs_review = blocked = 0

    for field_key, text in fields_to_check:
        for sentence in split_sentences(text):

            # Gate 0 blocked
            if (sentence[:80] in gate0_blocked or
                    sentence[:60] in gate0_blocked):
                details.append({
                    "field": field_key,
                    "sentence": sentence[:120],
                    "gate1_score": None,
                    "best_source_sentence": None,
                    "gate2_verdict": "BLOCKED",
                    "gate2_entail": None,
                    "gate2_contra": None,
                    "blocked_by": "gate0"
                })
                blocked += 1
                continue

            # Gate 1: find best source sentence
            g1_score, best_src = gate1_find_best(
                sentence, source_text
            )

            if not best_src:
                details.append({
                    "field": field_key,
                    "sentence": sentence[:120],
                    "gate1_score": g1_score,
                    "best_source_sentence": None,
                    "gate2_verdict": "NEEDS_REVIEW",
                    "gate2_entail": None,
                    "gate2_contra": None,
                    "blocked_by": "no_source_sentence"
                })
                needs_review += 1
                continue

            # Gate 2: NLI softmax decision
            verdict, entail, contra = gate2_nli(
                sentence, best_src
            )

            # If Gate 1 score very low, cap at NEEDS_REVIEW
            if g1_score < GATE1_FLOOR and verdict == "VERIFIED":
                verdict = "NEEDS_REVIEW"
                blocked_by = "gate1_floor_cap"
            else:
                blocked_by = None

            if verdict == "VERIFIED":
                verified += 1
            elif verdict == "NEEDS_REVIEW":
                needs_review += 1
            else:
                blocked += 1

            details.append({
                "field": field_key,
                "sentence": sentence[:120],
                "gate1_score": g1_score,
                "best_source_sentence": best_src[:120],
                "gate2_verdict": verdict,
                "gate2_entail": entail,
                "gate2_contra": contra,
                "blocked_by": blocked_by
            })

    total = verified + needs_review + blocked
    if total == 0:
        overall = "NO_SENTENCES"
    elif blocked > 0:
        overall = "BLOCKED"
    elif needs_review > 0:
        overall = "NEEDS_REVIEW"
    else:
        overall = "VERIFIED"

    return {
        "case_id": case_id,
        "status": "CHECKED",
        "overall": overall,
        "verified": verified,
        "needs_review": needs_review,
        "blocked": blocked,
        "total_sentences": total,
        "details": details
    }

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":

    # Find cases file
    cases_file = Path("data/step12/STEP-12B FINAL 225_gate0_corrected.json")
    if not cases_file.exists():
        raise FileNotFoundError(f"Locked cases file not found: {cases_file.resolve()}")
    print(f"Using locked cases file: {cases_file}")

    cases = load_json_safe(cases_file)
    if len(cases) != 225:
        raise ValueError(f"Expected 225 locked cases, found {len(cases)}")
    print(f"Cases loaded: {len(cases)}")

    field_map = discover_fields(cases)

    gate0_raw = load_json_safe(
        Path("data/step13a/gate0_results.json")
    )
    print(f"Gate 0 results: {len(gate0_raw)} records")

    print(f"\nVerifying {len(cases)} cases...")
    print("Gate 1: cosine candidate finder")
    print("Gate 2: NLI softmax (entail≥0.35, contra≥0.50)")
    print("=" * 60)

    all_results = []
    for i, case in enumerate(cases):
        result = verify_case(case, field_map, gate0_raw)
        all_results.append(result)
        print(
            f"[{i+1:3d}/{len(cases)}] "
            f"{result['case_id']:12s} "
            f"V={result.get('verified',0):3d} "
            f"NR={result.get('needs_review',0):3d} "
            f"B={result.get('blocked',0):3d} "
            f"→ {result.get('overall','?')}"
        )

    # Summary
    total_v = sum(
        r.get("verified", 0) for r in all_results
    )
    total_nr = sum(
        r.get("needs_review", 0) for r in all_results
    )
    total_b = sum(
        r.get("blocked", 0) for r in all_results
    )
    total = total_v + total_nr + total_b
    if total != EXPECTED_SENTENCES:
        raise ValueError(
            f"Expected exact Gate 0 scope of {EXPECTED_SENTENCES} sentences, found {total}"
        )

    cases_v = sum(
        1 for r in all_results
        if r.get("overall") == "VERIFIED"
    )
    cases_nr = sum(
        1 for r in all_results
        if r.get("overall") == "NEEDS_REVIEW"
    )
    cases_b = sum(
        1 for r in all_results
        if r.get("overall") == "BLOCKED"
    )

    print("\n" + "=" * 60)
    print("STEP 13 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total cases            : {len(cases)}")
    print(f"Total sentences        : {total}")
    if total:
        print(
            f"VERIFIED               : {total_v} "
            f"({total_v/total*100:.1f}%)"
        )
        print(
            f"NEEDS_REVIEW           : {total_nr} "
            f"({total_nr/total*100:.1f}%)"
        )
        print(
            f"BLOCKED                : {total_b} "
            f"({total_b/total*100:.1f}%)"
        )
    print(f"\nCases fully VERIFIED   : {cases_v}")
    print(f"Cases NEEDS_REVIEW     : {cases_nr}")
    print(f"Cases with BLOCKED     : {cases_b}")
    print(f"\nThresholds: entail≥{ENTAIL_THRESHOLD}, "
          f"contra≥{CONTRA_THRESHOLD}")

    # Save
    out = Path("data/step13_final")
    with open(
        out / "gate_all_results.json", "w",
        encoding="utf-8"
    ) as f:
        json.dump(all_results, f, indent=2,
                  ensure_ascii=False)
    print("\nSaved: data/step13_final/gate_all_results.json")

    review = [
        r for r in all_results
        if r.get("overall") in ("NEEDS_REVIEW", "BLOCKED")
    ]
    if review:
        with open(
            out / "needs_human_review.json", "w",
            encoding="utf-8"
        ) as f:
            json.dump(review, f, indent=2,
                      ensure_ascii=False)
        print(
            f"Human review: {len(review)} cases → "
            f"needs_human_review.json"
        )

    print("\nStep 13: COMPLETE")
