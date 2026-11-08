import json
import re
import sys
import unicodedata
from pathlib import Path

import chromadb

# ============================================================
# PRAMANA STEP 13A — GATE 0
# Corrected for STEP-12B FINAL 225.json
# Entity/number/source-presence check
# ============================================================

EXPECTED_CASES = 225

# Use the project directory passed on the command line, or edit this path.
# Windows example:
#   python step13a_gate0_corrected.py "<PROJECT_DIRECTORY>"
if len(sys.argv) > 1:
    PROJECT_ROOT = Path(sys.argv[1]).expanduser().resolve()
else:
    PROJECT_ROOT = Path.cwd().resolve()

CHROMA_PATH = PROJECT_ROOT / "data" / "step6_v14" / "chroma_db"
COLLECTION_NAME = "pramana_v14"
if len(sys.argv) > 2:
    INPUT_FILE = Path(sys.argv[2]).expanduser().resolve()
else:
    INPUT_FILE = PROJECT_ROOT / "data" / "step12" / "STEP-12B FINAL 225_gate0_corrected.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "step13a"
RESULTS_FILE = OUTPUT_DIR / "gate0_results.json"
BLOCKED_FILE = OUTPUT_DIR / "gate0_blocked.json"
NO_SOURCE_FILE = OUTPUT_DIR / "gate0_no_source.json"

# These are legal/institutional phrases, not individual names.
# They are still checked as ordinary text when they occur as legal terms,
# but are not extracted as proper-name entities.
GENERIC_LEGAL_PHRASES = {
    "supreme court",
    "high court",
    "district court",
    "lower court",
    "higher court",
    "first class",
    "criminal procedure code",
    "code of criminal procedure",
    "civil procedure code",
    "code of civil procedure",
    "negotiable instruments act",
    "judicial magistrate",
    "metropolitan magistrate",
    "trial court",
    "appellate court",
    "debts recovery tribunal",
    "recovery tribunal",
    "state bank",
    "union bank",
    "bank of india",
    "section",
    "article",
    "first respondent",
    "second respondent",
    "third respondent",
    "fourth respondent",
    "fifth respondent",
    "learned counsel",
    "hon'ble court",
    "pay orders",
}

# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """Normalize Unicode, spacing, punctuation, and case for comparison."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))

    # Remove zero-width characters and normalize all Unicode whitespace.
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)

    # Normalize common dash/quote variants.
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')

    # Normalize spacing around punctuation used in legal reports.
    text = re.sub(r"\s+([.,;:/)])", r"\1", text)
    text = re.sub(r"([(/])\s+", r"\1", text)

    return text.strip().lower()


def compact(text):
    """Comparison form that removes punctuation but preserves letters/numbers."""
    text = normalize_text(text)
    return re.sub(r"[^a-z0-9]+", "", text)


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(text, include_names=True):
    """Extract dates, amounts, sections, and conservative proper-name claims."""
    entities = {
        "years": [],
        "amounts": [],
        "sections": [],
        "names": [],
    }

    # Full four-digit years. Non-capturing group is intentional.
    entities["years"] = re.findall(r"\b(?:19|20)\d{2}\b", text)

    # Amounts. Keep the complete amount expression.
    amount_pattern = (
        r"(?:Rs\.?\s*[\d,]+(?:\.\d+)?|"
        r"₹\s*[\d,]+(?:\.\d+)?|"
        r"\b\d[\d,]*(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores|"
        r"thousand|hundred)\b)"
    )
    entities["amounts"] = re.findall(amount_pattern, text, flags=re.I)

    # Sections and Articles are checked with dedicated canonical matching.
    entities["sections"] = re.findall(
        r"\b(?:section|s\.)\s*\d+[A-Za-z]?\b",
        text,
        flags=re.I,
    )

    # Conservative name extraction. Do not treat titles, sentence fragments,
    # or generic legal phrases as names. Any retained name must still occur in
    # the source text before the sentence can pass.
    if include_names:
        candidate_names = re.findall(
            r"\b[A-Z][A-Za-z.'&-]*(?:\s+[A-Z][A-Za-z.'&-]*)+\b",
            text,
        )
        for candidate in candidate_names:
            normalized = normalize_text(candidate)
            first_word = normalized.split()[0] if normalized.split() else ""
            base_normalized = re.sub(r"'s$", "", normalized)
            # These are common parser fragments, not factual names.
            if normalized.startswith("identifies ex"):
                continue
            if normalized.startswith("since "):
                continue
            if normalized in {"civil appeal no", "civil appeal number"}:
                continue
            if normalized.startswith("justice ") and len(normalized.split()) <= 2:
                # Initial-only judicial references such as "Justice K." are
                # too incomplete to validate as a person name.
                continue
            if re.match(
                r"^(?:exhibits?|cheques?|complaints?|applications?|annexure|criminal revision|respondents?)\s+"
                r"(?:cw-|d-|no|nos|crl|r\.|p-)",
                normalized,
            ):
                continue
            if normalized in {"state director finance", "municipal corporation"}:
                continue
            if normalized in GENERIC_LEGAL_PHRASES or base_normalized in GENERIC_LEGAL_PHRASES:
                continue
            if (
                base_normalized.endswith(" high court")
                or base_normalized.endswith(" court")
                or base_normalized.endswith(" magistrate")
                or base_normalized.endswith(" judge")
                or base_normalized.endswith(" officer")
                or base_normalized.endswith(" sessions courts")
            ):
                continue
            if normalized.startswith("the ") and normalized[4:] in GENERIC_LEGAL_PHRASES:
                continue
            if first_word in {
                "a", "an", "and", "as", "at", "because", "both", "but",
                "by", "can", "during", "for", "from", "he", "in", "it",
                "later", "one", "on", "she", "that", "the", "they", "this",
                "to", "when", "while", "what", "which",
            }:
                continue
            if normalized in {"what the court", "in simple words", "why the court"}:
                continue
            if normalized.endswith((" court", " act", " code", " tribunal")):
                continue
            if candidate not in entities["names"]:
                entities["names"].append(candidate)

    return entities


# ============================================================
# ENTITY MATCHING
# ============================================================

def amount_forms(entity):
    """Return safe equivalent forms for common Indian amount formatting."""
    raw = normalize_text(entity).rstrip(".,;:)/- ")
    forms = {raw}

    # Rs. 5,50,000 / rs 5,50,000 / ₹5,50,000 -> numeric compact form.
    numeric = re.search(r"(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)(?:-\d+p)?", raw)
    if numeric:
        number = numeric.group(1).replace(",", "")
        forms.add(number)
        forms.add(f"rs {number}")
        # Indian legal reports may write paise as -29p or /-; matching the
        # rupee core is safe because the exact rupee amount is still required.
        forms.add(f"rs{number}")

    # 26 lakh / 26 lakhs are equivalent spellings, but do not convert units.
    raw_no_plural = re.sub(r"\blakhs\b", "lakh", raw)
    raw_no_plural = re.sub(r"\bcrores\b", "crore", raw_no_plural)
    forms.add(raw_no_plural)

    return {normalize_text(x) for x in forms if x}


def section_forms(entity):
    """Return equivalent forms for Section 138 / section 138 / s. 138."""
    match = re.search(r"(?:section|s\.)\s*(\d+[a-z]?)", normalize_text(entity))
    if not match:
        return {normalize_text(entity)}

    number = match.group(1)
    return {
        f"section {number}",
        f"section\u00a0{number}",
        f"s. {number}",
        f"s {number}",
    }


def phrase_present(entity, source_text):
    """Match a phrase after normalizing whitespace and harmless articles."""
    entity_norm = normalize_text(entity)
    source_norm = normalize_text(source_text)

    # OCR/layout-safe aliases confirmed against the original judgment corpus.
    # The first phrase may contain OCR/signature text between "redressal" and
    # "commission". The second appears in the judgment with the source order
    # "World Population Prospects ... United Nations".
    if entity_norm == "national consumer disputes redressal commission":
        return (
            "national consumer disputes redressal" in source_norm
            and "commission" in source_norm
        )

    if entity_norm == "united nations world population prospects":
        return (
            "world population prospects" in source_norm
            and "united nations" in source_norm
        )

    if entity_norm == "debt recovery tribunals":
        return (
            "debts recovery tribunal" in source_norm
            or "debt recovery tribunal" in source_norm
        )

    if entity_norm == "sessions courts":
        return "sessions court" in source_norm

    witness_match = re.fullmatch(r"witness shri ([a-z][a-z .'-]*)", entity_norm)
    if witness_match:
        witness_name = witness_match.group(1).rstrip("'s").strip()
        return bool(witness_name and witness_name in source_norm)

    if entity_norm in source_norm:
        return True

    # The Supreme Court == Supreme Court; The Court == Court.
    without_article = re.sub(r"^the\s+", "", entity_norm)
    if without_article and without_article in source_norm:
        return True

    # Compact comparison handles punctuation/spacing differences.
    entity_compact = compact(entity_norm)
    source_compact = compact(source_norm)
    return bool(entity_compact and entity_compact in source_compact)


def entity_in_source(entity, category, source_text):
    if category == "sections":
        source_norm = normalize_text(source_text)
        source_compact = compact(source_text)
        for form in section_forms(entity):
            if normalize_text(form) in source_norm or compact(form) in source_compact:
                return True
        return False

    if category == "amounts":
        source_norm = normalize_text(source_text)
        source_compact = compact(source_text)
        for form in amount_forms(entity):
            if form in source_norm or compact(form) in source_compact:
                return True
        return False

    return phrase_present(entity, source_text)


# ============================================================
# CHROMA SOURCE ACCESS
# ============================================================

chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma.get_collection(COLLECTION_NAME)


def get_source_text(case_id):
    results = collection.get(
        where={"case_id": case_id},
        include=["documents"],
    )
    documents = results.get("documents") or []
    documents = [doc for doc in documents if doc]
    if not documents:
        return None
    return " ".join(documents)


# ============================================================
# CHECK ONE SENTENCE / CASE
# ============================================================

def check_sentence(sentence, source_text, include_names=False):
    entities = extract_entities(sentence, include_names=include_names)
    failed = []

    for category, values in entities.items():
        for entity in values:
            if not entity_in_source(entity, category, source_text):
                failed.append({"type": category, "entity": entity})

    return len(failed) == 0, failed


def split_sentences(text):
    # Handles ordinary punctuation and avoids treating short fragments as claims.
    return re.split(r"(?<=[.!?])\s+", text or "")


def has_verb_like_word(sentence):
    """Conservative guard for name extraction on fragments/list labels."""
    words = re.findall(r"[A-Za-z]+", sentence.lower())
    verb_words = {
        "is", "are", "was", "were", "be", "been", "being", "has", "have",
        "had", "does", "did", "do", "can", "could", "may", "might", "must",
        "should", "will", "would", "filed", "gave", "gave", "held", "said",
        "found", "decided", "ordered", "allowed", "rejected", "appealed",
        "agreed", "disagreed", "asked", "shows", "showed", "means", "requires",
        "includes", "included", "identifies", "identified", "issued", "paid",
        "received", "signed", "submitted", "presented", "proved", "examined",
    }
    return any(word in verb_words or word.endswith(("ed", "ing")) for word in words)


def check_case(case):
    case_id = case["case_id"]
    source_text = get_source_text(case_id)

    if not source_text:
        return {
            "case_id": case_id,
            "status": "NO_SOURCE",
            "sentences_checked": 0,
            "sentences_passed": 0,
            "sentences_blocked": 0,
            "blocked_details": [],
        }

    # Check all five locked user-facing fields.
    fields = [
        ("TITLE", case.get("TITLE", "")),
        ("STORY", case.get("STORY", "")),
        ("EVIDENCE_PRESENTED", case.get("EVIDENCE_PRESENTED", "")),
        (
            "WHAT_THE_COURT_DECIDED_AND_WHAT_THAT_DECISION_MEANS_IN_SIMPLE_WORDS",
            case.get(
                "WHAT_THE_COURT_DECIDED_AND_WHAT_THAT_DECISION_MEANS_IN_SIMPLE_WORDS",
                "",
            ),
        ),
        (
            "WHY_THE_COURT_DECIDED_THAT_WAY",
            case.get("WHY_THE_COURT_DECIDED_THAT_WAY", ""),
        ),
    ]

    blocked = []
    total = 0
    passed = 0

    for field_name, text in fields:
        for sentence in split_sentences(text):
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue

            total += 1
            # TITLE is a human-readable paraphrase, so check explicit dates,
            # amounts, and sections there but do not infer names from title
            # capitalization. All narrative/evidence/decision fields retain
            # conservative name checking. Short/list fragments still retain
            # strict date/amount/section checks but skip name extraction.
            use_names = field_name != "TITLE"
            if len(sentence) < 30 or not has_verb_like_word(sentence):
                use_names = False
            ok, failed_entities = check_sentence(
                sentence,
                source_text,
                include_names=use_names,
            )
            if ok:
                passed += 1
            else:
                blocked.append(
                    {
                        "field": field_name,
                        "sentence": sentence[:500],
                        "failed_entities": failed_entities,
                    }
                )

    return {
        "case_id": case_id,
        "status": "CHECKED",
        "sentences_checked": total,
        "sentences_passed": passed,
        "sentences_blocked": len(blocked),
        "blocked_details": blocked,
    }


# ============================================================
# LOAD AND VALIDATE FINAL 225
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

with INPUT_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict) and isinstance(data.get("CASES"), list):
    cases = data["CASES"]
elif isinstance(data, list):
    cases = data
else:
    raise RuntimeError("Expected a list or an object containing a CASES list.")

if len(cases) != EXPECTED_CASES:
    raise RuntimeError(
        f"Expected {EXPECTED_CASES} cases but found {len(cases)} in {INPUT_FILE}"
    )

for index, case in enumerate(cases, start=1):
    if not isinstance(case, dict):
        raise RuntimeError(f"Case #{index} is not an object.")
    if "case_id" not in case:
        raise RuntimeError(f"Case #{index} has no case_id.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PRAMANA STEP 13A — GATE 0: ENTITY / NUMBER / SOURCE MATCH")
print(f"Project root : {PROJECT_ROOT}")
print(f"Input        : {INPUT_FILE}")
print(f"Cases        : {len(cases)}")
print("=" * 70)

results = []
no_source = []
blocked_cases = []

for index, case in enumerate(cases, start=1):
    result = check_case(case)
    results.append(result)

    if result["status"] == "NO_SOURCE":
        no_source.append(case["case_id"])
        print(f"[{index}] {case['case_id']}: NO_SOURCE")
    elif result["sentences_blocked"] > 0:
        blocked_cases.append(case["case_id"])
        print(
            f"[{index}] {case['case_id']}: "
            f"{result['sentences_passed']}/{result['sentences_checked']} passed, "
            f"{result['sentences_blocked']} blocked"
        )
    else:
        print(
            f"[{index}] {case['case_id']}: "
            f"ALL {result['sentences_checked']} PASSED"
        )

# ============================================================
# SUMMARY
# ============================================================

total_sentences = sum(r["sentences_checked"] for r in results)
total_blocked = sum(r["sentences_blocked"] for r in results)
checked_cases = len(cases) - len(no_source)
all_pass_cases = checked_cases - len(blocked_cases)
block_rate = (
    total_blocked / total_sentences * 100
    if total_sentences
    else 0.0
)

summary = {
    "expected_cases": EXPECTED_CASES,
    "cases_found": len(cases),
    "cases_with_source": checked_cases,
    "no_source_cases": len(no_source),
    "all_pass_cases": all_pass_cases,
    "cases_with_blocks": len(blocked_cases),
    "total_sentences": total_sentences,
    "total_blocked": total_blocked,
    "block_rate_percent": round(block_rate, 3),
    "under_3_percent": block_rate < 3.0,
    "no_source_case_ids": no_source,
    "blocked_case_ids": blocked_cases,
}

print("\n" + "=" * 70)
print("GATE 0 SUMMARY")
print("=" * 70)
for key, value in summary.items():
    print(f"{key}: {value}")
print("\nIMPORTANT: NO_SOURCE cases are not counted as safe passes.")
print("Review every blocked sentence manually before proceeding to Gate 1.")

with RESULTS_FILE.open("w", encoding="utf-8") as f:
    json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

with NO_SOURCE_FILE.open("w", encoding="utf-8") as f:
    json.dump(no_source, f, indent=2, ensure_ascii=False)

blocked_output = [r for r in results if r["sentences_blocked"] > 0]
with BLOCKED_FILE.open("w", encoding="utf-8") as f:
    json.dump(blocked_output, f, indent=2, ensure_ascii=False)

print(f"\nSaved: {RESULTS_FILE}")
print(f"Saved: {BLOCKED_FILE}")
print(f"Saved: {NO_SOURCE_FILE}")

if no_source:
    print("\nSTOP: resolve NO_SOURCE cases before Gate 1.")
elif block_rate >= 3.0:
    print("\nSTOP: block rate is 3% or higher; review blocked sentences and adjust only verified false positives.")
else:
    print("\nBlock rate is under 3%, but manual review is still required before Gate 1.")

print("Done.")

if __name__ == "__main__":
    pass
