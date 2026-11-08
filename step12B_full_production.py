"""
PRAMANA — STEP 12B
EXACT 225-CASE STORY PRODUCTION

Purpose
-------
Generate the final user-facing Story for the EXACT 225 user-locked in-domain
case IDs in ONE production run. The 538-case corpus remains frozen; the
225-case target manifest is the production authority.

The V3 Story prompt is frozen. The 50-case validation was only a prompt-quality
validation exercise; production does NOT depend on having 45/50 validation
results saved first.

Safety / integrity
------------------
- Model: openai/gpt-oss-120b
- Frozen ChromaDB is READ ONLY.
- Chroma must contain exactly 4,855 records.
- The five known out-of-domain cases are identified deterministically from the
  frozen 50-case selection and excluded.
- All other 225 cases are production targets.
- No fixed Story word-count requirement.
- No automatic style score is used as an acceptance gate.
- Facts must come only from supplied source text.
- Missing/unclear facts are omitted rather than guessed.
- Progress is saved after every case.
- The script is resume-safe and can continue after interruption.
- A failed case is never counted as successfully generated.
- Final automated PASS requires all 225 cases successfully generated and Chroma
  unchanged.
- Human semantic review remains the final quality gate.

Important API note
------------------
The observed Groq free-tier TPM limit is 8,000 tokens. The source context and
completion caps below are API-safety controls, NOT Story-length requirements.
The script deliberately leaves safety headroom and waits between requests.
"""

import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import chromadb
from groq import Groq


# ================================================================
# CONFIGURATION
# ================================================================

MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "V4_EVIDENCE_FIXED"
TOKEN_USAGE_LOGGING = True

CHROMA_PATH = Path("data/step6_v14/chroma_db")
COLLECTION_NAME = "pramana_v14"
EXPECTED_CHROMA_COUNT = 4855

OUTPUT_DIR = Path("data/step12")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "step12_full_225_story_results.json"
CSV_PATH = OUTPUT_DIR / "step12_full_225_story_results.csv"
SELECTION_PATH = OUTPUT_DIR / "step12_full_225_selection_manifest.csv"
MANUAL_REVIEW_PATH = OUTPUT_DIR / "step12_full_225_manual_review.md"
QC_PATH = OUTPUT_DIR / "step12_full_225_qc_report.txt"

EXPECTED_CORPUS_CASES = 538

# Historical Step 12A validation only. Not used for production targeting.
VALIDATION_SELECTED_CASES = 50
EXCLUDED_VALIDATION_INDICES = {9, 10, 18, 25, 40}
EXPECTED_VALIDATION_OOD_CASES = len(EXCLUDED_VALIDATION_INDICES)

# FROZEN PRODUCTION TARGET
EXPECTED_TARGET_CASES = 225

TARGET_CASE_IDS = ["1990_340", "1992_583", "1992_706", "1993_143", "1993_238", "1993_390", "1993_67", "1994_1053", "1995_336", "1996_1214", "1996_1275", "1996_249", "1996_563", "1997_838", "1998_1029", "1998_1030", "1998_531", "1998_759", "1998_822", "1999_1048", "1999_160", "1999_169", "1999_186", "1999_252", "1999_32", "1999_363", "1999_431", "1999_638", "1999_726", "1999_784", "1999_808", "1999_928", "2000_111", "2000_1119", "2000_113", "2000_151", "2000_31", "2000_372", "2000_461", "2000_737", "2000_739", "2001_1", "2001_1006", "2001_1011", "2001_1095", "2001_1122", "2001_121", "2001_133", "2001_185", "2001_469", "2001_64", "2001_640", "2001_747", "2001_787", "2001_79", "2001_800", "2001_909", "2002_411", "2002_502", "2002_504", "2002_738", "2003_1035", "2003_1238", "2003_1318", "2003_160", "2003_689", "2003_747", "2004_122", "2004_235", "2004_535", "2004_669", "2004_989", "2005_601", "2005_700", "2005_733", "2005_734", "2005_909", "2006_267", "2006_73", "2006_985", "2007_1118", "2007_1326", "2007_1370", "2007_1405", "2007_1451", "2007_1477", "2007_22", "2007_293", "2007_298", "2007_5", "2007_621", "2007_72", "2007_769", "2007_780", "2007_789", "2007_862", "2007_902", "2008_1017", "2008_1102", "2008_1116", "2008_1190", "2008_1191", "2008_1195", "2008_132", "2008_1472", "2008_1488", "2008_1579", "2008_1588", "2008_1617", "2008_1720", "2008_1832", "2008_2009", "2008_203", "2008_2098", "2008_2155", "2008_2439", "2008_2482", "2008_2808", "2008_404", "2008_460", "2008_612", "2008_641", "2008_978", "2009_1040", "2009_1117", "2009_1308", "2009_1413", "2009_1517", "2009_1631", "2009_179", "2009_1792", "2009_1852", "2009_2138", "2009_285", "2009_426", "2009_445", "2009_467", "2009_517", "2009_61", "2009_702", "2009_738", "2009_743", "2009_827", "2009_887", "2010_1250", "2010_1266", "2010_29", "2010_3", "2010_319", "2010_595", "2010_769", "2010_812", "2010_973", "2011_100", "2011_102", "2011_1183", "2011_193", "2011_256", "2011_292", "2011_53", "2011_6", "2011_770", "2011_802", "2011_895", "2012_164", "2012_176", "2012_252", "2012_287", "2012_31", "2012_484", "2012_60", "2013_1034", "2013_1054", "2013_326", "2013_358", "2013_513", "2013_640", "2013_67", "2013_854", "2013_9", "2013_914", "2013_921", "2013_938", "2014_28", "2014_295", "2014_341", "2014_344", "2014_349", "2014_366", "2014_430", "2014_474", "2014_513", "2014_515", "2014_596", "2014_655", "2014_674", "2015_131", "2015_30", "2015_600", "2015_606", "2015_737", "2015_95", "2016_347", "2016_429", "2016_91", "2017_124", "2017_194", "2017_264", "2017_281", "2017_515", "2017_692", "2017_96", "2018_313", "2018_319", "2018_938", "2018_962", "2019_1019", "2019_1128", "2019_1157", "2019_209", "2019_222", "2019_419", "2019_49", "2020_134", "2020_139"]


# Authoritative Stage 6 DOMAIN set from PRAMANA_STAGE4_5_6_STRICT_AUDIT.
# This is embedded so Step 12B cannot silently substitute another selection.
STAGE6_DOMAIN_CASE_IDS = ["1947_1", "1947_274", "1947_63", "1948_0", "1948_9", "1960_38", "1961_432", "1962_336", "1962_347", "1963_279", "1964_181", "1965_108", "1966_263", "1969_250", "1969_258", "1982_92", "1983_327", "1987_170", "1987_257", "1987_619", "1990_340", "1992_583", "1992_706", "1993_143", "1993_238", "1993_390", "1993_67", "1994_1053", "1995_336", "1996_1214", "1996_1275", "1996_249", "1996_563", "1997_838", "1998_1029", "1998_1030", "1998_531", "1998_759", "1998_822", "1999_1048", "1999_160", "1999_169", "1999_186", "1999_252", "1999_32", "1999_363", "1999_431", "1999_638", "1999_726", "1999_784", "1999_808", "1999_928", "2000_111", "2000_1119", "2000_113", "2000_151", "2000_31", "2000_372", "2000_461", "2000_737", "2000_739", "2001_1", "2001_1006", "2001_1011", "2001_1095", "2001_1122", "2001_121", "2001_133", "2001_185", "2001_469", "2001_64", "2001_640", "2001_747", "2001_787", "2001_79", "2001_800", "2001_909", "2002_411", "2002_502", "2002_504", "2002_738", "2003_1035", "2003_1238", "2003_1318", "2003_160", "2003_689", "2003_747", "2004_122", "2004_235", "2004_535", "2004_669", "2004_989", "2005_601", "2005_700", "2005_733", "2005_734", "2005_909", "2006_267", "2006_73", "2006_985", "2007_1118", "2007_1326", "2007_1370", "2007_1405", "2007_1451", "2007_1477", "2007_22", "2007_293", "2007_298", "2007_5", "2007_621", "2007_72", "2007_769", "2007_780", "2007_789", "2007_862", "2007_902", "2008_1017", "2008_1102", "2008_1116", "2008_1190", "2008_1191", "2008_1195", "2008_132", "2008_1472", "2008_1488", "2008_1579", "2008_1588", "2008_1617", "2008_1720", "2008_1832", "2008_2009", "2008_203", "2008_2098", "2008_2155", "2008_2439", "2008_2482", "2008_2808", "2008_404", "2008_460", "2008_612", "2008_641", "2008_978", "2009_1040", "2009_1117", "2009_1308", "2009_1413", "2009_1517", "2009_1631", "2009_179", "2009_1792", "2009_1852", "2009_2138", "2009_285", "2009_426", "2009_445", "2009_467", "2009_517", "2009_61", "2009_702", "2009_738", "2009_743", "2009_827", "2009_887", "2010_1250", "2010_1266", "2010_29", "2010_3", "2010_319", "2010_595", "2010_769", "2010_812", "2010_973", "2011_100", "2011_102", "2011_1183", "2011_193", "2011_256", "2011_292", "2011_53", "2011_6", "2011_770", "2011_802", "2011_895", "2012_164", "2012_176", "2012_252", "2012_287", "2012_31", "2012_484", "2012_60", "2013_1034", "2013_1054", "2013_326", "2013_358", "2013_513", "2013_640", "2013_67", "2013_854", "2013_9", "2013_914", "2013_921", "2013_938", "2014_28", "2014_295", "2014_341", "2014_344", "2014_349", "2014_366", "2014_430", "2014_474", "2014_513", "2014_515", "2014_596", "2014_655", "2014_674", "2015_131", "2015_30", "2015_600", "2015_606", "2015_737", "2015_95", "2016_347", "2016_429", "2016_91", "2017_124", "2017_194", "2017_264", "2017_281", "2017_515", "2017_692", "2017_96", "2018_313", "2018_319", "2018_938", "2018_962", "2019_1019", "2019_1128", "2019_1157", "2019_209", "2019_222", "2019_419", "2019_49", "2020_134", "2020_139"]


# API context-safety caps only; NEVER Story word-count requirements.
MAX_SOURCE_CHARS = 4200
MAX_COMPLETION_TOKENS = 800

MAX_ATTEMPTS = 3

# Groq Free Plan limits used by this production script.
# These are organization-level limits for the GPT-OSS models used here.
GROQ_TPM_LIMIT = 8_000
GROQ_TPD_LIMIT = 200_000

# Pramana safety buffer. This is NOT a Groq requirement.
# The script stops before intentionally consuming the final 1,000 tokens.
GROQ_TPD_SAFETY_MARGIN = 1_000

# IMPORTANT:
# The local TPD counter must start from the amount already consumed by this
# Groq organization TODAY if the script is started mid-day.
#
# The known dashboard usage at the time this script was prepared was 197,892.
# Set this to 0 ONLY after the Groq daily allowance has reset, or replace it
# with the current dashboard's organization usage before the first run.
GROQ_TPD_BOOTSTRAP_USAGE = 197_892

# No fixed successful-request delay. TPM pacing is controlled from Groq's
# x-ratelimit-remaining-tokens and x-ratelimit-reset-tokens headers.
FALLBACK_TPM_DELAY_SECONDS = 10.0
TPM_RESET_BUFFER_SECONDS = 0.5

# Retry backoff if a 429 does not provide a usable Retry-After/reset header.
RATE_LIMIT_FALLBACK_DELAYS = (5, 15, 30)

# Conservative estimate used only to prevent obviously oversized requests
# and to reserve enough local TPD budget before each request.
ESTIMATED_CHARS_PER_TOKEN = 3.0
MAX_ESTIMATED_REQUEST_TOKENS = 7600

TPD_USAGE_PATH = OUTPUT_DIR / "groq_tpd_usage.json"

# Persistent evidence-upgrade resume state. This is separate from the final
# results file so completed evidence chunks survive a quota stop/restart.
EVIDENCE_PROGRESS_PATH = OUTPUT_DIR / "evidence_upgrade_progress.json"

# Runtime-only telemetry from the most recent Groq response.
last_groq_headers = {}


# ================================================================
# GROQ STARTUP
# ================================================================

client = None

def get_groq_client():
    global client
    if client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        client = Groq(api_key=api_key)
    return client


# ================================================================
# ABSOLUTE STORY PROMPT — V3
# ================================================================

STORY_PROMPT = r"""
You are the Story writer for PRAMANA, an Indian legal information and research system designed to help ordinary people understand real court cases.

ABSOLUTE RULE: Use ONLY the supplied SOURCE TEXT. Never invent, infer, or fill gaps.
Explain everything so a 5th-class student can understand it, while preserving legally important facts and exact legal terminology when necessary.

The final output MUST contain EXACTLY these five sections and no others:

TITLE: <short, clear title>

STORY:
<natural chronological narrative of what happened; no headings inside this section>

EVIDENCE PRESENTED:
<exact evidence actually presented, produced, marked, filed, proved, testified to, or otherwise placed before the court, ONLY to the extent stated by the SOURCE TEXT>

COURT DECISION AND WHAT IT MEANS IN SIMPLE WORDS:
<what the Court actually decided AND immediately explain what that decision means in ordinary English; these are ONE combined section, not two separate sections>

WHY THE COURT DECIDED THAT WAY:
<the Court's actual reasoning, explained simply and only where supported by SOURCE TEXT>

EVIDENCE RULES — HIGHEST PRIORITY:
- Identify the actual evidence in this case, not merely the legal issue.
- Preserve exact evidence names/details when stated: documents, cheques, receipts, notices, agreements, certificates, reports, exhibits, affidavits, witness testimony, admissions, statements, photographs, records, etc.
- Distinguish evidence from procedure. Cross-examination, examination-in-chief and re-examination are procedures involving testimony; explain them simply and state exactly what the source says happened.
- For each important technical term, keep the legal term and immediately explain it in simple English.
- Example: "Cross-examination (questions asked to a witness by the other side to test what the witness said). In this case, the source states that ..."
- State exactly what was submitted/given/produced/testified, if the source says so.
- Do NOT convert a general statement of law into evidence in this particular case.
- Do NOT call an allegation an established fact.
- If the available SOURCE TEXT does not state what evidence was presented, explicitly say: "The available source does not specify the exact evidence presented."
- Never guess what a document contained, what a witness said, or what an exhibit proved.

STORY RULES:
- Begin directly with the first important incident.
- Use simple natural English, short clear sentences, and chronological order where possible.
- Preserve names, dates, amounts, courts, Acts, sections, orders, decrees and other legally material details when supported.
- Do not write fiction, dialogue, motives, emotions, or consequences not supported by the source.
- Do not use outside knowledge.
- Do not use bullet points or numbered lists inside STORY.
- No citations, metadata, comments, or discussion of the task.
- There is NO word limit.

COURT DECISION RULES:
- State exactly what happened to the appeal, petition, complaint, conviction, order, decree, claim, or other proceeding.
- Say which side succeeded or failed only if the source establishes it.
- Combine the decision and its simple meaning into the ONE required section above.
- Explain WHY in the separate final section.

SOURCE TEXT:
{source_text}

CASE ID: {case_id}
DECISION YEAR: {year}
"""

# ================================================================
# BASIC UTILITIES
# ================================================================

def clean_text(text):
    text = str(text or "").strip()
    text = re.sub(r"^\s*```(?:text|markdown)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text).strip()
    return text


def word_count(text):
    return len(re.findall(r"\b[\w₹$'-]+\b", text or ""))


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def chunk_number(chunk_id):
    match = re.search(r"_C(\d+)$", str(chunk_id))
    return int(match.group(1)) if match else 10**9


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_chroma_directory():
    parts = []
    for path in sorted(CHROMA_PATH.rglob("*")):
        if path.is_file():
            try:
                parts.append(
                    f"{path.relative_to(CHROMA_PATH)}:{sha256_file(path)}"
                )
            except OSError:
                pass
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# ================================================================
# CHROMA LOADING — READ ONLY
# ================================================================

def load_chunks(collection):
    data = collection.get(include=["documents", "metadatas"])

    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not (len(ids) == len(documents) == len(metadatas)):
        raise RuntimeError("Chroma ids/documents/metadatas length mismatch.")

    rows = []

    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        metadata = metadata or {}
        case_id = str(metadata.get("case_id", "")).strip()

        if not case_id:
            raise RuntimeError(f"Missing case_id for chunk {chunk_id}")

        text = str(document or "").strip()
        if not text:
            raise RuntimeError(f"Empty document for chunk {chunk_id}")

        rows.append({
            "chunk_id": str(chunk_id),
            "case_id": case_id,
            "year": safe_int(metadata.get("decision_year", 0)),
            "text": text,
            "chars": len(text),
        })

    return rows


# ================================================================
# SIGNALS FOR DIVERSITY SELECTION ONLY
# ================================================================

def factual_score(text):
    t = text.lower()
    patterns = [
        r"\bpetitioner\b", r"\brespondent\b", r"\bappellant\b",
        r"\bdefendant\b", r"\bplaintiff\b", r"\bfiled\b",
        r"\bclaimed\b", r"\bpaid\b", r"\bagreement\b", r"\bnotice\b",
        r"\border\b", r"\bappeal\b", r"\bjudgment\b", r"\bheld\b",
        r"\bdecided\b", r"\bcourt\b", r"\bdispute\b", r"\bevidence\b",
        r"\bsection\b", r"\bact\b", r"\bdismissed\b", r"\ballowed\b",
        r"\bconvicted\b", r"\bacquitted\b",
    ]
    score = sum(1 for pattern in patterns if re.search(pattern, t))
    score += min(5, len(re.findall(r"\b\d{4}\b", text)))
    score += min(5, len(re.findall(r"\b\d+(?:,\d{3})*\b", text)))
    return score


def build_source_context(chunks):
    full_text = "\n\n".join(x["text"] for x in chunks)

    if len(full_text) <= MAX_SOURCE_CHARS:
        return full_text

    # Three-region coverage is safer than only first+last because a long
    # judgment can contain the central dispute/reasoning in the middle.
    marker1 = "\n\n[Middle portion selected for context.]\n\n"
    marker2 = "\n\n[Ending portion selected for context.]\n\n"
    available = MAX_SOURCE_CHARS - len(marker1) - len(marker2)
    region = max(1, available // 3)

    beginning = full_text[:region]
    middle_start = max(0, (len(full_text) - region) // 2)
    middle = full_text[middle_start:middle_start + region]
    ending = full_text[-region:]

    context = beginning + marker1 + middle + marker2 + ending
    return context[:MAX_SOURCE_CHARS]


# ================================================================
# CASE CONSTRUCTION
# ================================================================

def build_cases(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)

    if len(grouped) != EXPECTED_CORPUS_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_CORPUS_CASES} unique cases in frozen Chroma; "
            f"found {len(grouped)}."
        )

    cases = []

    for case_id, chunks in grouped.items():
        chunks = sorted(
            chunks,
            key=lambda x: (chunk_number(x["chunk_id"]), x["chunk_id"]),
        )

        year = next((x["year"] for x in chunks if x["year"]), 0)
        all_text = " ".join(x["text"] for x in chunks)
        chars = len(all_text)
        chunk_count = len(chunks)

        participant_signals = sum(
            len(re.findall(pattern, all_text, flags=re.I))
            for pattern in [
                r"\bpetitioner\b", r"\brespondent\b", r"\bappellant\b",
                r"\bdefendant\b", r"\bplaintiff\b",
            ]
        )

        chronology_signals = len(re.findall(
            r"\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|"
            r"July|August|September|October|November|December)\b",
            all_text,
            flags=re.I,
        ))

        action_signals = len(re.findall(
            r"\b(?:filed|paid|sold|bought|arrested|terminated|dismissed|"
            r"appealed|issued|agreed|refused|claimed|ordered|held|decided|"
            r"convicted|acquitted|granted|rejected)\b",
            all_text,
            flags=re.I,
        ))

        procedural_signals = len(re.findall(
            r"\b(?:appeal|petition|writ|review|tribunal|high court|supreme court|"
            r"trial court|proceeding|judgment|decree)\b",
            all_text,
            flags=re.I,
        ))

        issue_signals = len(re.findall(
            r"\b(?:whether|issue|question|section|law|act|right|liable|"
            r"valid|invalid)\b",
            all_text,
            flags=re.I,
        ))

        document_signals = len(re.findall(
            r"\b(?:agreement|notice|certificate|document|exhibit|receipt|"
            r"report|letter|record)\b",
            all_text,
            flags=re.I,
        ))

        complexity_score = (
            min(25, chunk_count * 2)
            + min(20, chars // 5000)
            + min(10, participant_signals)
            + min(10, chronology_signals)
            + min(10, action_signals // 3)
            + min(10, procedural_signals // 3)
            + min(10, issue_signals // 5)
            + min(5, document_signals // 3)
        )

        representative = max(
            chunks,
            key=lambda x: (factual_score(x["text"]), -chunk_number(x["chunk_id"])),
        )

        cases.append({
            "case_id": case_id,
            "year": year,
            "chunk_count": chunk_count,
            "total_chars": chars,
            "representative_chunk_id": representative["chunk_id"],
            "story_source": build_source_context(chunks),
            "full_source": "\n\n".join(x["text"] for x in chunks),
            "complexity_score": complexity_score,
        })

    return cases


# ================================================================
# COMPLEXITY LABELING
# ================================================================

def assign_complexity(cases):
    ordered = sorted(
        cases,
        key=lambda x: (x["complexity_score"], x["case_id"]),
    )

    n = len(ordered)

    for index, case in enumerate(ordered):
        fraction = index / max(1, n - 1)
        if fraction < 1 / 3:
            label = "SIMPLE"
        elif fraction >= 2 / 3:
            label = "COMPLEX"
        else:
            label = "MODERATE"
        case["complexity_label"] = label

    return cases


# ================================================================
# MAXIMUM-RANGE 50-CASE SELECTION
# ================================================================

def select_50(cases):
    """
    Maximum-range deterministic selection.

    1. Sort all cases chronologically.
    2. Split the full corpus into five contiguous chronological strata.
    3. Select exactly 10 from every stratum.
    4. Within each stratum sample 3 SIMPLE, 4 MODERATE, 3 COMPLEX.

    This deliberately covers early, middle and recent portions of the frozen
    538-case corpus instead of taking the first 50 cases.
    """
    ordered = sorted(cases, key=lambda x: (x["year"], x["case_id"]))

    if len(ordered) < VALIDATION_SELECTED_CASES:
        raise RuntimeError(f"Only {len(ordered)} cases available.")

    base = len(ordered) // 5
    remainder = len(ordered) % 5
    strata = []
    cursor = 0

    for band in range(5):
        size = base + (1 if band < remainder else 0)
        pool = ordered[cursor:cursor + size]
        cursor += size

        if not pool:
            raise RuntimeError(f"Chronological stratum {band} is empty.")

        for case in pool:
            case["year_band"] = band
        strata.append(pool)

    selected = []

    for band, pool in enumerate(strata):
        if len(pool) < 10:
            raise RuntimeError(
                f"Chronological stratum {band} has only {len(pool)} cases."
            )

        ordered_by_complexity = sorted(
            pool,
            key=lambda x: (x["complexity_score"], x["case_id"]),
        )
        n = len(ordered_by_complexity)

        def sample_bucket(start, end, count):
            left = int(n * start)
            right = max(left + 1, int(n * end))
            bucket = ordered_by_complexity[left:right]
            if len(bucket) <= count:
                return bucket
            positions = [
                round(i * (len(bucket) - 1) / max(1, count - 1))
                for i in range(count)
            ]
            return [bucket[p] for p in positions]

        chosen = (
            sample_bucket(0.0, 1 / 3, 3)
            + sample_bucket(1 / 3, 2 / 3, 4)
            + sample_bucket(2 / 3, 1.0, 3)
        )

        unique = {}
        for case in chosen:
            unique[case["case_id"]] = case
        chosen = list(unique.values())

        if len(chosen) < 10:
            chosen_ids = {x["case_id"] for x in chosen}
            remaining = [
                x for x in ordered_by_complexity
                if x["case_id"] not in chosen_ids
            ]
            chosen.extend(remaining[:10 - len(chosen)])

        if len(chosen) != 10:
            raise RuntimeError(
                f"Stratum {band} selection produced {len(chosen)} cases."
            )

        selected.extend(chosen)

    if len(selected) != VALIDATION_SELECTED_CASES:
        raise RuntimeError(
            f"Selection produced {len(selected)} cases; expected {VALIDATION_SELECTED_CASES}."
        )

    if len({x["case_id"] for x in selected}) != VALIDATION_SELECTED_CASES:
        raise RuntimeError("Selection contains duplicate case IDs.")

    selected.sort(key=lambda x: (x["year"], x["case_id"]))

    for index, case in enumerate(selected, 1):
        case["validation_index"] = index

    return selected


# ================================================================
# PRODUCTION MANIFEST
# ================================================================

def validate_target_manifest(target_ids, stage6_ids, chroma_case_ids):
    """Fail-closed validation of the exact 225-case production manifest."""
    target_ids = list(target_ids)
    stage6_ids = list(stage6_ids)
    chroma_case_ids = set(chroma_case_ids)

    target_set = set(target_ids)
    stage6_set = set(stage6_ids)

    duplicates = sorted({x for x in target_ids if target_ids.count(x) > 1})
    expected_target_order = [
        case_id for case_id in stage6_ids
        if safe_int(str(case_id).split("_", 1)[0], 0) >= 1990
    ]

    in_stage6 = [x for x in target_ids if x in stage6_set]
    missing_stage6 = [x for x in target_ids if x not in stage6_set]
    missing_chroma = [x for x in target_ids if x not in chroma_case_ids]
    extra_chroma = sorted(chroma_case_ids - target_set)

    print("\n" + "=" * 72)
    print("STEP 12B — EXACT 225 TARGET MANIFEST VALIDATION")
    print("=" * 72)
    print(f"TARGET COUNT              = {len(target_ids)}")
    print(f"TARGET UNIQUE COUNT       = {len(target_set)}")
    print(f"TARGET DUPLICATES         = {len(duplicates)}")
    print(f"TARGET IN STAGE-6 SET     = {len(in_stage6)}/{len(target_ids)}")
    print(f"TARGET MISSING FROM STAGE-6 = {len(missing_stage6)}")
    print(f"TARGET IN CHROMA          = {len(target_ids) - len(missing_chroma)}/{len(target_ids)}")
    print(f"TARGET MISSING FROM CHROMA= {len(missing_chroma)}")
    print("TARGET ORDER              = PRESERVED")

    if len(target_ids) != EXPECTED_TARGET_CASES:
        raise RuntimeError(f"TARGET COUNT FAILURE: {len(target_ids)} != {EXPECTED_TARGET_CASES}")
    if len(target_set) != EXPECTED_TARGET_CASES:
        raise RuntimeError("TARGET DUPLICATE FAILURE: target IDs are not unique.")
    if duplicates:
        raise RuntimeError(f"TARGET DUPLICATES: {duplicates}")
    if len(stage6_ids) != 245 or len(stage6_set) != 245:
        raise RuntimeError(
            f"STAGE-6 AUTHORITY FAILURE: expected 245 unique DOMAIN IDs; "
            f"found {len(stage6_ids)} list / {len(stage6_set)} unique."
        )
    if target_ids != expected_target_order:
        raise RuntimeError(
            "TARGET ORDER/CONTENT FAILURE: target list is not exactly the "
            "Stage-6 DOMAIN list filtered to years >= 1990."
        )
    if missing_stage6:
        raise RuntimeError(f"TARGET IDs NOT IN STAGE-6: {missing_stage6}")
    if missing_chroma:
        raise RuntimeError(f"TARGET IDs MISSING FROM CHROMA: {missing_chroma}")

    decade_counts = {}
    for case_id in target_ids:
        year = safe_int(case_id.split("_", 1)[0], 0)
        decade = (year // 10) * 10
        decade_counts[decade] = decade_counts.get(decade, 0) + 1

    expected_decades = {1990: 32, 2000: 112, 2010: 79, 2020: 2}
    if decade_counts != expected_decades:
        raise RuntimeError(
            f"TARGET DECADE COUNT FAILURE: {decade_counts} != {expected_decades}"
        )

    if "1992_92" in target_set:
        raise RuntimeError("FORBIDDEN TARGET FAILURE: 1992_92 must not be generated.")

    print("1990s TARGETS             = 32 PASS")
    print("2000s TARGETS             = 112 PASS")
    print("2010s TARGETS             = 79 PASS")
    print("2020s TARGETS             = 2 PASS")
    print("MANIFEST STATUS           = PASS")
    print("=" * 72)
    return True


def build_production_targets(all_cases):
    """Build the EXACT user-specified 225-case production set."""

    if len(all_cases) != EXPECTED_CORPUS_CASES:
        raise RuntimeError(
            f"Expected frozen corpus to contain {EXPECTED_CORPUS_CASES} cases; "
            f"found {len(all_cases)}."
        )

    # One case_id must map to exactly one frozen case.
    by_id = {str(case["case_id"]).strip(): case for case in all_cases}

    if len(by_id) != len(all_cases):
        raise RuntimeError("Frozen corpus contains duplicate case IDs.")

    # TARGET_CASE_IDS is intentionally an ORDERED LIST.
    # Use a set only for validation; never use it to build production order.
    target_id_set = set(TARGET_CASE_IDS)

    if len(TARGET_CASE_IDS) != EXPECTED_TARGET_CASES:
        raise RuntimeError(
            f"Production target list must contain exactly "
            f"{EXPECTED_TARGET_CASES} IDs; "
            f"found {len(TARGET_CASE_IDS)}."
        )

    if len(target_id_set) != EXPECTED_TARGET_CASES:
        raise RuntimeError(
            "Production target list contains duplicate case IDs."
        )

    missing = sorted(target_id_set - set(by_id))

    if missing:
        raise RuntimeError(
            f"{len(missing)} requested target case IDs are missing "
            f"from frozen Chroma: {missing[:20]}"
        )

    # IMPORTANT: list order is the frozen production order.
    targets = [by_id[cid] for cid in TARGET_CASE_IDS]

    if len(targets) != EXPECTED_TARGET_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGET_CASES} production targets; "
            f"built {len(targets)}."
        )

    for production_index, case in enumerate(targets, start=1):
        case["production_index"] = production_index

    # Fail closed if anything changed the target identity or order.
    actual_target_ids = [case["case_id"] for case in targets]

    if actual_target_ids != TARGET_CASE_IDS:
        raise RuntimeError(
            "Production target ordering/content differs from "
            "the frozen 225-ID list."
        )

    return targets

def save_manifest(all_cases, targets):
    target_by_id = {x["case_id"]: x for x in targets}
    fields = [
        "production_index", "case_id", "year", "chunk_count", "total_chars",
        "complexity_label", "complexity_score", "representative_chunk_id",
        "production_status",
    ]
    with open(SELECTION_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in targets:
            writer.writerow({
                "production_index": case["production_index"],
                "case_id": case["case_id"], "year": case["year"],
                "chunk_count": case["chunk_count"], "total_chars": case["total_chars"],
                "complexity_label": case["complexity_label"],
                "complexity_score": case["complexity_score"],
                "representative_chunk_id": case["representative_chunk_id"],
                "production_status": "PRODUCTION_GENERATE",
            })


# ================================================================
# RESULT PERSISTENCE
# ================================================================

def save_results(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp = RESULTS_PATH.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(RESULTS_PATH)


def save_csv(results):
    fields = [
        "production_index", "case_id", "year", "complexity_label",
        "complexity_score", "chunk_count", "total_chars",
        "representative_chunk_id", "model", "prompt_version",
        "source_chars_sent", "prompt_chars", "generation_attempts",
        "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "tpd_accounted_tokens",
        "word_count", "status", "title", "story", "evidence",
        "decision_meaning", "why", "error",
        "generated_at", "generation_seconds",
    ]
    temp = CSV_PATH.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})
        f.flush()
        os.fsync(f.fileno())
    temp.replace(CSV_PATH)


def save_manual_review(results):
    parts = [
        "# PRAMANA — STEP 12B EXACT 225-CASE STORY MANUAL REVIEW",
        "",
        "CORE REVIEW TEST",
        "",
        "Can a 5th-class student who has never studied law understand the real "
        "incident, understand the Court's decision and explain the case to "
        "another person after reading it once?",
        "",
        "Check especially: factual accuracy, simple English, natural storytelling, "
        "clarity of the Court's decision, clarity of the Court's reason, and "
        "whether the Story is memorable without becoming fictional.",
        "",
    ]
    for result in sorted(results, key=lambda x: safe_int(x.get("production_index"), 999999)):
        if result.get("status") != "GENERATED":
            continue
        parts.extend([
            "---", "",
            f"## {result.get('production_index')}. {result.get('case_id')}", "",
            f"Year: {result.get('year', '')}",
            f"Complexity: {result.get('complexity_label', '')}",
            f"Story words: {result.get('word_count', '')} (observation only)", "",
            f"### {result.get('title', '')}", "",
            "### STORY", "",
            result.get("story", "[NO STORY]"), "",
            "### EVIDENCE PRESENTED", "",
            result.get("evidence", "[NO EVIDENCE SECTION]"), "",
            "### COURT DECISION AND WHAT IT MEANS IN SIMPLE WORDS", "",
            result.get("decision_meaning", "[NO DECISION SECTION]"), "",
            "### WHY THE COURT DECIDED THAT WAY", "",
            result.get("why", "[NO REASONING SECTION]"), "",
            "### HUMAN REVIEW", "",
            "5th-class understandable: ______________________________",
            "Can retell after one reading: _________________________",
            "Facts accurate: _______________________________________",
            "Legal meaning preserved: ______________________________",
            "Court decision crystal clear: _________________________",
            "Court reasoning clear: ________________________________",
            "Simple English: _______________________________________",
            "Natural storytelling: __________________________________",
            "Needs edit: ___________________________________________",
            "Wrong / reject: ________________________________________",
            "Comments: _____________________________________________", "",
        ])
    MANUAL_REVIEW_PATH.write_text("\n".join(parts), encoding="utf-8")
    return MANUAL_REVIEW_PATH


# ================================================================
# MODEL OUTPUT PARSING
# ================================================================

def parse_model_output(raw):
    text = clean_text(raw)

    patterns = {
        "title": r"(?is)^\s*TITLE\s*:\s*(.+?)\s*\n+\s*STORY\s*:",
        "story": r"(?is)\n+\s*STORY\s*:\s*(.+?)\s*\n+\s*EVIDENCE PRESENTED\s*:",
        "evidence": r"(?is)\n+\s*EVIDENCE PRESENTED\s*:\s*(.+?)\s*\n+\s*COURT DECISION AND WHAT IT MEANS IN SIMPLE WORDS\s*:",
        "decision_meaning": r"(?is)\n+\s*COURT DECISION AND WHAT IT MEANS IN SIMPLE WORDS\s*:\s*(.+?)\s*\n+\s*WHY THE COURT DECIDED THAT WAY\s*:",
        "why": r"(?is)\n+\s*WHY THE COURT DECIDED THAT WAY\s*:\s*(.+?)\s*$",
    }
    values = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            raise RuntimeError(f"Model output missing required section: {key}")
        values[key] = clean_text(m.group(1))
        if not values[key]:
            raise RuntimeError(f"Model returned empty required section: {key}")

    return (
        values["title"],
        values["story"],
        values["evidence"],
        values["decision_meaning"],
        values["why"],
    )


# ================================================================
# GROQ ERROR HANDLING
# ================================================================

class DailyQuotaExhausted(RuntimeError):
    """Raised when the local or Groq daily token budget is exhausted."""


def _parse_duration_seconds(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        pass

    # Accept Groq-style duration strings such as 1m30s, 2m, 0.5s.
    match = re.fullmatch(
        r"(?:(\d+(?:\.\d+)?)h)?"
        r"(?:(\d+(?:\.\d+)?)m)?"
        r"(?:(\d+(?:\.\d+)?)s)?",
        text,
        flags=re.I,
    )
    if not match:
        return None

    hours = float(match.group(1) or 0)
    minutes = float(match.group(2) or 0)
    seconds = float(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _header(headers, name):
    if not headers:
        return None

    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return value
    return None


def get_tpm_reset_seconds(headers):
    return _parse_duration_seconds(
        _header(headers, "x-ratelimit-reset-tokens")
    )


def get_rpd_reset_seconds(headers):
    return _parse_duration_seconds(
        _header(headers, "x-ratelimit-reset-requests")
    )


def get_remaining_tpm(headers):
    value = _header(headers, "x-ratelimit-remaining-tokens")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_remaining_rpd(headers):
    value = _header(headers, "x-ratelimit-remaining-requests")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_retry_after_seconds(headers):
    return _parse_duration_seconds(
        _header(headers, "retry-after")
    )


def print_rate_limit_status(headers):
    if not headers:
        return

    remaining_tpm = get_remaining_tpm(headers)
    tpm_reset = get_tpm_reset_seconds(headers)
    remaining_rpd = get_remaining_rpd(headers)
    rpd_reset = get_rpd_reset_seconds(headers)

    tpm_reset_text = (
        f"{tpm_reset:.2f}s" if tpm_reset is not None else "unknown"
    )
    rpd_reset_text = (
        f"{rpd_reset:.2f}s" if rpd_reset is not None else "unknown"
    )

    print(
        "GROQ LIMITS: "
        f"TPM remaining={remaining_tpm if remaining_tpm is not None else 'unknown'} | "
        f"TPM reset={tpm_reset_text}"
    )
    print(
        "             "
        f"RPD remaining={remaining_rpd if remaining_rpd is not None else 'unknown'} | "
        f"RPD reset={rpd_reset_text}"
    )


def estimate_request_tokens(prompt):
    return int(len(prompt) / ESTIMATED_CHARS_PER_TOKEN) + MAX_COMPLETION_TOKENS + 1


def wait_for_tpm_capacity(headers, next_estimated_tokens):
    """
    Do not impose a fixed delay after every successful request.

    If Groq says fewer TPM tokens remain than the next estimated request needs,
    wait for the actual TPM reset window reported by Groq.
    """
    remaining_tpm = get_remaining_tpm(headers)
    if remaining_tpm is None:
        return

    if remaining_tpm >= next_estimated_tokens:
        return

    reset_seconds = get_tpm_reset_seconds(headers)
    if reset_seconds is None:
        delay = FALLBACK_TPM_DELAY_SECONDS
        print(
            f"    TPM capacity insufficient "
            f"({remaining_tpm} remaining < {next_estimated_tokens} estimated); "
            f"reset header unavailable, waiting {delay:.1f}s"
        )
    else:
        delay = max(0.5, reset_seconds + TPM_RESET_BUFFER_SECONDS)
        print(
            f"    TPM capacity insufficient "
            f"({remaining_tpm} remaining < {next_estimated_tokens} estimated); "
            f"waiting {delay:.2f}s for TPM reset"
        )

    time.sleep(delay)


def is_rate_limit(exc):
    text = str(exc).lower()
    return any(token in text for token in [
        "429",
        "rate limit",
        "rate_limit_exceeded",
        "tokens per minute",
        "too many requests",
    ])


def is_daily_token_limit(exc):
    text = str(exc).lower()
    return any(token in text for token in [
        "tokens per day",
        "token per day",
        "tpd",
        "daily token",
        "daily tokens",
    ])


def is_request_too_large(exc):
    text = str(exc).lower()
    return any(token in text for token in [
        "413",
        "request too large",
        "request_too_large",
        "context length",
    ])


# ================================================================
# LOCAL GROQ DAILY TOKEN BUDGET
# ================================================================

def load_tpd_usage():
    today = time.strftime("%Y-%m-%d")

    if not TPD_USAGE_PATH.exists():
        usage = {
            "date": today,
            "total_tokens": int(GROQ_TPD_BOOTSTRAP_USAGE),
            "bootstrapped": True,
        }
        save_tpd_usage(usage)
        return usage

    try:
        data = json.loads(TPD_USAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        raise RuntimeError(
            f"Could not read local Groq TPD usage file: {TPD_USAGE_PATH}"
        )

    if data.get("date") != today:
        usage = {
            "date": today,
            "total_tokens": 0,
            "bootstrapped": False,
        }
        save_tpd_usage(usage)
        return usage

    return {
        "date": today,
        "total_tokens": safe_int(data.get("total_tokens"), 0),
        "bootstrapped": bool(data.get("bootstrapped", False)),
    }


def save_tpd_usage(usage):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp = TPD_USAGE_PATH.with_suffix(".tmp")

    with open(temp, "w", encoding="utf-8") as f:
        json.dump(usage, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    temp.replace(TPD_USAGE_PATH)


def record_tpd_usage(total_tokens):
    """
    Record actual Groq total_tokens after a completed request.

    If Groq does not return usage, the caller must pass a conservative estimate.
    """
    usage = load_tpd_usage()
    usage["total_tokens"] += int(total_tokens)
    save_tpd_usage(usage)
    return usage


def get_tpd_remaining():
    usage = load_tpd_usage()
    effective_limit = GROQ_TPD_LIMIT - GROQ_TPD_SAFETY_MARGIN
    return max(0, effective_limit - usage["total_tokens"])


def assert_tpd_capacity(estimated_tokens):
    usage = load_tpd_usage()
    effective_limit = GROQ_TPD_LIMIT - GROQ_TPD_SAFETY_MARGIN
    remaining = effective_limit - usage["total_tokens"]

    print(
        "PRAMANA DAILY TOKEN BUDGET: "
        f"{usage['total_tokens']:,} / {effective_limit:,} used | "
        f"{max(0, remaining):,} remaining"
    )

    if estimated_tokens > remaining:
        raise DailyQuotaExhausted(
            "GROQ_TPD_BUDGET_EXHAUSTED: "
            f"estimated next request={estimated_tokens:,}, "
            f"remaining local daily budget={max(0, remaining):,}. "
            "Stop this run and resume after the Groq daily quota resets."
        )


def extract_response_usage(response):
    usage = getattr(response, "usage", None)

    prompt_tokens = (
        getattr(usage, "prompt_tokens", None)
        if usage is not None else None
    )
    completion_tokens = (
        getattr(usage, "completion_tokens", None)
        if usage is not None else None
    )
    total_tokens = (
        getattr(usage, "total_tokens", None)
        if usage is not None else None
    )

    cached_tokens = None
    if usage is not None:
        prompt_details = getattr(
            usage,
            "prompt_tokens_details",
            None,
        )
        if prompt_details is not None:
            cached_tokens = getattr(
                prompt_details,
                "cached_tokens",
                None,
            )

    return (
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cached_tokens,
    )


# ================================================================
# STORY GENERATION
# ================================================================

def generate_story(case):
    global last_groq_headers

    prompt = STORY_PROMPT.format(
        case_id=case["case_id"],
        year=case["year"] or "not provided",
        source_text=case["story_source"],
    )

    if len(case["story_source"]) > MAX_SOURCE_CHARS:
        raise RuntimeError("Internal source-context cap failure.")

    estimated_input_tokens = int(
        len(prompt) / ESTIMATED_CHARS_PER_TOKEN
    ) + 1
    estimated_total_tokens = (
        estimated_input_tokens + MAX_COMPLETION_TOKENS
    )

    if estimated_total_tokens > MAX_ESTIMATED_REQUEST_TOKENS:
        raise RuntimeError(
            f"Estimated request too large for safety budget: "
            f"{estimated_total_tokens} > {MAX_ESTIMATED_REQUEST_TOKENS} tokens."
        )

    # Protect the daily quota BEFORE making the request.
    assert_tpd_capacity(estimated_total_tokens)

    # Protect the rolling TPM window using the previous response's headers.
    wait_for_tpm_capacity(
        last_groq_headers,
        estimated_total_tokens,
    )

    last_exc = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw_response = (
                get_groq_client()
                .chat.completions
                .with_raw_response
                .create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    reasoning_effort="low",
                    include_reasoning=False,
                )
            )

            # Preserve Groq's actual HTTP rate-limit telemetry.
            last_groq_headers = dict(raw_response.headers)
            print_rate_limit_status(last_groq_headers)

            response = raw_response.parse()

            (
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cached_tokens,
            ) = extract_response_usage(response)

            # If usage is available, account the ACTUAL total.
            # If unavailable, use the conservative request estimate so the
            # local safety counter cannot silently undercount this request.
            accounted_tokens = (
                int(total_tokens)
                if total_tokens is not None
                else int(estimated_total_tokens)
            )
            record_tpd_usage(accounted_tokens)

            raw = response.choices[0].message.content or ""

            try:
                title, story, evidence, decision_meaning, why = parse_model_output(raw)
            except Exception:
                # The API request has already consumed tokens. They remain
                # accounted even if the model output fails local parsing.
                raise

            return {
                "title": title,
                "story": story,
                "evidence": evidence,
                "decision_meaning": decision_meaning,
                "why": why,
                "attempts": attempt,
                "source_chars_sent": len(case["story_source"]),
                "prompt_chars": len(prompt),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": cached_tokens,
                "tpd_accounted_tokens": accounted_tokens,
            }

        except DailyQuotaExhausted:
            raise

        except Exception as exc:
            last_exc = exc

            if is_daily_token_limit(exc):
                raise DailyQuotaExhausted(
                    "Groq reported that the daily token allowance is exhausted: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            # A deterministic oversized request must never be blindly retried.
            if is_request_too_large(exc):
                break

            if attempt >= MAX_ATTEMPTS:
                break

            if is_rate_limit(exc):
                error_response = getattr(exc, "response", None)
                error_headers = getattr(
                    error_response,
                    "headers",
                    {},
                )

                retry_after = get_retry_after_seconds(error_headers)
                tpm_reset = get_tpm_reset_seconds(error_headers)

                if retry_after is not None:
                    delay = max(0.5, retry_after + TPM_RESET_BUFFER_SECONDS)
                    reason = "Retry-After"
                elif tpm_reset is not None:
                    delay = max(0.5, tpm_reset + TPM_RESET_BUFFER_SECONDS)
                    reason = "TPM reset"
                else:
                    delay = RATE_LIMIT_FALLBACK_DELAYS[
                        min(
                            attempt - 1,
                            len(RATE_LIMIT_FALLBACK_DELAYS) - 1,
                        )
                    ]
                    reason = "fallback"

                print(
                    f"    Rate limit; waiting {delay:.2f}s "
                    f"({reason}) before retry "
                    f"{attempt + 1}/{MAX_ATTEMPTS}"
                )
                time.sleep(delay)
            else:
                delay = 3 * attempt
                print(
                    f"    Groq error; waiting {delay}s before retry "
                    f"{attempt + 1}/{MAX_ATTEMPTS}"
                )
                time.sleep(delay)

    raise RuntimeError(
        f"Story generation failed after {MAX_ATTEMPTS} attempts: "
        f"{type(last_exc).__name__}: {last_exc}"
    )


# ================================================================
# RESUME SUPPORT
# ================================================================

def load_existing_results():
    if not RESULTS_PATH.exists():
        return []

    try:
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read existing results: {exc}")

    if not isinstance(data, list):
        raise RuntimeError("Existing results JSON is not a list.")

    return data


def result_key(result):
    return (
        safe_int(result.get("production_index"), 10**9),
        str(result.get("case_id", "")),
    )


# ================================================================
# PREFLIGHT
# ================================================================

def preflight(all_cases, targets, chroma_count, results):
    if chroma_count != EXPECTED_CHROMA_COUNT:
        raise RuntimeError(f"Chroma mismatch: {chroma_count} != {EXPECTED_CHROMA_COUNT}")
    if len(all_cases) != EXPECTED_CORPUS_CASES:
        raise RuntimeError(f"Expected {EXPECTED_CORPUS_CASES} cases; found {len(all_cases)}")
    if len(targets) != EXPECTED_TARGET_CASES:
        raise RuntimeError(f"Expected {EXPECTED_TARGET_CASES} requested targets; found {len(targets)}")
    print("\n" + "=" * 72)
    print("STEP 12B EXACT 225-CASE PRE-FLIGHT")
    print("=" * 72)
    print(f"Frozen Chroma records    : {chroma_count} PASS")
    print(f"Frozen corpus cases      : {len(all_cases)} PASS")
    print(f"User-specified targets   : {len(targets)} PASS")
    print(f"Existing saved results   : {len(results)}")
    print(f"Model                    : {MODEL}")
    print(f"Source context cap       : {MAX_SOURCE_CHARS} chars")
    print(f"Completion cap           : {MAX_COMPLETION_TOKENS} tokens")
    print(f"Groq TPM limit            : {GROQ_TPM_LIMIT:,}")
    print(f"Groq TPD limit            : {GROQ_TPD_LIMIT:,}")
    print(f"TPD safety margin         : {GROQ_TPD_SAFETY_MARGIN:,}")
    print("Story word-count gate    : NONE")
    print("Automatic style gate     : NONE")
    print("Human review gate        : YES")
    print("\nPRE-FLIGHT STATUS         : PASS")


# ================================================================
# QC REPORT
# ================================================================

def write_qc_report(all_cases, targets, results, initial_hash, final_hash):
    generated = [r for r in results if r.get("status") == "GENERATED"]
    failures = [r for r in results if r.get("status") == "GENERATION_FAILED"]
    counts = [word_count(r.get("story", "")) for r in generated]
    target_ids = {x["case_id"] for x in targets}
    successful_ids = {
        r.get("case_id") for r in generated
        if r.get("model") == MODEL and r.get("prompt_version") == PROMPT_VERSION
    }
    complete = (
        len(successful_ids) == EXPECTED_TARGET_CASES
        and target_ids == successful_ids
        and not failures
        and initial_hash == final_hash
    )
    lines = [
        "PRAMANA — STEP 12B EXACT 225-CASE STORY PRODUCTION QC REPORT",
        "=" * 72,
        f"Model: {MODEL}", f"Prompt version: {PROMPT_VERSION}",
        f"Frozen Chroma records: {EXPECTED_CHROMA_COUNT}",
        f"Frozen corpus cases: {len(all_cases)}",
        f"User-specified production cases: {EXPECTED_TARGET_CASES}",
        f"Generated: {len(successful_ids)}", f"Failures: {len(failures)}", "",
        f"Local TPD ledger: {load_tpd_usage()['total_tokens']:,} / "
        f"{GROQ_TPD_LIMIT - GROQ_TPD_SAFETY_MARGIN:,}",
        "Story word-count requirement: NONE",
        "Word counts are observation only, never an acceptance gate.",
        "Automatic Story style/quality validation: NONE",
        "Human semantic review: REQUIRED", "",
        f"Chroma SHA256 before: {initial_hash}",
        f"Chroma SHA256 after : {final_hash}",
        f"Chroma unchanged     : {'PASS' if initial_hash == final_hash else 'FAIL'}",
    ]
    if counts:
        lines += ["", "Observed Story word counts:", f"Minimum: {min(counts)}",
                  f"Maximum: {max(counts)}", f"Mean: {sum(counts)/len(counts):.2f}"]
    if failures:
        lines += ["", "FAILURES:"]
        for r in failures:
            lines.append(f"{r.get('production_index')} {r.get('case_id')}: {r.get('error')}")
    lines += ["", f"AUTOMATED STATUS: {'PASS' if complete else 'INCOMPLETE'}",
              "", "Automated PASS means generation and integrity checks completed.",
              "It does NOT guarantee semantic accuracy; human review is required."]
    QC_PATH.write_text("\n".join(lines), encoding="utf-8")
    return complete


# ================================================================
# EXISTING RESULT UPGRADE — ADD EVIDENCE WITHOUT REGENERATING STORY
# ================================================================

EVIDENCE_PROMPT = r"""
You are the Evidence extractor for PRAMANA.

Read ONLY the SOURCE TEXT below. Do not use outside knowledge and do not guess.
The goal is to state exactly what evidence was actually placed before the court in THIS case.

Return exactly:
EVIDENCE PRESENTED:
<one or more clear paragraphs>

Rules:
- Identify actual evidence described in the source: documents, cheques, receipts, notices, agreements, certificates, reports, exhibits, affidavits, witness testimony, admissions, statements, records, etc.
- Preserve exact names, dates, amounts, exhibit numbers and document descriptions when the source states them.
- Distinguish evidence from legal procedure. Cross-examination, examination-in-chief and re-examination are procedures involving witness testimony; explain the term simply and state exactly what the source says occurred.
- For every important legal/technical term, write the term and immediately give a simple explanation understandable to a 5th-class student.
- Then state exactly what happened in THIS case.
- Do not say a document proves something unless the source says it did or the judgment expressly records that finding.
- Do not turn allegations into facts.
- Do not invent missing evidence.
- If the source does not specify the exact evidence presented, write exactly: "The available source does not specify the exact evidence presented."
- No bullets or numbered lists.
- No citations, metadata, commentary, or discussion of this instruction.

SOURCE TEXT:
{source_text}

EXISTING STORY (context only; SOURCE TEXT remains the sole factual authority):
{existing_story}
"""


def build_full_source(case):
    """Return the complete frozen Chroma text for this case, without the 4200-char story cap."""
    # story_source is deliberately capped for ordinary story generation.
    # For evidence extraction we need the entire case because evidence can appear
    # anywhere in a long judgment.
    return case.get("full_source", case["story_source"])


EVIDENCE_CHUNK_PROMPT = r"""
You are the Evidence extractor for PRAMANA.

Read ONLY the SOURCE TEXT in this chunk. Do not use outside knowledge and do not guess.
The full case may be divided into multiple chunks. Extract ONLY evidence explicitly stated in this chunk.

Return exactly:
EVIDENCE PRESENTED:
<evidence paragraphs>

If this chunk contains no explicit evidence information, return exactly:
EVIDENCE PRESENTED:
NO_EVIDENCE_IN_THIS_CHUNK

Rules:
- Identify actual evidence described in the source: documents, cheques, receipts, notices, agreements, certificates, reports, exhibits, affidavits, witness testimony, admissions, statements, records, photographs, etc.
- Preserve exact names, dates, amounts, exhibit numbers and document descriptions when stated.
- Distinguish evidence from procedure. Cross-examination, examination-in-chief and re-examination are procedures involving testimony; explain the term simply and state exactly what the source says occurred.
- For each important technical term, keep the legal term and immediately explain it in simple language understandable to a 5th-class student.
- State exactly what happened in THIS case, only to the extent supported by THIS chunk.
- Do not say a document proves something unless this chunk says it did or the judgment expressly records that finding.
- Do not turn allegations into facts.
- Do not invent missing evidence.
- No bullets or numbered lists.
- No citations, metadata, or commentary.

Chunk {chunk_number} of {chunk_count}.

SOURCE TEXT:
{source_text}
"""

def _split_source_adaptively(source_text, max_chars):
    """Split source on paragraph boundaries; hard-split oversized paragraphs."""
    source_text = clean_text(source_text)
    if not source_text:
        return [""]

    paragraphs = re.split(r"\n\s*\n+", source_text)
    chunks, current = [], ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for pos in range(0, len(para), max_chars):
                chunks.append(para[pos:pos + max_chars])
            continue

        candidate = para if not current else current + "\n\n" + para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)
    return chunks or [source_text[:max_chars]]


def _request_evidence_chunk(prompt, estimated_total_tokens):
    """Send one safely bounded evidence extraction request."""
    global last_groq_headers

    assert_tpd_capacity(estimated_total_tokens)
    wait_for_tpm_capacity(last_groq_headers, estimated_total_tokens)

    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw_response = (
                get_groq_client().chat.completions.with_raw_response.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_completion_tokens=500,
                    reasoning_effort="low",
                    include_reasoning=False,
                )
            )
            last_groq_headers = dict(raw_response.headers)
            print_rate_limit_status(last_groq_headers)

            response = raw_response.parse()
            prompt_tokens, completion_tokens, total_tokens, cached_tokens = extract_response_usage(response)
            accounted_tokens = int(total_tokens) if total_tokens is not None else int(estimated_total_tokens)
            record_tpd_usage(accounted_tokens)

            raw = clean_text(response.choices[0].message.content or "")
            m = re.search(r"(?is)^\s*EVIDENCE PRESENTED\s*:\s*(.+?)\s*$", raw)
            if not m:
                raise RuntimeError("Evidence model output missing EVIDENCE PRESENTED section.")

            evidence = clean_text(m.group(1))
            if evidence.upper() == "NO_EVIDENCE_IN_THIS_CHUNK":
                evidence = ""

            return {
                "evidence": evidence,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": cached_tokens,
                "tpd_accounted_tokens": accounted_tokens,
                "attempts": attempt,
                "prompt_chars": len(prompt),
            }

        except DailyQuotaExhausted:
            raise
        except Exception as exc:
            last_exc = exc
            if is_daily_token_limit(exc):
                raise DailyQuotaExhausted(
                    "Groq reported that the daily token allowance is exhausted "
                    f"during Evidence extraction: {type(exc).__name__}: {exc}"
                ) from exc
            if is_request_too_large(exc):
                raise
            if attempt >= MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Evidence chunk request failed after {MAX_ATTEMPTS} attempts: "
                    f"{type(last_exc).__name__}: {last_exc}"
                ) from last_exc

            if is_rate_limit(exc):
                error_response = getattr(exc, "response", None)
                headers = getattr(error_response, "headers", {})
                retry_after = get_retry_after_seconds(headers)
                tpm_reset = get_tpm_reset_seconds(headers)
                if retry_after is not None:
                    delay = max(0.5, retry_after + TPM_RESET_BUFFER_SECONDS)
                elif tpm_reset is not None:
                    delay = max(0.5, tpm_reset + TPM_RESET_BUFFER_SECONDS)
                else:
                    delay = RATE_LIMIT_FALLBACK_DELAYS[min(attempt - 1, len(RATE_LIMIT_FALLBACK_DELAYS) - 1)]
                print(f"    Evidence rate limit; waiting {delay:.2f}s before retry {attempt + 1}/{MAX_ATTEMPTS}")
                time.sleep(delay)
            else:
                delay = 3 * attempt
                print(f"    Evidence request error; waiting {delay}s before retry {attempt + 1}/{MAX_ATTEMPTS}")
                time.sleep(delay)

    raise RuntimeError(f"Evidence chunk request failed: {type(last_exc).__name__}: {last_exc}")
def evidence_prompt_for_chunk(source_text, chunk_number, chunk_count):
    """Build a prompt for exactly one evidence-source chunk.

    The caller is responsible for enforcing MAX_SOURCE_CHARS on source_text.
    This function never expands, joins, or retrieves additional source text.
    """
    source_text = str(source_text or "")
    if not source_text:
        raise ValueError("evidence_prompt_for_chunk received empty source_text.")

    chunk_number = safe_int(chunk_number, 0)
    chunk_count = safe_int(chunk_count, 0)

    if chunk_count < 1:
        raise ValueError(f"Invalid chunk_count: {chunk_count}")
    if chunk_number < 1 or chunk_number > chunk_count:
        raise ValueError(
            f"Invalid chunk_number: {chunk_number} for chunk_count={chunk_count}"
        )

    if len(source_text) > MAX_SOURCE_CHARS:
        raise ValueError(
            f"Evidence source chunk exceeds MAX_SOURCE_CHARS: "
            f"{len(source_text):,} > {MAX_SOURCE_CHARS:,}"
        )

    return EVIDENCE_CHUNK_PROMPT.format(
        source_text=source_text,
        chunk_number=chunk_number,
        chunk_count=chunk_count,
    )

def _load_json_object(path, description):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read {description}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{description} must contain a JSON object.")
    return data


def _save_json_object(path, data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def load_evidence_progress():
    """Load persistent per-case evidence chunk progress."""
    data = _load_json_object(EVIDENCE_PROGRESS_PATH, "evidence progress file")
    if not data:
        return {"version": 1, "cases": {}}
    version = safe_int(data.get("version"), 0)
    cases = data.get("cases", {})
    if version != 1 or not isinstance(cases, dict):
        raise RuntimeError(
            f"Invalid evidence progress file format: {EVIDENCE_PROGRESS_PATH}"
        )
    return {"version": 1, "cases": cases}


def save_evidence_progress(progress):
    """Atomically persist evidence chunk progress after every successful chunk."""
    _save_json_object(EVIDENCE_PROGRESS_PATH, progress)


def _source_sha256(text):
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _normalise_completed_chunks(value):
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        n = safe_int(item, 0)
        if n >= 1 and n not in output:
            output.append(n)
    return sorted(output)


def enrich_existing_evidence(case, old, evidence_progress=None):
    """
    Extract Evidence Presented from the complete frozen source with persistent
    chunk-level resume support.

    A successful chunk is written to EVIDENCE_PROGRESS_PATH immediately. If the
    process stops because of Groq TPD, the next run skips every completed chunk
    and starts at the first incomplete chunk. The final upgraded case is written
    to the main results file only after every chunk is complete.
    """
    source_text = build_full_source(case)

    if not source_text:
        return {
            "evidence": "The available source does not specify the exact evidence presented.",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
            "tpd_accounted_tokens": 0,
            "attempts": 0,
            "prompt_chars": 0,
            "evidence_chunks": 0,
        }

    chunks = _split_source_adaptively(source_text, MAX_SOURCE_CHARS)
    if any(len(chunk) > MAX_SOURCE_CHARS for chunk in chunks):
        raise RuntimeError("Internal evidence chunking failure: oversized chunk.")

    total_chunks = len(chunks)
    case_id = str(case["case_id"])
    source_hash = _source_sha256(source_text)

    if evidence_progress is None:
        evidence_progress = load_evidence_progress()

    cases_progress = evidence_progress.setdefault("cases", {})
    saved = cases_progress.get(case_id)

    # Existing progress is valid only for the exact same frozen source and the
    # same deterministic chunk size. Otherwise discard it rather than mixing
    # fragments from different source versions.
    if (
        not isinstance(saved, dict)
        or saved.get("source_sha256") != source_hash
        or safe_int(saved.get("chunk_size"), 0) != MAX_SOURCE_CHARS
        or safe_int(saved.get("total_chunks"), 0) != total_chunks
    ):
        saved = {
            "source_sha256": source_hash,
            "chunk_size": MAX_SOURCE_CHARS,
            "total_chunks": total_chunks,
            "completed_chunks": [],
            "evidence_fragments": {},
            "chunk_stats": {},
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        cases_progress[case_id] = saved
        save_evidence_progress(evidence_progress)

    completed = set(_normalise_completed_chunks(saved.get("completed_chunks")))
    fragments_by_chunk = saved.setdefault("evidence_fragments", {})
    if not isinstance(fragments_by_chunk, dict):
        fragments_by_chunk = {}
        saved["evidence_fragments"] = fragments_by_chunk

    chunk_stats = saved.setdefault("chunk_stats", {})
    if not isinstance(chunk_stats, dict):
        chunk_stats = {}
        saved["chunk_stats"] = chunk_stats

    # Remove impossible/stale chunk numbers before resuming.
    completed = {n for n in completed if 1 <= n <= total_chunks}
    completed = {n for n in completed if str(n) in fragments_by_chunk}
    saved["completed_chunks"] = sorted(completed)

    pending = [n for n in range(1, total_chunks + 1) if n not in completed]
    if pending:
        first_pending = pending[0]
        if completed:
            print(
                f"    Evidence resume: {case_id} | "
                f"completed {min(completed)}-{max(completed)} of {total_chunks} | "
                f"starting chunk {first_pending}"
            )
        else:
            print(
                f"    Evidence resume: {case_id} | "
                f"completed 0 of {total_chunks} | starting chunk 1"
            )
    else:
        print(
            f"    Evidence resume: {case_id} | "
            f"all {total_chunks} chunks already complete; assembling"
        )

    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "tpd_accounted_tokens": 0,
        "attempts": 0,
        "prompt_chars": 0,
    }

    # Include usage from previously completed chunks so the final upgraded
    # result contains accurate aggregate evidence telemetry after a resume.
    for number in sorted(completed):
        stats = chunk_stats.get(str(number), {})
        if not isinstance(stats, dict):
            continue
        for key in totals:
            value = stats.get(key)
            if value is not None:
                totals[key] += int(value)

    # Only incomplete deterministic chunks are sent to Groq. This is the key
    # protection against repeatedly paying for chunks 1..N after a restart.
    for chunk_number in pending:
        chunk = chunks[chunk_number - 1]
        if not chunk:
            saved["completed_chunks"] = sorted(set(saved["completed_chunks"]) | {chunk_number})
            fragments_by_chunk[str(chunk_number)] = ""
            saved["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_evidence_progress(evidence_progress)
            continue

        prompt = evidence_prompt_for_chunk(
            chunk,
            chunk_number,
            total_chunks,
        )
        estimated_total = estimate_request_tokens(prompt)

        if estimated_total > MAX_ESTIMATED_REQUEST_TOKENS:
            raise RuntimeError(
                f"Evidence prompt exceeds safety budget: "
                f"{estimated_total:,} > {MAX_ESTIMATED_REQUEST_TOKENS:,} tokens."
            )

        try:
            print(
                f"    Evidence chunk {chunk_number}/{total_chunks} | "
                f"source={len(chunk):,} chars | "
                f"estimated={estimated_total:,} tokens"
            )
            result = _request_evidence_chunk(prompt, estimated_total)
        except DailyQuotaExhausted:
            # DO NOT mark this chunk complete. All earlier successful chunks
            # are already persisted in evidence_upgrade_progress.json.
            raise
        except Exception as exc:
            if not is_request_too_large(exc):
                raise
            raise RuntimeError(
                f"Groq rejected deterministic evidence chunk {chunk_number}/{total_chunks} "
                f"as too large despite the {MAX_SOURCE_CHARS}-character source cap."
            ) from exc

        fragment = clean_text(result.get("evidence", ""))
        fragments_by_chunk[str(chunk_number)] = fragment
        chunk_stats[str(chunk_number)] = {
            key: result.get(key) for key in totals
        }
        completed.add(chunk_number)
        saved["completed_chunks"] = sorted(completed)
        saved["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # CRITICAL: persist immediately after this chunk succeeds.
        save_evidence_progress(evidence_progress)

        for key in totals:
            value = result.get(key)
            if value is not None:
                totals[key] += int(value)

        print(
            "    Evidence chunk complete and checkpointed: "
            f"{result['total_tokens'] or result['tpd_accounted_tokens']} tokens | "
            f"checkpoint {len(completed)}/{total_chunks}"
        )

    # Assemble fragments strictly in source order.
    deduped = []
    seen = set()
    for number in range(1, total_chunks + 1):
        fragment = clean_text(fragments_by_chunk.get(str(number), ""))
        if not fragment:
            continue
        for paragraph in re.split(r"\n\s*\n+", fragment):
            paragraph = clean_text(paragraph)
            if not paragraph:
                continue
            fingerprint = re.sub(r"\s+", " ", paragraph).lower().strip()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(paragraph)

    evidence_text = "\n\n".join(deduped).strip()
    if not evidence_text:
        evidence_text = (
            "The available source does not specify the exact evidence presented."
        )

    # Every chunk is complete. Remove the checkpoint so it cannot be mistaken
    # for unfinished work on a later run.
    cases_progress.pop(case_id, None)
    save_evidence_progress(evidence_progress)

    return {
        "evidence": evidence_text,
        "prompt_tokens": totals["prompt_tokens"] or None,
        "completion_tokens": totals["completion_tokens"] or None,
        "total_tokens": totals["total_tokens"] or None,
        "cached_tokens": totals["cached_tokens"] or None,
        "tpd_accounted_tokens": totals["tpd_accounted_tokens"],
        "attempts": totals["attempts"],
        "prompt_chars": totals["prompt_chars"],
        "evidence_chunks": total_chunks,
    }

def split_legacy_story(story):
    """Split the existing V3 story into narrative, combined decision/meaning, and why."""
    text = clean_text(story)
    # Match the old bold/plain headings case-insensitively.
    pat_dec = re.compile(r"(?is)\n*\s*\**What the Court decided\**\s*\n*")
    pat_mean = re.compile(r"(?is)\n*\s*\**What that decision means in simple words\**\s*\n*")
    pat_why = re.compile(r"(?is)\n*\s*\**Why the Court decided that way\**\s*\n*")

    m_dec = pat_dec.search(text)
    m_mean = pat_mean.search(text)
    m_why = pat_why.search(text)

    if not m_dec:
        return text, "The available source does not contain a separately identified Court-decision section in the existing story.", "The existing story does not contain a separately identified reasoning section."

    narrative = text[:m_dec.start()].strip()
    decision_start = m_dec.end()
    if m_mean:
        decision_text = text[decision_start:m_mean.start()].strip()
        meaning_start = m_mean.end()
        meaning_end = m_why.start() if m_why else len(text)
        meaning_text = text[meaning_start:meaning_end].strip()
        decision_meaning = decision_text
        if meaning_text:
            decision_meaning += "\n\nIn simple words: " + meaning_text
    else:
        end = m_why.start() if m_why else len(text)
        decision_meaning = text[decision_start:end].strip()

    why = text[m_why.end():].strip() if m_why else "The existing story does not contain a separately identified reasoning section."
    return narrative, decision_meaning, why


def upgrade_existing_results(results, cases_by_id):
    """Upgrade already-GENERATED target results without regenerating their Story."""
    changed = 0
    quota_stopped = False
    evidence_progress = load_evidence_progress()

    for result in sorted(results, key=lambda x: safe_int(x.get("production_index"), 999999)):
        if result.get("status") != "GENERATED":
            continue
        if result.get("evidence") and result.get("decision_meaning") and result.get("why"):
            continue

        case_id = str(result.get("case_id", "")).strip()
        case = cases_by_id.get(case_id)
        if not case:
            raise RuntimeError(
                f"Cannot upgrade existing result {case_id}: case not found in frozen Chroma."
            )

        print("\n" + "-" * 72)
        print(f"UPGRADING EXISTING CASE WITH EVIDENCE: {case_id}")
        try:
            evidence = enrich_existing_evidence(case, result, evidence_progress)
            narrative, decision_meaning, why = split_legacy_story(result.get("story", ""))

            result["story"] = narrative
            result["evidence"] = evidence["evidence"]
            result["decision_meaning"] = decision_meaning
            result["why"] = why
            result["evidence_prompt_chars"] = evidence["prompt_chars"]
            result["evidence_prompt_tokens"] = evidence["prompt_tokens"]
            result["evidence_completion_tokens"] = evidence["completion_tokens"]
            result["evidence_total_tokens"] = evidence["total_tokens"]
            result["evidence_cached_tokens"] = evidence["cached_tokens"]
            result["evidence_tpd_accounted_tokens"] = evidence["tpd_accounted_tokens"]
            result["evidence_generation_attempts"] = evidence["attempts"]
            result["evidence_chunks"] = evidence["evidence_chunks"]
            result["prompt_version"] = PROMPT_VERSION
            result["word_count"] = word_count(result["story"])
            changed += 1

            # Persist the completed upgraded case immediately. The chunk
            # checkpoint has already been removed by enrich_existing_evidence.
            current = sorted(results, key=lambda x: safe_int(x.get("production_index"), 999999))
            save_results(current)
            save_csv(current)
            save_manual_review(current)

            print(f"Evidence added to {case_id}; upgraded result checkpointed.")

        except DailyQuotaExhausted as exc:
            print("\n" + "=" * 72)
            print("GROQ DAILY TOKEN LIMIT — CLEAN STOP DURING EXISTING-CASE UPGRADE")
            print("=" * 72)
            print(str(exc))
            progress = evidence_progress.get("cases", {}).get(case_id, {})
            completed = _normalise_completed_chunks(progress.get("completed_chunks"))
            total = safe_int(progress.get("total_chunks"), 0)
            next_chunk = next((n for n in range(1, total + 1) if n not in set(completed)), None)
            if next_chunk is not None:
                print(
                    f"Evidence checkpoint preserved: {case_id} | "
                    f"completed {len(completed)}/{total} | next chunk {next_chunk}"
                )
            print("The main case result was NOT marked as failed.")
            print("Resume after the Groq daily allowance resets.")
            print("=" * 72)
            quota_stopped = True
            break

    return results, changed, quota_stopped

def self_test_evidence_chunking():
    """Fail fast before any Groq request if the evidence path is inconsistent."""
    probe = "Evidence probe. " * 20
    prompt = evidence_prompt_for_chunk(probe, 1, 1)

    if "SOURCE TEXT:" not in prompt:
        raise RuntimeError("Evidence prompt self-test failed: SOURCE TEXT missing.")

    if len(probe) > MAX_SOURCE_CHARS:
        raise RuntimeError("Evidence prompt self-test failed: probe is oversized.")

    test_source = "Paragraph one.\n\nParagraph two." * 1000
    chunks = _split_source_adaptively(test_source, MAX_SOURCE_CHARS)

    if not chunks:
        raise RuntimeError("Evidence chunk self-test failed: no chunks produced.")

    if any(len(c) > MAX_SOURCE_CHARS for c in chunks):
        raise RuntimeError(
            "Evidence chunk self-test failed: a chunk exceeds MAX_SOURCE_CHARS."
        )

    print(
        f"Evidence path self-test: PASS "
        f"(max source chunk={max(len(c) for c in chunks):,} chars)"
    )

    # Verify the persistent-progress data shape without touching the real
    # progress file or making any API request.
    probe_progress = {
        "version": 1,
        "cases": {
            "SELF_TEST": {
                "source_sha256": _source_sha256("probe"),
                "chunk_size": MAX_SOURCE_CHARS,
                "total_chunks": 2,
                "completed_chunks": [1],
                "evidence_fragments": {"1": "probe evidence"},
                "chunk_stats": {"1": {"total_tokens": 10}},
            }
        },
    }
    if _normalise_completed_chunks(probe_progress["cases"]["SELF_TEST"]["completed_chunks"]) != [1]:
        raise RuntimeError("Evidence resume self-test failed: completed chunk normalization.")
    print("Evidence resume self-test: PASS (chunk checkpoints supported)")



# ================================================================
# MAIN
# ================================================================

def main():
    self_test_evidence_chunking()
    print("=" * 72)
    print("PRAMANA — STEP 12B EXACT 225-CASE STORY PRODUCTION")
    print("=" * 72)
    print(f"Model                    : {MODEL}")
    print("Story format              : V4 EVIDENCE FIXED")
    print(f"Frozen Chroma            : {CHROMA_PATH}")
    print(f"Expected Chroma records  : {EXPECTED_CHROMA_COUNT}")
    print(f"Frozen corpus cases      : {EXPECTED_CORPUS_CASES}")
    print(f"EXACT requested cases    : {EXPECTED_TARGET_CASES}")
    print(f"Source context cap       : {MAX_SOURCE_CHARS} chars")
    print(f"Completion cap           : {MAX_COMPLETION_TOKENS} tokens")
    print("Story word limit         : NONE")
    print("Production mode          : EXACT FROZEN 225 CASE IDs")
    print("TPM pacing               : DYNAMIC — Groq response headers")
    print(f"TPM limit                : {GROQ_TPM_LIMIT:,} tokens")
    print(f"TPD limit                : {GROQ_TPD_LIMIT:,} tokens")
    print(f"TPD safety margin        : {GROQ_TPD_SAFETY_MARGIN:,} tokens")
    print(f"TPD usage log            : {TPD_USAGE_PATH}")
    print(f"Evidence resume log      : {EVIDENCE_PROGRESS_PATH}")
    print("")

    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma.get_collection(COLLECTION_NAME)
    actual_count = collection.count()

    if actual_count != EXPECTED_CHROMA_COUNT:
        raise RuntimeError(
            f"FROZEN CHROMA FAILURE: expected "
            f"{EXPECTED_CHROMA_COUNT}, found {actual_count}"
        )

    initial_hash = hash_chroma_directory()

    rows = load_chunks(collection)

    # Build the complete frozen corpus once. No target is generated until the
    # exact manifest has passed every identity/order/Chroma check.
    cases = assign_complexity(build_cases(rows))

    chroma_case_ids = {case["case_id"] for case in cases}
    validate_target_manifest(
        TARGET_CASE_IDS,
        STAGE6_DOMAIN_CASE_IDS,
        chroma_case_ids,
    )

    targets = build_production_targets(cases)
    save_manifest(cases, targets)

    raw_results = load_existing_results()
    existing = {}

    for r in raw_results:
        key = str(r.get("case_id", "")).strip()
        if key:
            existing[key] = r

    results = sorted(
        existing.values(),
        key=lambda x: safe_int(x.get("production_index"), 999999),
    )

    preflight(cases, targets, actual_count, results)

    if "--manifest-only" in sys.argv:
        print("\nSTEP 12B MANIFEST-ONLY RUN = PASS")
        print("No Groq request was made.")
        return

    # Initialize / display the local daily token ledger before any request.
    tpd_usage = load_tpd_usage()
    print(
        "INITIAL DAILY TOKEN LEDGER : "
        f"{tpd_usage['total_tokens']:,} / "
        f"{GROQ_TPD_LIMIT - GROQ_TPD_SAFETY_MARGIN:,} used"
    )

    target_ids = {x["case_id"] for x in targets}

    # Do not let unrelated old Step-12 results contaminate the final count.
    existing = {
        k: v for k, v in existing.items()
        if k in target_ids
    }

    completed_now = 0
    skipped_existing = 0
    failures_now = 0
    quota_stopped = False

    # Upgrade already-generated V3 stories in place: do NOT regenerate their Story.
    # Evidence is extracted from the complete frozen case source, and the old
    # decision/meaning/reasoning headings are normalized into the fixed V4 format.
    cases_by_id = {str(c["case_id"]): c for c in targets}
    existing_values = list(existing.values())
    existing_values, upgraded_now, upgrade_quota_stopped = upgrade_existing_results(existing_values, cases_by_id)
    existing = {str(r.get("case_id")): r for r in existing_values if str(r.get("case_id", "")) in target_ids}
    if upgraded_now:
        save_results(sorted(existing.values(), key=lambda x: safe_int(x.get("production_index"), 999999)))
        save_csv(sorted(existing.values(), key=lambda x: safe_int(x.get("production_index"), 999999)))
        save_manual_review(sorted(existing.values(), key=lambda x: safe_int(x.get("production_index"), 999999)))
    if upgrade_quota_stopped:
        quota_stopped = True

    print("\n" + "=" * 72)
    print("GENERATING ONLY THE 225 REQUESTED STORIES WITH 120B")
    print("=" * 72)

    for position, case in enumerate(targets, 1):
        if quota_stopped:
            break
        case_id = case["case_id"]
        old = existing.get(case_id)

        if (
            old
            and old.get("status") == "GENERATED"
            and old.get("model") == MODEL
            and old.get("prompt_version") == PROMPT_VERSION
            and old.get("evidence")
            and old.get("decision_meaning")
            and old.get("why")
        ):
            print(
                f"[{position}/{len(targets)}] {case_id} "
                "| ALREADY COMPLETE — SKIP"
            )
            skipped_existing += 1
            continue

        print("\n" + "-" * 72)
        print(
            f"[{position}/{len(targets)}] {case_id} | "
            f"year={case['year'] or 'None'} | "
            f"{case['complexity_label']} | "
            f"score={case['complexity_score']}"
        )
        print(f"Source chunks in case : {case['chunk_count']}")
        print(f"Full case chars       : {case['total_chars']}")
        print(f"Source sent chars     : {len(case['story_source'])}")
        print("Generating TITLE + STORY + EVIDENCE with 120B...")

        started = time.time()

        try:
            generated = generate_story(case)

            result = {
                "production_index": position,
                "case_id": case_id,
                "year": case["year"],
                "complexity_label": case["complexity_label"],
                "complexity_score": case["complexity_score"],
                "chunk_count": case["chunk_count"],
                "total_chars": case["total_chars"],
                "representative_chunk_id": case["representative_chunk_id"],
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "source_chars_sent": generated["source_chars_sent"],
                "prompt_chars": generated["prompt_chars"],
                "generation_attempts": generated["attempts"],
                "prompt_tokens": generated.get("prompt_tokens"),
                "completion_tokens": generated.get("completion_tokens"),
                "total_tokens": generated.get("total_tokens"),
                "cached_tokens": generated.get("cached_tokens"),
                "tpd_accounted_tokens": generated.get("tpd_accounted_tokens"),
                "word_count": word_count(generated["story"]),
                "status": "GENERATED",
                "title": generated["title"],
                "story": generated["story"],
                "evidence": generated["evidence"],
                "decision_meaning": generated["decision_meaning"],
                "why": generated["why"],
                "error": "",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "generation_seconds": round(time.time() - started, 2),
            }

            existing[case_id] = result
            completed_now += 1

            print(f"TITLE: {result['title']}")
            print(
                "ACTUAL TOKENS: "
                f"prompt={result.get('prompt_tokens')} | "
                f"completion={result.get('completion_tokens')} | "
                f"total={result.get('total_tokens')} | "
                f"cached={result.get('cached_tokens')} | "
                f"TPD accounted={result.get('tpd_accounted_tokens')}"
            )

            current_tpd = load_tpd_usage()
            print(
                "DAILY TOKEN LEDGER: "
                f"{current_tpd['total_tokens']:,} / "
                f"{GROQ_TPD_LIMIT - GROQ_TPD_SAFETY_MARGIN:,}"
            )

            print(
                f"STORY ({result['word_count']} words — no word limit):"
            )
            print(result["story"])

        except KeyboardInterrupt:
            print("\nCTRL+C received. Saving progress before exit...")
            results = sorted(
                existing.values(),
                key=lambda x: safe_int(
                    x.get("production_index"),
                    999999,
                ),
            )
            save_results(results)
            save_csv(results)
            save_manual_review(results)
            raise

        except DailyQuotaExhausted as exc:
            # This is NOT a failed case. No GENERATION_FAILED record is created.
            # The case remains resumable on the next run.
            quota_stopped = True
            print("\n" + "=" * 72)
            print("GROQ DAILY TOKEN LIMIT — CLEAN STOP")
            print("=" * 72)
            print(str(exc))
            print(
                "The current case was NOT marked as failed. "
                "No fake failure record was written."
            )
            print(
                "Existing successful stories have already been preserved. "
                "Resume the same script after the daily allowance resets."
            )
            print("=" * 72)
            break

        except Exception as exc:
            failures_now += 1
            error = f"{type(exc).__name__}: {exc}"

            existing[case_id] = {
                "production_index": position,
                "case_id": case_id,
                "year": case["year"],
                "complexity_label": case["complexity_label"],
                "complexity_score": case["complexity_score"],
                "chunk_count": case["chunk_count"],
                "total_chars": case["total_chars"],
                "representative_chunk_id": case["representative_chunk_id"],
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "source_chars_sent": len(case["story_source"]),
                "prompt_chars": "",
                "generation_attempts": "",
                "prompt_tokens": "",
                "completion_tokens": "",
                "total_tokens": "",
                "cached_tokens": "",
                "tpd_accounted_tokens": "",
                "word_count": 0,
                "status": "GENERATION_FAILED",
                "title": "",
                "story": "",
                "evidence": "",
                "decision_meaning": "",
                "why": "",
                "error": error,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "generation_seconds": round(
                    time.time() - started,
                    2,
                ),
            }

            print(f"ERROR: {error}")
            print(
                "Failure saved; rerun the same script to retry this case."
            )

        results = sorted(
            existing.values(),
            key=lambda x: safe_int(
                x.get("production_index"),
                999999,
            ),
        )
        save_results(results)
        save_csv(results)
        save_manual_review(results)

        print(
            f"Progress saved: {len(results)} target result records."
        )

    # Always save and run final QC, including a clean quota stop.
    results = sorted(
        existing.values(),
        key=lambda x: safe_int(
            x.get("production_index"),
            999999,
        ),
    )

    save_results(results)
    save_csv(results)
    save_manual_review(results)

    final_hash = hash_chroma_directory()
    complete = write_qc_report(
        cases,
        targets,
        results,
        initial_hash,
        final_hash,
    )

    successful_ids = {
        r.get("case_id")
        for r in results
        if (
            r.get("status") == "GENERATED"
            and r.get("model") == MODEL
            and r.get("prompt_version") == PROMPT_VERSION
        )
    }

    print("\n" + "=" * 72)
    print("STEP 12B FINAL QC")
    print("=" * 72)
    print(f"Required cases             : {EXPECTED_TARGET_CASES}")
    print(f"Generated this run         : {completed_now}")
    print(f"Already present            : {skipped_existing}")
    print(f"Failures this run          : {failures_now}")
    print(
        f"Successful stories         : "
        f"{len(target_ids & successful_ids)} / "
        f"{EXPECTED_TARGET_CASES}"
    )

    generated_results = [
        r for r in results
        if (
            r.get("status") == "GENERATED"
            and r.get("model") == MODEL
            and r.get("prompt_version") == PROMPT_VERSION
        )
    ]

    def _sum_numeric(field):
        values = [r.get(field) for r in generated_results]
        nums = [
            int(v)
            for v in values
            if v not in (None, "", "")
        ]
        return sum(nums) if nums else None

    print(
        "Actual token usage         : "
        f"prompt={_sum_numeric('prompt_tokens')} | "
        f"completion={_sum_numeric('completion_tokens')} | "
        f"total={_sum_numeric('total_tokens')} | "
        f"cached={_sum_numeric('cached_tokens')}"
    )

    final_tpd = load_tpd_usage()
    print(
        "Local TPD ledger            : "
        f"{final_tpd['total_tokens']:,} / "
        f"{GROQ_TPD_LIMIT - GROQ_TPD_SAFETY_MARGIN:,}"
    )

    print(f"Chroma count               : {actual_count}")
    print(
        f"Chroma unchanged           : "
        f"{'PASS' if initial_hash == final_hash else 'FAIL'}"
    )
    print("Story word limit           : NONE")
    print("Human review required      : YES")
    print(
        f"Quota clean stop            : "
        f"{'YES' if quota_stopped else 'NO'}"
    )
    print(
        f"AUTOMATED STATUS           : "
        f"{'PASS' if complete else 'INCOMPLETE'}"
    )
    print("")

    print(f"Results                    : {RESULTS_PATH}")
    print(f"CSV                        : {CSV_PATH}")
    print(f"Manifest                   : {SELECTION_PATH}")
    print(f"Manual review              : {MANUAL_REVIEW_PATH}")
    print(f"QC report                  : {QC_PATH}")
    print(f"TPD usage ledger           : {TPD_USAGE_PATH}")

    if not complete:
        if quota_stopped:
            print(
                "\nSTEP 12B stopped because the Groq daily token "
                "budget was exhausted. This is an expected resumable stop."
            )
            return

        raise RuntimeError(
            "STEP 12B is incomplete. Review the saved QC report "
            "and rerun the same script to retry incomplete cases."
        )


if __name__ == "__main__":
    main()
