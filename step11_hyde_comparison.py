"""
PRAMANA — STEP 11
HyDE vs Direct-Query Retrieval Comparison
VERSION: FINAL_V3_SOURCE_CONTROLLED_PROCEDURAL_VALIDATION

Frozen inputs:
    data/step7/gold_eval.csv
    data/step7/gold_candidates.csv
    data/step6_v14/chroma_db

Experiment:
    Baseline = candidate_query -> BGE -> Chroma Top-10
    HyDE     = candidate_query -> Groq GPT-OSS-120B -> BGE -> Chroma Top-10

IMPORTANT:
- gold_candidates.csv intentionally contains 50 candidate queries (G001-G050).
- Step 7B gold_eval.csv contains the frozen verified 25-query gold set.
- This script selects ONLY the 25 query IDs present in gold_eval.csv.
- It does NOT require gold_candidates.csv to contain only those 25 IDs.
- No Step 7/8/9/10 artifacts are modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT ROOT — ROBUST TO SCRIPT LOCATION
# ============================================================

def find_project_root() -> Path:
    here = Path(__file__).resolve()

    candidates = [here.parent] + list(here.parents)

    for candidate in candidates:
        if (
            (candidate / "data" / "step7" / "gold_eval.csv").exists()
            and
            (candidate / "data" / "step6_v14" / "chroma_db").exists()
        ):
            return candidate

    # Expected layout fallback: script directly inside PRAMANA.
    if here.parent.name.upper() == "PRAMANA":
        return here.parent

    raise RuntimeError(
        "Could not locate PRAMANA project root. "
        "Expected a directory containing "
        "data\\step7\\gold_eval.csv and "
        "data\\step6_v14\\chroma_db."
    )


PROJECT_ROOT = find_project_root()

STEP7_DIR = PROJECT_ROOT / "data" / "step7"
STEP6_DIR = PROJECT_ROOT / "data" / "step6_v14"
STEP11_DIR = PROJECT_ROOT / "data" / "step11"

GOLD_EVAL_PATH = STEP7_DIR / "gold_eval.csv"
GOLD_CANDIDATES_PATH = STEP7_DIR / "gold_candidates.csv"
CHROMA_PATH = STEP6_DIR / "chroma_db"

HYDE_CACHE_PATH = STEP11_DIR / "step11_hyde_generation.csv"
COMPARISON_PATH = STEP11_DIR / "step11_comparison_results.csv"
SUMMARY_PATH = STEP11_DIR / "step11_summary.json"
QC_PATH = STEP11_DIR / "step11_qc_report.txt"

CHROMA_COLLECTION = "pramana_v14"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 10

HYDE_MIN_WORDS = 75
HYDE_MAX_WORDS = 95
HYDE_TARGET_MIN = 78
HYDE_TARGET_MAX = 84
HYDE_MAX_ATTEMPTS = 5

# Stay comfortably below the documented Groq RPM limit.
MIN_SECONDS_BETWEEN_GROQ_CALLS = 3.1
POST_SUCCESS_SLEEP = 0.5


# ============================================================
# FROZEN STEP 7B GOLD QUERY SET
# ============================================================

EXPECTED_DIRECT_IDS = {
    "G004", "G006", "G008", "G011", "G013", "G015",
    "G018", "G019", "G022", "G023", "G028", "G030",
    "G032", "G034", "G037", "G038", "G045", "G048",
}

EXPECTED_PARTIAL_IDS = {
    "G017", "G029", "G031", "G039", "G043", "G044", "G049",
}

EXPECTED_QUERY_IDS = (
    EXPECTED_DIRECT_IDS | EXPECTED_PARTIAL_IDS
)


HYDE_COLUMNS = [
    "query_id",
    "candidate_query",
    "hyde_text",
    "word_count",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "latency_seconds",
    "format_pass",
    "word_range_pass",
    "heading_pass",
    "outcome_pass",
    "sections_pass",
    "dates_pass",
    "years_pass",
    "time_periods_pass",
    "artifact_pass",
    "procedural_terms_pass",
    "validation_pass",
    "validation_reason",
]


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_directories():
    STEP11_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2009", " ")
    text = text.replace("\u202f", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove accidental markdown fences only.
    text = re.sub(
        r"^\s*```(?:text|markdown)?\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
        flags=re.I,
    )

    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip())) if text.strip() else 0


def extract_sections(text: str) -> List[str]:
    pattern = re.compile(
        r"\bsection\s+([0-9]{1,3}[A-Za-z]?(?:\([0-9A-Za-z]+\))?)",
        re.I,
    )
    return sorted(set(m.group(1) for m in pattern.finditer(text or "")))


def extract_years(text: str) -> List[str]:
    return sorted(set(re.findall(r"\b(?:18|19|20)\d{2}\b", text or "")))


def extract_dates(text: str) -> List[str]:
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-](?:18|19|20)\d{2}\b",
        r"\b(?:18|19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2}\b",
    ]
    found = []
    for p in patterns:
        found.extend(re.findall(p, text or ""))
    return sorted(set(found))


def extract_time_periods(text: str) -> List[str]:
    patterns = [
        r"\b\d+\s+(?:day|days|week|weeks|month|months|year|years)\b",
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?:day|days|week|weeks|month|months|year|years)\b",
    ]
    found = []
    for p in patterns:
        found.extend(re.findall(p, text or "", flags=re.I))
    return sorted(set(x.casefold() for x in found))


def has_heading(raw_text: str) -> bool:
    raw = (raw_text or "").strip()

    if not raw:
        return False

    nonempty_lines = [
        x.strip()
        for x in raw.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        if x.strip()
    ]

    if len(nonempty_lines) != 1:
        return True

    patterns = [
        r"^(hyde|answer|response|summary|analysis|legal principle)\s*:",
        r"^#+\s+",
        r"^\*\*.+\*\*$",
        r"^[-*]\s+",
        r"^\d+\.\s+",
    ]

    return any(
        re.search(p, raw, flags=re.I)
        for p in patterns
    )


def has_outcome_language(text: str) -> bool:
    patterns = [
        r"\bthe court\s+(?:allowed|dismissed)\b",
        r"\bthe appeal\s+(?:is|was)\s+(?:allowed|dismissed)\b",
        r"\bthe petition\s+(?:is|was)\s+(?:allowed|dismissed)\b",
        r"\bthe accused\s+(?:is|was)\s+(?:convicted|acquitted)\b",
        r"\bthe appellant\s+(?:succeeds|failed|fails)\b",
        r"\bthe respondent\s+(?:succeeds|failed|fails)\b",
        r"\bthe conviction\s+(?:is|was)\s+(?:upheld|set aside|confirmed)\b",
        r"\bthe decree\s+(?:is|was)\s+(?:set aside|confirmed|upheld)\b",
        r"\bthe prosecution\s+(?:is|was)\s+(?:quashed|dismissed)\b",
        r"\bthe complaint\s+(?:is|was)\s+(?:dismissed|allowed)\b",
    ]
    return any(
        re.search(p, text or "", flags=re.I)
        for p in patterns
    )


def extract_procedural_mechanisms(text: str) -> List[str]:
    """
    Extract only highly specific procedural mechanisms that the HyDE
    validator deliberately controls.

    General legal words such as:
        proceeding, prosecution, trial, appeal, complaint, petition,
        writ, remedy, liability, conviction
    are NOT treated as unsupported mechanisms by themselves.

    This avoids rejecting legitimate Indian judicial vocabulary merely
    because it is procedural in character.
    """
    patterns = {
        "summons": r"\bsummons\b",
        "warrant": r"\bwarrant\b",
        "attachment": r"\battachment\b",
        "garnishment": r"\bgarnishment\b",
        "execution proceedings": r"\bexecution proceedings\b",
        "execution petition": r"\bexecution petition\b",
        "interim injunction": r"\binterim injunction\b",
        "temporary injunction": r"\btemporary injunction\b",
        "contempt proceedings": r"\bcontempt proceedings\b",
    }

    found = []

    for label, pattern in patterns.items():
        if re.search(pattern, text or "", flags=re.I):
            found.append(label)

    return sorted(set(found))


def has_procedural_overreach(
    text: str,
    candidate_query: str,
) -> Tuple[bool, List[str]]:
    """
    A procedural mechanism is unsupported only when the mechanism appears
    in the HyDE but is absent from the source query.

    This is source-controlled rather than a blanket rejection list.
    """
    hyde_mechanisms = set(
        extract_procedural_mechanisms(text)
    )

    source_mechanisms = set(
        extract_procedural_mechanisms(candidate_query)
    )

    unsupported = sorted(
        hyde_mechanisms - source_mechanisms
    )

    return (
        len(unsupported) == 0,
        unsupported,
    )


def has_artifact(text: str) -> bool:
    patterns = [
        r"\bmaintainproceedings\b",
        r"\blocus standiupon\b",
        r"\bmaintainabilityof\b",
        r"\bmaintainableof\b",
        r"\bbarof\b",
        r"\batthe\b",
        r"\blimitationhas\b",
        r"\bthethe\b",
        r"\bandthe\b",
        r"\bwiththe\b",
        r"\bforthe\b",
        r"\bfromthe\b",
    ]
    return any(
        re.search(p, text or "", flags=re.I)
        for p in patterns
    )


# ============================================================
# HYDE VALIDATION
# ============================================================

def validate_hyde(
    hyde_text: str,
    candidate_query: str,
) -> Dict[str, object]:

    raw = hyde_text or ""
    text = clean_text(raw)
    wc = word_count(text)

    format_pass = bool(text)

    raw_lines = [
        x.strip()
        for x in raw.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        if x.strip()
    ]

    if raw_lines and len(raw_lines) != 1:
        format_pass = False

    if re.search(r"```|^\s*[-*]\s+", raw, flags=re.M):
        format_pass = False

    word_range_pass = (
        HYDE_MIN_WORDS <= wc <= HYDE_MAX_WORDS
    )

    heading_pass = not has_heading(raw)
    outcome_pass = not has_outcome_language(text)

    # Sections may be inherited ONLY from the source query.
    allowed_sections = {
        x.casefold()
        for x in extract_sections(candidate_query)
    }

    hyde_sections = {
        x.casefold()
        for x in extract_sections(text)
    }

    unsupported_sections = sorted(
        hyde_sections - allowed_sections
    )

    sections_pass = not unsupported_sections

    allowed_dates = {
        x.casefold()
        for x in extract_dates(candidate_query)
    }

    hyde_dates = {
        x.casefold()
        for x in extract_dates(text)
    }

    dates_pass = hyde_dates.issubset(allowed_dates)

    allowed_years = set(
        extract_years(candidate_query)
    )

    hyde_years = set(
        extract_years(text)
    )

    years_pass = hyde_years.issubset(allowed_years)

    allowed_periods = set(
        extract_time_periods(candidate_query)
    )

    hyde_periods = set(
        extract_time_periods(text)
    )

    time_periods_pass = hyde_periods.issubset(
        allowed_periods
    )

    artifact_pass = not has_artifact(text)

    procedural_terms_pass, unsupported_procedural_terms = (
        has_procedural_overreach(
            text,
            candidate_query,
        )
    )

    reasons = []

    if not format_pass:
        reasons.append("format")

    if not word_range_pass:
        reasons.append(
            f"word_count={wc} outside "
            f"{HYDE_MIN_WORDS}-{HYDE_MAX_WORDS}"
        )

    if not heading_pass:
        reasons.append("heading")

    if not outcome_pass:
        reasons.append("outcome")

    if not sections_pass:
        reasons.append(
            "unsupported section(s): "
            + ", ".join(unsupported_sections)
        )

    if not dates_pass:
        reasons.append("unsupported date(s)")

    if not years_pass:
        reasons.append("unsupported year(s)")

    if not time_periods_pass:
        reasons.append("unsupported time period(s)")

    if not artifact_pass:
        reasons.append("artifact")

    if not procedural_terms_pass:
        reasons.append(
            "unsupported procedural mechanism(s): "
            + ", ".join(unsupported_procedural_terms)
        )

    validation_pass = not reasons

    return {
        "hyde_text": text,
        "word_count": wc,
        "format_pass": format_pass,
        "word_range_pass": word_range_pass,
        "heading_pass": heading_pass,
        "outcome_pass": outcome_pass,
        "sections_pass": sections_pass,
        "dates_pass": dates_pass,
        "years_pass": years_pass,
        "time_periods_pass": time_periods_pass,
        "artifact_pass": artifact_pass,
        "procedural_terms_pass": procedural_terms_pass,
        "validation_pass": validation_pass,
        "validation_reason": (
            "PASS" if validation_pass else "; ".join(reasons)
        ),
    }


# ============================================================
# PROMPTS
# ============================================================

def build_generation_prompt(candidate_query: str) -> str:
    return f"""
Generate a HyDE passage for legal information retrieval.

SOURCE QUERY:
{candidate_query}

Write ONE paragraph of formal Indian judicial/legal language
expressing only the general legal issue or principle in the source query.

STRICT RULES:
- Output ONLY one paragraph.
- Target 78–84 words.
- MUST remain between 75 and 95 words.
- Use concise sentences.
- No headings.
- No bullets.
- No advice.
- No personal answer.
- No invented facts, parties, dates, amounts, years, outcomes,
  limitation periods, deadlines, or procedural events.
- Do not introduce a statutory section unless it appears in the source query.
- Do not introduce a numerical time period unless it appears in the source query.
- Do not introduce a case name.
- Do not explain the task.
- Do not add introductory or concluding padding.

Return ONLY the paragraph.
""".strip()


def build_retry_prompt(
    candidate_query: str,
    previous_text: str,
    reason: str,
) -> str:

    if previous_text and word_count(previous_text) > HYDE_MAX_WORDS:
        return f"""
Compress the following HyDE paragraph for legal information retrieval.

SOURCE QUERY:
{candidate_query}

PREVIOUS PARAGRAPH:
{previous_text}

VALIDATION FAILURE:
{reason}

Rewrite it into ONE paragraph of 78–84 words.

STRICT RULES:
- MUST be 75–95 words.
- Preserve the same legal issue and general proposition.
- Remove repetition and unnecessary wording.
- Do not add facts.
- Do not add parties.
- Do not add dates, years, amounts, sections, deadlines,
  limitation periods, outcomes, or procedural events.
- Do not change the legal issue.
- No heading.
- No bullets.
- No explanation.
- Output ONLY the paragraph.

Return ONLY the rewritten paragraph.
""".strip()

    return f"""
Generate a completely new HyDE paragraph.

SOURCE QUERY:
{candidate_query}

PREVIOUS VALIDATION FAILURE:
{reason}

STRICT RULES:
- Output ONLY one paragraph.
- Target 78–84 words.
- MUST be 75–95 words.
- Formal Indian judicial/legal language.
- State only the general legal issue or principle.
- No invented facts, parties, dates, amounts, years, outcomes,
  limitation periods, deadlines, or procedural events.
- No new statutory section unless present in the source query.
- No new numerical time period unless present in the source query.
- No case name.
- No heading.
- No bullets.
- No explanation.
- No introductory phrase.
- No concluding padding.

Return ONLY the paragraph.
""".strip()


# ============================================================
# GROQ
# ============================================================

class GroqRateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_call

        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        self.last_call = time.time()


def get_groq_client() -> Groq:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found."
        )

    return Groq(api_key=api_key)


def call_groq(
    client: Groq,
    limiter: GroqRateLimiter,
    prompt: str,
) -> Dict[str, object]:

    limiter.wait()

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
        reasoning_effort="low",
        max_completion_tokens=500,
    )

    latency = time.perf_counter() - start

    choice = response.choices[0]
    content = choice.message.content or ""

    usage = getattr(
        response,
        "usage",
        None,
    )

    return {
        "content": content,
        "finish_reason": getattr(
            choice,
            "finish_reason",
            "",
        ),
        "prompt_tokens": getattr(
            usage,
            "prompt_tokens",
            0,
        ) if usage else 0,
        "completion_tokens": getattr(
            usage,
            "completion_tokens",
            0,
        ) if usage else 0,
        "latency_seconds": round(
            latency,
            6,
        ),
    }


# ============================================================
# STEP 7B DATA
# ============================================================

def load_gold_data():
    gold = pd.read_csv(GOLD_EVAL_PATH)
    candidates = pd.read_csv(GOLD_CANDIDATES_PATH)

    required_gold = {
        "query_id",
        "source_case_id",
        "source_chunk_id",
        "relevance_type",
    }

    required_candidates = {
        "query_id",
        "candidate_query",
        "source_case_id",
        "source_chunk_id",
    }

    missing = required_gold - set(gold.columns)

    if missing:
        raise RuntimeError(
            "Missing gold_eval columns: "
            + ", ".join(sorted(missing))
        )

    missing = required_candidates - set(candidates.columns)

    if missing:
        raise RuntimeError(
            "Missing gold_candidates columns: "
            + ", ".join(sorted(missing))
        )

    gold["query_id"] = gold["query_id"].astype(str)
    candidates["query_id"] = candidates["query_id"].astype(str)

    gold_ids = set(gold["query_id"])

    # Exact frozen 25-query Step 7B gold set.
    if gold_ids != EXPECTED_QUERY_IDS:
        raise RuntimeError(
            "Gold query IDs do not exactly match the frozen "
            "25-query Step 7B set."
        )

    if len(gold) != 25:
        raise RuntimeError(
            f"Expected 25 gold rows, found {len(gold)}."
        )

    labels = set(
        gold["relevance_type"].astype(str).str.upper()
    )

    if labels != {"DIRECT", "PARTIAL"}:
        raise RuntimeError(
            "Unexpected relevance_type values: "
            + ", ".join(sorted(labels))
        )

    direct_ids = set(
        gold.loc[
            gold["relevance_type"].astype(str).str.upper() == "DIRECT",
            "query_id",
        ]
    )

    partial_ids = set(
        gold.loc[
            gold["relevance_type"].astype(str).str.upper() == "PARTIAL",
            "query_id",
        ]
    )

    if direct_ids != EXPECTED_DIRECT_IDS:
        raise RuntimeError(
            "DIRECT gold composition mismatch."
        )

    if partial_ids != EXPECTED_PARTIAL_IDS:
        raise RuntimeError(
            "PARTIAL gold composition mismatch."
        )

    # --------------------------------------------------------
    # CRITICAL CORRECTION:
    # gold_candidates.csv contains 50 candidates.
    # We select the 25 candidates corresponding to the
    # frozen Step 7B gold IDs.
    # --------------------------------------------------------

    candidate_subset = candidates[
        candidates["query_id"].isin(gold_ids)
    ].copy()

    if len(candidate_subset) != 25:
        found = set(candidate_subset["query_id"])

        raise RuntimeError(
            "Could not map exactly 25 frozen gold queries "
            "to gold_candidates.csv.\n"
            f"Expected IDs: {sorted(gold_ids)}\n"
            f"Found IDs:    {sorted(found)}"
        )

    if candidate_subset["query_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate candidate rows for the frozen 25 IDs."
        )

    # Ensure every gold ID has exactly one candidate.
    candidate_ids = set(
        candidate_subset["query_id"]
    )

    if candidate_ids != gold_ids:
        raise RuntimeError(
            "Candidate subset does not cover all gold IDs."
        )

    candidates_by_id = {
        str(row["query_id"]): str(
            row["candidate_query"]
        )
        for _, row in candidate_subset.iterrows()
    }

    gold_by_id = {
        str(row["query_id"]): row
        for _, row in gold.iterrows()
    }

    return (
        gold,
        candidate_subset,
        gold_by_id,
        candidates_by_id,
    )


# ============================================================
# CACHE
# ============================================================

def load_hyde_cache() -> Dict[str, Dict[str, object]]:
    if not HYDE_CACHE_PATH.exists():
        return {}

    df = pd.read_csv(
        HYDE_CACHE_PATH,
        dtype=str,
    )

    cache = {}

    for _, row in df.iterrows():
        qid = str(
            row.get("query_id", "")
        ).strip()

        if not qid:
            continue

        cache[qid] = {
            column: (
                ""
                if pd.isna(row.get(column, ""))
                else row.get(column, "")
            )
            for column in HYDE_COLUMNS
        }

    return cache


def save_hyde_cache(
    cache: Dict[str, Dict[str, object]]
):
    rows = []

    def query_sort_key(qid):
        match = re.search(r"\d+", qid)
        return int(match.group()) if match else 9999

    for qid in sorted(
        cache,
        key=query_sort_key,
    ):
        rows.append(
            {
                column: cache[qid].get(
                    column,
                    "",
                )
                for column in HYDE_COLUMNS
            }
        )

    df = pd.DataFrame(
        rows,
        columns=HYDE_COLUMNS,
    )

    tmp = HYDE_CACHE_PATH.with_suffix(
        ".tmp.csv"
    )

    df.to_csv(
        tmp,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    tmp.replace(HYDE_CACHE_PATH)


def refresh_cached_metadata(
    cache: Dict[str, Dict[str, object]],
    candidates_by_id: Dict[str, str],
) -> int:

    valid = 0
    changed = False

    for qid in list(cache.keys()):

        if qid not in candidates_by_id:
            continue

        validation = validate_hyde(
            str(
                cache[qid].get(
                    "hyde_text",
                    "",
                )
                or ""
            ),
            candidates_by_id[qid],
        )

        before = dict(cache[qid])

        cache[qid]["candidate_query"] = (
            candidates_by_id[qid]
        )

        for key, value in validation.items():
            cache[qid][key] = value

        if cache[qid] != before:
            changed = True

        if validation["validation_pass"]:
            valid += 1

    if changed:
        save_hyde_cache(cache)

    return valid


# ============================================================
# GENERATION
# ============================================================

def make_cache_record(
    qid: str,
    candidate_query: str,
    api_result: Dict[str, object],
    validation: Dict[str, object],
) -> Dict[str, object]:

    return {
        "query_id": qid,
        "candidate_query": candidate_query,
        "hyde_text": validation["hyde_text"],
        "word_count": validation["word_count"],
        "finish_reason": api_result.get(
            "finish_reason",
            "",
        ),
        "prompt_tokens": api_result.get(
            "prompt_tokens",
            0,
        ),
        "completion_tokens": api_result.get(
            "completion_tokens",
            0,
        ),
        "latency_seconds": api_result.get(
            "latency_seconds",
            0,
        ),
        "format_pass": validation["format_pass"],
        "word_range_pass": validation["word_range_pass"],
        "heading_pass": validation["heading_pass"],
        "outcome_pass": validation["outcome_pass"],
        "sections_pass": validation["sections_pass"],
        "dates_pass": validation["dates_pass"],
        "years_pass": validation["years_pass"],
        "time_periods_pass": validation["time_periods_pass"],
        "artifact_pass": validation["artifact_pass"],
        "procedural_terms_pass": validation[
            "procedural_terms_pass"
        ],
        "validation_pass": validation[
            "validation_pass"
        ],
        "validation_reason": validation[
            "validation_reason"
        ],
    }


def generate_hyde(
    client: Groq,
    limiter: GroqRateLimiter,
    qid: str,
    candidate_query: str,
    existing: Optional[Dict[str, object]],
):

    previous_text = ""

    if existing:
        previous_text = str(
            existing.get(
                "hyde_text",
                "",
            )
            or ""
        )

    last_reason = "no valid generation"

    total_latency = 0.0

    for attempt in range(
        1,
        HYDE_MAX_ATTEMPTS + 1,
    ):

        if attempt == 1:
            prompt = build_generation_prompt(
                candidate_query
            )
        else:
            prompt = build_retry_prompt(
                candidate_query,
                previous_text,
                last_reason,
            )

        try:
            api_result = call_groq(
                client,
                limiter,
                prompt,
            )

        except Exception as exc:

            last_reason = (
                f"API error: "
                f"{type(exc).__name__}: {exc}"
            )

            print(
                f"    API error "
                f"(attempt {attempt}/{HYDE_MAX_ATTEMPTS}): "
                f"{last_reason}"
            )

            if attempt < HYDE_MAX_ATTEMPTS:
                continue

            raise RuntimeError(
                f"HyDE generation for {qid} failed "
                f"after {HYDE_MAX_ATTEMPTS} attempts: "
                f"{last_reason}"
            )

        total_latency += float(
            api_result.get(
                "latency_seconds",
                0,
            )
            or 0
        )

        previous_text = str(
            api_result.get(
                "content",
                "",
            )
            or ""
        )

        validation = validate_hyde(
            previous_text,
            candidate_query,
        )

        if validation["validation_pass"]:

            api_result["latency_seconds"] = round(
                total_latency,
                6,
            )

            return make_cache_record(
                qid,
                candidate_query,
                api_result,
                validation,
            )

        last_reason = str(
            validation["validation_reason"]
        )

        print(
            f"    HyDE validation FAIL "
            f"(attempt {attempt}): "
            f"{last_reason}"
        )

        if attempt < HYDE_MAX_ATTEMPTS:
            print(
                f"    Retry "
                f"{attempt + 1}/{HYDE_MAX_ATTEMPTS}: "
                f"regenerating/compressing..."
            )

    raise RuntimeError(
        f"HyDE generation for {qid} failed validation "
        f"after {HYDE_MAX_ATTEMPTS} attempts. "
        f"Reason: {last_reason}"
    )


# ============================================================
# BGE + CHROMA
# ============================================================

def load_chroma():
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name=CHROMA_COLLECTION
    )

    count = collection.count()

    if count != 4855:
        raise RuntimeError(
            f"Frozen Chroma count mismatch: "
            f"expected 4855, found {count}"
        )

    return collection


def load_bge():
    return SentenceTransformer(
        MODEL_NAME
    )


def embed(
    model,
    text: str,
) -> np.ndarray:

    vector = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    if vector.shape != (384,):
        raise RuntimeError(
            f"Expected 384-dimensional embedding; "
            f"got {vector.shape}"
        )

    if not np.isfinite(vector).all():
        raise RuntimeError(
            "Embedding contains NaN/Inf."
        )

    norm = float(
        np.linalg.norm(vector)
    )

    if not np.isclose(
        norm,
        1.0,
        atol=1e-3,
    ):
        raise RuntimeError(
            f"Embedding normalization failure: norm={norm}"
        )

    return vector


def retrieve(
    collection,
    vector: np.ndarray,
) -> List[Dict[str, object]]:

    result = collection.query(
        query_embeddings=[
            vector.tolist()
        ],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    if len(ids) != TOP_K:
        raise RuntimeError(
            f"Expected {TOP_K} results; "
            f"got {len(ids)}"
        )

    if len(set(ids)) != TOP_K:
        raise RuntimeError(
            "Duplicate chunk IDs returned by Chroma."
        )

    rows = []

    for rank in range(TOP_K):

        metadata = (
            metadatas[rank]
            if metadatas[rank] is not None
            else {}
        )

        distance = float(
            distances[rank]
        )

        rows.append(
            {
                "rank": rank + 1,
                "chunk_id": str(
                    ids[rank]
                ),
                "case_id": str(
                    metadata.get(
                        "case_id",
                        metadata.get(
                            "source_case_id",
                            "",
                        ),
                    )
                ),
                "distance": distance,
                "similarity": 1.0 - distance,
                "text": documents[rank],
            }
        )

    return rows


def find_ranks(
    results,
    gold_case_id: str,
    gold_chunk_id: str,
):

    case_rank = None
    chunk_rank = None

    for row in results:

        rank = int(row["rank"])

        if (
            case_rank is None
            and str(row["case_id"]) == gold_case_id
        ):
            case_rank = rank

        if (
            chunk_rank is None
            and str(row["chunk_id"]) == gold_chunk_id
        ):
            chunk_rank = rank

    return case_rank, chunk_rank


def hit(rank, k):
    return int(
        rank is not None
        and rank <= k
    )


def rr(rank):
    return (
        0.0
        if rank is None
        else 1.0 / rank
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directories()

    print("=" * 72)
    print("PRAMANA — STEP 11 HYDE VS BASELINE")
    print("=" * 72)
    print(
        f"Project root           : {PROJECT_ROOT}"
    )
    print(
        f"Groq model             : {GROQ_MODEL}"
    )
    print(
        f"Embedding model        : {MODEL_NAME}"
    )
    print(
        f"HyDE word range        : "
        f"{HYDE_MIN_WORDS}–{HYDE_MAX_WORDS}"
    )
    print(
        f"HyDE generation target : "
        f"{HYDE_TARGET_MIN}–{HYDE_TARGET_MAX}"
    )
    print(
        f"HyDE max attempts      : "
        f"{HYDE_MAX_ATTEMPTS}"
    )
    print(
        f"Top-K                  : {TOP_K}"
    )
    print(
        f"Chroma collection      : "
        f"{CHROMA_COLLECTION}"
    )
    print()

    # --------------------------------------------------------
    # Verify frozen inputs.
    # --------------------------------------------------------

    if not GOLD_EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Missing frozen gold_eval.csv:\n{GOLD_EVAL_PATH}"
        )

    if not GOLD_CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            f"Missing gold_candidates.csv:\n"
            f"{GOLD_CANDIDATES_PATH}"
        )

    gold_hash_before = sha256_file(
        GOLD_EVAL_PATH
    )

    candidates_hash_before = sha256_file(
        GOLD_CANDIDATES_PATH
    )

    print("INPUT INTEGRITY")
    print(
        f"step7_gold_eval       : "
        f"{gold_hash_before}"
    )
    print(
        f"step7_gold_candidates : "
        f"{candidates_hash_before}"
    )
    print()

    (
        gold,
        candidate_subset,
        gold_by_id,
        candidates_by_id,
    ) = load_gold_data()

    # Informational only: prove that the source file really contains
    # 50 candidates and that only the frozen 25 are selected.
    all_candidates = pd.read_csv(
        GOLD_CANDIDATES_PATH,
        dtype=str,
    )

    print(
        f"Gold queries loaded    : {len(gold)}"
    )
    print(
        f"Candidate rows in file : "
        f"{len(all_candidates)}"
    )
    print(
        f"Candidate IDs in file  : "
        f"{all_candidates['query_id'].nunique()}"
    )
    print(
        f"Candidates selected    : "
        f"{len(candidate_subset)}"
    )
    print(
        "Candidate mapping      : 25/25 PASS"
    )
    print(
        f"DIRECT                 : "
        f"{len(EXPECTED_DIRECT_IDS)}"
    )
    print(
        f"PARTIAL                : "
        f"{len(EXPECTED_PARTIAL_IDS)}"
    )
    print()

    # --------------------------------------------------------
    # Groq client.
    # --------------------------------------------------------

    groq_client = get_groq_client()

    limiter = GroqRateLimiter(
        MIN_SECONDS_BETWEEN_GROQ_CALLS
    )

    # --------------------------------------------------------
    # Cache.
    # --------------------------------------------------------

    cache = load_hyde_cache()

    current_valid = refresh_cached_metadata(
        cache,
        candidates_by_id,
    )

    # Count only the frozen 25.
    current_valid = sum(
        1
        for qid in EXPECTED_QUERY_IDS
        if qid in cache
        and validate_hyde(
            str(
                cache[qid].get(
                    "hyde_text",
                    "",
                )
                or ""
            ),
            candidates_by_id[qid],
        )["validation_pass"]
    )

    print(
        f"Current-valid cached HyDE: "
        f"{current_valid}/25"
    )

    ordered_ids = sorted(
        EXPECTED_QUERY_IDS,
        key=lambda x: int(
            re.search(r"\d+", x).group()
        ),
    )

    # --------------------------------------------------------
    # Generate all 25 valid HyDE records.
    # --------------------------------------------------------

    for index, qid in enumerate(
        ordered_ids,
        start=1,
    ):

        candidate_query = candidates_by_id[qid]

        existing = cache.get(qid)

        if existing:

            current_validation = validate_hyde(
                str(
                    existing.get(
                        "hyde_text",
                        "",
                    )
                    or ""
                ),
                candidate_query,
            )

            if current_validation[
                "validation_pass"
            ]:

                # Persist refreshed metadata.
                cache[qid]["candidate_query"] = (
                    candidate_query
                )

                for key, value in current_validation.items():
                    cache[qid][key] = value

                save_hyde_cache(cache)

                print(
                    f"[{index}/25] {qid}: "
                    f"cached valid HyDE — skipped"
                )

                continue

        print(
            f"[{index}/25] {qid}: "
            f"generating/repairing HyDE..."
        )

        record = generate_hyde(
            groq_client,
            limiter,
            qid,
            candidate_query,
            existing,
        )

        cache[qid] = record

        # Crash-safe persistence after EVERY success.
        save_hyde_cache(cache)

        print(
            f"    PASS — "
            f"{record['word_count']} words, "
            f"{float(record['latency_seconds']):.3f}s"
        )

        time.sleep(
            POST_SUCCESS_SLEEP
        )

    # --------------------------------------------------------
    # Final persisted cache validation.
    # --------------------------------------------------------

    print()
    print("HYDE CACHE FINAL VALIDATION")

    cache = load_hyde_cache()

    if set(cache.keys()) != EXPECTED_QUERY_IDS:
        missing = EXPECTED_QUERY_IDS - set(cache.keys())
        extra = set(cache.keys()) - EXPECTED_QUERY_IDS

        raise RuntimeError(
            "Persisted HyDE cache query set mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra:   {sorted(extra)}"
        )

    all_hyde_valid = True

    final_word_counts = []

    for qid in ordered_ids:

        validation = validate_hyde(
            str(
                cache[qid].get(
                    "hyde_text",
                    "",
                )
                or ""
            ),
            candidates_by_id[qid],
        )

        if not validation["validation_pass"]:

            all_hyde_valid = False

            print(
                f"    FAIL {qid}: "
                f"{validation['validation_reason']}"
            )

        final_word_counts.append(
            int(validation["word_count"])
        )

    if not all_hyde_valid:
        raise RuntimeError(
            "Final persisted HyDE validation failed."
        )

    print(
        "    All 25 HyDE records: PASS"
    )

    # --------------------------------------------------------
    # Frozen retrieval stack.
    # --------------------------------------------------------

    print()
    print("FROZEN RETRIEVAL STACK")

    collection = load_chroma()

    print(
        f"Chroma records        : "
        f"{collection.count()}"
    )

    print(
        f"Loading BGE model      : "
        f"{MODEL_NAME}"
    )

    model = load_bge()

    # --------------------------------------------------------
    # Retrieval comparison.
    # --------------------------------------------------------

    comparison_rows = []

    for index, qid in enumerate(
        ordered_ids,
        start=1,
    ):

        candidate_query = candidates_by_id[qid]

        gold_row = gold_by_id[qid]

        relevance_type = str(
            gold_row["relevance_type"]
        ).upper()

        gold_case_id = str(
            gold_row["source_case_id"]
        )

        gold_chunk_id = str(
            gold_row["source_chunk_id"]
        )

        hyde_text = clean_text(
            str(
                cache[qid]["hyde_text"]
            )
        )

        print()
        print(
            f"[{index}/25] {qid} "
            f"({relevance_type})"
        )

        # BASELINE
        baseline_vector = embed(
            model,
            candidate_query,
        )

        baseline_results = retrieve(
            collection,
            baseline_vector,
        )

        baseline_case_rank, baseline_chunk_rank = find_ranks(
            baseline_results,
            gold_case_id,
            gold_chunk_id,
        )

        # HYDE
        hyde_vector = embed(
            model,
            hyde_text,
        )

        hyde_results = retrieve(
            collection,
            hyde_vector,
        )

        hyde_case_rank, hyde_chunk_rank = find_ranks(
            hyde_results,
            gold_case_id,
            gold_chunk_id,
        )

        # Case movement.
        if baseline_case_rank is None and hyde_case_rank is not None:
            case_result = "IMPROVED"
        elif baseline_case_rank is not None and hyde_case_rank is None:
            case_result = "WORSENED"
        elif (
            baseline_case_rank is not None
            and hyde_case_rank is not None
        ):
            if hyde_case_rank < baseline_case_rank:
                case_result = "IMPROVED"
            elif hyde_case_rank > baseline_case_rank:
                case_result = "WORSENED"
            else:
                case_result = "TIE"
        else:
            case_result = "TIE"

        # Chunk movement.
        if baseline_chunk_rank is None and hyde_chunk_rank is not None:
            chunk_result = "IMPROVED"
        elif baseline_chunk_rank is not None and hyde_chunk_rank is None:
            chunk_result = "WORSENED"
        elif (
            baseline_chunk_rank is not None
            and hyde_chunk_rank is not None
        ):
            if hyde_chunk_rank < baseline_chunk_rank:
                chunk_result = "IMPROVED"
            elif hyde_chunk_rank > baseline_chunk_rank:
                chunk_result = "WORSENED"
            else:
                chunk_result = "TIE"
        else:
            chunk_result = "TIE"

        print(
            f"    Baseline case rank : {baseline_case_rank}"
        )
        print(
            f"    HyDE case rank     : {hyde_case_rank}"
        )
        print(
            f"    Case result        : {case_result}"
        )
        print(
            f"    Baseline chunk rank: {baseline_chunk_rank}"
        )
        print(
            f"    HyDE chunk rank    : {hyde_chunk_rank}"
        )
        print(
            f"    Chunk result       : {chunk_result}"
        )

        comparison_rows.append(
            {
                "query_id": qid,
                "relevance_type": relevance_type,
                "candidate_query": candidate_query,
                "hyde_text": hyde_text,
                "hyde_word_count": word_count(hyde_text),
                "gold_case_id": gold_case_id,
                "gold_chunk_id": gold_chunk_id,

                "baseline_case_rank": (
                    ""
                    if baseline_case_rank is None
                    else baseline_case_rank
                ),
                "hyde_case_rank": (
                    ""
                    if hyde_case_rank is None
                    else hyde_case_rank
                ),
                "baseline_chunk_rank": (
                    ""
                    if baseline_chunk_rank is None
                    else baseline_chunk_rank
                ),
                "hyde_chunk_rank": (
                    ""
                    if hyde_chunk_rank is None
                    else hyde_chunk_rank
                ),

                "baseline_case_recall_at_1": hit(
                    baseline_case_rank, 1
                ),
                "baseline_case_recall_at_5": hit(
                    baseline_case_rank, 5
                ),
                "baseline_case_recall_at_10": hit(
                    baseline_case_rank, 10
                ),

                "hyde_case_recall_at_1": hit(
                    hyde_case_rank, 1
                ),
                "hyde_case_recall_at_5": hit(
                    hyde_case_rank, 5
                ),
                "hyde_case_recall_at_10": hit(
                    hyde_case_rank, 10
                ),

                "baseline_chunk_recall_at_1": hit(
                    baseline_chunk_rank, 1
                ),
                "baseline_chunk_recall_at_5": hit(
                    baseline_chunk_rank, 5
                ),
                "baseline_chunk_recall_at_10": hit(
                    baseline_chunk_rank, 10
                ),

                "hyde_chunk_recall_at_1": hit(
                    hyde_chunk_rank, 1
                ),
                "hyde_chunk_recall_at_5": hit(
                    hyde_chunk_rank, 5
                ),
                "hyde_chunk_recall_at_10": hit(
                    hyde_chunk_rank, 10
                ),

                "baseline_case_mrr": rr(
                    baseline_case_rank
                ),
                "hyde_case_mrr": rr(
                    hyde_case_rank
                ),

                "baseline_chunk_mrr": rr(
                    baseline_chunk_rank
                ),
                "hyde_chunk_mrr": rr(
                    hyde_chunk_rank
                ),

                "case_result": case_result,
                "chunk_result": chunk_result,
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Metrics.
    # --------------------------------------------------------

    def mean_col(column):
        return float(
            comparison_df[column]
            .astype(float)
            .mean()
        )

    summary = {
        "experiment": "Step 11 — HyDE vs Direct Query Retrieval",
        "model": GROQ_MODEL,
        "embedding_model": MODEL_NAME,
        "top_k": TOP_K,
        "query_count": len(comparison_df),
        "direct_count": int(
            (
                comparison_df["relevance_type"]
                == "DIRECT"
            ).sum()
        ),
        "partial_count": int(
            (
                comparison_df["relevance_type"]
                == "PARTIAL"
            ).sum()
        ),
    }

    metrics = [
        "baseline_case_recall_at_1",
        "baseline_case_recall_at_5",
        "baseline_case_recall_at_10",
        "hyde_case_recall_at_1",
        "hyde_case_recall_at_5",
        "hyde_case_recall_at_10",
        "baseline_chunk_recall_at_1",
        "baseline_chunk_recall_at_5",
        "baseline_chunk_recall_at_10",
        "hyde_chunk_recall_at_1",
        "hyde_chunk_recall_at_5",
        "hyde_chunk_recall_at_10",
        "baseline_case_mrr",
        "hyde_case_mrr",
        "baseline_chunk_mrr",
        "hyde_chunk_mrr",
    ]

    for metric in metrics:
        summary[metric] = mean_col(metric)

    summary["case_improved"] = int(
        (
            comparison_df["case_result"]
            == "IMPROVED"
        ).sum()
    )

    summary["case_worsened"] = int(
        (
            comparison_df["case_result"]
            == "WORSENED"
        ).sum()
    )

    summary["case_tie"] = int(
        (
            comparison_df["case_result"]
            == "TIE"
        ).sum()
    )

    summary["chunk_improved"] = int(
        (
            comparison_df["chunk_result"]
            == "IMPROVED"
        ).sum()
    )

    summary["chunk_worsened"] = int(
        (
            comparison_df["chunk_result"]
            == "WORSENED"
        ).sum()
    )

    summary["chunk_tie"] = int(
        (
            comparison_df["chunk_result"]
            == "TIE"
        ).sum()
    )

    for label in ["DIRECT", "PARTIAL"]:

        subset = comparison_df[
            comparison_df["relevance_type"]
            == label
        ]

        prefix = label.lower()

        summary[
            f"{prefix}_baseline_case_recall_at_10"
        ] = float(
            subset[
                "baseline_case_recall_at_10"
            ].mean()
        )

        summary[
            f"{prefix}_hyde_case_recall_at_10"
        ] = float(
            subset[
                "hyde_case_recall_at_10"
            ].mean()
        )

        summary[
            f"{prefix}_baseline_chunk_recall_at_10"
        ] = float(
            subset[
                "baseline_chunk_recall_at_10"
            ].mean()
        )

        summary[
            f"{prefix}_hyde_chunk_recall_at_10"
        ] = float(
            subset[
                "hyde_chunk_recall_at_10"
            ].mean()
        )

    # --------------------------------------------------------
    # Final integrity.
    # --------------------------------------------------------

    gold_hash_after = sha256_file(
        GOLD_EVAL_PATH
    )

    candidates_hash_after = sha256_file(
        GOLD_CANDIDATES_PATH
    )

    upstream_unchanged = (
        gold_hash_before == gold_hash_after
        and
        candidates_hash_before == candidates_hash_after
    )

    comparison_count_pass = (
        len(comparison_df) == 25
    )

    unique_query_pass = (
        comparison_df["query_id"].nunique()
        == 25
    )

    label_composition_pass = (
        summary["direct_count"] == 18
        and
        summary["partial_count"] == 7
    )

    chroma_count = collection.count()
    chroma_pass = chroma_count == 4855

    hyde_word_range_pass = all(
        HYDE_MIN_WORDS <= wc <= HYDE_MAX_WORDS
        for wc in final_word_counts
    )

    final_status = (
        "PASS"
        if all([
            all_hyde_valid,
            hyde_word_range_pass,
            comparison_count_pass,
            unique_query_pass,
            label_composition_pass,
            chroma_pass,
            upstream_unchanged,
        ])
        else "FAIL"
    )

    summary["hyde_word_count_min"] = min(
        final_word_counts
    )

    summary["hyde_word_count_max"] = max(
        final_word_counts
    )

    summary["hyde_word_count_mean"] = float(
        np.mean(final_word_counts)
    )

    summary["chroma_records"] = chroma_count
    summary["upstream_unchanged"] = upstream_unchanged
    summary["final_status"] = final_status

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # QC report.
    # --------------------------------------------------------

    with QC_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PRAMANA — STEP 11 QC REPORT\n"
        )
        f.write("=" * 60 + "\n\n")

        f.write(
            f"Project root           : {PROJECT_ROOT}\n"
        )
        f.write(
            f"Groq model             : {GROQ_MODEL}\n"
        )
        f.write(
            f"Embedding model        : {MODEL_NAME}\n"
        )
        f.write(
            f"HyDE accepted range    : "
            f"{HYDE_MIN_WORDS}-{HYDE_MAX_WORDS}\n"
        )
        f.write(
            f"HyDE target range      : "
            f"{HYDE_TARGET_MIN}-{HYDE_TARGET_MAX}\n"
        )
        f.write(
            f"HyDE max attempts      : "
            f"{HYDE_MAX_ATTEMPTS}\n"
        )
        f.write(
            f"Top-K                  : {TOP_K}\n\n"
        )

        f.write("DATASET\n")
        f.write(
            f"Candidate rows in source file : "
            f"{len(all_candidates)}\n"
        )
        f.write(
            f"Frozen gold queries           : 25\n"
        )
        f.write(
            f"Candidates selected           : "
            f"{len(candidate_subset)}\n"
        )
        f.write(
            f"DIRECT                       : "
            f"{summary['direct_count']}\n"
        )
        f.write(
            f"PARTIAL                      : "
            f"{summary['partial_count']}\n\n"
        )

        f.write("HYDE\n")
        f.write(
            f"All 25 valid                : "
            f"{'PASS' if all_hyde_valid else 'FAIL'}\n"
        )
        f.write(
            f"Word range 75-95            : "
            f"{'PASS' if hyde_word_range_pass else 'FAIL'}\n"
        )
        f.write(
            f"Word count min              : "
            f"{min(final_word_counts)}\n"
        )
        f.write(
            f"Word count max              : "
            f"{max(final_word_counts)}\n"
        )
        f.write(
            f"Word count mean             : "
            f"{np.mean(final_word_counts):.2f}\n\n"
        )

        f.write("RETRIEVAL\n")
        f.write(
            f"Chroma records              : "
            f"{chroma_count}\n"
        )
        f.write(
            f"Chroma count                : "
            f"{'PASS' if chroma_pass else 'FAIL'}\n"
        )
        f.write(
            f"Comparison records          : "
            f"{len(comparison_df)}/25\n"
        )
        f.write(
            f"Comparison count            : "
            f"{'PASS' if comparison_count_pass else 'FAIL'}\n"
        )
        f.write(
            f"Unique queries              : "
            f"{'PASS' if unique_query_pass else 'FAIL'}\n"
        )
        f.write(
            f"Label composition           : "
            f"{'PASS' if label_composition_pass else 'FAIL'}\n\n"
        )

        f.write("OVERALL\n")
        f.write(
            f"Baseline case Recall@1     : "
            f"{summary['baseline_case_recall_at_1']:.4f}\n"
        )
        f.write(
            f"HyDE case Recall@1         : "
            f"{summary['hyde_case_recall_at_1']:.4f}\n"
        )
        f.write(
            f"Baseline case Recall@5     : "
            f"{summary['baseline_case_recall_at_5']:.4f}\n"
        )
        f.write(
            f"HyDE case Recall@5         : "
            f"{summary['hyde_case_recall_at_5']:.4f}\n"
        )
        f.write(
            f"Baseline case Recall@10    : "
            f"{summary['baseline_case_recall_at_10']:.4f}\n"
        )
        f.write(
            f"HyDE case Recall@10        : "
            f"{summary['hyde_case_recall_at_10']:.4f}\n"
        )
        f.write(
            f"Baseline chunk Recall@10   : "
            f"{summary['baseline_chunk_recall_at_10']:.4f}\n"
        )
        f.write(
            f"HyDE chunk Recall@10       : "
            f"{summary['hyde_chunk_recall_at_10']:.4f}\n"
        )
        f.write(
            f"Baseline case MRR          : "
            f"{summary['baseline_case_mrr']:.4f}\n"
        )
        f.write(
            f"HyDE case MRR              : "
            f"{summary['hyde_case_mrr']:.4f}\n\n"
        )

        f.write("RANK MOVEMENT\n")
        f.write(
            f"Case improved              : "
            f"{summary['case_improved']}\n"
        )
        f.write(
            f"Case worsened              : "
            f"{summary['case_worsened']}\n"
        )
        f.write(
            f"Case tie                   : "
            f"{summary['case_tie']}\n"
        )
        f.write(
            f"Chunk improved             : "
            f"{summary['chunk_improved']}\n"
        )
        f.write(
            f"Chunk worsened             : "
            f"{summary['chunk_worsened']}\n"
        )
        f.write(
            f"Chunk tie                  : "
            f"{summary['chunk_tie']}\n\n"
        )

        f.write("UPSTREAM INTEGRITY\n")
        f.write(
            f"Step 7 gold unchanged      : "
            f"{'PASS' if gold_hash_before == gold_hash_after else 'FAIL'}\n"
        )
        f.write(
            f"Candidates unchanged       : "
            f"{'PASS' if candidates_hash_before == candidates_hash_after else 'FAIL'}\n"
        )
        f.write(
            f"Upstream unchanged         : "
            f"{'PASS' if upstream_unchanged else 'FAIL'}\n\n"
        )

        f.write("FINAL STATUS\n")
        f.write(
            f"{final_status}\n"
        )

    # --------------------------------------------------------
    # Terminal final QC.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 11 FINAL QC")
    print("=" * 72)

    print(
        f"Candidate rows in source : "
        f"{len(all_candidates)}"
    )

    print(
        f"Frozen gold queries      : 25"
    )

    print(
        f"Candidates selected      : "
        f"{len(candidate_subset)}/25"
    )

    print(
        f"HyDE valid               : "
        f"{sum(1 for qid in ordered_ids if validate_hyde(cache[qid]['hyde_text'], candidates_by_id[qid])['validation_pass'])}/25"
    )

    print(
        f"HyDE word range 75–95    : "
        f"{'PASS' if hyde_word_range_pass else 'FAIL'}"
    )

    print(
        f"HyDE word count min      : "
        f"{min(final_word_counts)}"
    )

    print(
        f"HyDE word count max      : "
        f"{max(final_word_counts)}"
    )

    print(
        f"HyDE word count mean     : "
        f"{np.mean(final_word_counts):.2f}"
    )

    print(
        f"Chroma records           : "
        f"{chroma_count}"
    )

    print(
        f"Comparison records       : "
        f"{len(comparison_df)}/25"
    )

    print(
        f"Label composition        : "
        f"{'PASS' if label_composition_pass else 'FAIL'}"
    )

    print()
    print("RETRIEVAL RESULTS")

    print(
        f"Baseline case Recall@1   : "
        f"{summary['baseline_case_recall_at_1']:.4f}"
    )

    print(
        f"HyDE case Recall@1       : "
        f"{summary['hyde_case_recall_at_1']:.4f}"
    )

    print(
        f"Baseline case Recall@5   : "
        f"{summary['baseline_case_recall_at_5']:.4f}"
    )

    print(
        f"HyDE case Recall@5       : "
        f"{summary['hyde_case_recall_at_5']:.4f}"
    )

    print(
        f"Baseline case Recall@10  : "
        f"{summary['baseline_case_recall_at_10']:.4f}"
    )

    print(
        f"HyDE case Recall@10      : "
        f"{summary['hyde_case_recall_at_10']:.4f}"
    )

    print(
        f"Baseline chunk Recall@10 : "
        f"{summary['baseline_chunk_recall_at_10']:.4f}"
    )

    print(
        f"HyDE chunk Recall@10     : "
        f"{summary['hyde_chunk_recall_at_10']:.4f}"
    )

    print()
    print("RANK MOVEMENT")

    print(
        f"Case improved            : "
        f"{summary['case_improved']}"
    )

    print(
        f"Case worsened            : "
        f"{summary['case_worsened']}"
    )

    print(
        f"Case tie                 : "
        f"{summary['case_tie']}"
    )

    print(
        f"Chunk improved           : "
        f"{summary['chunk_improved']}"
    )

    print(
        f"Chunk worsened           : "
        f"{summary['chunk_worsened']}"
    )

    print(
        f"Chunk tie                : "
        f"{summary['chunk_tie']}"
    )

    print()
    print("UPSTREAM INTEGRITY")

    print(
        f"Step 7 gold unchanged    : "
        f"{'PASS' if gold_hash_before == gold_hash_after else 'FAIL'}"
    )

    print(
        f"Candidates unchanged     : "
        f"{'PASS' if candidates_hash_before == candidates_hash_after else 'FAIL'}"
    )

    print(
        f"FINAL STATUS             : "
        f"{final_status}"
    )

    print()
    print(
        f"HyDE cache saved         : "
        f"{HYDE_CACHE_PATH}"
    )

    print(
        f"Comparison saved         : "
        f"{COMPARISON_PATH}"
    )

    print(
        f"Summary saved            : "
        f"{SUMMARY_PATH}"
    )

    print(
        f"QC report saved          : "
        f"{QC_PATH}"
    )

    if final_status != "PASS":
        print()
        print(
            "Step 11 is NOT complete."
        )
        sys.exit(1)

    print()
    print(
        "STEP 11 FINAL STATUS: PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted by user.")
        print("Existing HyDE cache has been preserved.")
        sys.exit(130)
    except Exception as exc:
        print()
        print("FATAL:", exc)
        print()
        print("Step 11 is NOT complete.")
        sys.exit(1)