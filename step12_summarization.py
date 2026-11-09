"""
PRAMANA — STEP 12: BASELINE SUMMARIZATION
===============================================================

FINAL DESIGN
------------
Step 11 established that direct/Baseline retrieval outperformed HyDE.
Therefore Step 12 uses:

    Step 8 legal query
        -> BGE-small-en-v1.5
        -> frozen ChromaDB V14
        -> Top-3 chunks
        -> Story + Structured JSON

For every retrieved chunk this script generates BOTH:
1. A 100–160 prose-word citizen-friendly Story.
2. A strict structured JSON object.

Important:
- HyDE is NOT used.
- Step 8 is read-only.
- Step 6 ChromaDB is read-only.
- Steps 4–11 are untouched.
- Full retrieved Chroma chunk text is used, not a preview.
- Source tags are added deterministically by Python.
- JSON generation uses strict JSON response format when supported,
  with a robust extraction fallback.
- Existing valid case summaries are reused.
- Progress is saved after every completed query.
- Final QC reloads the persisted file and validates everything again.
"""

import os
import csv
import json
import re
import time
import hashlib
from pathlib import Path

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer


# ================================================================
# CONFIGURATION
# ================================================================

MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

STEP8_PATH = Path("data/step8/step8_results.csv")
CHROMA_PATH = Path("data/step6_v14/chroma_db")
COLLECTION_NAME = "pramana_v14"

OUTPUT_DIR = Path("data/step12")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "step12_summaries.json"
QC_PATH = OUTPUT_DIR / "step12_qc_report.txt"

EXPECTED_CHROMA_COUNT = 4855
EXPECTED_EMBEDDING_DIM = 384

TOP_K = 3

# Locked Step 12 test set from the completed end-to-end evaluation.
TEST_QUERY_IDS = [
    "S08-001",
    "S08-012",
    "S08-013",
    "S08-014",
    "S08-015",
]

# Revised after observing real model behavior:
STORY_MIN_WORDS = 100
STORY_MAX_WORDS = 160

MAX_ATTEMPTS = 3

# Enough context for a full legal chunk while avoiding pathological
# prompt growth.
MAX_SOURCE_CHARS = 20000


# ================================================================
# API / MODELS
# ================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set."
    )

client = Groq(api_key=GROQ_API_KEY)

print("Loading BGE model...")
BGE = SentenceTransformer(EMBEDDING_MODEL)

print("Opening frozen ChromaDB...")
chroma = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma.get_collection(
    COLLECTION_NAME
)

actual_chroma_count = collection.count()

if actual_chroma_count != EXPECTED_CHROMA_COUNT:
    raise RuntimeError(
        "FROZEN CHROMA INTEGRITY FAILURE: "
        f"expected {EXPECTED_CHROMA_COUNT}, "
        f"found {actual_chroma_count}"
    )


# ================================================================
# STRICT STRUCTURED JSON SCHEMA
# ================================================================

SUMMARY_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "facts",
        "evidence",
        "reasoning",
        "outcome",
    ],
    "properties": {
        "facts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {
            "type": "array",
            "items": {"type": "string"},
        },
        "outcome": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


# ================================================================
# PROMPTS
# ================================================================

STORY_PROMPT = r"""
You are explaining a real Indian court judgment excerpt to an
ordinary citizen with no legal training.

CASE ID: {case_id}
YEAR: {year}
SOURCE CHUNK: {chunk_id}

SOURCE TEXT:
{source_text}

Write ONE coherent plain-language paragraph of approximately 115–130 WORDS.\nThe absolute hard limit is 100–160 words.

Explain only what the source text supports:
- the dispute or legal issue;
- facts or evidence actually mentioned;
- legal reasoning actually expressed;
- the outcome ONLY if the source text actually states an outcome.

STRICT RULES:
1. Write only the substantive explanation.
2. Do NOT add source tags. Python will add them automatically.
3. Do NOT invent or infer facts.
4. Do NOT invent names, dates, amounts, sections, case names,
   procedural events, or outcomes.
5. Do NOT turn a general legal principle into a fact about the parties.
6. Do NOT give legal advice.
7. Do NOT tell the citizen what they should do.
8. Do NOT claim that this judgment determines the citizen's own case.
9. If the excerpt does not establish the final outcome, say so.
10. Do not use headings, bullets, numbered lists, markdown, or
    quotation marks around the answer.
11. The final sentence must be exactly:
    This is not legal advice.
12. Output ONLY the paragraph.

The 100–160 word requirement applies to prose words.
Source-attribution metadata added later is NOT part of that count.
"""


JSON_PROMPT = r"""
Extract structured information ONLY from the supplied Indian court
judgment excerpt.

CASE ID: {case_id}
YEAR: {year}
SOURCE CHUNK: {chunk_id}

SOURCE TEXT:
{source_text}

TASK
Return exactly ONE JSON object with exactly these four top-level keys:
facts, evidence, reasoning, outcome.

All four keys are REQUIRED.
If a category is not explicitly supported by the source, return [].

EXACT SHAPE
{{
  "facts": [],
  "evidence": [],
  "reasoning": [],
  "outcome": []
}}

Each non-empty array MUST contain plain strings only.
Do NOT create objects inside arrays.
Do NOT create a source_chunk field.
Python adds source_chunk after successful validation.

CONTENT RULES
- facts = explicitly stated facts only.
- evidence = explicitly mentioned evidence only.
- reasoning = legal reasoning explicitly expressed in the excerpt only.
- outcome = outcome only when explicitly stated in the excerpt.
- Use ONLY information explicitly supported by the supplied excerpt.
- Do not invent or infer facts, names, dates, amounts, sections, case names,
  procedural events, reasoning, or outcomes.
- Do not turn a general legal principle into a fact about the parties.
- Do not give legal advice.
- Do not speculate.
- Do not use information outside the supplied source text.

OUTPUT RULES
- Exactly one JSON object.
- Exactly four top-level keys.
- Every value is an array.
- Every array element is a string.
- Empty categories must be [].
- No fifth key.
- No markdown.
- No explanation outside the JSON object.
"""



# ================================================================
# UTILITY FUNCTIONS
# ================================================================

SOURCE_TAG_RE = re.compile(
    r"\s*\[Source:\s*[^\]]+\]\s*"
)

TAGGED_SENTENCE_RE = re.compile(
    r"(?P<sentence>.*?[.!?])\s*"
    r"(?P<tag>\[Source:\s*[^\]]+\])"
    r"(?=\s|$)",
    re.DOTALL,
)


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def clean_text(text):
    text = (text or "").strip()

    text = re.sub(
        r"^```(?:text|markdown)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def prose_without_tags(text):
    return SOURCE_TAG_RE.sub(
        " ",
        text or "",
    ).strip()


def word_count(text):
    return len(
        re.findall(
            r"\b[\w₹$'-]+\b",
            prose_without_tags(text),
        )
    )


def split_prose_sentences(text):
    body = prose_without_tags(text)

    body = re.sub(
        r"\s*This is not legal advice\.\s*$",
        "",
        body,
        flags=re.I,
    ).strip()

    return [
        s.strip()
        for s in re.split(
            r"(?<=[.!?])\s+",
            body,
        )
        if s.strip()
    ]


def split_tagged_sentences(text):
    """
    Extract substantive sentence + source-tag pairs from a
    programmatically tagged Story.
    """
    if not text:
        return []

    return [
        (
            match.group("sentence").strip(),
            match.group("tag").strip(),
        )
        for match in TAGGED_SENTENCE_RE.finditer(text)
    ]


def add_source_tags(story, case_id, chunk_id):
    """
    Source attribution is deterministic and is never delegated
    to the LLM.
    """

    story = clean_text(story)

    if not story:
        return ""

    disclaimer = "This is not legal advice."

    # Remove any accidental model-generated disclaimer.
    story = re.sub(
        r"\s*" + re.escape(disclaimer) + r"\s*$",
        "",
        story,
        flags=re.I,
    ).strip()

    # Remove any accidental model-generated tags.
    story = SOURCE_TAG_RE.sub(
        " ",
        story,
    ).strip()

    sentences = split_prose_sentences(story)

    if not sentences:
        return disclaimer

    tag = (
        f"[Source: {case_id}, "
        f"Chunk {chunk_id}]"
    )

    tagged = [
        f"{sentence} {tag}"
        for sentence in sentences
    ]

    tagged.append(disclaimer)

    return " ".join(tagged)


# ================================================================
# VALIDATION
# ================================================================

def local_compress_story(story, case_id, chunk_id):
    """
    Deterministic local Story compressor.

    Purpose:
    --------
    If the LLM repeatedly returns a Story above 160 words, do not make
    another paid/network call just to shave words. Instead, remove
    complete, lower-priority sentences locally.

    This function NEVER rewrites or invents language. It only removes
    complete model-generated sentences, preserving their exact wording
    and source attribution.

    The disclaimer is always retained as the final sentence.

    Target:
    -------
    <= 150 prose words where possible, leaving a safety margin below
    the 160-word hard ceiling.
    """

    if not story:
        return story

    expected_tag = (
        f"[Source: {case_id}, "
        f"Chunk {chunk_id}]"
    )

    # Remove disclaimer temporarily.
    body = re.sub(
        r"\s*This is not legal advice\.\s*$",
        "",
        story.strip(),
        flags=re.I,
    ).strip()

    # Extract sentence/tag units. Because source tags are inserted by
    # Python, each substantive sentence should be independently removable.
    units = split_tagged_sentences(body)

    if not units:
        return story

    # If the parser did not recover all sentences, do not attempt
    # destructive compression.
    if len(units) != len(split_prose_sentences(body)):
        return story

    current = list(units)

    def render(items):
        if not items:
            return "This is not legal advice."

        return (
            " ".join(
                f"{sentence.strip()} {tag.strip()}"
                for sentence, tag in items
            )
            + " This is not legal advice."
        )

    # We want a comfortable margin, not exactly 160.
    target = 150

    # Never remove the first substantive sentence. It normally establishes
    # the subject of the excerpt.
    #
    # Prefer removing sentences that are:
    # 1. repetitive/meta,
    # 2. comparatively long,
    # 3. less likely to contain an explicit outcome or core legal reasoning.
    def removal_score(index, sentence):
        text = sentence.lower()
        score = 0

        if index == 0:
            return -10_000

        # Preserve explicit outcome language where possible.
        if re.search(
            r"\b("
            r"held|hold|ordered|orders|allowed|dismissed|"
            r"disposed|rejected|confirmed|set aside|"
            r"decreed|judgment|decision|conclusion|"
            r"appeal was|appeal is|petition was|petition is"
            r")\b",
            text,
        ):
            score -= 100

        # Preserve core legal reasoning.
        if re.search(
            r"\b("
            r"court observed|court held|court found|"
            r"court considered|therefore|accordingly|"
            r"reason|principle|law|limitation|"
            r"maintainable|entitled|liable|"
            r"legal|statute|section"
            r")\b",
            text,
        ):
            score -= 25

        # Sentences that explicitly announce context rather than substance
        # are safer candidates for removal.
        if re.search(
            r"\b("
            r"it is important to note|"
            r"in this context|"
            r"it may be noted|"
            r"the judgment also explains|"
            r"the passage indicates|"
            r"the excerpt shows"
            r")\b",
            text,
        ):
            score += 40

        # Prefer removing longer sentences when several are otherwise equal.
        score += min(len(sentence.split()), 60) * 0.25

        # Later sentences are slightly safer than the opening sentence.
        score += index * 0.5

        return score

    while word_count(render(current)) > target and len(current) > 2:

        candidates = [
            (
                removal_score(index, sentence),
                index,
            )
            for index, (sentence, _) in enumerate(current)
            if index != 0
        ]

        if not candidates:
            break

        _, remove_index = max(
            candidates,
            key=lambda x: x[0],
        )

        del current[remove_index]

    compressed = render(current)

    # Only accept the local compression if it remains inside the
    # hard range and attribution remains exact.
    ok, _ = validate_story(
        compressed,
        case_id,
        chunk_id,
    )

    if ok:
        return compressed

    return story


def validate_story(story, case_id, chunk_id):
    reasons = []

    if not story:
        return False, ["empty story"]

    wc = word_count(story)

    if wc < STORY_MIN_WORDS or wc > STORY_MAX_WORDS:
        reasons.append(
            f"word_count={wc} outside "
            f"{STORY_MIN_WORDS}-{STORY_MAX_WORDS}"
        )

    if "```" in story:
        reasons.append("markdown/code fence")

    if re.search(
        r"(?m)^\s*[-*]\s+",
        story,
    ):
        reasons.append("bullet/list format")

    if not story.rstrip().endswith(
        "This is not legal advice."
    ):
        reasons.append(
            'missing final "This is not legal advice."'
        )

    if re.search(
        r"(?i)\b("
        r"I recommend|you should|you must|"
        r"you can file|you need to|"
        r"consult a lawyer|hire a lawyer|"
        r"take legal action"
        r")\b",
        story,
    ):
        reasons.append("possible legal advice")

    expected_tag = (
        f"[Source: {case_id}, "
        f"Chunk {chunk_id}]"
    )

    tagged = [
        (sentence, tag)
        for sentence, tag
        in TAGGED_SENTENCE_RE.findall(story)
    ]

    prose_sentences = split_prose_sentences(story)

    if not tagged:
        reasons.append(
            "no source-tagged substantive sentences"
        )
    else:
        if len(tagged) != len(prose_sentences):
            reasons.append(
                "not every substantive sentence has a source tag"
            )

        for _, tag in tagged:
            if tag != expected_tag:
                reasons.append(
                    "incorrect source attribution"
                )
                break

    return len(reasons) == 0, reasons


def validate_model_json(data):
    """
    Validate the exact form produced by the Groq strict schema:
    four required arrays containing strings only.
    """
    required = {
        "facts",
        "evidence",
        "reasoning",
        "outcome",
    }

    if not isinstance(data, dict):
        return False, ["JSON root is not an object"]

    reasons = []

    if set(data.keys()) != required:
        reasons.append(
            "JSON keys are not exactly "
            "facts/evidence/reasoning/outcome"
        )

    for category in (
        "facts",
        "evidence",
        "reasoning",
        "outcome",
    ):
        value = data.get(category)

        if not isinstance(value, list):
            reasons.append(
                f"{category} is not an array"
            )
            continue

        for item in value:
            if not isinstance(item, str):
                reasons.append(
                    f"{category} contains non-string item"
                )
            elif not item.strip():
                reasons.append(
                    f"{category} contains empty sentence"
                )

    return len(reasons) == 0, reasons


def canonicalize_structured_json(data, chunk_id):
    """
    Convert valid model JSON into the persisted Step 12 format.
    source_chunk is attached deterministically by Python.
    """
    ok, reasons = validate_model_json(data)

    if not ok:
        raise RuntimeError(
            "Cannot canonicalize invalid model JSON: "
            + "; ".join(reasons)
        )

    source_chunk = str(chunk_id)

    return {
        category: [
            {
                "sentence": sentence.strip(),
                "source_chunk": source_chunk,
            }
            for sentence in data[category]
        ]
        for category in (
            "facts",
            "evidence",
            "reasoning",
            "outcome",
        )
    }


def validate_structured_json(data, chunk_id):
    """
    Validate the persisted Step 12 representation:
    four required arrays of {sentence, source_chunk} objects.
    """
    required = {
        "facts",
        "evidence",
        "reasoning",
        "outcome",
    }

    if not isinstance(data, dict):
        return False, ["JSON root is not an object"]

    reasons = []

    if set(data.keys()) != required:
        reasons.append(
            "JSON keys are not exactly "
            "facts/evidence/reasoning/outcome"
        )

    for category in (
        "facts",
        "evidence",
        "reasoning",
        "outcome",
    ):
        value = data.get(category)

        if not isinstance(value, list):
            reasons.append(
                f"{category} is not an array"
            )
            continue

        for item in value:
            if not isinstance(item, dict):
                reasons.append(
                    f"{category} contains non-object item"
                )
                continue

            if set(item.keys()) != {
                "sentence",
                "source_chunk",
            }:
                reasons.append(
                    f"{category} item has incorrect keys"
                )
                continue

            sentence = item.get("sentence")
            source = item.get("source_chunk")

            if (
                not isinstance(sentence, str)
                or not sentence.strip()
            ):
                reasons.append(
                    f"{category} contains empty sentence"
                )

            if str(source) != str(chunk_id):
                reasons.append(
                    f"{category} has incorrect source_chunk"
                )

    return len(reasons) == 0, reasons



# ================================================================
# STEP 8 INPUT
# ================================================================

def load_step8():

    if not STEP8_PATH.exists():
        raise FileNotFoundError(
            f"Step 8 results not found: {STEP8_PATH}"
        )

    with open(
        STEP8_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    indexed = {}

    for row in rows:
        qid = str(
            row.get("query_id", "")
        ).strip()

        if qid:
            indexed[qid] = row

    missing = [
        qid
        for qid in TEST_QUERY_IDS
        if qid not in indexed
    ]

    if missing:
        raise RuntimeError(
            f"Missing Step 8 query IDs: {missing}"
        )

    return indexed


def get_value(row, *names):
    for name in names:
        value = row.get(name)

        if value is not None:
            value = str(value).strip()

            if value:
                return value

    return ""


# ================================================================
# FROZEN BASELINE RETRIEVAL
# ================================================================

def retrieve_top3(legal_query):

    if not legal_query:
        raise RuntimeError(
            "Cannot retrieve with empty legal query."
        )

    embedding = BGE.encode(
        legal_query,
        normalize_embeddings=True,
    )

    if embedding.shape[0] != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected BGE dimension "
            f"{EXPECTED_EMBEDDING_DIM}; "
            f"got {embedding.shape[0]}"
        )

    norm = float(
        sum(float(x) ** 2 for x in embedding) ** 0.5
    )

    if abs(norm - 1.0) > 1e-4:
        raise RuntimeError(
            f"Embedding normalization failure: norm={norm}"
        )

    results = collection.query(
        query_embeddings=[
            embedding.tolist()
        ],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if len(ids) != TOP_K:
        raise RuntimeError(
            f"Expected {TOP_K} chunks; got {len(ids)}"
        )

    output = []

    for rank, (
        chunk_id,
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            ids,
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):

        if not document:
            raise RuntimeError(
                f"Empty document returned for {chunk_id}"
            )

        case_id = metadata.get("case_id")

        if not case_id:
            raise RuntimeError(
                f"Missing case_id for {chunk_id}"
            )

        output.append({
            "rank": rank,
            "chunk_id": str(chunk_id),
            "case_id": str(case_id),
            "year": metadata.get(
                "decision_year"
            ),
            "similarity": round(
                1.0 - float(distance),
                6,
            ),
            "source_text": str(document),
        })

    return output


# ================================================================
# RATE-LIMIT HANDLING
# ================================================================

class ProviderRateLimitError(RuntimeError):
    """Provider quota/rate-limit reached; stop safely and resume later."""


def is_rate_limit_error(exc):
    """
    Detect Groq/OpenAI-compatible 429 rate-limit failures without
    depending on one exact SDK exception class.
    """
    text = str(exc).lower()

    return (
        "rate limit" in text
        or "rate_limit_exceeded" in text
        or "tokens per day" in text
        or "tokens per minute" in text
        or "error code: 429" in text
        or "429" in text
    )


def raise_if_rate_limited(exc):
    if is_rate_limit_error(exc):
        raise ProviderRateLimitError(
            "Groq rate limit reached. "
            "Stopping immediately so no additional retry calls "
            "are wasted. Resume later; completed cached results "
            "will be reused."
        ) from exc


# ================================================================
# LLM CALLS
# ================================================================

def call_story_model(case, corrective_instruction=""):
    """
    Generate ONLY the prose story.

    Story and JSON are deliberately separate calls. This prevents a
    failure in structured JSON generation from invalidating an already
    correct Story.
    """

    prompt = STORY_PROMPT.format(
        case_id=case["case_id"],
        year=(
            case["year"]
            if case["year"] is not None
            else "not provided"
        ),
        chunk_id=case["chunk_id"],
        source_text=case["source_text"][
            :MAX_SOURCE_CHARS
        ],
    )

    if corrective_instruction:
        prompt += (
            "\n\nCORRECTION REQUIRED:\n"
            + corrective_instruction
        )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_completion_tokens=500,
            reasoning_effort="low",
        )
    except Exception as exc:
        raise_if_rate_limited(exc)
        raise

    raw = clean_text(
        response.choices[0].message.content
        or ""
    )

    return add_source_tags(
        raw,
        case["case_id"],
        case["chunk_id"],
    )


def _json_prompt(case, corrective_instruction=""):
    prompt = JSON_PROMPT.format(
        case_id=case["case_id"],
        year=(
            case["year"]
            if case["year"] is not None
            else "not provided"
        ),
        chunk_id=case["chunk_id"],
        source_text=case["source_text"][
            :MAX_SOURCE_CHARS
        ],
    )

    if corrective_instruction:
        prompt += (
            "\n\nCORRECTION REQUIRED:\n"
            + corrective_instruction
            + "\nReturn the corrected JSON object only."
        )

    return prompt


def _parse_json_object(raw):
    """
    Parse JSON conservatively.

    A JSON array, scalar, or malformed object is NEVER accepted.
    """
    if not raw:
        return None

    raw = raw.strip()

    # Remove markdown fences if the model nevertheless emits them.
    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.I,
    )
    raw = re.sub(
        r"\s*```$",
        "",
        raw,
    ).strip()

    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    # Conservative balanced-object extraction.
    start = raw.find("{")

    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(raw)):

        char = raw[index]

        if in_string:

            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False

            continue

        if char == '"':
            in_string = True

        elif char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:

                candidate = raw[
                    start:index + 1
                ]

                try:
                    value = json.loads(
                        candidate
                    )

                    return (
                        value
                        if isinstance(value, dict)
                        else None
                    )

                except json.JSONDecodeError:
                    return None

    return None


def _request_json_mode(case, corrective_instruction=""):
    """
    Deprecated and intentionally unused.
    Step 12 requires strict JSON Schema only.
    """
    raise RuntimeError(
        "JSON Object Mode is disabled. Step 12 requires strict JSON Schema."
    )


def _request_strict_schema(case, corrective_instruction=""):
    """
    One Groq Strict Structured Outputs request for GPT-OSS 20B.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": _json_prompt(
                    case,
                    corrective_instruction,
                ),
            }
        ],
        temperature=0.0,
        max_completion_tokens=1200,
        reasoning_effort="low",
        include_reasoning=False,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "legal_summary",
                "strict": True,
                "schema": SUMMARY_JSON_SCHEMA,
            },
        },
    )

    return response.choices[0].message.content or ""


def generate_story_with_retry(case):
    """
    Generate a valid Story with targeted retry instructions.

    Retry strategy:
    - Attempt 1: normal concise generation, target 115–130 words.
    - Attempt 2: stronger compression target, 105–125 words.
    - Attempt 3: final LLM attempt, target 100–115 words.
    - If the final LLM response is still too long, apply a deterministic
      local extractive compressor. This does not call the API and cannot
      invent new facts.
    """

    last_reasons = []
    last_story = ""

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        print(
            f"      Story attempt "
            f"{attempt}/{MAX_ATTEMPTS}"
        )

        try:

            correction_parts = []

            if attempt == 1:
                correction_parts.append(
                    "Target approximately 115–130 prose words. "
                    "Stay comfortably below the 160-word hard maximum."
                )

            elif attempt == 2:
                correction_parts.append(
                    "The previous response was too long. "
                    "Rewrite it substantially shorter. "
                    "Target 105–125 prose words. "
                    "Keep only the most important source-supported facts, "
                    "evidence, reasoning, and outcome. "
                    "Remove repetition and secondary details."
                )

            else:
                correction_parts.append(
                    "FINAL LENGTH CORRECTION. "
                    "Produce a concise paragraph of 100–115 prose words. "
                    "Remove secondary details, repetition, and unnecessary "
                    "legal explanation. Preserve only information explicitly "
                    "supported by the source text."
                )

            if last_reasons:
                correction_parts.append(
                    "Previous validation errors: "
                    + "; ".join(last_reasons)
                )

            correction = "\n".join(
                correction_parts
            )

            story = call_story_model(
                case,
                correction,
            )

            last_story = story

            ok, reasons = validate_story(
                story,
                case["case_id"],
                case["chunk_id"],
            )

            if ok:
                print(
                    f"      Story PASS — "
                    f"{word_count(story)} prose words"
                )
                return story

            last_reasons = reasons

            print(
                "      Story validation FAIL: "
                + "; ".join(reasons)
            )

        except ProviderRateLimitError:
            print(
                "      RATE LIMIT — stopping immediately. "
                "No further Story retries will be attempted."
            )
            raise

        except Exception as exc:

            last_reasons = [
                f"{type(exc).__name__}: {exc}"
            ]

            print(
                "      Story ERROR: "
                + last_reasons[0]
            )

        if attempt < MAX_ATTEMPTS:
            time.sleep(1)

    # ------------------------------------------------------------
    # FINAL LOCAL COMPRESSOR
    # ------------------------------------------------------------
    #
    # This is deliberately AFTER the three LLM attempts. It is used
    # only when the generated Story is otherwise structurally valid
    # but outside the word range.
    #
    # It never invents, paraphrases, or calls the API.
    # It only removes complete source-tagged sentences.

    if last_story:
        print(
            "      Local Story compressor..."
        )

        compressed = local_compress_story(
            last_story,
            case["case_id"],
            case["chunk_id"],
        )

        compressed_ok, compressed_reasons = (
            validate_story(
                compressed,
                case["case_id"],
                case["chunk_id"],
            )
        )

        if compressed_ok:
            print(
                f"      Local compression PASS — "
                f"{word_count(compressed)} prose words"
            )
            return compressed

        print(
            "      Local compression FAIL: "
            + "; ".join(
                compressed_reasons
            )
        )

        last_reasons = (
            compressed_reasons
        )

    raise RuntimeError(
        "Story generation failed after "
        f"{MAX_ATTEMPTS} attempts plus local compression: "
        + "; ".join(last_reasons)
    )


def generate_json_with_retry(case):
    """
    Generate model JSON with Groq Strict Structured Outputs only.
    After model validation, convert it to the persisted PRAMANA format.
    """
    last_reasons = []

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        print(
            f"      JSON attempt "
            f"{attempt}/{MAX_ATTEMPTS}"
        )

        if attempt == 1:
            correction = (
                "Return exactly four required keys: facts, evidence, "
                "reasoning, outcome. All four must be present. "
                "Each value must be an array of strings. "
                "Use [] when unsupported."
            )
        elif attempt == 2:
            correction = (
                "STRICT CORRECTION: Return exactly "
                '{"facts":[],"evidence":[],"reasoning":[],"outcome":[]}. '
                "All four keys are required. Every array element must be "
                "a plain string. Do not create objects. Do not add keys."
            )
        else:
            correction = (
                "FINAL STRICT RETRY: Return only one JSON object with "
                "exactly facts, evidence, reasoning, and outcome. "
                "All four values must be arrays of strings. "
                "Use [] when necessary. No extra text."
            )

        if last_reasons:
            correction += (
                "\nPrevious provider/validation feedback: "
                + "; ".join(last_reasons)
            )

        try:
            raw = _request_strict_schema(
                case,
                correction,
            )
        except ProviderRateLimitError:
            raise
        except Exception as exc:
            raise_if_rate_limited(exc)
            last_reasons = [
                f"{type(exc).__name__}: {exc}"
            ]
            print(
                "      JSON ERROR: "
                + last_reasons[0]
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(1)
            continue

        if not raw.strip():
            last_reasons = [
                "empty strict JSON response"
            ]
        else:
            parsed = _parse_json_object(raw)

            if parsed is None:
                last_reasons = [
                    "strict response was not a parseable JSON object"
                ]
            else:
                ok, reasons = validate_model_json(parsed)

                if ok:
                    canonical = canonicalize_structured_json(
                        parsed,
                        case["chunk_id"],
                    )
                    print(
                        "      JSON PASS (strict schema)"
                    )
                    return canonical

                last_reasons = reasons

        print(
            "      JSON validation FAIL: "
            + "; ".join(last_reasons)
        )

        if attempt < MAX_ATTEMPTS:
            time.sleep(1)

    raise RuntimeError(
        "Strict JSON generation failed after "
        f"{MAX_ATTEMPTS} attempts: "
        + "; ".join(last_reasons)
    )


def generate_case_summary(case):

    # IMPORTANT:
    # Story and JSON are independently generated and independently
    # retried. A JSON failure never discards a valid Story.

    story = generate_story_with_retry(
        case
    )

    structured = generate_json_with_retry(
        case
    )

    story_ok, story_reasons = validate_story(
        story,
        case["case_id"],
        case["chunk_id"],
    )

    json_ok, json_reasons = (
        validate_structured_json(
            structured,
            case["chunk_id"],
        )
    )

    if not story_ok:
        raise RuntimeError(
            "Internal Story validation failure after "
            "successful generation: "
            + "; ".join(story_reasons)
        )

    if not json_ok:
        raise RuntimeError(
            "Internal JSON validation failure after "
            "successful generation: "
            + "; ".join(json_reasons)
        )

    return {
        "story": story,
        "story_word_count": word_count(
            story
        ),
        "structured": structured,
        "story_validation": {
            "pass": True,
            "reasons": [],
        },
        "json_validation": {
            "pass": True,
            "reasons": [],
        },
    }


# ================================================================
# CACHE
# ================================================================

def load_cache():

    if not RESULTS_PATH.exists():
        return {}

    try:

        with open(
            RESULTS_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if not isinstance(data, list):
            return {}

        return {
            str(record["query_id"]): record
            for record in data
            if record.get("query_id")
            and record.get("status") == "PASS"
        }

    except Exception:
        return {}


def save_results(results):

    temp_path = RESULTS_PATH.with_suffix(
        ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    temp_path.replace(
        RESULTS_PATH
    )


# ================================================================
# FINAL PERSISTED QC
# ================================================================

def validate_all_persisted(results):

    expected_ids = set(TEST_QUERY_IDS)

    actual_ids = {
        str(r.get("query_id"))
        for r in results
    }

    reasons = []

    if actual_ids != expected_ids:
        reasons.append(
            "query coverage mismatch"
        )

    for result in results:

        qid = str(
            result.get("query_id", "")
        )

        if result.get("status") != "PASS":
            reasons.append(
                f"{qid}: status is not PASS"
            )
            continue

        retrieval = result.get(
            "retrieval",
            [],
        )

        if len(retrieval) != TOP_K:
            reasons.append(
                f"{qid}: retrieval count != {TOP_K}"
            )

        retrieval_ids = [
            str(x.get("chunk_id"))
            for x in retrieval
        ]

        if len(set(retrieval_ids)) != TOP_K:
            reasons.append(
                f"{qid}: duplicate retrieval chunk IDs"
            )

        summaries = result.get(
            "summaries",
            [],
        )

        if len(summaries) != TOP_K:
            reasons.append(
                f"{qid}: summary count != {TOP_K}"
            )
            continue

        summary_ids = [
            str(x.get("chunk_id"))
            for x in summaries
        ]

        if set(summary_ids) != set(retrieval_ids):
            reasons.append(
                f"{qid}: summary/retrieval chunk mismatch"
            )

        for item in summaries:

            case_id = str(
                item.get("case_id", "")
            )

            chunk_id = str(
                item.get("chunk_id", "")
            )

            story = item.get(
                "story",
                "",
            )

            structured = item.get(
                "structured"
            )

            story_ok, story_reasons = (
                validate_story(
                    story,
                    case_id,
                    chunk_id,
                )
            )

            json_ok, json_reasons = (
                validate_structured_json(
                    structured,
                    chunk_id,
                )
            )

            if not story_ok:
                reasons.append(
                    f"{qid}/{chunk_id} Story: "
                    + "; ".join(story_reasons)
                )

            if not json_ok:
                reasons.append(
                    f"{qid}/{chunk_id} JSON: "
                    + "; ".join(json_reasons)
                )

    return (
        len(reasons) == 0,
        reasons,
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 72)
    print(
        "PRAMANA — STEP 12 BASELINE SUMMARIZATION"
    )
    print("=" * 72)

    print(
        f"Model                    : {MODEL}"
    )
    print(
        f"Embedding model          : {EMBEDDING_MODEL}"
    )
    print(
        "Retrieval strategy       : "
        "BASELINE / DIRECT QUERY"
    )
    print(
        "HyDE                     : DISABLED"
    )
    print(
        f"Top-K                    : {TOP_K}"
    )
    print(
        f"Story word range         : "
        f"{STORY_MIN_WORDS}–{STORY_MAX_WORDS}"
    )
    print(
        f"Max attempts             : {MAX_ATTEMPTS}"
    )
    print(
        f"Chroma collection        : "
        f"{COLLECTION_NAME}"
    )
    print(
        f"Chroma records           : "
        f"{collection.count()}"
    )

    step8_hash_before = sha256_file(
        STEP8_PATH
    )

    print(
        f"Step 8 SHA256             : "
        f"{step8_hash_before}"
    )

    step8 = load_step8()

    print(
        f"Step 8 records loaded     : "
        f"{len(step8)}"
    )

    cache = load_cache()

    print(
        f"Current valid cached cases: "
        f"{len(cache)}/{len(TEST_QUERY_IDS)}"
    )

    final_results = []

    for number, query_id in enumerate(
        TEST_QUERY_IDS,
        start=1,
    ):

        row = step8[query_id]

        citizen_query = get_value(
            row,
            "citizen_query",
            "query",
            "candidate_query",
        )

        legal_query = get_value(
            row,
            "legal_query",
        )

        domain = get_value(
            row,
            "domain",
            "classification",
        )

        print("")
        print(
            f"[{number}/{len(TEST_QUERY_IDS)}] "
            f"{query_id}"
        )

        if domain.upper() == "OUT_OF_SCOPE":
            raise RuntimeError(
                f"{query_id} is OUT_OF_SCOPE."
            )

        if not legal_query:
            raise RuntimeError(
                f"{query_id} has no Step 8 legal_query."
            )

        cached = cache.get(query_id)

        # Cache is accepted only if it is linked to the exact
        # current Step 8 legal query and contains all Top-3 summaries.
        if cached:

            if (
                str(
                    cached.get(
                        "legal_query",
                        "",
                    )
                ).strip()
                == legal_query
                and len(
                    cached.get(
                        "summaries",
                        [],
                    )
                ) == TOP_K
            ):

                valid, _ = (
                    validate_all_persisted(
                        [cached]
                    )
                )

                if valid:
                    print(
                        "    Cached valid Step 12 "
                        "result — skipped"
                    )
                    final_results.append(
                        cached
                    )
                    continue

        print(
            f"    Citizen query : "
            f"{citizen_query[:100]}"
        )
        print(
            f"    Legal query   : "
            f"{legal_query[:140]}"
        )

        # --------------------------------------------------------
        # Baseline retrieval
        # --------------------------------------------------------

        print(
            "    Frozen Baseline retrieval..."
        )

        retrieval = retrieve_top3(
            legal_query
        )

        print("    Top-3:")

        for item in retrieval:
            print(
                f"      [{item['rank']}] "
                f"{item['case_id']} / "
                f"{item['chunk_id']} "
                f"sim={item['similarity']}"
            )

        # --------------------------------------------------------
        # Story + JSON for each retrieved chunk
        # --------------------------------------------------------

        summaries = []

        for item in retrieval:

            print(
                f"    Summarizing "
                f"{item['case_id']} / "
                f"{item['chunk_id']}"
            )

            try:
                generated = (
                    generate_case_summary(item)
                )
            except ProviderRateLimitError:
                print("")
                print(
                    "    RATE LIMIT REACHED — stopping Step 12 safely."
                )
                print(
                    "    Completed query results remain persisted."
                )
                print(
                    "    Re-run later to resume from the cache."
                )
                raise

            summaries.append({
                "rank": item["rank"],
                "case_id": item["case_id"],
                "year": item["year"],
                "chunk_id": item["chunk_id"],
                "similarity": item["similarity"],
                "story": generated["story"],
                "story_word_count": (
                    generated[
                        "story_word_count"
                    ]
                ),
                "structured": (
                    generated["structured"]
                ),
                "story_validation": (
                    generated[
                        "story_validation"
                    ]
                ),
                "json_validation": (
                    generated[
                        "json_validation"
                    ]
                ),
            })

            time.sleep(1)

        result = {
            "query_id": query_id,
            "query": citizen_query,
            "legal_query": legal_query,
            "retrieval_strategy": (
                "BASELINE_DIRECT_QUERY"
            ),
            "hyde_used": False,
            "top_k": TOP_K,
            "retrieval": [
                {
                    "rank": item["rank"],
                    "case_id": item["case_id"],
                    "year": item["year"],
                    "chunk_id": item["chunk_id"],
                    "similarity": item["similarity"],
                }
                for item in retrieval
            ],
            "summaries": summaries,
            "status": "PASS",
        }

        final_results.append(result)

        save_results(final_results)

        print(
            f"    Progress saved: "
            f"{len(final_results)}/"
            f"{len(TEST_QUERY_IDS)}"
        )

    # ------------------------------------------------------------
    # Final persisted QC
    # ------------------------------------------------------------

    save_results(final_results)

    persisted_ok, persisted_reasons = (
        validate_all_persisted(
            final_results
        )
    )

    step8_hash_after = sha256_file(
        STEP8_PATH
    )

    step8_unchanged = (
        step8_hash_before
        == step8_hash_after
    )

    chroma_ok = (
        collection.count()
        == EXPECTED_CHROMA_COUNT
    )

    total_summaries = sum(
        len(
            result.get(
                "summaries",
                [],
            )
        )
        for result in final_results
    )

    story_pass = 0
    json_pass = 0
    attribution_pass = 0
    words = []

    for result in final_results:

        for item in result.get(
            "summaries",
            [],
        ):

            story = item.get(
                "story",
                "",
            )

            if item.get(
                "story_validation",
                {},
            ).get("pass"):
                story_pass += 1

            if item.get(
                "json_validation",
                {},
            ).get("pass"):
                json_pass += 1

            words.append(
                word_count(story)
            )

            expected_tag = (
                f"[Source: {item['case_id']}, "
                f"Chunk {item['chunk_id']}]"
            )

            tagged = split_tagged_sentences(
                story
            )

            if tagged and all(
                tag == expected_tag
                for _, tag in tagged
            ):
                attribution_pass += 1

    expected_summaries = (
        len(TEST_QUERY_IDS) * TOP_K
    )

    query_coverage = (
        len(final_results)
        == len(TEST_QUERY_IDS)
    )

    retrieval_count_ok = (
        total_summaries
        == expected_summaries
    )

    print("")
    print("=" * 72)
    print("STEP 12 FINAL QC")
    print("=" * 72)

    print(
        f"Test queries              : "
        f"{len(final_results)}/"
        f"{len(TEST_QUERY_IDS)}"
    )

    print(
        f"Query coverage            : "
        f"{'PASS' if query_coverage else 'FAIL'}"
    )

    print(
        f"Retrieved top-{TOP_K} summaries  : "
        f"{total_summaries}/"
        f"{expected_summaries}"
    )

    print(
        f"Retrieval/summary count   : "
        f"{'PASS' if retrieval_count_ok else 'FAIL'}"
    )

    print(
        f"Story validation          : "
        f"{story_pass}/{expected_summaries}"
    )

    print(
        f"Source attribution        : "
        f"{attribution_pass}/{expected_summaries}"
    )

    print(
        f"Structured JSON validation: "
        f"{json_pass}/{expected_summaries}"
    )

    if words:
        print(
            f"Story word count min      : "
            f"{min(words)}"
        )
        print(
            f"Story word count max      : "
            f"{max(words)}"
        )
        print(
            f"Story word count mean     : "
            f"{sum(words) / len(words):.2f}"
        )

    print(
        "HyDE used                 : NO"
    )

    print(
        f"Chroma records            : "
        f"{collection.count()}"
    )

    print(
        f"Frozen Chroma integrity   : "
        f"{'PASS' if chroma_ok else 'FAIL'}"
    )

    print(
        f"Step 8 unchanged          : "
        f"{'PASS' if step8_unchanged else 'FAIL'}"
    )

    print(
        f"Persisted output integrity: "
        f"{'PASS' if persisted_ok else 'FAIL'}"
    )

    final_pass = (
        query_coverage
        and retrieval_count_ok
        and persisted_ok
        and step8_unchanged
        and chroma_ok
        and story_pass == expected_summaries
        and json_pass == expected_summaries
        and attribution_pass == expected_summaries
    )

    print(
        f"FINAL STATUS              : "
        f"{'PASS' if final_pass else 'FAIL'}"
    )

    # ------------------------------------------------------------
    # QC report
    # ------------------------------------------------------------

    report = [
        "PRAMANA — STEP 12 QC REPORT",
        "=" * 72,
        "",
        f"Model: {MODEL}",
        f"Embedding model: {EMBEDDING_MODEL}",
        "Retrieval strategy: BASELINE / DIRECT QUERY",
        "HyDE used: NO",
        f"Top-K: {TOP_K}",
        f"Story word range: "
        f"{STORY_MIN_WORDS}–{STORY_MAX_WORDS}",
        "",
        f"Test queries: "
        f"{len(final_results)}/"
        f"{len(TEST_QUERY_IDS)}",
        f"Retrieved summaries: "
        f"{total_summaries}/"
        f"{expected_summaries}",
        f"Story validation: "
        f"{story_pass}/"
        f"{expected_summaries}",
        f"Source attribution: "
        f"{attribution_pass}/"
        f"{expected_summaries}",
        f"Structured JSON validation: "
        f"{json_pass}/"
        f"{expected_summaries}",
    ]

    if words:
        report.extend([
            f"Story word count min: {min(words)}",
            f"Story word count max: {max(words)}",
            f"Story word count mean: "
            f"{sum(words) / len(words):.2f}",
        ])

    report.extend([
        f"Chroma records: "
        f"{collection.count()}",
        f"Frozen Chroma integrity: "
        f"{'PASS' if chroma_ok else 'FAIL'}",
        f"Step 8 unchanged: "
        f"{'PASS' if step8_unchanged else 'FAIL'}",
        f"Persisted output integrity: "
        f"{'PASS' if persisted_ok else 'FAIL'}",
        "",
        f"FINAL STATUS: "
        f"{'PASS' if final_pass else 'FAIL'}",
    ])

    if persisted_reasons:
        report.extend([
            "",
            "PERSISTED VALIDATION ERRORS:",
            *persisted_reasons,
        ])

    with open(
        QC_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(report)
        )

    print("")
    print(
        f"Results saved             : "
        f"{RESULTS_PATH}"
    )
    print(
        f"QC report saved           : "
        f"{QC_PATH}"
    )

    if not final_pass:
        raise SystemExit(
            "STEP 12 FINAL STATUS: FAIL"
        )

    print("")
    print(
        "STEP 12 FINAL STATUS: PASS"
    )
    print(
        "Step 12 is ready for manual semantic review."
    )


if __name__ == "__main__":
    try:
        main()
    except ProviderRateLimitError as exc:
        print("")
        print("=" * 72)
        print("STEP 12 PAUSED — GROQ RATE LIMIT")
        print("=" * 72)
        print(str(exc))
        print(
            "Previously completed query-level results were preserved."
        )
        print(
            "Run the same script again after the quota resets; "
            "valid cached results will be skipped."
        )
        raise SystemExit(2)