from pathlib import Path
from collections import defaultdict
import json
import re
import sys
import hashlib
from datetime import datetime, timezone

import pandas as pd
import spacy
from transformers import AutoTokenizer


# ============================================================
# PRAMANA — STEP 4 V14 PRODUCTION CHUNKER
# ============================================================
#
# SOURCE:
#   Frozen V14 forensic corpus
#
# PURPOSE:
#   Convert V14 judgments into retrieval chunks.
#
# GUARANTEES:
#   - V14 is read-only
#   - all 538 V14 cases must be represented
#   - no case may silently fail
#   - exact character provenance
#   - zero gaps
#   - zero overlaps
#   - sentence/whitespace-aware boundaries
#   - hard BGE <= 512 token ceiling
#   - independent token validation
#   - exact reconstruction of every case
#   - duplicate chunk IDs forbidden
#   - output written only after all QC gates pass
#
# ============================================================


ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# FROZEN V14 INPUT
# ------------------------------------------------------------

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "ildc_domain_classification_v6"
    / "v14"
    / "ildc_multi_domain_relevant_v14_forensic.parquet"
)

# ------------------------------------------------------------
# NEW V14 DOWNSTREAM OUTPUT
# ------------------------------------------------------------

OUT = ROOT / "data" / "step4_v14"

PARQUET = OUT / "chunks.parquet"
JSONL = OUT / "chunks.jsonl"
MANIFEST = OUT / "chunk_manifest.csv"
QC = OUT / "step4_qc_report.txt"


# ------------------------------------------------------------
# MODEL / LIMITS
# ------------------------------------------------------------

MODEL = "BAAI/bge-small-en-v1.5"

LIMIT = 512

EXPECTED_CASES = 538


# ============================================================
# HELPERS
# ============================================================

def die(message):
    print()
    print("=" * 90)
    print("FATAL — STEP 4 V14 BUILD ABORTED")
    print("=" * 90)
    print(message)
    print("=" * 90)
    sys.exit(1)


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def count_tokens(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )

    return len(encoded["input_ids"])


def make_chunk_id(case_id, index):
    return f"{case_id}_C{index:04d}"


# ============================================================
# BOUNDARY CREATION
# ============================================================

def make_boundaries(text, doc):
    """
    Create exact character boundaries.

    Preferred boundaries:
        1. sentence ends
        2. whitespace ends
        3. document end

    Every returned position refers directly to the immutable
    V14 source string.
    """

    boundaries = set()

    # Sentence boundaries.
    for sentence in doc.sents:
        end = sentence.end_char

        if end > 0:
            boundaries.add(end)

    # Whitespace boundaries.
    for match in re.finditer(
        r"\s+",
        text
    ):
        boundaries.add(match.end())

    # Absolute end.
    boundaries.add(len(text))

    return sorted(boundaries)


# ============================================================
# TOKEN-SAFE END SEARCH
# ============================================================

def find_fitting_boundary(
    text,
    start,
    boundaries,
    tokenizer,
):
    """
    Find the furthest boundary whose complete chunk is
    <= 512 BGE tokens.

    IMPORTANT:
    We validate the final selected boundary independently.

    If no sentence/whitespace boundary fits, we perform an
    exact character-level fallback search.

    No chunk is ever allowed to exceed 512 tokens.
    """

    candidates = [
        position
        for position in boundaries
        if start < position <= len(text)
    ]

    if not candidates:
        die(
            f"No usable boundary after character {start}."
        )

    # --------------------------------------------------------
    # Whole remaining text.
    # --------------------------------------------------------

    if count_tokens(
        tokenizer,
        text[start:]
    ) <= LIMIT:

        return len(text)

    # --------------------------------------------------------
    # First attempt:
    # binary search over preferred boundaries.
    # --------------------------------------------------------

    low = 0
    high = len(candidates) - 1
    best = None

    while low <= high:

        middle = (
            low + high
        ) // 2

        end = candidates[middle]

        token_count = count_tokens(
            tokenizer,
            text[start:end]
        )

        if token_count <= LIMIT:

            best = end
            low = middle + 1

        else:

            high = middle - 1

    # --------------------------------------------------------
    # Validate binary-search result.
    # --------------------------------------------------------

    if best is not None:

        if count_tokens(
            tokenizer,
            text[start:best]
        ) <= LIMIT:

            return best

    # --------------------------------------------------------
    # Exact fallback.
    #
    # This handles unusual tokenizer behaviour and ensures
    # that a bad monotonicity assumption can never create an
    # invalid chunk.
    # --------------------------------------------------------

    for end in reversed(candidates):

        if end <= start:
            continue

        if count_tokens(
            tokenizer,
            text[start:end]
        ) <= LIMIT:

            return end

    # --------------------------------------------------------
    # If even the first available boundary cannot fit, find
    # an exact character position.
    # --------------------------------------------------------

    # There must be at least one character that can fit unless
    # tokenizer behaviour is pathological.
    for end in range(
        min(
            len(text),
            start + 2000
        ),
        start,
        -1
    ):

        piece = text[start:end]

        if count_tokens(
            tokenizer,
            piece
        ) <= LIMIT:

            return end

    die(
        "Unable to construct a <=512-token chunk "
        f"starting at character {start}."
    )


# ============================================================
# CASE CHUNKING
# ============================================================

def chunk_case(
    case_id,
    text,
    doc,
    tokenizer,
):
    """
    Split one immutable V14 judgment.

    Each chunk is literally:

        text[start:end]

    Therefore provenance is exact by construction.
    """

    if not text:
        die(
            f"Empty text encountered for {case_id}."
        )

    boundaries = make_boundaries(
        text,
        doc
    )

    chunks = []

    start = 0

    while start < len(text):

        end = find_fitting_boundary(
            text,
            start,
            boundaries,
            tokenizer,
        )

        if end <= start:
            die(
                f"Non-advancing boundary in {case_id}: "
                f"{start} -> {end}"
            )

        piece = text[start:end]

        token_count = count_tokens(
            tokenizer,
            piece
        )

        if token_count > LIMIT:

            die(
                f"Internal token violation in "
                f"{case_id}: {token_count}"
            )

        chunks.append(
            {
                "start": start,
                "end": end,
                "text": piece,
                "token_count": token_count,
            }
        )

        start = end

    # --------------------------------------------------------
    # Exact reconstruction.
    # --------------------------------------------------------

    reconstructed = "".join(
        chunk["text"]
        for chunk in chunks
    )

    if reconstructed != text:

        die(
            f"Exact reconstruction failed for {case_id}."
        )

    # --------------------------------------------------------
    # Boundary continuity.
    # --------------------------------------------------------

    expected = 0

    for chunk in chunks:

        if chunk["start"] != expected:

            die(
                f"Internal gap/overlap in {case_id}: "
                f"expected {expected}, "
                f"got {chunk['start']}"
            )

        expected = chunk["end"]

    if expected != len(text):

        die(
            f"Chunk coverage incomplete for {case_id}: "
            f"ended at {expected}, "
            f"expected {len(text)}"
        )

    return chunks


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("PRAMANA — STEP 4 V14 PRODUCTION CHUNKER")
    print("=" * 90)

    print()
    print("SOURCE")
    print("Frozen V14                  :", INPUT)
    print("Output directory            :", OUT)
    print("Tokenizer                   :", MODEL)
    print("Hard token ceiling          :", LIMIT)
    print("Expected cases              :", EXPECTED_CASES)

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("V14 INPUT VALIDATION")
    print("=" * 90)

    if not INPUT.exists():

        die(
            "Frozen V14 file not found:\n"
            + str(INPUT)
        )

    try:

        df = pd.read_parquet(
            INPUT
        )

    except Exception as exc:

        die(
            "Could not read V14:\n"
            + str(exc)
        )

    if len(df) != EXPECTED_CASES:

        die(
            f"Expected {EXPECTED_CASES} V14 cases, "
            f"found {len(df)}."
        )

    if "id" not in df.columns:

        die(
            "V14 does not contain 'id'."
        )

    if "text" not in df.columns:

        die(
            "V14 does not contain 'text'."
        )

    df = df.copy()

    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].astype(str)

    if df["id"].duplicated().any():

        die(
            "Duplicate case IDs in V14."
        )

    if df["text"].isna().any():

        die(
            "Null text detected in V14."
        )

    if (df["text"].str.len() == 0).any():

        die(
            "Empty text detected in V14."
        )

    input_ids = list(
        df["id"]
    )

    if len(set(input_ids)) != EXPECTED_CASES:

        die(
            "V14 case-ID uniqueness failure."
        )

    print(
        "V14 records                 :",
        len(df)
    )

    print(
        "Case IDs                    : PASS"
    )

    print(
        "Text availability            : PASS"
    )

    print(
        "Case uniqueness              : PASS"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print()
    print("=" * 90)
    print("TOKENIZER INITIALIZATION")
    print("=" * 90)

    print(
        "Loading tokenizer            :",
        MODEL
    )

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            use_fast=True,
        )

    except Exception as exc:

        die(
            "Tokenizer initialization failed:\n"
            + str(exc)
        )

    # Avoid the harmless Transformers warning when a long
    # complete judgment is inspected.
    tokenizer.model_max_length = 10**9

    print(
        "Tokenizer                   : PASS"
    )

    # ========================================================
    # SPACY
    # ========================================================

    print()
    print("=" * 90)
    print("SPACY INITIALIZATION")
    print("=" * 90)

    print(
        "Loading spaCy                : en_core_web_sm"
    )

    try:

        nlp = spacy.load(
            "en_core_web_sm"
        )

    except Exception as exc:

        die(
            "spaCy initialization failed:\n"
            + str(exc)
        )

    print(
        "spaCy pipeline               :",
        nlp.pipe_names
    )

    # ========================================================
    # OUTPUT SAFETY
    # ========================================================

    print()
    print("=" * 90)
    print("OUTPUT SAFETY")
    print("=" * 90)

    # Never overwrite an existing V14 downstream build.
    existing = [
        PARQUET,
        JSONL,
        MANIFEST,
        QC,
    ]

    existing_files = [
        path
        for path in existing
        if path.exists()
    ]

    if existing_files:

        die(
            "Existing Step 4 V14 output detected.\n"
            "Delete/rename the previous output before "
            "creating a new immutable build:\n"
            + "\n".join(
                str(path)
                for path in existing_files
            )
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CHUNKING
    # ========================================================

    print()
    print("=" * 90)
    print("CHUNKING V14")
    print("=" * 90)

    all_chunks = []

    failed_cases = []

    case_chunk_counts = {}

    case_hashes = {}

    for number, (_, row) in enumerate(
        df.iterrows(),
        1
    ):

        case_id = str(
            row["id"]
        )

        text = str(
            row["text"]
        )

        case_hashes[
            case_id
        ] = sha256_text(
            text
        )

        try:

            doc = nlp(
                text
            )

            chunks = chunk_case(
                case_id,
                text,
                doc,
                tokenizer,
            )

            case_chunk_counts[
                case_id
            ] = len(chunks)

            for index, chunk in enumerate(
                chunks
            ):

                chunk_id = make_chunk_id(
                    case_id,
                    index
                )

                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "case_id": case_id,
                        "chunk_index": index,
                        "start_char": chunk["start"],
                        "end_char": chunk["end"],
                        "token_count": chunk["token_count"],
                        "word_count": len(
                            chunk["text"].split()
                        ),
                        "text": chunk["text"],
                        "source_text_sha256": case_hashes[
                            case_id
                        ],
                    }
                )

        except Exception as exc:

            failed_cases.append(
                (
                    case_id,
                    str(exc)
                )
            )

        if (
            number % 25 == 0
            or number == EXPECTED_CASES
        ):

            print(
                f"Processed {number}/{EXPECTED_CASES} "
                f"judgments | "
                f"Chunks so far: {len(all_chunks)}"
            )

    # --------------------------------------------------------
    # FAIL CLOSED IF ANY CASE FAILED
    # --------------------------------------------------------

    if failed_cases:

        print()

        for case_id, error in failed_cases:
            print(
                f"FAILED {case_id}: {error}"
            )

        die(
            f"{len(failed_cases)} case(s) failed "
            "during chunking. "
            "No Step 4 dataset will be accepted."
        )

    # ========================================================
    # BASIC CHUNK VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("BASIC CHUNK VALIDATION")
    print("=" * 90)

    if not all_chunks:

        die(
            "Zero chunks were produced."
        )

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in all_chunks
    ]

    duplicate_chunk_ids = (
        len(chunk_ids)
        - len(set(chunk_ids))
    )

    if duplicate_chunk_ids:

        die(
            f"Duplicate chunk IDs: "
            f"{duplicate_chunk_ids}"
        )

    represented_cases = {
        chunk["case_id"]
        for chunk in all_chunks
    }

    missing_cases = (
        set(input_ids)
        - represented_cases
    )

    extra_cases = (
        represented_cases
        - set(input_ids)
    )

    if missing_cases:

        die(
            "Missing cases after chunking:\n"
            + "\n".join(
                sorted(missing_cases)
            )
        )

    if extra_cases:

        die(
            "Unexpected cases after chunking:\n"
            + "\n".join(
                sorted(extra_cases)
            )
        )

    print(
        "Total chunks               :",
        len(all_chunks)
    )

    print(
        "Cases represented          :",
        len(represented_cases)
    )

    print(
        "All 538 cases represented  : PASS"
    )

    print(
        "Duplicate chunk IDs        : PASS"
    )

    # ========================================================
    # INDEPENDENT TOKEN VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("INDEPENDENT BGE TOKEN VALIDATION")
    print("=" * 90)

    token_failures = []

    token_counts = []

    for chunk in all_chunks:

        actual = count_tokens(
            tokenizer,
            chunk["text"]
        )

        token_counts.append(
            actual
        )

        if actual != chunk["token_count"]:

            token_failures.append(
                (
                    chunk["chunk_id"],
                    "metadata mismatch",
                    chunk["token_count"],
                    actual,
                )
            )

        if actual > LIMIT:

            token_failures.append(
                (
                    chunk["chunk_id"],
                    "over 512",
                    LIMIT,
                    actual,
                )
            )

        if actual <= 0:

            token_failures.append(
                (
                    chunk["chunk_id"],
                    "zero tokens",
                    1,
                    actual,
                )
            )

    if token_failures:

        for failure in token_failures[:20]:
            print(
                "TOKEN FAILURE:",
                failure
            )

        die(
            f"{len(token_failures)} token validation failure(s)."
        )

    max_tokens = max(
        token_counts
    )

    min_tokens = min(
        token_counts
    )

    print(
        "Maximum actual tokens       :",
        max_tokens
    )

    print(
        "Chunks >512 tokens          : 0"
    )

    print(
        "Independent token validation: PASS"
    )

    # ========================================================
    # PROVENANCE VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("PROVENANCE VALIDATION")
    print("=" * 90)

    provenance_failures = []
    overlap_failures = []
    gap_failures = []
    order_failures = []
    reconstruction_failures = []

    chunks_by_case = defaultdict(list)

    for chunk in all_chunks:

        chunks_by_case[
            chunk["case_id"]
        ].append(
            chunk
        )

    for case_id in input_ids:

        source = text_by_id = str(
            df.loc[
                df["id"] == case_id,
                "text"
            ].iloc[0]
        )

        case_chunks = sorted(
            chunks_by_case.get(
                case_id,
                []
            ),
            key=lambda x: x["chunk_index"]
        )

        if not case_chunks:

            gap_failures.append(
                case_id
            )

            continue

        expected_start = 0

        reconstructed_parts = []

        for expected_index, chunk in enumerate(
            case_chunks
        ):

            start = int(
                chunk["start_char"]
            )

            end = int(
                chunk["end_char"]
            )

            if chunk["chunk_index"] != expected_index:

                order_failures.append(
                    case_id
                )

            if start < expected_start:

                overlap_failures.append(
                    case_id
                )

            elif start > expected_start:

                gap_failures.append(
                    case_id
                )

            if start < 0 or end > len(source):

                provenance_failures.append(
                    case_id
                )

            else:

                expected_text = source[
                    start:end
                ]

                if expected_text != chunk["text"]:

                    provenance_failures.append(
                        case_id
                    )

            if (
                chunk["source_text_sha256"]
                != sha256_text(source)
            ):

                provenance_failures.append(
                    case_id
                )

            reconstructed_parts.append(
                chunk["text"]
            )

            expected_start = end

        reconstructed = "".join(
            reconstructed_parts
        )

        if expected_start != len(source):

            gap_failures.append(
                case_id
            )

        if reconstructed != source:

            reconstruction_failures.append(
                case_id
            )

    # Deduplicate case-level failures.
    provenance_failures = sorted(
        set(provenance_failures)
    )

    overlap_failures = sorted(
        set(overlap_failures)
    )

    gap_failures = sorted(
        set(gap_failures)
    )

    order_failures = sorted(
        set(order_failures)
    )

    reconstruction_failures = sorted(
        set(reconstruction_failures)
    )

    print(
        "Provenance failures          :",
        len(provenance_failures)
    )

    print(
        "Overlap failures             :",
        len(overlap_failures)
    )

    print(
        "Gap failures                 :",
        len(gap_failures)
    )

    print(
        "Order failures               :",
        len(order_failures)
    )

    print(
        "Reconstruction failures      :",
        len(reconstruction_failures)
    )

    if (
        provenance_failures
        or overlap_failures
        or gap_failures
        or order_failures
        or reconstruction_failures
    ):

        die(
            "Provenance/structure validation failed."
        )

    print(
        "Exact provenance             : PASS"
    )

    print(
        "No overlaps                  : PASS"
    )

    print(
        "No gaps                      : PASS"
    )

    print(
        "Sentence/chunk order         : PASS"
    )

    print(
        "Exact reconstruction         : PASS"
    )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    word_counts = [
        chunk["word_count"]
        for chunk in all_chunks
    ]

    token_counts = [
        chunk["token_count"]
        for chunk in all_chunks
    ]

    words = pd.Series(
        word_counts
    )

    tokens = pd.Series(
        token_counts
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest_rows = []

    for vector_id, chunk in enumerate(
        all_chunks
    ):

        manifest_rows.append(
            {
                "vector_id": vector_id,
                "chunk_id": chunk["chunk_id"],
                "case_id": chunk["case_id"],
                "chunk_index": chunk["chunk_index"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "token_count": chunk["token_count"],
                "word_count": chunk["word_count"],
                "source_text_sha256": chunk[
                    "source_text_sha256"
                ],
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    # ========================================================
    # FINAL QC GATES
    # ========================================================

    gates = {
        "input_records_538": (
            len(df) == EXPECTED_CASES
        ),
        "all_judgments_represented": (
            len(represented_cases)
            == EXPECTED_CASES
        ),
        "no_missing_cases": (
            len(missing_cases) == 0
        ),
        "no_extra_cases": (
            len(extra_cases) == 0
        ),
        "no_failed_cases": (
            len(failed_cases) == 0
        ),
        "no_duplicate_chunk_ids": (
            duplicate_chunk_ids == 0
        ),
        "all_chunks_le_512_tokens": (
            max_tokens <= LIMIT
        ),
        "independent_token_validation": (
            len(token_failures) == 0
        ),
        "exact_provenance": (
            len(provenance_failures) == 0
        ),
        "no_overlap": (
            len(overlap_failures) == 0
        ),
        "no_gaps": (
            len(gap_failures) == 0
        ),
        "sentence_order_valid": (
            len(order_failures) == 0
        ),
        "exact_reconstruction": (
            len(reconstruction_failures) == 0
        ),
    }

    final_pass = all(
        gates.values()
    )

    # ========================================================
    # WRITE ONLY AFTER ALL CORE QC PASSES
    # ========================================================

    if not final_pass:

        die(
            "Pre-write QC failed. "
            "No Step 4 V14 artifacts will be written."
        )

    print()
    print("=" * 90)
    print("WRITING STEP 4 V14")
    print("=" * 90)

    chunk_df = pd.DataFrame(
        all_chunks
    )

    # Remove internal helper column from JSON/parquet if desired
    # only after validation; retain provenance hash explicitly.
    chunk_df.to_parquet(
        PARQUET,
        index=False,
    )

    with open(
        JSONL,
        "w",
        encoding="utf-8"
    ) as handle:

        for chunk in all_chunks:

            handle.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest.to_csv(
        MANIFEST,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # RELOAD VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("RELOAD VALIDATION")
    print("=" * 90)

    try:

        reloaded = pd.read_parquet(
            PARQUET
        )

    except Exception as exc:

        die(
            "Could not reload chunks.parquet:\n"
            + str(exc)
        )

    if len(reloaded) != len(all_chunks):

        die(
            "Reloaded Parquet row count mismatch."
        )

    if list(
        reloaded["chunk_id"]
    ) != chunk_ids:

        die(
            "Reloaded chunk ID order mismatch."
        )

    if list(
        reloaded["case_id"]
    ) != [
        chunk["case_id"]
        for chunk in all_chunks
    ]:

        die(
            "Reloaded case order mismatch."
        )

    print(
        "Parquet reload              : PASS"
    )

    print(
        "Chunk order                 : PASS"
    )

    print(
        "Case order                  : PASS"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    report = [
        "=" * 90,
        "PRAMANA — STEP 4 FINAL QC (V14)",
        "=" * 90,
        "",
        "BUILD UTC                   : "
        + datetime.now(
            timezone.utc
        ).isoformat(),
        "",
        "SOURCE",
        "-" * 90,
        f"V14 source                  : {INPUT}",
        f"V14 records                 : {len(df)}",
        "",
        "OUTPUT",
        "-" * 90,
        f"Output chunks               : {len(all_chunks)}",
        f"Judgments represented      : {len(represented_cases)}",
        f"Failed judgments            : {len(failed_cases)}",
        "",
        "TOKEN CONTROL",
        "-" * 90,
        f"Tokenizer                   : {MODEL}",
        f"Hard token ceiling          : {LIMIT}",
        f"Minimum actual tokens       : {min_tokens}",
        f"Median tokens               : {tokens.median():.1f}",
        f"Mean tokens                 : {tokens.mean():.1f}",
        f"Maximum actual tokens       : {max_tokens}",
        f"Chunks >512 tokens          : {len(token_failures)}",
        "",
        "WORD DISTRIBUTION",
        "-" * 90,
        f"Minimum words               : {int(words.min())}",
        f"Median words                : {words.median():.1f}",
        f"Mean words                  : {words.mean():.1f}",
        f"Maximum words               : {int(words.max())}",
        "",
        "PROVENANCE / STRUCTURE",
        "-" * 90,
        f"Provenance failures         : {len(provenance_failures)}",
        f"Overlap failures            : {len(overlap_failures)}",
        f"Gap failures                : {len(gap_failures)}",
        f"Order failures              : {len(order_failures)}",
        f"Reconstruction failures     : {len(reconstruction_failures)}",
        f"Duplicate chunk IDs         : {duplicate_chunk_ids}",
        "",
        "FINAL QC GATE",
        "-" * 90,
    ]

    for name, passed in gates.items():

        report.append(
            f"{name:35s}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    report.extend(
        [
            "",
            f"parquet_reload               : PASS",
            f"chunk_order_reload           : PASS",
            f"case_order_reload            : PASS",
            "",
            f"FINAL STATUS                 : "
            f"{'PASS' if final_pass else 'FAIL'}",
            "",
            "=" * 90,
            "OUTPUT FILES",
            "=" * 90,
            f"Parquet                     : {PARQUET}",
            f"JSONL                       : {JSONL}",
            f"Manifest                    : {MANIFEST}",
            f"QC report                   : {QC}",
            "=" * 90,
        ]
    )

    QC.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print(
        "\n".join(report)
    )

    if not final_pass:

        sys.exit(1)


if __name__ == "__main__":
    main()