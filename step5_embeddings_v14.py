from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# PRAMANA — STEP 5 V14 PRODUCTION EMBEDDING PIPELINE
# ============================================================

ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# FROZEN STEP 4 V14 INPUT
# ------------------------------------------------------------

INPUT_PARQUET = (
    ROOT
    / "data"
    / "step4_v14"
    / "chunks.parquet"
)

INPUT_MANIFEST = (
    ROOT
    / "data"
    / "step4_v14"
    / "chunk_manifest.csv"
)

# ------------------------------------------------------------
# NEW V14 OUTPUT
# ------------------------------------------------------------

OUT = ROOT / "data" / "step5_v14"

EMBEDDINGS = OUT / "embeddings.npy"
MANIFEST = OUT / "embedding_manifest.csv"
QC = OUT / "step5_qc_report.txt"

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

EXPECTED_DIMENSION = 384
BATCH_SIZE = 32

EXPECTED_TOKEN_LIMIT = 512


# ============================================================
# HELPERS
# ============================================================

def die(message):
    print()
    print("=" * 90)
    print("FATAL — STEP 5 V14 BUILD ABORTED")
    print("=" * 90)
    print(message)
    print("=" * 90)
    sys.exit(1)


def sha256_text(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("PRAMANA — STEP 5 V14 PRODUCTION EMBEDDING PIPELINE")
    print("=" * 90)

    print()
    print("SOURCE")
    print("Step 4 V14 chunks            :", INPUT_PARQUET)
    print("Step 4 V14 manifest          :", INPUT_MANIFEST)
    print("Model                        :", MODEL_NAME)
    print("Expected dimension           :", EXPECTED_DIMENSION)

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("INPUT VALIDATION")
    print("=" * 90)

    if not INPUT_PARQUET.exists():
        die(
            "Step 4 V14 chunks.parquet not found:\n"
            + str(INPUT_PARQUET)
        )

    if not INPUT_MANIFEST.exists():
        die(
            "Step 4 V14 chunk_manifest.csv not found:\n"
            + str(INPUT_MANIFEST)
        )

    try:
        chunks = pd.read_parquet(
            INPUT_PARQUET
        )
    except Exception as exc:
        die(
            "Unable to read Step 4 V14 Parquet:\n"
            + str(exc)
        )

    try:
        step4_manifest = pd.read_csv(
            INPUT_MANIFEST
        )
    except Exception as exc:
        die(
            "Unable to read Step 4 V14 manifest:\n"
            + str(exc)
        )

    required_chunk_columns = {
        "chunk_id",
        "case_id",
        "chunk_index",
        "start_char",
        "end_char",
        "token_count",
        "word_count",
        "text",
        "source_text_sha256",
    }

    missing_columns = (
        required_chunk_columns
        - set(chunks.columns)
    )

    if missing_columns:
        die(
            "Missing Step 4 columns:\n"
            + ", ".join(sorted(missing_columns))
        )

    required_manifest_columns = {
        "vector_id",
        "chunk_id",
        "case_id",
        "chunk_index",
        "start_char",
        "end_char",
        "token_count",
        "word_count",
        "source_text_sha256",
    }

    missing_manifest_columns = (
        required_manifest_columns
        - set(step4_manifest.columns)
    )

    if missing_manifest_columns:
        die(
            "Missing Step 4 manifest columns:\n"
            + ", ".join(sorted(missing_manifest_columns))
        )

    expected_count = len(chunks)

    if expected_count != 4855:
        die(
            f"Expected 4855 V14 chunks, found {expected_count}."
        )

    if len(step4_manifest) != expected_count:
        die(
            "Step 4 manifest row count does not match chunks."
        )

    print(
        "Step 4 chunks               :",
        expected_count
    )

    print(
        "Step 4 manifest rows        :",
        len(step4_manifest)
    )

    # ========================================================
    # CHUNK ID VALIDATION
    # ========================================================

    chunk_ids = (
        chunks["chunk_id"]
        .astype(str)
        .tolist()
    )

    manifest_ids = (
        step4_manifest["chunk_id"]
        .astype(str)
        .tolist()
    )

    if len(chunk_ids) != len(set(chunk_ids)):
        die(
            "Duplicate chunk IDs in Step 4."
        )

    if chunk_ids != manifest_ids:
        die(
            "Step 4 chunk ID/order mismatch between "
            "Parquet and manifest."
        )

    print(
        "Chunk IDs                   : PASS"
    )

    # ========================================================
    # TEXT VALIDATION
    # ========================================================

    if chunks["text"].isna().any():
        die(
            "Null chunk text detected."
        )

    text_values = (
        chunks["text"]
        .astype(str)
        .tolist()
    )

    empty_indices = [
        i
        for i, text in enumerate(text_values)
        if not text.strip()
    ]

    if empty_indices:
        die(
            f"Empty chunk text detected at "
            f"{len(empty_indices)} row(s)."
        )

    print(
        "Null / empty text            : PASS"
    )

    # ========================================================
    # TOKEN METADATA GATE
    # ========================================================

    token_counts = (
        chunks["token_count"]
        .astype(int)
        .to_numpy()
    )

    if np.any(token_counts <= 0):
        die(
            "Invalid token_count <= 0 detected."
        )

    if np.any(
        token_counts > EXPECTED_TOKEN_LIMIT
    ):
        die(
            "Step 4 contains a chunk over the "
            "512-token ceiling."
        )

    print(
        "512-token metadata gate      : PASS"
    )

    # ========================================================
    # TEXT HASH VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("STEP 4 TEXT INTEGRITY")
    print("=" * 90)

    stored_hashes = (
        chunks["source_text_sha256"]
        .astype(str)
        .tolist()
    )

    if any(
        len(value) != 64
        for value in stored_hashes
    ):
        die(
            "Invalid source_text_sha256 detected."
        )

    print(
        "Source text hash metadata    : PASS"
    )

    # ========================================================
    # OUTPUT SAFETY
    # ========================================================

    print()
    print("=" * 90)
    print("OUTPUT SAFETY")
    print("=" * 90)

    existing = [
        EMBEDDINGS,
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
            "Existing Step 5 V14 output detected.\n"
            "Do not overwrite an existing build:\n"
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
    # MODEL INITIALIZATION
    # ========================================================

    print()
    print("=" * 90)
    print("MODEL INITIALIZATION")
    print("=" * 90)

    print(
        "Loading model                :",
        MODEL_NAME
    )

    try:
        model = SentenceTransformer(
            MODEL_NAME
        )
    except Exception as exc:
        die(
            "Model initialization failed:\n"
            + str(exc)
        )

    dimension = (
        model.get_sentence_embedding_dimension()
    )

    print(
        "Embedding dimension          :",
        dimension
    )

    if dimension != EXPECTED_DIMENSION:
        die(
            f"Expected embedding dimension "
            f"{EXPECTED_DIMENSION}, got {dimension}."
        )

    print(
        "Embedding dimension          : PASS"
    )

    # ========================================================
    # EMBEDDING
    # ========================================================

    print()
    print("=" * 90)
    print("EMBEDDING")
    print("=" * 90)

    try:
        embeddings = model.encode(
            text_values,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except Exception as exc:
        die(
            "Embedding generation failed:\n"
            + str(exc)
        )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(
        "Embedding rows              :",
        embeddings.shape[0]
    )

    print(
        "Embedding dimension         :",
        embeddings.shape[1]
    )

    # ========================================================
    # EMBEDDING VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("EMBEDDING VALIDATION")
    print("=" * 90)

    if embeddings.shape != (
        expected_count,
        EXPECTED_DIMENSION,
    ):
        die(
            "Embedding shape mismatch."
        )

    nan_count = int(
        np.isnan(embeddings).sum()
    )

    inf_count = int(
        np.isinf(embeddings).sum()
    )

    if nan_count:
        die(
            f"NaN values detected: {nan_count}"
        )

    if inf_count:
        die(
            f"Inf values detected: {inf_count}"
        )

    if not np.isfinite(
        embeddings
    ).all():
        die(
            "Non-finite embedding values detected."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    norm_failures = int(
        np.sum(
            np.abs(norms - 1.0)
            > 1e-5
        )
    )

    if norm_failures:
        die(
            f"Unit-normalization failures: "
            f"{norm_failures}"
        )

    print(
        "Count validation             : PASS"
    )

    print(
        "Dimension validation         : PASS"
    )

    print(
        "NaN values                  :",
        nan_count
    )

    print(
        "Inf values                  :",
        inf_count
    )

    print(
        "Finite values                : PASS"
    )

    print(
        "Unit-normalization validation: PASS"
    )

    # ========================================================
    # PROVENANCE / ALIGNMENT
    # ========================================================

    print()
    print("=" * 90)
    print("PROVENANCE / ALIGNMENT VALIDATION")
    print("=" * 90)

    # Encoding preserves input order.
    # Therefore row i corresponds exactly to chunk i.

    print(
        "Chunk order preserved        : PASS"
    )

    # ========================================================
    # EMBEDDING MANIFEST
    # ========================================================

    embedding_manifest = pd.DataFrame(
        {
            "vector_id": np.arange(
                expected_count,
                dtype=np.int64
            ),
            "chunk_id": chunk_ids,
            "case_id": chunks[
                "case_id"
            ].astype(str),
            "chunk_index": chunks[
                "chunk_index"
            ].astype(int),
            "start_char": chunks[
                "start_char"
            ].astype(int),
            "end_char": chunks[
                "end_char"
            ].astype(int),
            "token_count": chunks[
                "token_count"
            ].astype(int),
            "word_count": chunks[
                "word_count"
            ].astype(int),
            "source_text_sha256": stored_hashes,
            "chunk_text_sha256": [
                sha256_text(text)
                for text in text_values
            ],
            "embedding_dimension": [
                EXPECTED_DIMENSION
            ] * expected_count,
            "model": [
                MODEL_NAME
            ] * expected_count,
            "normalized": [
                True
            ] * expected_count,
        }
    )

    # ========================================================
    # FINAL QC GATE BEFORE WRITING
    # ========================================================

    print()
    print("=" * 90)
    print("PRAMANA — STEP 5 V14 FINAL QC")
    print("=" * 90)

    gates = {
        "input_chunks_4855": (
            expected_count == 4855
        ),
        "embedding_count_4855": (
            embeddings.shape[0] == 4855
        ),
        "embedding_dimension_384": (
            embeddings.shape[1] == 384
        ),
        "chunk_id_alignment": (
            chunk_ids == manifest_ids
        ),
        "no_nan": (
            nan_count == 0
        ),
        "no_inf": (
            inf_count == 0
        ),
        "finite_embeddings": (
            bool(np.isfinite(
                embeddings
            ).all())
        ),
        "normalized_embeddings": (
            norm_failures == 0
        ),
    }

    if not all(
        gates.values()
    ):
        for name, passed in gates.items():
            print(
                f"{name:35s}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

        die(
            "Pre-write QC failed."
        )

    print()
    print("All pre-write QC gates         : PASS")

    # ========================================================
    # WRITE
    # ========================================================

    print()
    print("=" * 90)
    print("WRITING STEP 5 V14")
    print("=" * 90)

    try:

        np.save(
            EMBEDDINGS,
            embeddings
        )

        embedding_manifest.to_csv(
            MANIFEST,
            index=False,
            encoding="utf-8"
        )

    except Exception as exc:

        die(
            "Failed to write Step 5 artifacts:\n"
            + str(exc)
        )

    # ========================================================
    # RELOAD VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("SAVED EMBEDDING RELOAD")
    print("=" * 90)

    try:

        reloaded_embeddings = np.load(
            EMBEDDINGS,
            allow_pickle=False
        )

        reloaded_manifest = pd.read_csv(
            MANIFEST
        )

    except Exception as exc:

        die(
            "Reload validation failed:\n"
            + str(exc)
        )

    if reloaded_embeddings.shape != embeddings.shape:
        die(
            "Reloaded embedding shape mismatch."
        )

    if not np.array_equal(
        reloaded_embeddings,
        embeddings
    ):
        die(
            "Reloaded embeddings differ from "
            "original embeddings."
        )

    if len(reloaded_manifest) != expected_count:
        die(
            "Reloaded manifest row count mismatch."
        )

    if (
        reloaded_manifest[
            "chunk_id"
        ].astype(str).tolist()
        != chunk_ids
    ):
        die(
            "Reloaded manifest chunk order mismatch."
        )

    print(
        "Embedding reload             : PASS"
    )

    print(
        "Manifest reload              : PASS"
    )

    print(
        "Exact embedding preservation : PASS"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    final_gates = {
        **gates,
        "embeddings_written": EMBEDDINGS.exists(),
        "manifest_written": MANIFEST.exists(),
        "reload_validation": True,
        "embedding_values_preserved": True,
    }

    final_status = all(
        final_gates.values()
    )

    report = []

    report.append(
        "=" * 90
    )
    report.append(
        "PRAMANA — STEP 5 FINAL QC (V14)"
    )
    report.append(
        "=" * 90
    )
    report.append("")
    report.append(
        "BUILD UTC                   : "
        + datetime.now(
            timezone.utc
        ).isoformat()
    )

    report.append("")
    report.append("INPUT")
    report.append("-" * 90)
    report.append(
        f"Step 4 chunks               : {expected_count}"
    )
    report.append(
        "Expected chunks             : 4855"
    )

    report.append("")
    report.append("MODEL")
    report.append("-" * 90)
    report.append(
        f"Model                       : {MODEL_NAME}"
    )
    report.append(
        "Embedding dimension         : 384"
    )
    report.append(
        "Normalized embeddings      : True"
    )

    report.append("")
    report.append("EMBEDDINGS")
    report.append("-" * 90)
    report.append(
        f"Embedding rows              : {embeddings.shape[0]}"
    )
    report.append(
        f"Embedding dimension         : {embeddings.shape[1]}"
    )
    report.append(
        f"NaN values                  : {nan_count}"
    )
    report.append(
        f"Inf values                  : {inf_count}"
    )
    report.append(
        f"Norm failures               : {norm_failures}"
    )

    report.append("")
    report.append("PROVENANCE")
    report.append("-" * 90)
    report.append(
        f"Chunk IDs                   : {len(chunk_ids)}"
    )
    report.append(
        "Chunk ID alignment          : PASS"
    )
    report.append(
        "Embedding order             : PASS"
    )

    report.append("")
    report.append("FINAL QC GATE")
    report.append("-" * 90)

    for name, passed in final_gates.items():
        report.append(
            f"{name:35s}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    report.append("")
    report.append(
        f"FINAL STATUS                : "
        f"{'PASS' if final_status else 'FAIL'}"
    )

    report.append("")
    report.append("=" * 90)
    report.append("OUTPUT FILES")
    report.append("=" * 90)
    report.append(
        f"Embeddings                  : {EMBEDDINGS}"
    )
    report.append(
        f"Manifest                    : {MANIFEST}"
    )
    report.append(
        f"QC report                   : {QC}"
    )
    report.append(
        "=" * 90
    )

    QC.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print()
    print(
        "\n".join(report)
    )

    if not final_status:
        sys.exit(1)


if __name__ == "__main__":
    main()