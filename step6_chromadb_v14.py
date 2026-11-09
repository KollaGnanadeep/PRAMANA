from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PRAMANA — STEP 6 V14 PRODUCTION CHROMADB BUILD
# ============================================================
#
# FROZEN INPUTS:
#   Step 4 V14
#   Step 5 V14
#
# OUTPUT:
#   data/step6_v14/chroma_db
#
# FAIL-CLOSED:
#   Any integrity failure aborts the build.
#
# NO modification of V14 / Step 4 V14 / Step 5 V14.
# ============================================================


ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

STEP4_PARQUET = (
    ROOT
    / "data"
    / "step4_v14"
    / "chunks.parquet"
)

STEP4_MANIFEST = (
    ROOT
    / "data"
    / "step4_v14"
    / "chunk_manifest.csv"
)

STEP5_EMBEDDINGS = (
    ROOT
    / "data"
    / "step5_v14"
    / "embeddings.npy"
)

STEP5_MANIFEST = (
    ROOT
    / "data"
    / "step5_v14"
    / "embedding_manifest.csv"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUT = ROOT / "data" / "step6_v14"

CHROMA_DIR = OUT / "chroma_db"
MANIFEST_OUT = OUT / "step6_manifest.csv"
QC_OUT = OUT / "step6_qc_report.txt"

COLLECTION_NAME = "pramana_v14"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

EXPECTED_DIMENSION = 384
EXPECTED_CHUNKS = 4855

BATCH_SIZE = 256


# ============================================================
# HELPERS
# ============================================================

def die(message):
    print()
    print("=" * 90)
    print("FATAL — STEP 6 V14 BUILD ABORTED")
    print("=" * 90)
    print(message)
    print("=" * 90)
    sys.exit(1)


def sha256_text(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def metadata_value(value):
    """
    Chroma metadata supports primitive scalar values.
    """

    if pd.isna(value):
        return ""

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return str(value)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("PRAMANA — STEP 6 V14 PRODUCTION CHROMADB BUILD")
    print("=" * 90)

    print()
    print("INPUTS")
    print("Step 4 V14 chunks            :", STEP4_PARQUET)
    print("Step 4 V14 manifest          :", STEP4_MANIFEST)
    print("Step 5 V14 embeddings        :", STEP5_EMBEDDINGS)
    print("Step 5 V14 manifest          :", STEP5_MANIFEST)

    print()
    print("OUTPUT")
    print("ChromaDB                     :", CHROMA_DIR)
    print("Collection                   :", COLLECTION_NAME)

    # ========================================================
    # INPUT EXISTENCE
    # ========================================================

    print()
    print("=" * 90)
    print("INPUT VALIDATION")
    print("=" * 90)

    required_files = [
        STEP4_PARQUET,
        STEP4_MANIFEST,
        STEP5_EMBEDDINGS,
        STEP5_MANIFEST,
    ]

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:

        die(
            "Required input file(s) missing:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    # ========================================================
    # LOAD STEP 4
    # ========================================================

    try:

        chunks = pd.read_parquet(
            STEP4_PARQUET
        )

        step4_manifest = pd.read_csv(
            STEP4_MANIFEST
        )

    except Exception as exc:

        die(
            "Failed to load Step 4 V14 artifacts:\n"
            + str(exc)
        )

    # ========================================================
    # LOAD STEP 5
    # ========================================================

    try:

        embeddings = np.load(
            STEP5_EMBEDDINGS,
            allow_pickle=False
        )

        step5_manifest = pd.read_csv(
            STEP5_MANIFEST
        )

    except Exception as exc:

        die(
            "Failed to load Step 5 V14 artifacts:\n"
            + str(exc)
        )

    # ========================================================
    # BASIC COUNTS
    # ========================================================

    if len(chunks) != EXPECTED_CHUNKS:

        die(
            f"Expected {EXPECTED_CHUNKS} Step 4 chunks, "
            f"found {len(chunks)}."
        )

    if len(step4_manifest) != EXPECTED_CHUNKS:

        die(
            "Step 4 manifest count mismatch."
        )

    if embeddings.shape != (
        EXPECTED_CHUNKS,
        EXPECTED_DIMENSION,
    ):

        die(
            "Step 5 embedding shape mismatch: "
            f"{embeddings.shape}"
        )

    if len(step5_manifest) != EXPECTED_CHUNKS:

        die(
            "Step 5 manifest count mismatch."
        )

    print(
        "Step 4 chunks               :",
        len(chunks)
    )

    print(
        "Step 5 embeddings           :",
        embeddings.shape
    )

    print(
        "Step 4 manifest rows        :",
        len(step4_manifest)
    )

    print(
        "Step 5 manifest rows        :",
        len(step5_manifest)
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

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

    missing = (
        required_chunk_columns
        - set(chunks.columns)
    )

    if missing:

        die(
            "Missing Step 4 columns:\n"
            + ", ".join(sorted(missing))
        )

    required_step4_manifest_columns = {
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

    missing = (
        required_step4_manifest_columns
        - set(step4_manifest.columns)
    )

    if missing:

        die(
            "Missing Step 4 manifest columns:\n"
            + ", ".join(sorted(missing))
        )

    required_step5_manifest_columns = {
        "vector_id",
        "chunk_id",
        "case_id",
        "chunk_index",
        "start_char",
        "end_char",
        "token_count",
        "word_count",
        "source_text_sha256",
        "chunk_text_sha256",
        "embedding_dimension",
        "model",
        "normalized",
    }

    missing = (
        required_step5_manifest_columns
        - set(step5_manifest.columns)
    )

    if missing:

        die(
            "Missing Step 5 manifest columns:\n"
            + ", ".join(sorted(missing))
        )

    # ========================================================
    # STEP 4 ↔ STEP 5 ALIGNMENT
    # ========================================================

    print()
    print("=" * 90)
    print("STEP 4 ↔ STEP 5 ALIGNMENT")
    print("=" * 90)

    step4_ids = (
        chunks["chunk_id"]
        .astype(str)
        .tolist()
    )

    step4_manifest_ids = (
        step4_manifest["chunk_id"]
        .astype(str)
        .tolist()
    )

    step5_ids = (
        step5_manifest["chunk_id"]
        .astype(str)
        .tolist()
    )

    if len(step4_ids) != len(set(step4_ids)):

        die(
            "Duplicate chunk IDs in Step 4."
        )

    if len(step5_ids) != len(set(step5_ids)):

        die(
            "Duplicate chunk IDs in Step 5."
        )

    if step4_ids != step4_manifest_ids:

        die(
            "Step 4 Parquet ↔ manifest ID/order mismatch."
        )

    if step4_ids != step5_ids:

        die(
            "Step 4 ↔ Step 5 chunk ID/order mismatch."
        )

    # Check row-level provenance fields.
    comparison_columns = [
        "case_id",
        "chunk_index",
        "start_char",
        "end_char",
        "token_count",
        "word_count",
        "source_text_sha256",
    ]

    for column in comparison_columns:

        left = (
            chunks[column]
            .astype(str)
            .tolist()
        )

        right = (
            step5_manifest[column]
            .astype(str)
            .tolist()
        )

        if left != right:

            die(
                f"Step 4 ↔ Step 5 mismatch in "
                f"column: {column}"
            )

    print(
        "Chunk ID/order alignment     : PASS"
    )

    print(
        "Provenance metadata alignment: PASS"
    )

    # ========================================================
    # TEXT / HASH VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("TEXT PROVENANCE")
    print("=" * 90)

    text_values = (
        chunks["text"]
        .astype(str)
        .tolist()
    )

    chunk_hashes = [
        sha256_text(text)
        for text in text_values
    ]

    manifest_chunk_hashes = (
        step5_manifest[
            "chunk_text_sha256"
        ]
        .astype(str)
        .tolist()
    )

    if chunk_hashes != manifest_chunk_hashes:

        die(
            "Step 4 text ↔ Step 5 chunk-text hash mismatch."
        )

    source_hashes = (
        chunks["source_text_sha256"]
        .astype(str)
        .tolist()
    )

    manifest_source_hashes = (
        step5_manifest[
            "source_text_sha256"
        ]
        .astype(str)
        .tolist()
    )

    if source_hashes != manifest_source_hashes:

        die(
            "Source text hash mismatch."
        )

    print(
        "Step 4 text ↔ Step 5 hash   : PASS"
    )

    print(
        "Source provenance hashes    : PASS"
    )

    # ========================================================
    # EMBEDDING SANITY
    # ========================================================

    print()
    print("=" * 90)
    print("EMBEDDING SANITY")
    print("=" * 90)

    if not np.isfinite(
        embeddings
    ).all():

        die(
            "NaN or Inf values detected."
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
        "NaN / Inf                   : PASS"
    )

    print(
        "Unit normalization          : PASS"
    )

    # ========================================================
    # MODEL METADATA VALIDATION
    # ========================================================

    model_values = (
        step5_manifest["model"]
        .astype(str)
        .unique()
        .tolist()
    )

    if model_values != [MODEL_NAME]:

        die(
            "Unexpected embedding model metadata:\n"
            + repr(model_values)
        )

    dimensions = (
        step5_manifest[
            "embedding_dimension"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    if dimensions != [EXPECTED_DIMENSION]:

        die(
            "Unexpected embedding dimensions:\n"
            + repr(dimensions)
        )

    normalized_values = (
        step5_manifest[
            "normalized"
        ]
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    if normalized_values != ["true"]:

        die(
            "Step 5 embeddings are not marked normalized."
        )

    print(
        "Model metadata              : PASS"
    )

    print(
        "Embedding dimension metadata: PASS"
    )

    print(
        "Normalization metadata      : PASS"
    )

    # ========================================================
    # OUTPUT SAFETY
    # ========================================================

    print()
    print("=" * 90)
    print("CHROMADB INITIALIZATION")
    print("=" * 90)

    if CHROMA_DIR.exists():

        die(
            "Existing Step 6 V14 ChromaDB detected:\n"
            + str(CHROMA_DIR)
            + "\n\n"
            "Refusing to overwrite an existing build."
        )

    if MANIFEST_OUT.exists():

        die(
            "Existing Step 6 V14 manifest detected."
        )

    if QC_OUT.exists():

        die(
            "Existing Step 6 V14 QC report detected."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CHROMADB
    # ========================================================

    try:

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "model": MODEL_NAME,
                "embedding_dimension": EXPECTED_DIMENSION,
                "normalized": True,
                "source": "PRAMANA_V14",
            },
        )

    except Exception as exc:

        die(
            "ChromaDB initialization failed:\n"
            + str(exc)
        )

    print(
        "Collection                  :",
        COLLECTION_NAME
    )

    # ========================================================
    # INSERT
    # ========================================================

    print()
    print("=" * 90)
    print("CHROMADB INSERT")
    print("=" * 90)

    ids = step4_ids

    documents = text_values

    metadatas = []

    for index in range(
        EXPECTED_CHUNKS
    ):

        row = chunks.iloc[index]

        metadata = {
            "vector_id": int(index),
            "case_id": str(row["case_id"]),
            "chunk_index": int(
                row["chunk_index"]
            ),
            "start_char": int(
                row["start_char"]
            ),
            "end_char": int(
                row["end_char"]
            ),
            "token_count": int(
                row["token_count"]
            ),
            "word_count": int(
                row["word_count"]
            ),
            "source_text_sha256": str(
                row["source_text_sha256"]
            ),
            "chunk_text_sha256": sha256_text(
                row["text"]
            ),
            "model": MODEL_NAME,
            "embedding_dimension": EXPECTED_DIMENSION,
            "normalized": True,
        }

        metadatas.append(
            metadata
        )

    for start in range(
        0,
        EXPECTED_CHUNKS,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            EXPECTED_CHUNKS
        )

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[
                start:end
            ].tolist(),
            metadatas=metadatas[
                start:end
            ],
        )

        print(
            f"Inserted {end}/{EXPECTED_CHUNKS}"
        )

    # ========================================================
    # COLLECTION VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("COLLECTION VALIDATION")
    print("=" * 90)

    count = collection.count()

    if count != EXPECTED_CHUNKS:

        die(
            f"Chroma record count mismatch: "
            f"{count} != {EXPECTED_CHUNKS}"
        )

    print(
        "Chroma records              :",
        count
    )

    # Retrieve all IDs/documents/metadata.
    stored = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    stored_ids = [
        str(value)
        for value in stored["ids"]
    ]

    if set(stored_ids) != set(ids):

        die(
            "Chroma ID set does not exactly match Step 4."
        )

    if len(stored_ids) != len(set(stored_ids)):

        die(
            "Duplicate IDs detected in Chroma."
        )

    print(
        "Full ID set                 : PASS"
    )

    # --------------------------------------------------------
    # Validate each stored record by ID.
    # This avoids relying on Chroma return ordering.
    # --------------------------------------------------------

    lookup = collection.get(
        ids=ids,
        include=[
            "documents",
            "metadatas",
        ]
    )

    lookup_by_id = {
        str(lookup["ids"][i]): (
            lookup["documents"][i],
            lookup["metadatas"][i],
        )
        for i in range(
            len(lookup["ids"])
        )
    }

    if set(lookup_by_id) != set(ids):

        die(
            "Exact Chroma lookup ID set mismatch."
        )

    document_failures = 0
    metadata_failures = 0

    for index, chunk_id in enumerate(ids):

        document = lookup_by_id[
            chunk_id
        ][0]

        metadata = lookup_by_id[
            chunk_id
        ][1]

        if document != documents[index]:

            document_failures += 1

        expected_metadata = metadatas[
            index
        ]

        for key, expected_value in (
            expected_metadata.items()
        ):

            actual_value = metadata.get(
                key
            )

            if str(actual_value) != str(
                expected_value
            ):

                metadata_failures += 1
                break

    if document_failures:

        die(
            f"Exact document failures: "
            f"{document_failures}"
        )

    if metadata_failures:

        die(
            f"Exact metadata failures: "
            f"{metadata_failures}"
        )

    print(
        "Exact documents             : PASS"
    )

    print(
        "Exact metadata              : PASS"
    )

    # ========================================================
    # STORED EMBEDDING VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("STORED EMBEDDING VALIDATION")
    print("=" * 90)

    # Query Chroma for every ID in manageable batches.
    embedding_max_difference = 0.0

    stored_embedding_failures = 0

    for start in range(
        0,
        EXPECTED_CHUNKS,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            EXPECTED_CHUNKS
        )

        batch_ids = ids[
            start:end
        ]

        result = collection.get(
            ids=batch_ids,
            include=[
                "embeddings"
            ]
        )

        result_by_id = {
            str(result["ids"][i]):
            np.asarray(
                result["embeddings"][i],
                dtype=np.float32
            )
            for i in range(
                len(result["ids"])
            )
        }

        for offset, chunk_id in enumerate(
            batch_ids
        ):

            expected = embeddings[
                start + offset
            ]

            actual = result_by_id.get(
                chunk_id
            )

            if actual is None:

                stored_embedding_failures += 1
                continue

            if actual.shape != expected.shape:

                stored_embedding_failures += 1
                continue

            difference = float(
                np.max(
                    np.abs(
                        actual
                        - expected
                    )
                )
            )

            embedding_max_difference = max(
                embedding_max_difference,
                difference
            )

            if difference != 0.0:

                stored_embedding_failures += 1

    if stored_embedding_failures:

        die(
            "Stored embedding validation failures: "
            f"{stored_embedding_failures}"
        )

    print(
        "Embedding shape             : PASS"
    )

    print(
        "Embedding values            : PASS"
    )

    print(
        "Maximum numerical difference:",
        embedding_max_difference
    )

    # ========================================================
    # TARGETED RETRIEVAL TEST
    # ========================================================

    print()
    print("=" * 90)
    print("TARGETED RETRIEVAL TEST")
    print("=" * 90)

    test_indices = [
        0,
        min(100, EXPECTED_CHUNKS - 1),
        min(1000, EXPECTED_CHUNKS - 1),
        min(2500, EXPECTED_CHUNKS - 1),
        EXPECTED_CHUNKS - 1,
    ]

    retrieval_failures = []

    for test_index in test_indices:

        query_embedding = embeddings[
            test_index
        ].tolist()

        expected_id = ids[
            test_index
        ]

        result = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=10,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        result_ids = result[
            "ids"
        ][0]

        if not result_ids:

            retrieval_failures.append(
                expected_id
            )

            continue

        if len(result_ids) != 10:

            retrieval_failures.append(
                expected_id
            )

            continue

        if len(
            result_ids
        ) != len(
            set(result_ids)
        ):

            retrieval_failures.append(
                expected_id
            )

            continue

        if expected_id not in result_ids:

            retrieval_failures.append(
                expected_id
            )

            continue

        rank = (
            result_ids.index(
                expected_id
            )
            + 1
        )

        print(
            f"{expected_id} -> "
            f"rank={rank}"
        )

    if retrieval_failures:

        die(
            "Targeted retrieval failures:\n"
            + "\n".join(
                retrieval_failures
            )
        )

    print(
        "Targeted retrieval tests     :",
        len(test_indices)
    )

    print(
        "Retrieval validation         : PASS"
    )

    # ========================================================
    # STEP 6 MANIFEST
    # ========================================================

    manifest = pd.DataFrame(
        {
            "vector_id": np.arange(
                EXPECTED_CHUNKS,
                dtype=np.int64
            ),
            "chunk_id": ids,
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
            "source_text_sha256": source_hashes,
            "chunk_text_sha256": [
                sha256_text(text)
                for text in text_values
            ],
            "embedding_dimension": [
                EXPECTED_DIMENSION
            ] * EXPECTED_CHUNKS,
            "model": [
                MODEL_NAME
            ] * EXPECTED_CHUNKS,
            "normalized": [
                True
            ] * EXPECTED_CHUNKS,
            "chroma_collection": [
                COLLECTION_NAME
            ] * EXPECTED_CHUNKS,
        }
    )

    # ========================================================
    # FINAL QC
    # ========================================================

    final_gates = {
        "input_chunks_4855":
            len(chunks) == EXPECTED_CHUNKS,

        "embeddings_4855":
            embeddings.shape[0] == EXPECTED_CHUNKS,

        "embedding_dimension_384":
            embeddings.shape[1] == EXPECTED_DIMENSION,

        "no_nan":
            bool(np.isfinite(
                embeddings
            ).all()),

        "no_inf":
            bool(np.isfinite(
                embeddings
            ).all()),

        "normalized_embeddings":
            norm_failures == 0,

        "step4_step5_alignment":
            step4_ids == step5_ids,

        "chroma_count_4855":
            count == EXPECTED_CHUNKS,

        "chroma_id_set_exact":
            set(stored_ids) == set(ids),

        "no_duplicate_chroma_ids":
            len(stored_ids) == len(set(stored_ids)),

        "exact_documents":
            document_failures == 0,

        "exact_metadata":
            metadata_failures == 0,

        "stored_embedding_shape":
            stored_embedding_failures == 0,

        "stored_embeddings_finite":
            stored_embedding_failures == 0,

        "embedding_values_exact":
            embedding_max_difference == 0.0,

        "targeted_retrieval":
            len(retrieval_failures) == 0,
    }

    final_status = all(
        final_gates.values()
    )

    print()
    print("=" * 90)
    print("PRAMANA — STEP 6 FINAL QC (V14)")
    print("=" * 90)

    print()
    print("INPUT")
    print("-" * 90)
    print(
        "Step 4 V14 chunks           :",
        len(chunks)
    )
    print(
        "Step 5 V14 embeddings       :",
        embeddings.shape
    )

    print()
    print("CHROMADB")
    print("-" * 90)
    print(
        "Collection                  :",
        COLLECTION_NAME
    )
    print(
        "Records                     :",
        count
    )
    print(
        "Embedding dimension         :",
        EXPECTED_DIMENSION
    )

    print()
    print("PROVENANCE")
    print("-" * 90)
    print(
        "ID set                      : PASS"
    )
    print(
        "Exact documents             : PASS"
    )
    print(
        "Exact metadata              : PASS"
    )
    print(
        "Embedding values exact      : PASS"
    )

    print()
    print("RETRIEVAL")
    print("-" * 90)
    print(
        "Targeted tests              :",
        len(test_indices)
    )
    print(
        "Retrieval failures          :",
        len(retrieval_failures)
    )

    print()
    print("FINAL QC GATE")
    print("-" * 90)

    for name, passed in final_gates.items():

        print(
            f"{name:35s}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()
    print(
        "FINAL STATUS                 :",
        "PASS" if final_status else "FAIL"
    )

    # ========================================================
    # WRITE MANIFEST / QC
    # ========================================================

    if not final_status:

        die(
            "Final QC failed. "
            "Step 6 V14 is not accepted."
        )

    manifest.to_csv(
        MANIFEST_OUT,
        index=False,
        encoding="utf-8"
    )

    report = []

    report.append(
        "=" * 90
    )

    report.append(
        "PRAMANA — STEP 6 FINAL QC (V14)"
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
        f"Step 4 V14 chunks           : {len(chunks)}"
    )

    report.append(
        f"Step 5 V14 embeddings       : {embeddings.shape}"
    )

    report.append("")

    report.append("CHROMADB")
    report.append("-" * 90)

    report.append(
        f"Collection                  : {COLLECTION_NAME}"
    )

    report.append(
        f"Records                     : {count}"
    )

    report.append(
        f"Embedding dimension         : {EXPECTED_DIMENSION}"
    )

    report.append("")

    report.append("PROVENANCE")
    report.append("-" * 90)

    report.append(
        "ID set                      : PASS"
    )

    report.append(
        "Exact documents             : PASS"
    )

    report.append(
        "Exact metadata              : PASS"
    )

    report.append(
        "Embedding values exact      : PASS"
    )

    report.append("")

    report.append("RETRIEVAL")
    report.append("-" * 90)

    report.append(
        f"Targeted tests              : {len(test_indices)}"
    )

    report.append(
        f"Retrieval failures          : {len(retrieval_failures)}"
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
        f"FINAL STATUS                 : "
        f"{'PASS' if final_status else 'FAIL'}"
    )

    report.append("")

    report.append("=" * 90)

    report.append("OUTPUT")

    report.append("=" * 90)

    report.append(
        f"ChromaDB                    : {CHROMA_DIR}"
    )

    report.append(
        f"Manifest                    : {MANIFEST_OUT}"
    )

    report.append(
        f"QC report                   : {QC_OUT}"
    )

    report.append(
        "=" * 90
    )

    QC_OUT.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print()
    print(
        "Manifest written            : PASS"
    )

    print(
        "QC report written           : PASS"
    )

    print()
    print(
        "STEP 6 V14 BUILD COMPLETE."
    )


if __name__ == "__main__":
    main()