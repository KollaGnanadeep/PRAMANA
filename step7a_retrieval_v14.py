from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import time

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PRAMANA — STEP 7A V14 RETRIEVAL VALIDATION
# ============================================================

ROOT = Path(__file__).resolve().parent

CHROMA_DIR = (
    ROOT
    / "data"
    / "step6_v14"
    / "chroma_db"
)

COLLECTION_NAME = "pramana_v14"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

EXPECTED_RECORDS = 4855
EXPECTED_DIMENSION = 384

TOP_K = 10

OUT = ROOT / "data" / "step7_v14"

RESULTS_OUT = OUT / "retrieval_results.json"
QC_OUT = OUT / "step7a_qc_report.txt"


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [
    "compensation for wrongful termination",
    "constitutional validity of legislation",
    "principles of natural justice",
    "right to property under the Constitution",
    "burden of proof in criminal proceedings",
    "arbitration agreement and jurisdiction",
    "fundamental rights and Article 14",
    "appeal against judgment of the High Court",
    "interpretation of statutory provisions",
    "limitation period for filing a claim",
]


# ============================================================
# FAIL CLOSED
# ============================================================

def die(message):

    print()
    print("=" * 90)
    print("FATAL — STEP 7A V14 VALIDATION ABORTED")
    print("=" * 90)
    print(message)
    print("=" * 90)

    sys.exit(1)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("PRAMANA — STEP 7A V14 RETRIEVAL VALIDATION")
    print("=" * 90)

    print()
    print("DATABASE")
    print("ChromaDB                    :", CHROMA_DIR)
    print("Collection                  :", COLLECTION_NAME)

    # ========================================================
    # OUTPUT SAFETY
    # ========================================================

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CHROMADB VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("CHROMADB VALIDATION")
    print("=" * 90)

    if not CHROMA_DIR.exists():

        die(
            "Step 6 V14 ChromaDB does not exist:\n"
            + str(CHROMA_DIR)
        )

    try:

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

    except Exception as exc:

        die(
            "Unable to initialize ChromaDB:\n"
            + str(exc)
        )

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception as exc:

        die(
            "Expected collection does not exist:\n"
            + str(exc)
        )

    record_count = collection.count()

    print(
        "Collection                  :",
        COLLECTION_NAME
    )

    print(
        "Records                     :",
        record_count
    )

    if record_count != EXPECTED_RECORDS:

        die(
            f"Expected {EXPECTED_RECORDS} records, "
            f"found {record_count}."
        )

    print(
        "Record count                : PASS"
    )

    # ========================================================
    # MODEL INITIALIZATION
    # ========================================================

    print()
    print("=" * 90)
    print("MODEL INITIALIZATION")
    print("=" * 90)

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
        "Model                       :",
        MODEL_NAME
    )

    print(
        "Embedding dimension         :",
        dimension
    )

    if dimension != EXPECTED_DIMENSION:

        die(
            f"Model dimension mismatch: "
            f"{dimension} != {EXPECTED_DIMENSION}"
        )

    print(
        "Model dimension             : PASS"
    )

    # ========================================================
    # COLLECTION INTEGRITY
    # ========================================================

    print()
    print("=" * 90)
    print("COLLECTION INTEGRITY")
    print("=" * 90)

    sample_ids = [
        "1965_107_C0000",
        "1966_263_C0009",
        "2015_131_C0002",
        "2006_267_C0004",
        "1954_158_C0012",
    ]

    sample = collection.get(
        ids=sample_ids,
        include=[
            "documents",
            "metadatas",
        ],
    )

    if len(sample["ids"]) != len(sample_ids):

        die(
            "Sample collection lookup failed."
        )

    sample_document_failures = 0
    sample_metadata_failures = 0

    for i in range(
        len(sample["ids"])
    ):

        document = sample["documents"][i]
        metadata = sample["metadatas"][i]

        if not isinstance(
            document,
            str
        ) or not document.strip():

            sample_document_failures += 1

        if not isinstance(
            metadata,
            dict
        ):

            sample_metadata_failures += 1

        elif not metadata.get(
            "case_id"
        ):

            sample_metadata_failures += 1

    if sample_document_failures:

        die(
            "Sample document integrity failure."
        )

    if sample_metadata_failures:

        die(
            "Sample metadata integrity failure."
        )

    print(
        "Sample document integrity   : PASS"
    )

    print(
        "Sample metadata integrity   : PASS"
    )

    # ========================================================
    # SEMANTIC RETRIEVAL
    # ========================================================

    print()
    print("=" * 90)
    print("SEMANTIC RETRIEVAL")
    print("=" * 90)

    results = []

    duplicate_failures = 0
    malformed_failures = 0
    retrieval_failures = 0

    latencies = []

    for index, query in enumerate(
        TEST_QUERIES,
        start=1
    ):

        start_time = time.perf_counter()

        try:

            query_embedding = model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            query_embedding = (
                query_embedding.tolist()
            )

            result = collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=TOP_K,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

        except Exception as exc:

            retrieval_failures += 1

            print(
                f"{index:02d}. {query}"
            )

            print(
                "    ERROR:",
                str(exc)
            )

            continue

        latency = (
            time.perf_counter()
            - start_time
        )

        latencies.append(
            latency
        )

        result_ids = result.get(
            "ids"
        )

        result_documents = result.get(
            "documents"
        )

        result_metadatas = result.get(
            "metadatas"
        )

        result_distances = result.get(
            "distances"
        )

        # ----------------------------------------------------
        # Result shape validation
        # ----------------------------------------------------

        malformed = False

        if not isinstance(
            result_ids,
            list
        ) or len(result_ids) != 1:

            malformed = True

        if not isinstance(
            result_documents,
            list
        ) or len(result_documents) != 1:

            malformed = True

        if not isinstance(
            result_metadatas,
            list
        ) or len(result_metadatas) != 1:

            malformed = True

        if not isinstance(
            result_distances,
            list
        ) or len(result_distances) != 1:

            malformed = True

        if malformed:

            malformed_failures += 1

            print(
                f"{index:02d}. {query}"
            )

            print(
                "    MALFORMED RESULT"
            )

            continue

        ids = result_ids[0]
        documents = result_documents[0]
        metadatas = result_metadatas[0]
        distances = result_distances[0]

        if len(ids) != TOP_K:

            malformed_failures += 1

            print(
                f"{index:02d}. {query}"
            )

            print(
                f"    Expected {TOP_K} results, "
                f"got {len(ids)}"
            )

            continue

        if not (
            len(documents)
            == len(ids)
            == len(metadatas)
            == len(distances)
        ):

            malformed_failures += 1

            print(
                f"{index:02d}. {query}"
            )

            print(
                "    Result field lengths mismatch"
            )

            continue

        # ----------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------

        if len(ids) != len(set(ids)):

            duplicate_failures += 1

            print(
                f"{index:02d}. {query}"
            )

            print(
                "    DUPLICATE RESULT IDS"
            )

            continue

        # ----------------------------------------------------
        # Individual result integrity
        # ----------------------------------------------------

        for doc, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            if not isinstance(
                doc,
                str
            ) or not doc.strip():

                malformed = True
                break

            if not isinstance(
                metadata,
                dict
            ):

                malformed = True
                break

            if not metadata.get(
                "case_id"
            ):

                malformed = True
                break

            try:

                float(distance)

            except Exception:

                malformed = True
                break

        if malformed:

            malformed_failures += 1

            print(
                f"{index:02d}. {query}"
            )

            print(
                "    MALFORMED RESULT CONTENT"
            )

            continue

        top_result = ids[0]
        top_distance = float(
            distances[0]
        )

        print(
            f"{index:02d}. {query}"
        )

        print(
            f"    Top result : "
            f"{top_result} | "
            f"distance={top_distance:.6f} | "
            f"{latency:.4f}s"
        )

        results.append(
            {
                "query_number": index,
                "query": query,
                "top_result": top_result,
                "top_distance": top_distance,
                "latency_seconds": latency,
                "results": [
                    {
                        "rank": rank + 1,
                        "chunk_id": ids[rank],
                        "distance": float(
                            distances[rank]
                        ),
                        "case_id": metadatas[
                            rank
                        ].get(
                            "case_id"
                        ),
                    }
                    for rank in range(
                        TOP_K
                    )
                ],
            }
        )

    successful_queries = len(
        results
    )

    # ========================================================
    # RETRIEVAL QC
    # ========================================================

    if successful_queries != len(
        TEST_QUERIES
    ):

        retrieval_failures += (
            len(TEST_QUERIES)
            - successful_queries
        )

    if duplicate_failures:

        die(
            f"Duplicate-result failures: "
            f"{duplicate_failures}"
        )

    if malformed_failures:

        die(
            f"Malformed-result failures: "
            f"{malformed_failures}"
        )

    if retrieval_failures:

        die(
            f"Retrieval failures: "
            f"{retrieval_failures}"
        )

    # ========================================================
    # LATENCY
    # ========================================================

    minimum_latency = (
        min(latencies)
        if latencies
        else 0.0
    )

    mean_latency = (
        sum(latencies)
        / len(latencies)
        if latencies
        else 0.0
    )

    maximum_latency = (
        max(latencies)
        if latencies
        else 0.0
    )

    # ========================================================
    # FINAL QC
    # ========================================================

    final_gates = {

        "chromadb_exists":
            CHROMA_DIR.exists(),

        "collection_exists":
            collection is not None,

        "record_count_4855":
            record_count == EXPECTED_RECORDS,

        "embedding_dimension_384":
            dimension == EXPECTED_DIMENSION,

        "sample_document_integrity":
            sample_document_failures == 0,

        "successful_queries":
            successful_queries == len(
                TEST_QUERIES
            ),

        "top_k_results":
            all(
                len(
                    item["results"]
                ) == TOP_K
                for item in results
            ),

        "no_duplicate_results":
            duplicate_failures == 0,

        "retrieved_documents_integrity":
            malformed_failures == 0,

        "results_written":
            True,
    }

    final_status = all(
        final_gates.values()
    )

    # ========================================================
    # WRITE RESULTS
    # ========================================================

    payload = {

        "build_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "collection":
            COLLECTION_NAME,

        "model":
            MODEL_NAME,

        "embedding_dimension":
            dimension,

        "record_count":
            record_count,

        "top_k":
            TOP_K,

        "queries":
            results,

        "retrieval_qc":
            {
                "queries":
                    len(TEST_QUERIES),

                "successful_queries":
                    successful_queries,

                "duplicate_result_failures":
                    duplicate_failures,

                "malformed_result_failures":
                    malformed_failures,

                "retrieval_failures":
                    retrieval_failures,

                "minimum_latency_seconds":
                    minimum_latency,

                "mean_latency_seconds":
                    mean_latency,

                "maximum_latency_seconds":
                    maximum_latency,
            },

        "final_status":
            "PASS"
            if final_status
            else "FAIL",
    }

    RESULTS_OUT.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "=" * 90
    )

    report.append(
        "PRAMANA — STEP 7A RETRIEVAL QC (V14)"
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

    report.append("DATABASE")
    report.append("-" * 90)

    report.append(
        f"Collection                  : "
        f"{COLLECTION_NAME}"
    )

    report.append(
        f"Records                     : "
        f"{record_count}"
    )

    report.append("")

    report.append("MODEL")
    report.append("-" * 90)

    report.append(
        f"Model                       : "
        f"{MODEL_NAME}"
    )

    report.append(
        f"Dimension                   : "
        f"{dimension}"
    )

    report.append("")

    report.append("RETRIEVAL")
    report.append("-" * 90)

    report.append(
        f"Queries                     : "
        f"{len(TEST_QUERIES)}"
    )

    report.append(
        f"Successful queries          : "
        f"{successful_queries}"
    )

    report.append(
        f"Top-K                       : "
        f"{TOP_K}"
    )

    report.append(
        f"Duplicate-result failures   : "
        f"{duplicate_failures}"
    )

    report.append(
        f"Malformed-result failures   : "
        f"{malformed_failures}"
    )

    report.append(
        f"Retrieval failures          : "
        f"{retrieval_failures}"
    )

    report.append(
        f"Minimum latency             : "
        f"{minimum_latency:.4f}s"
    )

    report.append(
        f"Mean latency                : "
        f"{mean_latency:.4f}s"
    )

    report.append(
        f"Maximum latency             : "
        f"{maximum_latency:.4f}s"
    )

    report.append("")

    report.append("FINAL QC GATE")
    report.append("-" * 90)

    for name, passed in (
        final_gates.items()
    ):

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

    report.append("OUTPUT")

    report.append("=" * 90)

    report.append(
        f"Results                     : "
        f"{RESULTS_OUT}"
    )

    report.append(
        f"QC report                   : "
        f"{QC_OUT}"
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
        "\n".join(report)
    )

    if not final_status:

        sys.exit(1)


if __name__ == "__main__":
    main()