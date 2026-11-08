"""
PRAMANA — STEP 10
End-to-End Retrieval Integration Test

Frozen upstream artifacts:
    Step 8: data/step8/step8_results.csv
    Step 9: data/step9/step9_results.csv
    ChromaDB: data/step6_v14/chroma_db
    Collection: pramana_v14
    Embedding model: BAAI/bge-small-en-v1.5

IMPORTANT:
    This script is READ-ONLY against Steps 4–9.
    No LLM/API calls are made.
"""

from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

STEP8_PATH = PROJECT_ROOT / "data" / "step8" / "step8_results.csv"
STEP9_PATH = PROJECT_ROOT / "data" / "step9" / "step9_results.csv"

CHROMA_PATH = PROJECT_ROOT / "data" / "step6_v14" / "chroma_db"
COLLECTION_NAME = "pramana_v14"

OUTPUT_DIR = PROJECT_ROOT / "data" / "step10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = OUTPUT_DIR / "step10_e2e_results.json"
RESULTS_CSV = OUTPUT_DIR / "step10_e2e_results.csv"
QC_REPORT = OUTPUT_DIR / "step10_e2e_qc_report.txt"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EXPECTED_DIMENSION = 384
TOP_K = 5

# These are the five queries whose Step 9 outputs were
# already manually reviewed and frozen.
TEST_QUERY_IDS = [
    "S08-001",
    "S08-012",
    "S08-013",
    "S08-014",
    "S08-015",
]

# Separate Step 8 domain-rejection smoke test.
OUT_OF_SCOPE_QUERY_ID = "S08-016"


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    """Return SHA-256 of a file for integrity tracking."""
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    name: str,
) -> None:
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{name} is missing required columns: {missing}\n"
            f"Actual columns: {list(df.columns)}"
        )


def clean_string(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# LOAD FROZEN STEP 8
# ============================================================

def load_step8() -> pd.DataFrame:
    if not STEP8_PATH.exists():
        raise FileNotFoundError(
            f"Step 8 file not found:\n{STEP8_PATH}"
        )

    df = pd.read_csv(STEP8_PATH)

    require_columns(
        df,
        [
            "query_id",
            "query",
            "domain_result",
            "legal_query",
        ],
        "Step 8 results",
    )

    df["query_id"] = df["query_id"].astype(str).str.strip()

    return df


# ============================================================
# LOAD FROZEN STEP 9
# ============================================================

def load_step9() -> pd.DataFrame:
    if not STEP9_PATH.exists():
        raise FileNotFoundError(
            f"Step 9 file not found:\n{STEP9_PATH}"
        )

    df = pd.read_csv(STEP9_PATH)

    require_columns(
        df,
        [
            "query_id",
            "citizen_query",
            "legal_query",
            "hyde_text",
            "validation_pass",
        ],
        "Step 9 results",
    )

    df["query_id"] = df["query_id"].astype(str).str.strip()

    return df


# ============================================================
# VALIDATE STEP 8 / STEP 9 LINKAGE
# ============================================================

def build_frozen_pipeline_records(
    step8: pd.DataFrame,
    step9: pd.DataFrame,
) -> list[dict[str, Any]]:

    records = []

    for query_id in TEST_QUERY_IDS:

        step8_matches = step8[
            step8["query_id"] == query_id
        ]

        step9_matches = step9[
            step9["query_id"] == query_id
        ]

        if len(step8_matches) != 1:
            raise RuntimeError(
                f"{query_id}: expected exactly one Step 8 row, "
                f"found {len(step8_matches)}"
            )

        if len(step9_matches) != 1:
            raise RuntimeError(
                f"{query_id}: expected exactly one Step 9 row, "
                f"found {len(step9_matches)}"
            )

        s8 = step8_matches.iloc[0]
        s9 = step9_matches.iloc[0]

        domain = clean_string(s8["domain_result"])
        citizen_query = clean_string(s8["query"])
        legal_query_s8 = clean_string(s8["legal_query"])

        hyde_text = clean_string(s9["hyde_text"])
        legal_query_s9 = clean_string(s9["legal_query"])

        validation_pass = clean_string(
            s9["validation_pass"]
        ).lower()

        # ----------------------------------------------------
        # Frozen consistency checks
        # ----------------------------------------------------

        if domain != "IN_SCOPE":
            raise RuntimeError(
                f"{query_id}: Step 8 domain is not IN_SCOPE: "
                f"{domain}"
            )

        if validation_pass not in {
            "true",
            "1",
            "yes",
        }:
            raise RuntimeError(
                f"{query_id}: Step 9 validation_pass is not PASS: "
                f"{s9['validation_pass']}"
            )

        if not citizen_query:
            raise RuntimeError(
                f"{query_id}: empty citizen query"
            )

        if not legal_query_s8:
            raise RuntimeError(
                f"{query_id}: empty Step 8 legal query"
            )

        if not hyde_text:
            raise RuntimeError(
                f"{query_id}: empty Step 9 HyDE text"
            )

        # Step 9 should have received the same legal query
        # produced by Step 8.
        if legal_query_s8 != legal_query_s9:
            raise RuntimeError(
                f"{query_id}: Step 8 / Step 9 legal_query mismatch.\n"
                f"Step 8: {legal_query_s8}\n"
                f"Step 9: {legal_query_s9}"
            )

        records.append(
            {
                "query_id": query_id,
                "citizen_query": citizen_query,
                "legal_query": legal_query_s8,
                "hyde_text": hyde_text,
            }
        )

    return records


# ============================================================
# LOAD FROZEN CHROMADB
# ============================================================

def load_chroma():
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB path not found:\n{CHROMA_PATH}"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        COLLECTION_NAME
    )

    return client, collection


# ============================================================
# LOAD BGE
# ============================================================

def load_bge():
    print("\nLoading embedding model:")
    print(f"  {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # Verify dimension independently.
    test_embedding = model.encode(
        "PRAMANA dimension verification",
        normalize_embeddings=True,
    )

    actual_dimension = len(test_embedding)

    if actual_dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"Unexpected embedding dimension: "
            f"{actual_dimension}; expected {EXPECTED_DIMENSION}"
        )

    return model


# ============================================================
# EMBED + RETRIEVE
# ============================================================

def retrieve(
    model,
    collection,
    hyde_text: str,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, dict[str, Any]]:

    embedding = model.encode(
        hyde_text,
        normalize_embeddings=True,
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if embedding.ndim != 1:
        raise RuntimeError(
            f"Unexpected embedding shape: {embedding.shape}"
        )

    if len(embedding) != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"Embedding dimension {len(embedding)} "
            f"!= {EXPECTED_DIMENSION}"
        )

    if not np.isfinite(embedding).all():
        raise RuntimeError(
            "Embedding contains NaN or Inf"
        )

    norm = float(np.linalg.norm(embedding))

    if not np.isclose(norm, 1.0, atol=1e-5):
        raise RuntimeError(
            f"Normalized embedding norm is {norm}, "
            f"expected approximately 1.0"
        )

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return embedding, results


# ============================================================
# NORMALIZE CHROMA RESULTS
# ============================================================

def normalize_results(
    results: dict[str, Any],
) -> list[dict[str, Any]]:

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not (
        len(ids)
        == len(documents)
        == len(metadatas)
        == len(distances)
    ):
        raise RuntimeError(
            "Chroma result arrays have inconsistent lengths"
        )

    normalized = []

    for i in range(len(ids)):

        metadata = metadatas[i] or {}

        chunk_id = clean_string(ids[i])

        case_id = clean_string(
            metadata.get("case_id")
        )

        if not chunk_id:
            raise RuntimeError(
                f"Rank {i + 1}: empty chunk_id"
            )

        if not case_id:
            raise RuntimeError(
                f"Rank {i + 1}: empty case_id"
            )

        distance = float(distances[i])

        document = clean_string(documents[i])

        normalized.append(
            {
                "rank": i + 1,
                "chunk_id": chunk_id,
                "case_id": case_id,
                "distance": distance,
                "similarity": round(
                    1.0 - distance,
                    6,
                ),
                "preview": document[:300],
            }
        )

    return normalized


# ============================================================
# OUT-OF-SCOPE CHECK
# ============================================================

def test_out_of_scope(
    step8: pd.DataFrame,
) -> dict[str, Any]:

    matches = step8[
        step8["query_id"] == OUT_OF_SCOPE_QUERY_ID
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{OUT_OF_SCOPE_QUERY_ID}: expected exactly one "
            f"Step 8 result, found {len(matches)}"
        )

    row = matches.iloc[0]

    domain = clean_string(
        row["domain_result"]
    )

    query = clean_string(
        row["query"]
    )

    return {
        "query_id": OUT_OF_SCOPE_QUERY_ID,
        "query": query,
        "domain_result": domain,
        "retrieval_attempted": False,
        "status": (
            "PASS"
            if domain == "OUT_OF_SCOPE"
            else "FAIL"
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("PRAMANA — STEP 10")
    print("END-TO-END RETRIEVAL INTEGRATION TEST")
    print("=" * 72)

    print("\nFrozen upstream artifacts:")
    print(f"  Step 8 : {STEP8_PATH}")
    print(f"  Step 9 : {STEP9_PATH}")
    print(f"  Chroma : {CHROMA_PATH}")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Top-K  : {TOP_K}")

    # --------------------------------------------------------
    # Integrity hashes before execution
    # --------------------------------------------------------

    step8_hash_before = sha256_file(STEP8_PATH)
    step9_hash_before = sha256_file(STEP9_PATH)

    print("\nLoading frozen Step 8...")
    step8 = load_step8()
    print(f"  Step 8 records: {len(step8)}")

    print("\nLoading frozen Step 9...")
    step9 = load_step9()
    print(f"  Step 9 records: {len(step9)}")

    print("\nValidating Step 8 → Step 9 linkage...")
    records = build_frozen_pipeline_records(
        step8,
        step9,
    )
    print(
        f"  Valid frozen pipeline records: "
        f"{len(records)}/{len(TEST_QUERY_IDS)}"
    )

    print("\nLoading frozen ChromaDB...")
    _, collection = load_chroma()

    collection_count = collection.count()

    print(
        f"  Collection: {COLLECTION_NAME}"
    )
    print(
        f"  Records: {collection_count}"
    )

    if collection_count != 4855:
        raise RuntimeError(
            f"Unexpected ChromaDB count: "
            f"{collection_count}; expected 4855"
        )

    model = load_bge()

    # --------------------------------------------------------
    # OUT-OF-SCOPE TEST
    # --------------------------------------------------------

    print("\n" + "-" * 72)
    print("OUT-OF-SCOPE DOMAIN REJECTION TEST")
    print("-" * 72)

    out_result = test_out_of_scope(step8)

    print(
        f"Query ID     : {out_result['query_id']}"
    )
    print(
        f"Query        : {out_result['query']}"
    )
    print(
        f"Domain       : {out_result['domain_result']}"
    )
    print(
        f"Retrieval    : NOT ATTEMPTED"
    )
    print(
        f"Gate         : {out_result['status']}"
    )

    # --------------------------------------------------------
    # END-TO-END RETRIEVAL
    # --------------------------------------------------------

    all_results = []

    print("\n" + "=" * 72)
    print("IN-SCOPE END-TO-END RETRIEVAL")
    print("=" * 72)

    for index, record in enumerate(
        records,
        start=1,
    ):

        print("\n" + "-" * 72)
        print(
            f"[{index}/{len(records)}] "
            f"{record['query_id']}"
        )
        print("-" * 72)

        print(
            f"Citizen query:\n"
            f"  {record['citizen_query']}"
        )

        print(
            f"\nLegal query:\n"
            f"  {record['legal_query']}"
        )

        print(
            f"\nHyDE word count:\n"
            f"  {len(record['hyde_text'].split())}"
        )

        print(
            f"\nHyDE preview:\n"
            f"  {record['hyde_text'][:250]}..."
        )

        embedding, raw_results = retrieve(
            model,
            collection,
            record["hyde_text"],
            TOP_K,
        )

        retrieved = normalize_results(
            raw_results
        )

        if len(retrieved) != TOP_K:
            raise RuntimeError(
                f"{record['query_id']}: expected "
                f"{TOP_K} results, got {len(retrieved)}"
            )

        print("\nTop-5:")

        for item in retrieved:
            print(
                f"  [{item['rank']}] "
                f"{item['case_id']} "
                f"| chunk={item['chunk_id']} "
                f"| distance={item['distance']:.6f} "
                f"| similarity={item['similarity']:.6f}"
            )

            print(
                f"       {item['preview'][:150]}..."
            )

        all_results.append(
            {
                "query_id": record["query_id"],
                "citizen_query": record[
                    "citizen_query"
                ],
                "legal_query": record[
                    "legal_query"
                ],
                "hyde_text": record[
                    "hyde_text"
                ],
                "embedding_dimension": int(
                    len(embedding)
                ),
                "embedding_norm": float(
                    np.linalg.norm(embedding)
                ),
                "top_k": TOP_K,
                "results": retrieved,
                "status": "PASS",
            }
        )

    # --------------------------------------------------------
    # BUILD CSV
    # --------------------------------------------------------

    csv_rows = []

    for result in all_results:

        for item in result["results"]:

            csv_rows.append(
                {
                    "query_id": result["query_id"],
                    "citizen_query": result[
                        "citizen_query"
                    ],
                    "legal_query": result[
                        "legal_query"
                    ],
                    "rank": item["rank"],
                    "chunk_id": item["chunk_id"],
                    "case_id": item["case_id"],
                    "distance": item["distance"],
                    "similarity": item["similarity"],
                    "preview": item["preview"],
                }
            )

    pd.DataFrame(csv_rows).to_csv(
        RESULTS_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL INTEGRITY CHECK
    # --------------------------------------------------------

    step8_hash_after = sha256_file(STEP8_PATH)
    step9_hash_after = sha256_file(STEP9_PATH)

    upstream_unchanged = (
        step8_hash_before == step8_hash_after
        and step9_hash_before == step9_hash_after
    )

    # --------------------------------------------------------
    # FINAL QC
    # --------------------------------------------------------

    pipeline_count = len(all_results)

    embedding_pass = all(
        r["embedding_dimension"]
        == EXPECTED_DIMENSION
        and np.isclose(
            r["embedding_norm"],
            1.0,
            atol=1e-5,
        )
        for r in all_results
    )

    retrieval_pass = all(
        len(r["results"]) == TOP_K
        for r in all_results
    )

    unique_chunk_pass = all(
        len(
            {
                x["chunk_id"]
                for x in r["results"]
            }
        )
        == TOP_K
        for r in all_results
    )

    structure_pass = all(
        all(
            x["case_id"]
            and x["chunk_id"]
            and isinstance(
                x["rank"],
                int,
            )
            for x in r["results"]
        )
        for r in all_results
    )

    all_pass = all(
        [
            pipeline_count == len(TEST_QUERY_IDS),
            out_result["status"] == "PASS",
            embedding_pass,
            retrieval_pass,
            unique_chunk_pass,
            structure_pass,
            upstream_unchanged,
        ]
    )

    qc_lines = [
        "PRAMANA — STEP 10 QC REPORT",
        "=" * 60,
        "",
        f"Test queries              : {len(TEST_QUERY_IDS)}",
        f"Pipeline records          : {pipeline_count}/{len(TEST_QUERY_IDS)}",
        f"Out-of-scope test         : {out_result['status']}",
        f"Embedding dimension       : "
        f"{EXPECTED_DIMENSION} "
        f"({'PASS' if embedding_pass else 'FAIL'})",
        f"Embedding normalization   : "
        f"{'PASS' if embedding_pass else 'FAIL'}",
        f"Top-{TOP_K} retrieval        : "
        f"{'PASS' if retrieval_pass else 'FAIL'}",
        f"Unique chunks per query   : "
        f"{'PASS' if unique_chunk_pass else 'FAIL'}",
        f"Result structure          : "
        f"{'PASS' if structure_pass else 'FAIL'}",
        f"ChromaDB record count     : "
        f"{collection_count}",
        f"Upstream artifacts        : "
        f"{'UNCHANGED' if upstream_unchanged else 'MODIFIED'}",
        "",
        f"FINAL STATUS              : "
        f"{'PASS' if all_pass else 'FAIL'}",
    ]

    with QC_REPORT.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(qc_lines))

    output_json = {
        "step": "10",
        "status": "PASS" if all_pass else "FAIL",
        "model": MODEL_NAME,
        "embedding_dimension": EXPECTED_DIMENSION,
        "chroma_collection": COLLECTION_NAME,
        "chroma_count": collection_count,
        "top_k": TOP_K,
        "out_of_scope_test": out_result,
        "results": all_results,
        "qc": {
            "pipeline_count": pipeline_count,
            "embedding_pass": embedding_pass,
            "retrieval_pass": retrieval_pass,
            "unique_chunk_pass": unique_chunk_pass,
            "structure_pass": structure_pass,
            "upstream_unchanged": upstream_unchanged,
        },
    }

    with RESULTS_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output_json,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 10 FINAL QC")
    print("=" * 72)

    print(
        f"Test queries              : "
        f"{len(TEST_QUERY_IDS)}"
    )

    print(
        f"Pipeline records          : "
        f"{pipeline_count}/{len(TEST_QUERY_IDS)}"
    )

    print(
        f"Out-of-scope test         : "
        f"{out_result['status']}"
    )

    print(
        f"Embedding dimension       : "
        f"{EXPECTED_DIMENSION} "
        f"{'PASS' if embedding_pass else 'FAIL'}"
    )

    print(
        f"Embedding normalization   : "
        f"{'PASS' if embedding_pass else 'FAIL'}"
    )

    print(
        f"Top-{TOP_K} retrieval        : "
        f"{'PASS' if retrieval_pass else 'FAIL'}"
    )

    print(
        f"Unique chunks             : "
        f"{'PASS' if unique_chunk_pass else 'FAIL'}"
    )

    print(
        f"Result structure          : "
        f"{'PASS' if structure_pass else 'FAIL'}"
    )

    print(
        f"ChromaDB records          : "
        f"{collection_count}"
    )

    print(
        f"Upstream unchanged       : "
        f"{'PASS' if upstream_unchanged else 'FAIL'}"
    )

    print(
        f"\nFINAL STATUS              : "
        f"{'PASS' if all_pass else 'FAIL'}"
    )

    print(
        f"\nResults saved             : "
        f"{RESULTS_JSON}"
    )

    print(
        f"CSV saved                 : "
        f"{RESULTS_CSV}"
    )

    print(
        f"QC report saved           : "
        f"{QC_REPORT}"
    )

    if all_pass:
        print(
            "\nSTEP 10 FINAL STATUS: PASS"
        )
    else:
        print(
            "\nSTEP 10 FINAL STATUS: FAIL"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()