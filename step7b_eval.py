from pathlib import Path
import sys
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# =============================================================================
# PRAMANA — STEP 7B V14 GOLD EVALUATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

V14_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ildc_domain_classification_v6"
    / "v14"
    / "ildc_multi_domain_relevant_v14_forensic.parquet"
)

CHROMA_DIR = BASE_DIR / "data" / "step6_v14" / "chroma_db"
COLLECTION_NAME = "pramana_v14"

STEP7_DIR = BASE_DIR / "data" / "step7"
GOLD_PATH = STEP7_DIR / "gold_eval.csv"

RESULTS_PATH = STEP7_DIR / "step7b_eval_results.csv"
REPORT_PATH = STEP7_DIR / "step7b_qc_report.txt"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

EXPECTED_V14_CASES = 538
EXPECTED_CHROMA_RECORDS = 4855
EXPECTED_GOLD_ROWS = 25
EXPECTED_DIRECT = 18
EXPECTED_PARTIAL = 7
TOP_K = 10


def fatal(message):
    print()
    print("=" * 90)
    print("FATAL — STEP 7B GOLD EVALUATION ABORTED")
    print("=" * 90)
    print(message)
    print("=" * 90)
    sys.exit(1)


print("=" * 90)
print("PRAMANA — STEP 7B V14 GOLD EVALUATION")
print("=" * 90)

print()
print("READ ONLY")
print("V14 WILL NOT BE MODIFIED")
print("STEP 4 WILL NOT BE MODIFIED")
print("STEP 5 WILL NOT BE MODIFIED")
print("STEP 6 WILL NOT BE MODIFIED")
print("CHROMADB WILL NOT BE MODIFIED")
print("GOLD EVAL WILL NOT BE MODIFIED")


# =============================================================================
# V14
# =============================================================================

print()
print("=" * 90)
print("V14 VALIDATION")
print("=" * 90)

if not V14_PATH.exists():
    fatal(f"V14 source not found:\n{V14_PATH}")

v14 = pd.read_parquet(V14_PATH)

print(f"V14 source                  : {V14_PATH}")
print(f"V14 records                 : {len(v14)}")
print(f"V14 columns                 : {v14.columns.tolist()}")

if len(v14) != EXPECTED_V14_CASES:
    fatal(
        f"Expected {EXPECTED_V14_CASES} V14 cases, found {len(v14)}."
    )

if list(v14.columns) != ["id", "text"]:
    fatal(
        "Unexpected V14 schema.\n"
        f"Expected: ['id', 'text']\n"
        f"Found:    {v14.columns.tolist()}"
    )

v14_case_ids = set(v14["id"].astype(str))

if len(v14_case_ids) != EXPECTED_V14_CASES:
    fatal("V14 case IDs are not unique.")

print("Record count                : PASS")
print("Schema                      : PASS")
print("Case ID uniqueness          : PASS")


# =============================================================================
# GOLD
# =============================================================================

print()
print("=" * 90)
print("GOLD EVALUATION VALIDATION")
print("=" * 90)

if not GOLD_PATH.exists():
    fatal(f"Gold file not found:\n{GOLD_PATH}")

gold = pd.read_csv(GOLD_PATH)

print(f"Gold file                   : {GOLD_PATH}")
print(f"Gold rows                   : {len(gold)}")
print(f"Gold columns                : {gold.columns.tolist()}")

REQUIRED_GOLD_COLUMNS = [
    "query_id",
    "source_case_id",
    "source_chunk_id",
    "relevance_type",
]

missing = [
    c for c in REQUIRED_GOLD_COLUMNS
    if c not in gold.columns
]

if missing:
    fatal(
        "Gold file is missing required columns:\n"
        + "\n".join(f"  - {x}" for x in missing)
    )

if len(gold) != EXPECTED_GOLD_ROWS:
    fatal(
        f"Expected {EXPECTED_GOLD_ROWS} gold rows, "
        f"found {len(gold)}."
    )

if gold["query_id"].duplicated().any():
    fatal("Duplicate query IDs detected.")

gold["query_id"] = gold["query_id"].astype(str)
gold["source_case_id"] = gold["source_case_id"].astype(str)
gold["source_chunk_id"] = gold["source_chunk_id"].astype(str)
gold["relevance_type"] = (
    gold["relevance_type"]
    .astype(str)
    .str.upper()
    .str.strip()
)

unexpected = set(gold["relevance_type"]) - {"DIRECT", "PARTIAL"}

if unexpected:
    fatal(
        "Unexpected relevance_type values:\n"
        + "\n".join(sorted(unexpected))
    )

direct_count = (gold["relevance_type"] == "DIRECT").sum()
partial_count = (gold["relevance_type"] == "PARTIAL").sum()

print(f"DIRECT rows                 : {direct_count}")
print(f"PARTIAL rows                : {partial_count}")

if direct_count != EXPECTED_DIRECT:
    fatal(
        f"Expected {EXPECTED_DIRECT} DIRECT rows, "
        f"found {direct_count}."
    )

if partial_count != EXPECTED_PARTIAL:
    fatal(
        f"Expected {EXPECTED_PARTIAL} PARTIAL rows, "
        f"found {partial_count}."
    )

print("Gold row count              : PASS")
print("Gold schema                 : PASS")
print("Gold labels                 : PASS")


# =============================================================================
# GOLD CASE VALIDATION
# =============================================================================

print()
print("=" * 90)
print("GOLD CASE REFERENCE VALIDATION")
print("=" * 90)

for _, row in gold.iterrows():

    qid = row["query_id"]
    case_id = row["source_case_id"]

    if case_id not in v14_case_ids:
        fatal(
            f"{qid}: gold case not found in V14:\n{case_id}"
        )

print("All gold cases exist in V14 : PASS")


# =============================================================================
# CHROMA
# =============================================================================

print()
print("=" * 90)
print("FROZEN V14 CHROMADB VALIDATION")
print("=" * 90)

print(f"ChromaDB                    : {CHROMA_DIR}")
print(f"Collection                  : {COLLECTION_NAME}")

if not CHROMA_DIR.exists():
    fatal(f"ChromaDB not found:\n{CHROMA_DIR}")

try:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

except Exception as exc:
    fatal(f"Could not open ChromaDB:\n{exc}")

chroma_count = collection.count()

print(f"Chroma records              : {chroma_count}")

if chroma_count != EXPECTED_CHROMA_RECORDS:
    fatal(
        f"Expected {EXPECTED_CHROMA_RECORDS} Chroma records, "
        f"found {chroma_count}."
    )

print("Record count                : PASS")


# =============================================================================
# CHROMA GOLD CHUNK VALIDATION
# =============================================================================

print()
print("=" * 90)
print("GOLD CHUNK REFERENCE VALIDATION")
print("=" * 90)

try:
    chroma_data = collection.get(
        include=["metadatas"]
    )
except Exception as exc:
    fatal(f"Could not read Chroma metadata:\n{exc}")

chroma_ids = chroma_data["ids"]
chroma_metadatas = chroma_data["metadatas"]

if len(chroma_ids) != EXPECTED_CHROMA_RECORDS:
    fatal("Chroma ID count mismatch.")

if len(set(chroma_ids)) != EXPECTED_CHROMA_RECORDS:
    fatal("Duplicate Chroma IDs detected.")

chroma_id_set = set(str(x) for x in chroma_ids)

for _, row in gold.iterrows():

    qid = row["query_id"]
    chunk_id = row["source_chunk_id"]

    if chunk_id not in chroma_id_set:
        fatal(
            f"{qid}: gold chunk not found in frozen ChromaDB:\n"
            f"{chunk_id}"
        )

print("All gold chunks exist       : PASS")


# =============================================================================
# MODEL
# =============================================================================

print()
print("=" * 90)
print("MODEL INITIALIZATION")
print("=" * 90)

print(f"Model                       : {MODEL_NAME}")

try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as exc:
    fatal(f"Could not load model:\n{exc}")

dimension = model.get_sentence_embedding_dimension()

print(f"Embedding dimension         : {dimension}")

if dimension != 384:
    fatal(
        f"Expected dimension 384, found {dimension}."
    )

print("Model dimension             : PASS")


# =============================================================================
# QUERY CONSTRUCTION
#
# The finalized gold file does not retain candidate_query.
# Therefore reconstruct the exact candidate query from the frozen
# Chroma document associated with source_chunk_id.
# =============================================================================

print()
print("=" * 90)
print("GOLD QUERY RECONSTRUCTION")
print("=" * 90)

chroma_lookup = {
    str(chroma_ids[i]): {
        "metadata": chroma_metadatas[i]
    }
    for i in range(len(chroma_ids))
}

queries = []

for _, row in gold.iterrows():

    qid = row["query_id"]
    chunk_id = row["source_chunk_id"]
    case_id = row["source_case_id"]

    metadata = chroma_lookup[chunk_id]

    metadata_case = str(
        metadata["metadata"].get("case_id")
    )

    if metadata_case != case_id:
        fatal(
            f"{qid}: gold case/chunk mismatch.\n"
            f"Gold case: {case_id}\n"
            f"Chunk metadata case: {metadata_case}"
        )

    # Chroma documents must be retrieved separately.
    try:
        doc_data = collection.get(
            ids=[chunk_id],
            include=["documents"]
        )
    except Exception as exc:
        fatal(
            f"{qid}: could not retrieve gold chunk document:\n{exc}"
        )

    documents = doc_data.get("documents", [])

    if len(documents) != 1:
        fatal(
            f"{qid}: expected one document for {chunk_id}."
        )

    passage = str(documents[0])

    candidate_query = (
        "What legal principle or issue is addressed "
        "in this judgment regarding: "
        + passage[:1000]
    )

    queries.append(
        {
            "query_id": qid,
            "candidate_query": candidate_query,
            "relevant_case_id": case_id,
            "relevant_chunk_id": chunk_id,
            "relevance_type": row["relevance_type"],
        }
    )

print(
    f"Queries reconstructed       : {len(queries)}"
)

if len(queries) != EXPECTED_GOLD_ROWS:
    fatal("Gold query reconstruction count mismatch.")

print("Query reconstruction        : PASS")


# =============================================================================
# RETRIEVAL
# =============================================================================

print()
print("=" * 90)
print("GOLD QUERY RETRIEVAL")
print("=" * 90)

results = []

for n, row in enumerate(
    queries,
    start=1
):

    qid = row["query_id"]
    query = row["candidate_query"]
    relevant_case = row["relevant_case_id"]
    relevant_chunk = row["relevant_chunk_id"]
    relevance_type = row["relevance_type"]

    embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    retrieved = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=TOP_K,
        include=[
            "metadatas",
            "distances",
        ],
    )

    retrieved_ids = [
        str(x)
        for x in retrieved["ids"][0]
    ]

    retrieved_metadatas = retrieved["metadatas"][0]

    retrieved_cases = [
        str(m["case_id"])
        for m in retrieved_metadatas
    ]

    if len(retrieved_ids) != TOP_K:
        fatal(
            f"{qid}: expected {TOP_K} results, "
            f"got {len(retrieved_ids)}."
        )

    case_rank = next(
        (
            i + 1
            for i, case in enumerate(retrieved_cases)
            if case == relevant_case
        ),
        None,
    )

    chunk_rank = next(
        (
            i + 1
            for i, chunk in enumerate(retrieved_ids)
            if chunk == relevant_chunk
        ),
        None,
    )

    results.append(
        {
            "query_id": qid,
            "relevance_type": relevance_type,

            "relevant_case_id": relevant_case,
            "relevant_chunk_id": relevant_chunk,

            "case_rank": case_rank,
            "case_hit@1": case_rank is not None and case_rank <= 1,
            "case_hit@5": case_rank is not None and case_rank <= 5,
            "case_hit@10": case_rank is not None and case_rank <= 10,

            "chunk_rank": chunk_rank,
            "chunk_hit@1": chunk_rank is not None and chunk_rank <= 1,
            "chunk_hit@5": chunk_rank is not None and chunk_rank <= 5,
            "chunk_hit@10": chunk_rank is not None and chunk_rank <= 10,

            "reciprocal_rank": (
                1.0 / chunk_rank
                if chunk_rank is not None
                else 0.0
            ),

            "retrieved_chunk_ids": "|".join(
                retrieved_ids
            ),

            "retrieved_case_ids": "|".join(
                retrieved_cases
            ),
        }
    )

    print(
        f"{n:02d}/{EXPECTED_GOLD_ROWS} "
        f"{qid} "
        f"{relevance_type:<7} "
        f"case_rank={case_rank} "
        f"chunk_rank={chunk_rank}"
    )


# =============================================================================
# METRICS
# =============================================================================

results_df = pd.DataFrame(results)


def mean_metric(column, frame=None):

    if frame is None:
        frame = results_df

    if len(frame) == 0:
        return 0.0

    return float(
        frame[column].astype(float).mean()
    )


case_r1 = mean_metric("case_hit@1")
case_r5 = mean_metric("case_hit@5")
case_r10 = mean_metric("case_hit@10")

chunk_r1 = mean_metric("chunk_hit@1")
chunk_r5 = mean_metric("chunk_hit@5")
chunk_r10 = mean_metric("chunk_hit@10")

mrr = mean_metric("reciprocal_rank")

direct_df = results_df[
    results_df["relevance_type"] == "DIRECT"
]

partial_df = results_df[
    results_df["relevance_type"] == "PARTIAL"
]


# =============================================================================
# OUTPUT
# =============================================================================

print()
print("=" * 90)
print("PRAMANA — STEP 7B V14 GOLD EVALUATION RESULTS")
print("=" * 90)

print()
print("DATASET")
print("-" * 90)
print(f"V14 cases                   : {len(v14)}")
print(f"Chroma chunks               : {chroma_count}")
print(f"Gold queries                : {len(results_df)}")
print(f"DIRECT queries              : {len(direct_df)}")
print(f"PARTIAL queries             : {len(partial_df)}")

print()
print("CASE-LEVEL RETRIEVAL")
print("-" * 90)
print(f"Recall@1                    : {case_r1:.3f}")
print(f"Recall@5                    : {case_r5:.3f}")
print(f"Recall@10                   : {case_r10:.3f}")

print()
print("CHUNK-LEVEL RETRIEVAL")
print("-" * 90)
print(f"Recall@1                    : {chunk_r1:.3f}")
print(f"Recall@5                    : {chunk_r5:.3f}")
print(f"Recall@10                   : {chunk_r10:.3f}")

print()
print("CHUNK-LEVEL MRR")
print("-" * 90)
print(f"MRR                         : {mrr:.3f}")

print()
print("DIRECT / PARTIAL CHUNK RECALL@10")
print("-" * 90)
print(
    f"DIRECT                     : "
    f"{mean_metric('chunk_hit@10', direct_df):.3f}"
)
print(
    f"PARTIAL                    : "
    f"{mean_metric('chunk_hit@10', partial_df):.3f}"
)


# =============================================================================
# WRITE
# =============================================================================

STEP7_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    RESULTS_PATH,
    index=False,
    encoding="utf-8-sig"
)


report = f"""
PRAMANA — STEP 7B V14 GOLD EVALUATION QC REPORT
==========================================================================================

FROZEN INPUTS
------------------------------------------------------------------------------------------
V14 cases                   : {len(v14)}
Chroma chunks               : {chroma_count}
Gold queries                : {len(results_df)}

MODEL
------------------------------------------------------------------------------------------
Model                       : {MODEL_NAME}
Dimension                   : {dimension}
Normalized queries          : True

GOLD SET
------------------------------------------------------------------------------------------
DIRECT                      : {len(direct_df)}
PARTIAL                     : {len(partial_df)}

CASE-LEVEL RETRIEVAL
------------------------------------------------------------------------------------------
Recall@1                    : {case_r1:.3f}
Recall@5                    : {case_r5:.3f}
Recall@10                   : {case_r10:.3f}

CHUNK-LEVEL RETRIEVAL
------------------------------------------------------------------------------------------
Recall@1                    : {chunk_r1:.3f}
Recall@5                    : {chunk_r5:.3f}
Recall@10                   : {chunk_r10:.3f}

CHUNK-LEVEL MRR
------------------------------------------------------------------------------------------
MRR                         : {mrr:.3f}

DIRECT / PARTIAL CHUNK RECALL@10
------------------------------------------------------------------------------------------
DIRECT                      : {mean_metric('chunk_hit@10', direct_df):.3f}
PARTIAL                     : {mean_metric('chunk_hit@10', partial_df):.3f}

QC
------------------------------------------------------------------------------------------
V14_538                     : PASS
Chroma_4855                 : PASS
Gold_25                     : PASS
Gold_DIRECT_18              : PASS
Gold_PARTIAL_7              : PASS
Gold_case_references        : PASS
Gold_chunk_references       : PASS
Model_dimension_384         : PASS
Queries_evaluated_25        : PASS
Top10_retrieval              : PASS
Results_written             : PASS

FINAL STATUS                : PASS

OUTPUT
------------------------------------------------------------------------------------------
Results                     : {RESULTS_PATH}
QC report                   : {REPORT_PATH}
==========================================================================================
"""

REPORT_PATH.write_text(
    report.strip() + "\n",
    encoding="utf-8"
)

print()
print("=" * 90)
print("OUTPUT")
print("=" * 90)
print(f"Results                     : {RESULTS_PATH}")
print(f"QC report                   : {REPORT_PATH}")
print()
print("FINAL STATUS                : PASS")
print("=" * 90)