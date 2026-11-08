import os
import csv
import time
from datetime import datetime
from pathlib import Path

from groq import Groq


# =============================================================================
# PRAMANA — STEP 8 STEP-BACK + DOMAIN CHECK
# =============================================================================
#
# READ ONLY with respect to Steps 4–7.
#
# This script:
#   - DOES NOT modify V14
#   - DOES NOT modify Step 4
#   - DOES NOT modify Step 5
#   - DOES NOT modify Step 6
#   - DOES NOT modify ChromaDB
#   - DOES NOT modify Step 7 gold data
#
# It only creates/updates:
#   data/step8/step8_results.csv
#   data/step8/step8_qc_report.txt
#
# Resume-safe:
#   - SUCCESS rows are preserved
#   - RESTORED_SUCCESS rows are preserved
#   - QUOTA_BLOCKED rows are NOT treated as completed
#   - API_ERROR rows are NOT treated as completed
#   - S08-029 is intentionally re-evaluated once because the previous
#     successful response violated the domain anti-invention constraint.
#
# Model:
#   - Existing S08-001..012 are preserved
#   - Existing valid S08-013..028 and S08-030 are preserved
#   - S08-029 is re-evaluated with the hardened prompt
#
# =============================================================================


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
STEP8_DIR = BASE_DIR / "data" / "step8"

RESULTS_PATH = STEP8_DIR / "step8_results.csv"
QC_PATH = STEP8_DIR / "step8_qc_report.txt"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable not set."
    )

groq_client = Groq(api_key=GROQ_API_KEY)

MODEL = "openai/gpt-oss-120b"

INTER_CALL_DELAY = 1


# =============================================================================
# TARGETED RE-EVALUATION
# =============================================================================
#
# S08-029 previously produced:
#
#   DOMAIN: IN_SCOPE
#   LEGAL_QUERY: ...legally enforceable debt...dishonoured cheque...
#
# The citizen query contains no debt, loan, cheque, or Section 138 facts.
#
# Therefore S08-029 must be rerun after strengthening the prompt.
#
# No other completed query is intentionally rerun.
#
# =============================================================================

FORCE_REEVALUATE = {
    "S08-029",
}


STEP8_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# DOMAIN
# =============================================================================

DOMAIN_DESCRIPTION = """
Money-lending disputes (personal loans, informal debts,
promissory notes, repayment disputes) and cheque-bounce
disputes under Section 138 of the Negotiable Instruments Act,
India. Also includes legal-heir rights to recover a
deceased creditor's debt.
"""


# =============================================================================
# PROMPT
# =============================================================================

STEPBACK_PROMPT = """
You are a legal query translator for an Indian legal
research system.

Domain: {domain}

Citizen query: "{query}"

Your tasks:

1. Determine if this query is within the domain above.

CRITICAL DOMAIN-BOUNDARY RULE:

Classify the query ONLY from facts and legal subject matter
that are actually stated or clearly expressed in the citizen
query.

DO NOT invent, assume, infer, or import a factual context that
is not present in the citizen query.

In particular:

- Do NOT assume that an unspecified dispute concerns a debt.
- Do NOT assume that an unspecified dispute concerns a loan.
- Do NOT assume that an unspecified dispute concerns a cheque.
- Do NOT assume Section 138 of the Negotiable Instruments Act
  unless the citizen query itself mentions a cheque, cheque
  dishonour, cheque-bounce, Section 138, or an unmistakably
  equivalent cheque-related fact.
- Do NOT convert a general police, criminal, property, family,
  employment, consumer, tax, medical, accident, or other legal
  issue into an in-domain debt or cheque issue.
- Do NOT use the domain description as evidence that the
  citizen query belongs to the domain.
- The domain description defines the supported subject matter;
  it does NOT supply missing facts.

If the citizen query does not contain enough information to
establish that it concerns money-lending, debt recovery,
promissory notes, cheque-bounce/Section 138 matters, or
legal-heir recovery of a deceased creditor's debt, classify it
as OUT_OF_SCOPE.

If NOT in domain, respond with exactly:

DOMAIN: OUT_OF_SCOPE
LEGAL_QUERY: N/A

If IN domain, continue to task 2.

2. Translate the citizen query into a precise legal
question using proper Indian legal terminology.

For an IN_SCOPE query:

- Preserve the actual facts and subject matter stated by the
  citizen.
- You may generalize wording into appropriate legal terminology.
- You may clarify the legal issue implicit in the stated facts.
- You MUST NOT introduce new factual circumstances.
- You MUST NOT introduce a different legal subject merely
  because it is related to the supported domain.
- You MUST NOT add dates, amounts, parties, documents,
  transactions, statutory provisions, or procedural events
  that are not supported by the citizen query.
- Do not fabricate specific facts, dates, amounts, or outcomes.

Use terms such as these ONLY when supported by the citizen query:

- "legally enforceable debt"
- "dishonoured cheque under Section 138 NI Act"
- "legal heirs of deceased creditor"
- "recovery of principal amount"

FINAL DECISION RULE:

When there is uncertainty about whether the citizen query
belongs to the supported domain, do NOT force an IN_SCOPE
classification by supplying missing facts.

Instead, use OUT_OF_SCOPE unless the citizen query itself
establishes a connection to the supported domain.

Respond in EXACTLY this format (two lines only):

DOMAIN: IN_SCOPE or OUT_OF_SCOPE
LEGAL_QUERY: [your translated legal question or N/A]
"""


# =============================================================================
# EXACT 30-QUERY BENCHMARK
# =============================================================================

TEST_QUERIES = [

    # -------------------------------------------------------------------------
    # IN_SCOPE — 15
    # -------------------------------------------------------------------------

    {
        "id": "S08-001",
        "query": "my dad lent money to his friend 8 years ago and he is not paying back",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-002",
        "query": "cheque bounced for insufficient funds what can I do",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-003",
        "query": "father passed away he was owed money can our family still claim it",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-004",
        "query": "I gave someone a post dated cheque and now they are filing a case against me",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-005",
        "query": "friend borrowed 2 lakhs in 2019 gave promissory note not repaying",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-006",
        "query": "cheque dishonoured due to stop payment instruction by drawer",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-007",
        "query": "legal heir certificate for money recovery after death of creditor",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-008",
        "query": "I lent money with a written agreement they are refusing to pay back",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-009",
        "query": "section 138 NI Act case filed against me what are my defences",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-010",
        "query": "bank loan was given to friend on my guarantee now they want me to pay",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-011",
        "query": "recovery of business loan from partner who left the firm",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-012",
        "query": "cheque bounced multiple times for the same debt what happens",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-013",
        "query": "deceased uncle had given loan to someone can we as heirs file a case",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-014",
        "query": "promissory note signed 15 years ago can I still recover money",
        "expected": "IN_SCOPE",
    },

    {
        "id": "S08-015",
        "query": "person paid part of debt by cheque it bounced for balance amount",
        "expected": "IN_SCOPE",
    },


    # -------------------------------------------------------------------------
    # OUT_OF_SCOPE — 15
    # -------------------------------------------------------------------------

    {
        "id": "S08-016",
        "query": "landlord is not returning security deposit after I vacated",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-017",
        "query": "wife filed divorce case against me what should I do",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-018",
        "query": "accident compensation claim against truck driver",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-019",
        "query": "my employer is not paying my salary for last three months",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-020",
        "query": "property dispute between brothers after father died",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-021",
        "query": "criminal case filed against me for theft I did not commit",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-022",
        "query": "consumer complaint against mobile company for defective phone",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-023",
        "query": "custody of children after separation from spouse",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-024",
        "query": "builder not delivering flat on time what can I do",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-025",
        "query": "wrongful termination from government job",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-026",
        "query": "electricity bill dispute with power company",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-027",
        "query": "GST refund not received from income tax department",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-028",
        "query": "neighbour encroaching on my land boundary",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-029",
        "query": "police not registering my FIR what should I do",
        "expected": "OUT_OF_SCOPE",
    },

    {
        "id": "S08-030",
        "query": "medical negligence claim against hospital",
        "expected": "OUT_OF_SCOPE",
    },
]


# =============================================================================
# PREVIOUSLY VERIFIED RESULTS
# =============================================================================
#
# These results came from the earlier successful Step 8 Gemini run.
# They are preserved and NOT sent to Groq again.
#
# =============================================================================

RESTORED_RESULTS = {

    "S08-001": {
        "domain_result": "IN_SCOPE",
        "legal_query":
            "Whether a suit for the recovery of a personal loan or principal "
            "amount lent is maintainable after the lapse of several years "
            "under the Indian law of limitation governing legally enforceable debts?",
    },

    "S08-002": {
        "domain_result": "IN_SCOPE",
        "legal_query":
            "What are the legal remedies and procedure for the recovery of "
            "a debt based on an executed promissory note upon default in "
            "repayment by the borrower?",
    },

    "S08-003": {
        "domain_result": "IN_SCOPE",
        "legal_query":
            "What legal remedies and statutory procedures are available to "
            "a payee under Section 138 of the Negotiable Instruments Act, "
            "1881, upon the dishonour of a cheque due to insufficiency of funds?",
    },

    "S08-004": {
        "domain_result": "IN_SCOPE",
        "legal_query":
            "Whether prosecution under Section 138 of the Negotiable "
            "Instruments Act, 1881 is maintainable for the dishonour of a "
            "cheque issued towards the repayment of a loan in discharge of "
            "a legally enforceable debt or liability?",
    },

    "S08-005": {
        "domain_result": "IN_SCOPE",
        "legal_query":
            "Whether the legal heirs of a deceased creditor have the right "
            "to institute or maintain legal proceedings for the recovery "
            "of an outstanding debt owed to the deceased?",
    },

    "S08-006": {
        "domain_result": "IN_SCOPE",
        "legal_query":
            "What is the prescribed period of limitation for filing a suit "
            "for recovery of money lent, and whether a debt recovery action "
            "initiated eight years after the loan was disbursed is barred "
            "by limitation?",
    },

    "S08-007": {
        "domain_result": "IN_SCOPE",
        "legal_query":
            "What is the statutory procedure and legal remedy available "
            "to a payee under Section 138 of the Negotiable Instruments Act, "
            "1881, following the dishonour of a cheque issued in discharge "
            "of a legally enforceable debt or liability?",
    },

    "S08-008": {
        "domain_result": "IN_SCOPE",
        "legal_query":
            "Whether the dishonour of a post-dated cheque issued towards "
            "the repayment of a loan constitutes an offence under Section "
            "138 of the Negotiable Instruments Act, 1881 for the discharge "
            "of a legally enforceable debt or liability?",
    },

    "S08-009": {
        "domain_result": "IN_SCOPE",
        "legal_query":
            "What are the legal remedies available under Indian law for "
            "the recovery of a principal amount arising from a legally "
            "enforceable debt upon default of repayment by the borrower?",
    },

    "S08-010": {
        "domain_result": "IN_SCOPE",
        "legal_query":
            "Whether a debt constitutes a legally enforceable debt or "
            "liability under Section 138 of the Negotiable Instruments Act, "
            "1881, when the drawer disputes the underlying liability after "
            "issuing the cheque.",
    },

    "S08-011": {
        "domain_result": "IN_SCOPE",
        "legal_query":
            "What is the legal effect and statutory requirement of issuing "
            "a demand notice under Section 138 of the Negotiable Instruments "
            "Act, 1881, following the dishonour of a cheque?",
    },
}


# =============================================================================
# CSV FIELDNAMES
# =============================================================================

FIELDNAMES = [
    "query_id",
    "query",
    "expected",
    "api_status",
    "raw_response",
    "domain_result",
    "legal_query",
    "format_valid",
    "classification_correct",
    "latency_s",
    "error",
    "timestamp",
    "result_origin",
]


# =============================================================================
# LOAD EXISTING RESULTS
# =============================================================================

def load_existing_results():

    existing = {}

    if not RESULTS_PATH.exists():
        return existing

    with open(
        RESULTS_PATH,
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            qid = row.get("query_id")

            if not qid:
                continue

            status = row.get("api_status", "")

            if status in ("SUCCESS", "RESTORED_SUCCESS"):

                # -----------------------------------------------------------------
                # Targeted invalidation:
                #
                # S08-029 previously passed format but failed classification
                # because the model invented a debt/cheque context.
                #
                # It must NOT be treated as completed.
                # -----------------------------------------------------------------
                if qid in FORCE_REEVALUATE:
                    continue

                existing[qid] = row

    return existing


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def parse_response(text):

    domain = None
    legal_query = None

    if not text:
        return domain, legal_query

    lines = text.strip().splitlines()

    for line in lines:

        if line.startswith("DOMAIN:"):

            domain = line.replace(
                "DOMAIN:",
                "",
                1
            ).strip()

        elif line.startswith("LEGAL_QUERY:"):

            legal_query = line.replace(
                "LEGAL_QUERY:",
                "",
                1
            ).strip()

    return domain, legal_query


# =============================================================================
# FORMAT VALIDATION
# =============================================================================

def validate_format(text, domain, legal_query):

    if not text:
        return False

    lines = [
        line.strip()
        for line in text.strip().splitlines()
        if line.strip()
    ]

    if len(lines) != 2:
        return False

    if not lines[0].startswith("DOMAIN:"):
        return False

    if not lines[1].startswith("LEGAL_QUERY:"):
        return False

    if domain not in (
        "IN_SCOPE",
        "OUT_OF_SCOPE"
    ):
        return False

    if legal_query is None:
        return False

    if domain == "OUT_OF_SCOPE":

        if legal_query != "N/A":
            return False

    else:

        if not legal_query.strip():
            return False

    return True


# =============================================================================
# GROQ CALL
# =============================================================================

def call_groq(query_text):

    prompt = STEPBACK_PROMPT.format(
        domain=DOMAIN_DESCRIPTION,
        query=query_text
    )

    t0 = time.time()

    try:

        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            reasoning_effort="low",
            max_completion_tokens=500
        )

        latency = round(
            time.time() - t0,
            3
        )

        content = response.choices[0].message.content

        if not content or not content.strip():

            return (
                "API_ERROR",
                "",
                latency,
                "Model returned empty content."
            )

        text = content.strip()

        return (
            "SUCCESS",
            text,
            latency,
            ""
        )

    except Exception as exc:

        latency = round(
            time.time() - t0,
            3
        )

        error = str(exc)

        if (
            "429" in error
            or "rate_limit" in error.lower()
            or "rate limit" in error.lower()
            or "quota" in error.lower()
        ):

            return (
                "QUOTA_BLOCKED",
                "",
                latency,
                error
            )

        return (
            "API_ERROR",
            "",
            latency,
            error
        )


# =============================================================================
# RESTORE PREVIOUSLY VERIFIED RESULTS
# =============================================================================

def build_restored_rows(existing):

    rows = []

    now = datetime.utcnow().isoformat()

    for item in TEST_QUERIES:

        qid = item["id"]

        if qid not in RESTORED_RESULTS:
            continue

        if qid in existing:
            continue

        restored = RESTORED_RESULTS[qid]

        raw_response = (
            "DOMAIN: "
            + restored["domain_result"]
            + "\nLEGAL_QUERY: "
            + restored["legal_query"]
        )

        format_valid = validate_format(
            raw_response,
            restored["domain_result"],
            restored["legal_query"]
        )

        classification_correct = (
            format_valid
            and restored["domain_result"]
            == item["expected"]
        )

        row = {
            "query_id": qid,
            "query": item["query"],
            "expected": item["expected"],
            "api_status": "RESTORED_SUCCESS",
            "raw_response": raw_response,
            "domain_result": restored["domain_result"],
            "legal_query": restored["legal_query"],
            "format_valid": str(format_valid),
            "classification_correct":
                str(classification_correct),
            "latency_s": "",
            "error": "",
            "timestamp": now,
            "result_origin": "PREVIOUS_VERIFIED_RUN",
        }

        rows.append(row)

    return rows


# =============================================================================
# WRITE COMPLETE RESULT FILE
# =============================================================================

def write_results(rows):

    by_id = {}

    for row in rows:
        by_id[row["query_id"]] = row

    ordered = []

    for item in TEST_QUERIES:

        qid = item["id"]

        if qid in by_id:
            ordered.append(by_id[qid])

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()

        for row in ordered:
            writer.writerow(row)


# =============================================================================
# MAIN
# =============================================================================

def run_evaluation():

    print("=" * 90)
    print("PRAMANA — STEP 8 STEP-BACK + DOMAIN CHECK")
    print("=" * 90)

    print()
    print("READ ONLY")
    print("V14 WILL NOT BE MODIFIED")
    print("STEP 4 WILL NOT BE MODIFIED")
    print("STEP 5 WILL NOT BE MODIFIED")
    print("STEP 6 WILL NOT BE MODIFIED")
    print("CHROMADB WILL NOT BE MODIFIED")
    print("STEP 7 WILL NOT BE MODIFIED")

    print()
    print("=" * 90)
    print("CONFIGURATION")
    print("=" * 90)

    print("Provider                    : Groq")
    print(f"Model                       : {MODEL}")
    print(f"Results                     : {RESULTS_PATH}")
    print(f"Benchmark queries           : {len(TEST_QUERIES)}")
    print(
        "Targeted re-evaluation     : "
        + ", ".join(sorted(FORCE_REEVALUATE))
    )

    existing = load_existing_results()

    # Restore the previously verified Gemini results only if necessary.
    restored_rows = build_restored_rows(existing)

    for row in restored_rows:

        existing[row["query_id"]] = row

        print(
            f"RESTORE {row['query_id']} "
            f"from previous verified Gemini run"
        )

    all_rows = list(existing.values())

    write_results(all_rows)

    print()
    print(
        f"Completed/restored before Groq calls : "
        f"{len(existing)}"
    )

    # -------------------------------------------------------------------------
    # API EVALUATION
    # -------------------------------------------------------------------------

    for item in TEST_QUERIES:

        qid = item["id"]

        if qid in existing:

            print(
                f"{qid}: already completed — skipping"
            )

            continue

        if qid in FORCE_REEVALUATE:

            print(
                f"{qid}: targeted re-evaluation "
                f"with hardened domain-boundary prompt..."
            )

        else:

            print(
                f"{qid}: calling Groq ({MODEL})..."
            )

        api_status, raw, latency, error = call_groq(
            item["query"]
        )

        row = {
            "query_id": qid,
            "query": item["query"],
            "expected": item["expected"],
            "api_status": api_status,
            "raw_response": raw or "",
            "domain_result": "",
            "legal_query": "",
            "format_valid": "False",
            "classification_correct": "False",
            "latency_s":
                "" if latency is None else str(latency),
            "error": error or "",
            "timestamp":
                datetime.utcnow().isoformat(),
            "result_origin": "CURRENT_GROQ_RUN",
        }

        if api_status == "SUCCESS":

            domain, legal_query = parse_response(raw)

            format_valid = validate_format(
                raw,
                domain,
                legal_query
            )

            classification_correct = (
                format_valid
                and domain == item["expected"]
            )

            row["domain_result"] = (
                domain or ""
            )

            row["legal_query"] = (
                legal_query or ""
            )

            row["format_valid"] = str(
                format_valid
            )

            row["classification_correct"] = str(
                classification_correct
            )

            print(
                f"    DOMAIN={domain} "
                f"FORMAT={'PASS' if format_valid else 'FAIL'} "
                f"CLASS={'PASS' if classification_correct else 'FAIL'} "
                f"LATENCY={latency}s"
            )

            # Only completed API results are added to existing.
            existing[qid] = row

            # Persist immediately.
            all_rows = list(existing.values())
            write_results(all_rows)

            time.sleep(INTER_CALL_DELAY)

        elif api_status == "QUOTA_BLOCKED":

            print(
                "    QUOTA_BLOCKED — stopping API work "
                "for this run."
            )

            all_rows = list(existing.values()) + [row]
            write_results(all_rows)

            print()
            print(
                "Groq rate limit/quota block encountered."
            )
            print(
                "Stop this run and retry later."
            )

            break

        else:

            print(
                f"    API_ERROR — {error[:120]}"
            )

            all_rows = list(existing.values()) + [row]
            write_results(all_rows)

            continue

    # -------------------------------------------------------------------------
    # FINAL QC
    # -------------------------------------------------------------------------

    final_rows = load_all_rows()

    total = len(TEST_QUERIES)

    successful = [
        r for r in final_rows
        if r["api_status"]
        in ("SUCCESS", "RESTORED_SUCCESS")
    ]

    quota_blocked = [
        r for r in final_rows
        if r["api_status"] == "QUOTA_BLOCKED"
    ]

    api_errors = [
        r for r in final_rows
        if r["api_status"] == "API_ERROR"
    ]

    evaluated = len(successful)

    format_pass = sum(
        str(r["format_valid"]).lower() == "true"
        for r in successful
    )

    classification_pass = sum(
        str(r["classification_correct"]).lower() == "true"
        for r in successful
    )

    in_scope = [
        r for r in successful
        if r["expected"] == "IN_SCOPE"
    ]

    out_scope = [
        r for r in successful
        if r["expected"] == "OUT_OF_SCOPE"
    ]

    in_correct = sum(
        str(r["classification_correct"]).lower() == "true"
        for r in in_scope
    )

    out_correct = sum(
        str(r["classification_correct"]).lower() == "true"
        for r in out_scope
    )

    all_30_evaluated = evaluated == total

    all_format_valid = format_pass == evaluated

    all_classification_correct = (
        classification_pass == evaluated
    )

    no_quota_blocks = len(quota_blocked) == 0

    no_api_errors = len(api_errors) == 0

    final_pass = (
        all_30_evaluated
        and all_format_valid
        and all_classification_correct
        and no_quota_blocks
        and no_api_errors
    )

    # -------------------------------------------------------------------------
    # QC REPORT
    # -------------------------------------------------------------------------

    report = []

    report.append("=" * 90)
    report.append("PRAMANA — STEP 8 FINAL QC")
    report.append("=" * 90)
    report.append("")

    report.append("READ ONLY")
    report.append("V14 modified                 : NO")
    report.append("Step 4 modified              : NO")
    report.append("Step 5 modified              : NO")
    report.append("Step 6 modified              : NO")
    report.append("ChromaDB modified            : NO")
    report.append("Step 7 modified              : NO")

    report.append("")
    report.append("MODEL")
    report.append("-" * 90)
    report.append("Provider                     : Groq")
    report.append("Model                        : " + MODEL)
    report.append("Existing Gemini results      : S08-001..012")
    report.append("Groq evaluated results       : S08-013..030")
    report.append(
        "Targeted re-evaluation      : "
        + ", ".join(sorted(FORCE_REEVALUATE))
    )

    report.append("")
    report.append("DOMAIN SAFETY")
    report.append("-" * 90)
    report.append(
        "Hard anti-invention rule    : ENABLED"
    )
    report.append(
        "Classification basis        : Citizen query only"
    )
    report.append(
        "Missing facts may be inferred: NO"
    )
    report.append(
        "Unsupported context injection: PROHIBITED"
    )

    report.append("")
    report.append("DATASET")
    report.append("-" * 90)

    report.append(
        f"Benchmark queries            : {total}"
    )

    report.append(
        f"Successfully evaluated       : {evaluated}"
    )

    report.append(
        f"Quota blocked                : {len(quota_blocked)}"
    )

    report.append(
        f"API errors                   : {len(api_errors)}"
    )

    report.append(
        f"Format valid                 : "
        f"{format_pass}/{evaluated}"
    )

    report.append(
        f"Classification correct       : "
        f"{classification_pass}/{evaluated}"
    )

    report.append(
        f"IN_SCOPE correct             : "
        f"{in_correct}/{len(in_scope)}"
    )

    report.append(
        f"OUT_OF_SCOPE correct         : "
        f"{out_correct}/{len(out_scope)}"
    )

    report.append("")
    report.append("GATES")
    report.append("-" * 90)

    report.append(
        f"all_30_evaluated             : "
        f"{'PASS' if all_30_evaluated else 'FAIL'}"
    )

    report.append(
        f"all_format_valid             : "
        f"{'PASS' if all_format_valid else 'FAIL'}"
    )

    report.append(
        f"all_classification_correct   : "
        f"{'PASS' if all_classification_correct else 'FAIL'}"
    )

    report.append(
        f"no_quota_blocks              : "
        f"{'PASS' if no_quota_blocks else 'FAIL'}"
    )

    report.append(
        f"no_api_errors                : "
        f"{'PASS' if no_api_errors else 'FAIL'}"
    )

    report.append("")

    report.append(
        f"FINAL STATUS                 : "
        f"{'PASS' if final_pass else 'FAIL'}"
    )

    report.append("")

    report.append(
        "Generated                   : "
        + datetime.utcnow().isoformat()
    )

    report_text = "\n".join(report)

    QC_PATH.write_text(
        report_text,
        encoding="utf-8"
    )

    print()
    print(report_text)


# =============================================================================
# LOAD ALL ROWS
# =============================================================================

def load_all_rows():

    rows = []

    if not RESULTS_PATH.exists():
        return rows

    with open(
        RESULTS_PATH,
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_evaluation()