"""
PRAMANA — LIVE SEARCH API

Live pipeline:

    Citizen query
        ↓
    Step 8 — Step-Back + domain check
        ↓
    Step 9 — HyDE
        ↓
    Step 10 — BGE + ChromaDB retrieval
        ↓
    Supabase — verified case metadata

IMPORTANT:
- Steps 8, 9 and 10 are NOT modified.
- Benchmark main() functions are never called.
- Step 11 is never used as Step 8.
- Supabase service-role key remains server-side.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

TOP_K = 5
EXPANDED_TOP_K = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("pramana-api")


# ============================================================================
# ENVIRONMENT
# ============================================================================

def load_dotenv() -> None:
    env_path = BASE_DIR / ".env"

    if not env_path.exists():
        logger.warning("[ENV] .env not found: %s", env_path)
        return

    try:
        for raw in env_path.read_text(
            encoding="utf-8"
        ).splitlines():

            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = value.strip()

            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]

            if key and key not in os.environ:
                os.environ[key] = value

    except Exception:
        logger.exception("[ENV] Failed loading .env")


load_dotenv()


def env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return ""


SUPABASE_URL = env(
    "SUPABASE_URL",
    "supabase_url",
).rstrip("/")

SUPABASE_SERVICE_KEY = env(
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_SERVICE_KEY",
    "supabase_service_key",
    "supabase_service_role_key",
)


# ============================================================================
# FASTAPI
# ============================================================================

app = FastAPI(
    title="PRAMANA API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST MODEL
# ============================================================================

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )


# ============================================================================
# MODULE LOADING
# ============================================================================

def load_exact_pipeline_module(
    label: str,
    required_functions: list[str],
    preferred_names: list[str],
):
    """
    Load the correct pipeline module.

    CRITICAL:
    A file is accepted only when it contains ALL functions required
    for that pipeline stage.

    This prevents files such as:
        step11_hyde_comparison.py

    from being mistaken for Step 8 merely because they contain
    call_groq().
    """

    candidates: list[Path] = []

    # ------------------------------------------------------------
    # 1. Preferred filenames first
    # ------------------------------------------------------------

    for filename in preferred_names:

        path = BASE_DIR / filename

        if (
            path.exists()
            and path.is_file()
            and path.suffix == ".py"
            and path.name != Path(__file__).name
        ):
            if path not in candidates:
                candidates.append(path)

    # ------------------------------------------------------------
    # 2. Scan remaining Python files
    # ------------------------------------------------------------

    for path in sorted(
        BASE_DIR.glob("*.py")
    ):

        if path.name == Path(__file__).name:
            continue

        if path not in candidates:
            candidates.append(path)

    # ------------------------------------------------------------
    # 3. Require EVERY function
    # ------------------------------------------------------------

    matches: list[Path] = []

    for path in candidates:

        try:
            source = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        except Exception:
            continue

        if all(
            f"def {function}" in source
            for function in required_functions
        ):
            matches.append(path)

    if not matches:
        raise RuntimeError(
            f"Could not locate {label} module.\n"
            f"Required functions: "
            f"{required_functions}\n"
            f"Project directory: {BASE_DIR}"
        )

    # ------------------------------------------------------------
    # 4. There should be one real implementation.
    # ------------------------------------------------------------

    if len(matches) > 1:

        # Prefer a conventional exact filename.
        preferred_matches = [
            path
            for path in matches
            if path.name in preferred_names
        ]

        if len(preferred_matches) == 1:
            selected = preferred_matches[0]

        else:
            names = ", ".join(
                path.name
                for path in matches
            )

            raise RuntimeError(
                f"Multiple possible {label} modules found: "
                f"{names}\n"
                f"Refusing to guess."
            )

    else:
        selected = matches[0]

    # ------------------------------------------------------------
    # 5. Import selected module
    # ------------------------------------------------------------

    module_name = (
        f"_pramana_live_{label.lower()}_"
        f"{selected.stem}"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        selected,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create import specification for "
            f"{selected}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)

    except Exception as exc:
        raise RuntimeError(
            f"Failed importing {label} module "
            f"{selected.name}: {exc}"
        ) from exc

    # Final verification after import.
    missing = [
        function
        for function in required_functions
        if not hasattr(module, function)
    ]

    if missing:
        raise RuntimeError(
            f"Loaded {label} module {selected.name}, "
            f"but it is missing: {missing}"
        )

    logger.info(
        "[MODULE] %s → %s",
        label,
        selected.name,
    )

    return module


# ============================================================================
# PIPELINE STATE
# ============================================================================

step8_module = None
step9_module = None
step10_module = None

bge_model = None
chroma_client = None
chroma_collection = None


def initialize_pipeline() -> None:

    global step8_module
    global step9_module
    global step10_module
    global bge_model
    global chroma_client
    global chroma_collection

    if (
        step8_module is not None
        and step9_module is not None
        and step10_module is not None
        and bge_model is not None
        and chroma_collection is not None
    ):
        return

    logger.info("=" * 72)
    logger.info("[INIT] Loading PRAMANA live pipeline")
    logger.info("=" * 72)

    # ================================================================
    # STEP 8
    #
    # Requiring ALL THREE functions is the critical fix.
    # ================================================================

    step8_module = load_exact_pipeline_module(
        label="STEP8",
        required_functions=[
            "call_groq",
            "parse_response",
            "validate_format",
        ],
        preferred_names=[
            "step8.py",
            "step_8.py",
            "step8_live.py",
        ],
    )

    # ================================================================
    # STEP 9
    # ================================================================

    step9_module = load_exact_pipeline_module(
        label="STEP9",
        required_functions=[
            "generate_valid_hyde",
        ],
        preferred_names=[
            "step9.py",
            "step_9.py",
            "step9_hyde.py",
        ],
    )

    # ================================================================
    # STEP 10
    # ================================================================

    step10_module = load_exact_pipeline_module(
        label="STEP10",
        required_functions=[
            "load_chroma",
            "load_bge",
            "retrieve",
            "normalize_results",
        ],
        preferred_names=[
            "step10.py",
            "step_10.py",
        ],
    )

    # ================================================================
    # CHROMA
    # ================================================================

    logger.info(
        "[INIT] Loading ChromaDB through Step 10."
    )

    # Step 10's actual contract is:
    #
    #     return client, collection
    #
    chroma_client, chroma_collection = (
        step10_module.load_chroma()
    )

    if chroma_collection is None:
        raise RuntimeError(
            "Step 10 load_chroma() returned no collection."
        )

    logger.info(
        "[INIT] Chroma collection: %s",
        chroma_collection.name,
    )

    # ================================================================
    # BGE
    # ================================================================

    logger.info(
        "[INIT] Loading BGE through Step 10."
    )

    bge_model = step10_module.load_bge()

    if bge_model is None:
        raise RuntimeError(
            "Step 10 load_bge() returned no model."
        )

    logger.info(
        "[INIT] PRAMANA live pipeline ready."
    )

    logger.info("=" * 72)


# ============================================================================
# STEP 8
# ============================================================================

def run_step8(
    citizen_query: str,
) -> tuple[str, str]:

    logger.info(
        "[STEP 8] Starting Step-Back + domain check."
    )

    returned = step8_module.call_groq(
        citizen_query
    )

    # Actual Step 8 contract:
    #
    # (status, text, latency, error)
    #

    if not isinstance(returned, tuple):
        raise RuntimeError(
            "Step 8 call_groq() returned "
            f"{type(returned).__name__}; "
            "expected tuple."
        )

    if len(returned) != 4:
        raise RuntimeError(
            "Step 8 call_groq() returned "
            f"{len(returned)} values; expected 4."
        )

    status, text, latency, error = returned

    logger.info(
        "[STEP 8] API status=%s latency=%s",
        status,
        latency,
    )

    if status != "SUCCESS":
        raise RuntimeError(
            "Step 8 Groq call failed: "
            f"{error or status}"
        )

    if not isinstance(text, str):
        raise RuntimeError(
            "Step 8 returned non-string response text."
        )

    text = text.strip()

    if not text:
        raise RuntimeError(
            "Step 8 returned empty response text."
        )

    # ================================================================
    # ACTUAL STEP 8 CONTRACT
    #
    # parse_response(text)
    #
    # returns:
    #
    #     domain, legal_query
    #
    # ================================================================

    parsed = step8_module.parse_response(
        text
    )

    if not isinstance(parsed, tuple):
        raise RuntimeError(
            "Step 8 parse_response() returned "
            f"{type(parsed).__name__}; expected tuple."
        )

    if len(parsed) != 2:
        raise RuntimeError(
            "Step 8 parse_response() returned "
            f"{len(parsed)} values; expected 2."
        )

    domain, legal_query = parsed

    domain = (
        str(domain).strip()
        if domain is not None
        else ""
    )

    legal_query = (
        str(legal_query).strip()
        if legal_query is not None
        else ""
    )

    # Use Step 8's own validator.
    valid = step8_module.validate_format(
        text,
        domain,
        legal_query,
    )

    if not valid:
        raise RuntimeError(
            "Step 8 response failed its own "
            "validate_format() check."
        )

    logger.info(
        "[STEP 8] Domain=%s",
        domain,
    )

    if domain != "IN_SCOPE":

        raise HTTPException(
            status_code=422,
            detail=(
                "This legal problem is outside "
                "PRAMANA's supported domain."
            ),
        )

    if not legal_query:
        raise RuntimeError(
            "Step 8 generated an empty legal query."
        )

    logger.info(
        "[STEP 8] Legal query generated successfully."
    )

    return legal_query, text


# ============================================================================
# STEP 9
# ============================================================================

def run_step9(
    citizen_query: str,
    legal_query: str,
) -> str:

    logger.info(
        "[STEP 9] Starting HyDE generation."
    )

    # The Step 9 implementation supports a LIVE-ONLY fallback for a
    # HyDE that is outside the strict benchmark range but still passes
    # all substantive validation gates. This does NOT change the locked
    # Step 9 benchmark rules; it only allows the live search API to use
    # the explicitly marked LIVE_FALLBACK result.
    try:
        result = step9_module.generate_valid_hyde(
            citizen_query,
            legal_query,
            allow_live_fallback=True,
        )
    except TypeError as exc:
        # Fail clearly if the loaded Step 9 module does not expose the
        # live-fallback contract expected by this API. Do not silently
        # fall back to an unvalidated HyDE.
        raise RuntimeError(
            "Step 9 module does not support the required live-fallback "
            "contract (allow_live_fallback=True). Update the Step 9 module "
            "before starting the live API."
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            "Step 9 generate_valid_hyde() returned "
            f"{type(result).__name__}; expected dict."
        )

    api_status = str(
        result.get("api_status", "")
    ).strip()

    if api_status not in {"SUCCESS", "LIVE_FALLBACK"}:
        raise RuntimeError(
            "Step 9 failed: "
            f"{result.get('error') or api_status}"
        )

    if api_status == "LIVE_FALLBACK":
        logger.warning(
            "[STEP 9] LIVE_FALLBACK accepted for retrieval. "
            "This is not a benchmark PASS: %s",
            result.get("error", ""),
        )

    hyde_text = str(
        result.get("hyde_text", "")
    ).strip()

    if not hyde_text:
        raise RuntimeError(
            "Step 9 returned empty HyDE text."
        )

    validation = result.get(
        "validation"
    )

    if isinstance(validation, dict):

        if validation.get("pass") is False:
            raise RuntimeError(
                "Step 9 HyDE validation failed: "
                f"{validation.get('reason', '')}"
            )

    logger.info(
        "[STEP 9] Valid HyDE generated."
    )

    return hyde_text


# ============================================================================
# STEP 10
# ============================================================================

def unique_cases(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    seen: set[str] = set()
    output: list[dict[str, Any]] = []

    for record in records:

        case_id = str(
            record.get("case_id", "")
        ).strip()

        if not case_id:
            continue

        if case_id in seen:
            continue

        seen.add(case_id)
        output.append(record)

    return output


def run_step10(
    hyde_text: str,
    retrieval_k: int = TOP_K,
) -> list[dict[str, Any]]:

    logger.info(
        "[STEP 10] Starting ChromaDB retrieval (top_k=%d).",
        retrieval_k,
    )

    _embedding, raw_results = (
        step10_module.retrieve(
            bge_model,
            chroma_collection,
            hyde_text,
            top_k=retrieval_k,
        )
    )

    if not isinstance(raw_results, dict):
        raise RuntimeError(
            "Step 10 retrieve() returned "
            f"{type(raw_results).__name__}; "
            "expected dict."
        )

    normalized = step10_module.normalize_results(raw_results)

    if not isinstance(normalized, list):
        raise RuntimeError(
            "Step 10 normalize_results() did not return a list."
        )

    cases = unique_cases(normalized)

    logger.info(
        "[STEP 10] Unique cases from %d retrieved chunks: %d",
        retrieval_k,
        len(cases),
    )

    # For the normal call, return the top five. For the explicit
    # expanded call, return the full expanded candidate set so the
    # Supabase layer can skip missing case rows and still fill up to
    # five results while preserving Chroma ranking.
    limit = TOP_K if retrieval_k == TOP_K else retrieval_k
    selected = cases[:limit]

    if not selected:
        raise RuntimeError(
            "Step 10 returned no usable cases."
        )

    for rank, case in enumerate(selected, start=1):
        logger.info(
            "[STEP 10] Rank %d → %s",
            rank,
            case.get("case_id"),
        )

    return selected


# ============================================================================
# SUPABASE
# ============================================================================

def require_supabase() -> None:

    if not SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is missing from .env."
        )

    if not SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "Supabase service-role key is missing "
            "from .env."
        )


def supabase_headers() -> dict[str, str]:

    require_supabase()

    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": (
            f"Bearer {SUPABASE_SERVICE_KEY}"
        ),
        "Content-Type": "application/json",
    }


def as_int(value: Any) -> int:

    if value is None:
        return 0

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def fetch_result_cases(
    retrieved: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if not retrieved:
        return []

    case_ids: list[str] = []

    for row in retrieved:

        case_id = str(
            row.get("case_id", "")
        ).strip()

        if (
            case_id
            and case_id not in case_ids
        ):
            case_ids.append(case_id)

    if not case_ids:
        raise RuntimeError(
            "Step 10 returned no valid case IDs."
        )

    # Supabase/PostgREST `in` filter.
    encoded = ",".join(
        f'"{case_id}"'
        for case_id in case_ids
    )

    url = (
        f"{SUPABASE_URL}/rest/v1/cases"
        f"?select=case_id,title,"
        f"verified_count,needs_review_count,total_sentences"
        f"&case_id=in.({encoded})"
    )

    logger.info(
        "[SUPABASE] Fetching %d cases.",
        len(case_ids),
    )

    response = requests.get(
        url,
        headers=supabase_headers(),
        timeout=20,
    )

    if not response.ok:
        raise RuntimeError(
            "Supabase query failed: "
            f"HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    rows = response.json()

    if not isinstance(rows, list):
        raise RuntimeError(
            "Supabase returned a non-list response."
        )

    by_id = {
        str(row.get("case_id", "")).strip(): row
        for row in rows
    }

    output: list[dict[str, Any]] = []

    # Preserve Step 10 ranking.
    for retrieved_row in retrieved:

        case_id = str(
            retrieved_row.get("case_id", "")
        ).strip()

        row = by_id.get(case_id)

        if row is None:
            logger.warning(
                "[SUPABASE] Case not found: %s",
                case_id,
            )
            continue

        title = str(
            row.get("title", "")
        ).strip()

        if not title:
            continue

        output.append(
            {
                "case_id": case_id,
                "title": title,
                "verified_count": as_int(
                    row.get("verified_count")
                ),
                "needs_review_count": as_int(
                    row.get("needs_review_count")
                ),
                "total_sentences": as_int(
                    row.get("total_sentences")
                ),
            }
        )

        if len(output) == TOP_K:
            break

    if not output:
        raise RuntimeError(
            "None of the retrieved cases were found "
            "in Supabase."
        )

    logger.info(
        "[SUPABASE] Final result cases: %d",
        len(output),
    )

    return output


# ============================================================================
# LIVE SEARCH PIPELINE
# ============================================================================

def perform_live_search(
    citizen_query: str,
) -> list[dict[str, Any]]:

    logger.info("=" * 72)
    logger.info(
        "[SEARCH] New live query: %s",
        citizen_query,
    )
    logger.info("=" * 72)

    # ------------------------------------------------------------
    # STEP 8
    # ------------------------------------------------------------
    legal_query, _step8_text = run_step8(citizen_query)

    # ------------------------------------------------------------
    # STEP 9
    # ------------------------------------------------------------
    hyde_text = run_step9(citizen_query, legal_query)

    # ------------------------------------------------------------
    # STEP 10 + SUPABASE
    # ------------------------------------------------------------
    # First use the normal top-5 retrieval. If one or more of those
    # candidates has no corresponding authoritative Supabase case row,
    # expand the SAME Chroma retrieval to 30 candidates and keep the
    # highest-ranked candidates that actually exist in Supabase.
    #
    # This does not fabricate a case and does not replace semantic
    # retrieval with keyword search. It only prevents a missing DB row
    # from turning five valid retrieval slots into four UI results.
    retrieved = run_step10(hyde_text, retrieval_k=TOP_K)
    cases = fetch_result_cases(retrieved)

    if len(cases) < TOP_K:
        logger.warning(
            "[SEARCH] Only %d/%d retrieved cases have Supabase metadata; "
            "expanding Chroma retrieval to %d candidates.",
            len(cases),
            TOP_K,
            EXPANDED_TOP_K,
        )

        expanded_retrieved = run_step10(
            hyde_text,
            retrieval_k=EXPANDED_TOP_K,
        )
        cases = fetch_result_cases(expanded_retrieved)

    if len(cases) < TOP_K:
        logger.warning(
            "[SEARCH] Final result count is %d/%d because fewer than "
            "%d retrieved candidates have authoritative Supabase rows.",
            len(cases),
            TOP_K,
            TOP_K,
        )

    return cases


# ============================================================================
# CASE DETAILS
# ============================================================================

def supabase_get(table: str, params: dict[str, str]) -> list[dict[str, Any]]:
    """Read-only PostgREST helper used only by the Case Details endpoint."""
    require_supabase()
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        params=params,
        headers=supabase_headers(),
        timeout=20,
    )
    if not response.ok:
        raise RuntimeError(
            f"Supabase {table} lookup failed: HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    rows = response.json()
    if not isinstance(rows, list):
        raise RuntimeError(f"Supabase {table} returned a non-list response.")
    return rows


def _norm_source(text: Any) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _find_source_chunk(best_source: str, chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    needle = _norm_source(best_source)
    if not needle:
        return None
    for chunk in chunks:
        haystack = _norm_source(chunk.get("chunk_text"))
        if needle and (needle in haystack or needle[:100] in haystack):
            return chunk
    return None


def fetch_case_details(case_id: str) -> dict[str, Any]:
    """Return the existing case summary plus read-only, non-blocked proof records."""
    case_id = case_id.strip()
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id is required.")

    rows = supabase_get(
        "cases",
        {
            "select": "case_id,title,story,evidence,decision_and_meaning,why_court_decided,verified_count,needs_review_count,blocked_count,total_sentences",
            "case_id": f"eq.{case_id}",
            "limit": "1",
        },
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Case not found.")
    row = rows[0]

    claims = supabase_get(
        "claims",
        {
            "select": "sentence_id,case_id,field_name,sentence_text,sentence_order",
            "case_id": f"eq.{case_id}",
            "order": "sentence_order.asc",
            "limit": "1000",
        },
    )
    sentence_ids = [str(item.get("sentence_id", "")) for item in claims if item.get("sentence_id")]
    if not sentence_ids:
        visible_claims: list[dict[str, Any]] = []
    else:
        encoded_ids = ",".join(f'"{sid}"' for sid in sentence_ids)
        gate1_rows = supabase_get(
            "gate1_results",
            {
                "select": "sentence_id,best_source_sentence,retrieval_status",
                "sentence_id": f"in.({encoded_ids})",
                "limit": "1000",
            },
        )
        gate2_rows = supabase_get(
            "gate2_results",
            {
                "select": "sentence_id,verdict,blocked_by",
                "sentence_id": f"in.({encoded_ids})",
                "limit": "1000",
            },
        )
        chunks = supabase_get(
            "source_chunks",
            {
                "select": "chunk_id,chunk_text,paragraph_num,source_name",
                "case_id": f"eq.{case_id}",
                "limit": "1000",
            },
        )
        gate1_by_id = {str(item.get("sentence_id")): item for item in gate1_rows}
        gate2_by_id = {str(item.get("sentence_id")): item for item in gate2_rows}
        visible_claims = []
        for claim in claims:
            sid = str(claim.get("sentence_id", ""))
            g2 = gate2_by_id.get(sid, {})
            verdict = str(g2.get("verdict", "")).upper()
            if verdict == "BLOCKED":
                continue
            if verdict not in {"VERIFIED", "NEEDS_REVIEW"}:
                continue
            g1 = gate1_by_id.get(sid, {})
            best_source = str(g1.get("best_source_sentence") or "").strip()
            chunk = _find_source_chunk(best_source, chunks)
            visible_claims.append({
                "sentence_id": sid,
                "field_name": str(claim.get("field_name", "")),
                "sentence_text": str(claim.get("sentence_text", "")),
                "sentence_order": int(claim.get("sentence_order") or 0),
                "verdict": verdict,
                "original_statement": best_source,
                "source_chunk_id": str(chunk.get("chunk_id")) if chunk else "",
                "source_name": str((chunk or {}).get("source_name") or ""),
                "paragraph_num": (chunk or {}).get("paragraph_num"),
            })

    return {
        "case_id": str(row.get("case_id", "")),
        "title": str(row.get("title", "")),
        "story": str(row.get("story", "")),
        "evidence": str(row.get("evidence", "")),
        "decision_and_meaning": str(row.get("decision_and_meaning", "")),
        "why_court_decided": str(row.get("why_court_decided", "")),
        "verified_count": int(row.get("verified_count") or 0),
        "needs_review_count": int(row.get("needs_review_count") or 0),
        "blocked_count": int(row.get("blocked_count") or 0),
        "total_sentences": int(row.get("total_sentences") or 0),
        "claims": visible_claims,
    }


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/health")
def health() -> dict[str, str]:

    return {
        "status": "ok",
        "service": "PRAMANA API",
        "pipeline": (
            "Step 8 → Step 9 → Step 10 → Supabase"
        ),
    }


@app.post("/search")
def search(
    request: SearchRequest,
) -> dict[str, Any]:

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:

        initialize_pipeline()

        cases = perform_live_search(
            query
        )

        return {
            "query": query,
            "cases": cases,
        }

    except HTTPException:
        raise

    except Exception as exc:

        logger.error(
            "[SEARCH] LIVE SEARCH FAILED"
        )

        logger.error(
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "PRAMANA search failed. "
                "Check the API terminal for the "
                "exact failing stage."
            ),
        ) from exc


@app.get("/cases/{case_id}")
def get_case(
    case_id: str,
) -> dict[str, Any]:

    try:

        initialize_pipeline()

        return fetch_case_details(
            case_id
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.error(
            "[CASE] CASE LOOKUP FAILED"
        )

        logger.error(
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail="PRAMANA case lookup failed.",
        ) from exc


# ============================================================================
# SERVER
# ============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )