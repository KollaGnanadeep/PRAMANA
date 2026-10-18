/**
 * PRAMANA — Frontend API Layer
 *
 * Thin fetch() wrappers for the FastAPI backend running at http://127.0.0.1:8000
 * No external dependencies — pure fetch().
 */

const BASE_URL = 'http://127.0.0.1:8000';

// ─────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────

export class ApiError extends Error {
    status: number;
    detail: string;
    constructor(status: number, message: string, detail: string = '') {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.detail = detail;
    }
}

export type SearchCase = {
    case_id: string;
    title: string;
    verified_count: number;
    needs_review_count: number;
    total_sentences: number;
};

export type SearchResponse = {
    query: string;
    cases: SearchCase[];
};

export type CaseClaim = {
    sentence_id: string;
    field_name: string;
    sentence_text: string;
    sentence_order: number;
    verdict: 'VERIFIED' | 'NEEDS_REVIEW';
    original_statement: string;
    source_chunk_id: string;
    source_name: string;
    paragraph_num: number | null;
};

export type CaseDetails = {
    case_id: string;
    title: string;
    story: string;
    evidence: string;
    decision_and_meaning: string;
    why_court_decided: string;
    verified_count: number;
    needs_review_count: number;
    blocked_count: number;
    total_sentences: number;
    claims: CaseClaim[];
};

// ─────────────────────────────────────────────────────────
// searchCases — POST /search
// ─────────────────────────────────────────────────────────

export async function searchCases(query: string): Promise<SearchResponse> {
    const response = await fetch(`${BASE_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
    });

    if (!response.ok) {
        let detail = '';
        try {
            const errJson = await response.json();
            if (errJson && errJson.detail) {
                detail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
            }
        } catch {
            const text = await response.text().catch(() => '');
            detail = text.slice(0, 200);
        }
        throw new ApiError(response.status, `[PRAMANA] /search returned HTTP ${response.status}: ${detail}`, detail);
    }

    const data: SearchResponse = await response.json();
    return data;
}

// ─────────────────────────────────────────────────────────
// getCaseDetails — GET /cases/{case_id}
// ─────────────────────────────────────────────────────────

export async function getCaseDetails(caseId: string): Promise<CaseDetails> {
    const response = await fetch(`${BASE_URL}/cases/${encodeURIComponent(caseId)}`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
    });

    if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(
            `[PRAMANA] /cases/${caseId} returned HTTP ${response.status}: ${text.slice(0, 200)}`
        );
    }

    const data: CaseDetails = await response.json();
    return data;
}
