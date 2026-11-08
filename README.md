PRAMĀṆA: Evidence-Aware Legal Retrieval and Claim Verification for Indian Supreme Court Judgments

PRAMĀṆA is a research prototype designed to retrieve relevant Indian Supreme Court judgments, generate citizen-friendly case summaries, and assess whether individual claims in those summaries are supported by source text.

It combines semantic retrieval, source-grounded claim analysis, and Natural Language Inference (NLI).

> Research status:Experimental prototype. The public repository contains implementation code and selected evaluation results. The underlying judgment corpus, vector database, and Supabase schema and records are not published.

---

1. Project Overview

Legal judgments are lengthy and difficult for ordinary citizens to interpret. PRAMĀṆA explores how AI can make relevant judgments easier to find and their contents easier to understand, while providing a mechanism to examine factual claims against source material.

The system focuses on two connected tasks:

1. Legal retrieval: Find relevant Supreme Court judgments for a user's query.
2. Claim verification: Examine generated summary claims against retrieved source text.

PRAMĀṆA is a research prototype, not a legal advice system or a substitute for professional legal counsel.

2. What the System Does

| Component | Function |
|---|---|
| Legal query processing | Interprets a user's legal question |
| Step-Back prompting | Reformulates the query into a broader legal question |
| HyDE | Generates hypothetical document text for retrieval |
| Semantic retrieval | Retrieves relevant judgment chunks |
| Case summarisation | Produces structured, citizen-friendly summaries |
| Gate 0 | Checks key entities and numerical details against source text |
| Gate 1 | Finds the closest matching source sentence |
| Gate 2 | Uses NLI to assess claim-source relationships |

3. System Architecture

Retrieval workflow

1. User submits a legal question.
2. Step-Back processing reformulates the question.
3. HyDE generates hypothetical document text.
4. BGE embeddings represent the query.
5. ChromaDB retrieves relevant judgment chunks.
6. Relevant judgments are returned for further inspection.

Summary and verification workflow

The research pipeline generates structured summaries and extracts individual claims for evaluation.

The summary structure includes:

- Title
- Story
- Evidence
- Decision and Meaning
- Why the Court Decided That Way

Each extracted claim is evaluated against source material through the verification pipeline.

Important: The three verification gates described below are part of the research pipeline. They should not be interpreted as proof that every live user-facing answer is independently passed through all three gates.

4. Dataset Construction

The original dataset was derived from **34,816 Indian Legal Documents Corpus (ILDC) records** relating to Supreme Court judgments.

The curation process involved:

- Six classifier versions, from v2 to v6, to progressively refine domain filtering.
- Removal of out-of-domain records.
- Repair of systematic text corruption, including substitutions such as `company` becoming `con`.
- Manual removal of two out-of-domain records.

The resulting curated Supreme Court corpus contained **538 judgments**.

A separate 100-case Telangana and Andhra Pradesh High Court supplement was curated independently and is not included in the 538-judgment Supreme Court corpus.

### Summary and claim dataset

| Metric | Count |
|---|---:|
| Curated Supreme Court judgments | 538 |
| Generated case summaries | 225 |
| Extracted claims | 3,362 |
| Summary sections | 1,125 |
| Source chunks recorded in Supabase | 1,236 |

The 225 summaries comprised:
- 52 manually prepared reference summaries.
- 87 DeepSeek-generated summaries that were checked and corrected.
- 86 Manus-generated summaries that were checked and corrected.

ChatGPT was also used as an independent quality-checking tool.

## 5. Retrieval Implementation

| Component | Technology / Configuration |
|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embedding dimensions | 384 |
| Vector database | ChromaDB |
| Collection | `pramana_v14` |
| Distance metric | Cosine distance |
| Retrieved results | Top 5 judgment chunks |

### Retrieval evaluation

Evaluation used 25 manually verified gold queries selected from 50 source-derived candidate queries.

| Metric | Result |
|---|---:|
| Case-level Recall@1 | 0.960 |
| Case-level Recall@5 | 1.000 |
| Case-level Recall@10 | 1.000 |
| Chunk-level Recall@1 | 0.960 |
| Chunk-level Recall@5 | 1.000 |
| Chunk-level Recall@10 | 1.000 |
| Chunk-level MRR | 0.980 |

These results describe the evaluated query set and should not be interpreted as guaranteed performance on all legal queries.

### Step-Back and HyDE evaluation

Step-Back and Domain Check were evaluated on 30 queries: 12 using Gemini and 18 using Groq. The reported classification accuracy on this test set was 100%.

HyDE did not improve retrieval performance in the reported experiment:

> **HyDE reduced Recall@5 from 1.000 to 0.840.**

It was retained as an experimental language-bridging component despite this reduction.

## 6. Three-Gate Claim Verification

PRAMĀṆA uses three stages to examine whether generated claims are supported by source material.

### Gate 0 — Entity and Number Match

Checks important factual details against the cited source text, including:

- Dates
- Monetary amounts
- Legal section numbers
- Party names

The reported block rate was 0.65% across 3,362 claims. Two iterations of regex refinement were used to address formatting-related false positives.

### Gate 1 — Cosine Span Match

Initially designed as a blocking gate, Gate 1 was evaluated using a calibration set of 300 claim-source pairs.

Its best reported precision was 59.6% at the tested thresholds. Because this was not sufficiently reliable for a blocking decision, it was redesigned as a **candidate finder**.

Its role is to identify the closest source sentence for further evaluation by Gate 2.

### Gate 2 — Natural Language Inference

Gate 2 uses:

`cross-encoder/nli-deberta-v3-small`

It evaluates the relationship between a claim and its supporting source text using NLI scores.

Reported thresholds:

| NLI outcome | Threshold |
|---|---:|
| Entailment | ≥ 0.35 |
| Contradiction | ≥ 0.50 |

The original Gate 2 targets were:

| Metric | Target |
|---|---:|
| Precision | ≥ 98% |
| Recall | ≥ 90% |
| False Positive Rate | ≤ 2% |
| F1 Score | ≥ 0.93 |

**These targets were not achieved.** The final Gate 2 status was `gate2_passed: false`.

### Aggregate claim outcomes

| Outcome | Claims | Percentage |
|---|---:|---:|
| Verified | 686 | 20.4% |
| Needs Review | 2,155 | 64.1% |
| Blocked | 521 | 15.5% |
| **Total** | **3,362** | **100%** |

These are experimental classification outcomes, not a measurement of legal correctness.

### Calibration limitations

The 300 claim-source pairs were labelled by four AI evaluators: two Claude-based evaluators and two ChatGPT-based evaluators.

No legal expert participated in this labelling process, and no formal inter-annotator agreement statistic was reported.

The calibration results should therefore be interpreted as preliminary experimental evidence, not legal validation.

**Evaluation gap:** I did not measure whether the gates reduce unsupported sentences compared with an ungated summary.

## 7. Implementation Status

| Area | Status |
|---|---|
| Judgment corpus curation | Completed |
| Text cleaning and correction | Completed |
| BGE embeddings | Implemented |
| ChromaDB indexing | Implemented |
| Semantic retrieval evaluation | Completed on the reported test set |
| Step-Back and Domain Check evaluation | Completed on 30 queries |
| HyDE evaluation | Completed; Recall@5 decreased |
| Structured case summaries | 225 prepared |
| Claim extraction | 3,362 claims |
| Gate 0 | Implemented and evaluated |
| Gate 1 | Redesigned as a candidate finder |
| Gate 2 | Implemented; original targets not met |
| Ungated-summary baseline comparison | Not evaluated |
| Public database reproduction | Not available |

## 8. Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| Embeddings | BGE Small English v1.5 |
| Vector database | ChromaDB |
| Relational database | Supabase PostgreSQL |
| LLM integration | Groq |
| Claim verification | DeBERTa-based NLI cross-encoder |
| Language | Python 3.11.9 used during development |

## 9. API Endpoints

The backend includes the following routes:

| Endpoint | Purpose |
|---|---|
| `/health` | Health check |
| `/search` | Legal judgment retrieval |
| `/cases/{id}` | Retrieve case information |

Refer to the backend implementation for request and response schemas.

## 10. Repository Structure

The repository contains the frontend, backend, retrieval and processing modules, evaluation scripts, and supporting configuration files.

Key implementation files include:

- `api.py` — FastAPI application and API routes.
- `step8_stepback.py` — Step-Back processing.
- `step9_hyde.py` — HyDE generation.
- `step10_e2e.py` — Retrieval pipeline.
- `requirements.txt` — Python dependencies.
- `.env.example` — Environment variable template.

The exact repository contents and directory structure are available in the source tree.

## 11. Setup and Configuration

### Requirements

- Python 3.11
- Node.js and npm
- Supabase project for database-backed functionality
- Groq API key for workflows that call Groq

### Environment variables

Copy `.env.example` to `.env` and configure the required values.

| Variable | Purpose |
|---|---|
| `SUPABASE_URL` | Supabase project URL; required for database-backed functionality |
| `SUPABASE_SERVICE_ROLE_KEY` | Server-side Supabase credential; required for database-backed functionality |
| `GROQ_API_KEY` | Required only for workflows that call Groq |

**Security:** Never commit `.env` or expose the Supabase service-role key in frontend code.

### Important reproduction limitation

The public repository does not include the underlying Supabase schema and records, the private judgment corpus, or the ChromaDB index.

Consequently, a fresh clone cannot reproduce the complete research pipeline from the repository alone. Python 3.11.9 was used on the author's development machine; a complete fresh-clone installation and end-to-end reproduction have not been established.

## 12. Research Contributions

The project explores:

1. A curated legal judgment retrieval pipeline for Indian Supreme Court cases.
2. Semantic retrieval using BGE embeddings and ChromaDB.
3. Step-Back and HyDE query-processing experiments.
4. Structured citizen-friendly legal case summaries.
5. Claim-level source comparison using entity matching, cosine similarity, and NLI.
6. Experimental evaluation of verification thresholds and their limitations.

## 13. Limitations

- The curated corpus is limited to the selected judgments and legal domain.
- Retrieval metrics are based on a small manually verified query set.
- HyDE reduced Recall@5 in the reported experiment.
- Gate 1 did not achieve sufficient precision to remain a blocking gate.
- Gate 2 did not meet its original performance targets.
- Calibration labels were generated by AI evaluators rather than legal experts.
- No ungated-summary baseline comparison was conducted.
- The database schema, records, corpus, and vector index are not publicly available.
- Complete reproduction from a fresh clone has not been demonstrated.

## 14. Future Work

- Expand the judgment corpus and legal-domain coverage.
- Improve claim-source alignment.
- Conduct legal-expert annotation and validation.
- Recalibrate and independently evaluate NLI thresholds.
- Compare gated and ungated summaries.
- Evaluate on larger, independently prepared test sets.
- Publish reproducible database and indexing instructions where legally and technically permissible.

## 15. Author

**Author:** K. Gnanadeep

### AI assistance disclosure

AI-based tools were used for selected development activities, including code assistance, debugging, and documentation. The author is responsible for the project implementation, experimental methodology, interpretation of results, and claims made in this repository.

## 16. Demonstration

**Demo video:** `DEMO_VIDEO_URL_TO_BE_ADDED`

The demonstration video will be added as a GitHub Release asset.

## 17. Citation

A citation file, `CITATION.cff`, should be used for repository citation once included in the release.

## 18. Disclaimer

PRAMĀṆA is an experimental research prototype intended for educational and research purposes. It does not provide legal advice, determine legal rights, or replace consultation with a qualified legal professional.

Users should independently verify all judgments, citations, legal provisions, and interpretations against authoritative legal sources.