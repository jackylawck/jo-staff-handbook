# 🔄 AI System Lifecycle & Change Management Plan (`LIFECYCLE_MANAGEMENT.md`)

Aligned with **ISO/IEC 42001 Clause 8 & Annex A.10**.

## 1. System Maintenance Cycle
* **Routine Audit**: Bi-annual review of vector embeddings and retrieval precision.
* **Trigger-Based Update**: Immediate system update upon the release of new circulars or revisions to the *Staff Handbook*.

## 2. Change Management Workflow
1. **Policy Update**: When HR issues a new circular, update `policy_overrides.json` with the effective date and title summary without changing core algorithms.
2. **Handbook Re-Indexing**: When a full new edition of the handbook (e.g., 2027 edition) is published, upload the new PDF via Streamlit; the system automatically builds a fresh FAISS/BM25 index.
3. **Regression Testing**: Execute test queries (e.g., Typhoon arrangements, Training subsidies, Facial recognition) to ensure 100% chapter routing accuracy before wide deployment.
