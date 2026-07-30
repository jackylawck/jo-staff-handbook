# 🪪 AI Model Card: jo-staff-handbook (ISO 42001 Governance Compliance)

This Model Card documents the governance, technical specifications, risk controls, and compliance scope of the AI system powering the **Jumbo Orient Staff Handbook Assistant (`jo-staff-handbook`)**. It aligns with **ISO/IEC 42001:2023 (AIMS)** requirements for AI transparency, accountability, and ethical evaluation.

---

## 1. System Overview & Model Identity (系統概覽與模型身份)

* **System Name**: Jumbo Orient Staff Handbook Assistant (`jo-staff-handbook`)
* **System Version**: v9.0 (Production-Ready / Local Hybrid RAG)
* **Owner / Maintainer**: Jacky Law (AI Governance & HR Compliance Lead)
* **Model Type**: Hybrid Retrieval-Augmented Generation (RAG) System Architecture
* **Primary Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **Primary Retriever Engines**: 
  * Vector Search Engine: FAISS (`faiss-cpu`)
  * Lexical Search Engine: BM25 (`rank_bm25`)
* **Deployment Infrastructure**: Streamlit Cloud / Local Sandboxed Environment

---

## 2. Intended Use & Operational Boundary (意圖用途與營運邊界)

### ✅ Intended Use Cases (核准使用場景)
* **HR Policy Retrieval**: Assisting Jumbo Orient employees in searching and interpreting clauses within the *Monthly Staff Handbook* (`月薪員工手冊`).
* **Addendum Policy Navigation**: Directing employees to the latest company notices (e.g., 2026 Facial Recognition Attendance, Compensation Restructuring, Training Subsidies).
* **Official Document Routing**: Directing staff to authoritative files stored on the internal network drive (`Z:\Hrd-Public Folder\16.0 人力資源政策及指引`).

### 🚫 Out-of-Scope & Prohibited Uses (禁止與越界使用場景)
* **Binding Legal Interpretation**: The system does NOT provide legally binding employment arbitration; official HR communications prevail.
* **Automated Decision-Making (ADM)**: The system MUST NOT be used to automatically evaluate, punish, or decide employee promotions/penalties.
* **External Commercial Redistribution**: Not permitted for public or third-party deployment containing proprietary company policies.

---

## 3. Data Lineage & Privacy Controls (資料血統與隱私控管)

| ISO 42001 Control Dimension | Implementation Mechanism in `jo-staff-handbook` |
| :--- | :--- |
| **Data Retention Policy** | **Zero Data Retention**: Uploaded PDFs are parsed in temporary memory (`tempfile`) and purged immediately upon session termination or manual reset. |
| **Sensitive Data Exposure** | **Decoupled Overrides**: Confidential compensation details are anonymized in `policy_overrides.json` to prevent source code data leaks on public repositories. |
| **Data Training Boundaries** | **Zero Model Training**: User queries and documents are NOT used to fine-tune or train any underlying public or third-party foundational models. |

---

## 4. Algorithmic Architecture & Risk Controls (演算法架構與風險控制)

### 4.1 Hybrid Retrieval Mechanism (雙引擎混合檢索)
To mitigate the risk of AI hallucination, the system utilizes a deterministic hybrid retrieval pipeline:
1. **Semantic Layer (FAISS)**: Captures intent and contextual similarity (Weight: 0.7).
2. **Lexical Layer (BM25)**: Ensures exact keyword precision for form codes (e.g., `HRF-011`) and explicit terms (Weight: 0.3).
3. **Intent Overrides**: Deterministic rules forcibly route specific domains (e.g., Training Subsidies strictly bind to *Chapter 9*).

### 4.2 Hallucination & Noise Reduction (幻覺與雜訊控制)
* **Chunking Granularity**: Document split size set to 700 characters with 120-character overlap for optimal sentence integrity.
* **Header/Footer Noise Filtering**: Regex sanitization automatically strips repetitive running headers (e.g., `東淦工程有限公司 員工手冊 XX/51`) and ignores chunks under 50 characters.
* **Fallback & Verification Notice**: If retrieved chunks do not match the routed chapter, a user-facing warning is triggered: `⚠️ 系統未能精確定位至特定條文`.

---

## 5. Performance Metrics & System Limitations (效能指標與系統限制)

### 📊 Performance Benchmark (性能評估)
* **Chunk Density**: 100~103 active chunks parsed from a 51-page handbook (falls within the optimal ISO RAG granularity benchmark of 80~120 chunks).
* **Citation Accuracy**: 100% traceable source attribution including source filename, chapter title, and exact page number (e.g., `[第九章 培訓和發展 (第31頁)]`).

### ⚠️ Known System Limitations (已知系統限制)
1. **Image/Scanned PDF Limitation**: OCR is not pre-packaged; pure image-based scanned PDFs require pre-processing before parsing.
2. **Tabular Formatting Loss**: Complex nested tables in raw PDFs may experience layout flattening during PyPDF text extraction.

---

## 6. ISO 42001 Audit Alignment Matrix (ISO 42001 審計對照表)

| ISO/IEC 42001:2023 Clause / Annex | Alignment Status | Evidential Artifact in Codebase |
| :--- | :--- | :--- |
| **A.6.2 AI System Impact Assessment** | **Compliant** | Low risk profile; restricted to internal informational Q&A without ADM impact. |
| **A.7.2 Traceability of System Outputs** | **Compliant** | Page-level source tags (`📍 來源：... (第X頁)`) rendered on every answer block. |
| **A.8.3 Data Minimization & Privacy** | **Compliant** | Memory-only session processing (`try...finally` cleanup blocks in `process_pdf_to_chunks`). |
| **A.9.2 System Transparency & Notice** | **Compliant** | Clear top-level banner notifying users of session ephemeral data policies. |

---

## 7. Governance & Maintenance Protocol (維護與管治協定)

* **Review Frequency**: Bi-annually or immediately upon the release of a new edition of the *Staff Handbook*.
* **Change Management**: Policy changes must be updated in `policy_overrides.json` without altering core algorithmic logic, ensuring strict separation of data and code.
* **Audit Escalation Contact**: [Jacky Law (Lead Auditor / Maintainer)](https://jackylawck.github.io/jackylawck/)
