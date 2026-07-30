# 📊 Data Governance & Lineage Policy (`DATA_GOVERNANCE.md`)

Aligned with **EU AI Act Article 10** and **ISO/IEC 42001 Annex A.7**.

## 1. Data Sources & Inventory
* **Primary Authority**: *Monthly Staff Handbook* (`月薪員工手冊.pdf`) - Approved by Jumbo Orient Management.
* **Secondary Authority**: Dynamic Policy Overrides (`policy_overrides.json`) - Updated for 2026 addendums.

## 2. Data Cleaning & Sanitization Rules
During PDF parsing (`process_pdf_to_chunks`), the pipeline applies strict data quality controls:
* **Noise Suppression**: Removes running headers/footers (e.g., `東淦工程有限公司 員工手冊 XX/51`) using regular expressions to prevent vector distortion.
* **Minimum Length Filter**: Automatically discards text fragments under 50 characters to eliminate empty header artifacts.
* **Chunking Standards**: Standardized to 700 characters with 120-character overlap for optimal semantic coherence.

## 3. Data Privacy & Zero Retention
* **No External Training**: Data parsed in memory is processed locally and NEVER transmitted to third-party public LLM training datasets.
* **Session Disposal**: Memory allocations are automatically purged upon ending the browser session.
