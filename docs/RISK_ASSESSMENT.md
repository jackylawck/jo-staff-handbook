# ⚖️ AI System Risk Assessment Report (`RISK_ASSESSMENT.md`)

## 1. Regulatory Categorization
* **EU AI Act Risk Classification**: **Minimal / Limited Risk System**
  * *Rationale*: The system serves strictly as an informational HR policy navigation tool. It does NOT perform Automated Decision-Making (ADM), automated resume screening, workplace surveillance, or binding employment performance evaluations.
* **ISO/IEC 42001 Alignment**: Annex A.6 (AI Impact Assessment) & Risk Treatment.

## 2. Risk Matrix & Mitigation Controls

| Risk ID | Hazard / Risk Description | Severity | Likelihood | Mitigation Mechanism in `jo-staff-handbook` | Residual Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R-01** | **AI Hallucination**: Generating incorrect leave or compensation rules leading to labor disputes. | High | Low | **Intent Locking & Exact Routing**: Hardcodes domain bindings (e.g., Training Subsidies forcibly route to Chapter 9) and displays prominent notices deferring to official Z-Drive PDFs. | Negligible |
| **R-02** | **Stale Policy Misdirection**: Users relying on outdated manual clauses rather than 2026 notices. | High | Medium | **Top-Level Dynamic Overrides**: Automatically detects query keywords and displays highlighted addendums (e.g., 2026 Attendance) at the top of answers. | Very Low |
| **R-03** | **Sensitive Data Leakage**: Exposing confidential salary thresholds in public source code. | High | Low | **Decoupled Anonymization**: All policy overrides in GitHub repositories are anonymized, reserving full detailed figures for local internal files. | Zero |
