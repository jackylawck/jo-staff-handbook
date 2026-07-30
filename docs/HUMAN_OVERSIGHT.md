# 👥 Human Oversight & Transparency Protocol (`HUMAN_OVERSIGHT.md`)

Aligned with **EU AI Act Article 14 (Human Oversight)** and **ISO/IEC 42001 Annex A.8**.

## 1. Governance Principle
The AI system is categorized solely as a **Human-in-the-Loop (HITL) Decision-Support Tool**. Under no circumstances shall system outputs constitute binding legal advice or employment decisions without explicit HR review.

## 2. User Transparency & Disclaimers
* **Interface Banner**: Every query output displays a mandatory governance disclaimer:  
  > *"※ Notice: Updated policies take precedence over legacy manual clauses. For full regulatory files, refer to `Z:\Hrd-Public Folder\16.0 人力資源政策及指引`."*

## 3. Human Escalation Protocol (HR Exception Handling)
If an employee identifies a discrepancy or requires binding confirmation regarding AI responses:
1. **Reporting**: The employee contacts the HR Department via internal channels or system maintainer ([Jacky Law](https://jackylawck.github.io/jackylawck/)).
2. **HR Intervention**: An authorized HR officer reviews the official PDF documents on the Z-Drive and provides an official, written ruling.
3. **Feedback Loop**: If the discrepancy stems from an unindexed policy change, HR updates `policy_overrides.json` within 2 business days.
