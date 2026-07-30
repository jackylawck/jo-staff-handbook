# 🔒 Security & Data Governance Policy

## 1. Security Overview
The **Jumbo Orient Staff Handbook Assistant (`jo-staff-handbook`)** is designed with privacy-first and data-minimization principles, adhering to **ISO/IEC 27001** and **ISO/IEC 42001** standards.

## 2. Privacy & Data Protection Commitments
* **Zero Data Retention (ZDR)**: Uploaded document files and user chat queries are processed purely in ephemeral session memory. No data is stored permanently or transmitted to external AI training pipelines.
* **Sensitive Data Isolation**: Confidential corporate details are decoupled from the public code repository using anonymized local configurations (`policy_overrides.json`).
* **Session Destruction**: All session memory and vector indexes are completely purged upon closing the browser tab or clicking the "Clear Chat" button.

## 3. Reporting a Vulnerability
If you discover a security vulnerability or data exposure issue within this repository, please report it immediately to:
* **Maintainer**: [Jacky Law](https://jackylawck.github.io/jackylawck/)
* **Official Website**: [Jumbo Orient Construction](https://jumboorient.com.hk/)

Please do not open public GitHub issues for security vulnerabilities.
