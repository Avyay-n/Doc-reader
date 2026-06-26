---
name: general-document-extraction
description: Engineering guidelines and architectural patterns for building 100% general Document AI, OCR, and invoice extraction pipelines without hardcoding filenames or benchmark cheating.
---

# Universal Document AI Extraction Architecture

When designing or modifying document parsing pipelines (PDFs, invoices, receipts, medical bills), follow these mandatory engineering patterns:

## 1. Zero Hardcoded Routing (Generalization)
- **NEVER** use `if "doc1" in filename` or regex tailored specifically to benchmark sample files.
- **Universal Schema:** Map all varying document headers (`Particulars`, `Description`, `Charges`, `Rate`, `NetAmt`) into a standardized universal dictionary schema.

## 2. Sliding Window Table Chunking
- To prevent LLM attention degradation and table row truncation:
  - Count lines per page. If lines > 28, split table bodies into overlapping or adjacent chunks of ~18 lines.
  - Query the LLM per chunk and deduplicate reassembled items using tuple keys.

## 3. Deterministic Math Self-Correction Loop
- Always calculate `extracted_sum = sum(item['Price'] * item['Quantity'])`.
- Parse the raw document footer for `Gross Amount Claimed` or `Total Payable`.
- If `abs(extracted_sum - footer_total) > 5.00`, automatically trigger a targeted reconciliation re-prompt:
  > *"The extracted items total $X but the invoice total is $Y. Find the missing item worth $Z."*

## 4. Physical Ground Truth Hallucination Filter
- Before appending an extracted row to the ledger, verify that its numeric price (`float > 0`) or primary substring (`len > 4`) physically exists in the raw OCR text stream. Discard purely generative hallucinations.

## 5. Hybrid Quota Shielding (Cloud + Local Offline)
- When querying cloud endpoints (`Gemini`, `OpenAI`) in batch loops:
  - Implement sequential pacing intervals (~4.2s) to comply with free-tier requests-per-minute (RPM) limits.
  - Wrap API calls in `HTTP 429` exception handlers that pause 15s before retrying.
  - Implement seamless dynamic fallbacks to local offline CPU hardware (`Ollama`) upon cloud quota exhaustion.
